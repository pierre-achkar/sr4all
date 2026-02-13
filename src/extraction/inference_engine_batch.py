"""
Inference Engine for Systematic Review Information Extraction (Batched).

This module provides the QwenInference class, a wrapper around vLLM optimized for
Qwen 3 models running on H100 hardware. It handles:
1. Long-Context Optimization (YaRN + FP8 Cache).
2. Dynamic Configuration Patching (to support older vLLM versions).
3. Structured JSON Generation (enforcing Pydantic schemas).
4. Continuous Batching (High Throughput).

Usage:
    engine = QwenInference("Qwen/Qwen3-32B")
    results = engine.generate_batch([doc_text_1, doc_text_2, ...])
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoConfig

os.environ["VLLM_USE_V1"] = "0"

# 1. Native vLLM Imports
from vllm import LLM, SamplingParams

# 2. API Detection (Handle both Old and New vLLM)
try:
    from vllm.sampling_params import StructuredOutputsParams

    HAS_NEW_API = True
except ImportError:
    HAS_NEW_API = False

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import Project Assets
from extraction.schema import ReviewExtraction
from extraction.prompts import SYSTEM_PROMPT, USER_TEMPLATE_RAW

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("InferenceEngine")


# -----------------------------------------------------------------------------
# HELPER: Config Patcher
# -----------------------------------------------------------------------------
def ensure_yarn_config(model_path: str):
    """
    Patches the local HuggingFace `config.json` to enable YaRN (RoPE Scaling).
    This enables context windows >32k on Qwen models.
    """
    logger.info(f"Checking config for {model_path}...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        current_rope = getattr(config, "rope_scaling", None)

        target_rope = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }

        needs_patch = True
        if current_rope and isinstance(current_rope, dict):
            if (
                current_rope.get("rope_type") == "yarn"
                and current_rope.get("factor") == 4.0
            ):
                needs_patch = False

        if needs_patch:
            logger.info("Patching config.json to enable YaRN (RoPE Scaling)...")
            config.rope_scaling = target_rope

            from transformers.utils.hub import cached_file

            config_file = cached_file(model_path, "config.json")

            if config_file:
                with open(config_file, "w") as f:
                    f.write(config.to_json_string())
                logger.info("Successfully patched config.json in cache.")
            else:
                logger.warning(
                    "Could not locate config.json file on disk. YaRN might fail."
                )
        else:
            logger.info("Config already has YaRN enabled. Skipping patch.")

    except Exception as e:
        logger.error(f"Failed to patch config: {e}")


# -----------------------------------------------------------------------------
# INFERENCE CLASS
# -----------------------------------------------------------------------------
class QwenInference:
    """
    A robust inference engine for extracting structured data using Qwen models.
    Optimized for high-throughput batch processing.
    """

    def __init__(self, model_path: str, tensor_parallel: int = 2):
        """
        Initializes Native vLLM with H100 optimizations.
        """
        # Disable V1 engine for stability
        os.environ["VLLM_USE_V1"] = "0"

        # 1. PATCH CONFIG FIRST
        ensure_yarn_config(model_path)

        logger.info(f"Loading Qwen (Native) from {model_path}...")

        # 2. Initialize Engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            # --- Memory & Context Settings ---
            max_model_len=131072,  # Force 128k Context
            gpu_memory_utilization=0.90,  # Aggressive memory usage
            kv_cache_dtype="fp8",  # FP8 Cache reduces VRAM usage
            dtype="bfloat16",  # Native weights
            trust_remote_code=True,
            enforce_eager=False,
        )

        # 3. Initialize Tokenizer & Schema
        self.tokenizer = self.llm.get_tokenizer()
        self.json_schema = ReviewExtraction.model_json_schema()

        # 4. Prepare Sampling Params (Once)
        if HAS_NEW_API:
            # Modern vLLM (v0.6+)
            structured_params = StructuredOutputsParams(json=self.json_schema)
            self.sampling_params = SamplingParams(
                temperature=0.1, max_tokens=16384, structured_outputs=structured_params
            )
        else:
            # Legacy vLLM (< v0.6)
            self.sampling_params = SamplingParams(
                temperature=0.1, max_tokens=16384, guided_json=self.json_schema
            )

        api_status = "New StructuredOutputs" if HAS_NEW_API else "Legacy GuidedJSON"
        logger.info(f"Inference Engine Ready ({api_status}).")

    def generate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generates structured JSON extractions for a BATCH of documents.

        Args:
            texts (List[str]): List of document texts.

        Returns:
            List[Dict]: A list of result objects, one per input text:
            {
                "parsed": Dict or None,
                "raw": str,
                "error": str or None
            }
        """
        if not texts:
            return []

        # 1. Prepare Batch Prompts (CPU side)
        prompts = []
        for text in texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE_RAW.replace("{TEXT}", text)},
            ]

            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(full_prompt)

        # 2. Run Batch Inference (GPU side)
        # vLLM handles the continuous batching internally.
        try:
            # use_tqdm=False to keep logs clean in batch jobs
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        except Exception as e:
            logger.critical(f"Batch Generation Failed: {e}")
            # Fail safe: return error for all
            return [{"parsed": None, "raw": "", "error": str(e)} for _ in texts]

        # 3. Process Results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text

            result_entry = {"parsed": None, "raw": generated_text, "error": None}

            try:
                # Parse JSON
                result_entry["parsed"] = json.loads(generated_text)
            except json.JSONDecodeError as e:
                result_entry["error"] = f"JSON_PARSE_ERROR: {str(e)}"
            except Exception as e:
                result_entry["error"] = f"UNKNOWN_ERROR: {str(e)}"

            results.append(result_entry)

        return results

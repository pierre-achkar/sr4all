"""
Inference Engine for Systematic Review Information Extraction (Batched).

This module provides the QwenInference class, a wrapper around vLLM optimized for
Qwen 3 models running on H100 hardware. It handles:
1. Long-Context Optimization (YaRN via vLLM hf_overrides).
2. Structured output generation following the 2a2s LanguageEngine pattern.
3. Structured JSON Generation (enforcing Pydantic schemas).
4. Continuous Batching (High Throughput).

Usage:
    engine = QwenInference("Qwen/Qwen3.5-27B")
    results = engine.generate_batch([doc_text_1, doc_text_2, ...])
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
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


DEFAULT_VLLM_CONFIG = {
    "model_path": "Qwen/Qwen3.5-27B",
    "max_model_len": None,
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90,
    "dtype": "bfloat16",
    "kv_cache_dtype": "auto",
    "enforce_eager": True,
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 16384,
    "yarn_rope_scaling": True,
    "yarn_factor": 4.0,
    "native_ctx_length": None,
    "enable_thinking": False,
}


# -----------------------------------------------------------------------------
# INFERENCE CLASS
# -----------------------------------------------------------------------------
class QwenInference:
    """
    A robust inference engine for extracting structured data using Qwen models.
    Optimized for high-throughput batch processing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tensor_parallel: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes Native vLLM with H100 optimizations.
        """
        # Disable V1 engine for stability
        os.environ["VLLM_USE_V1"] = "0"

        self.config = dict(DEFAULT_VLLM_CONFIG)
        if config:
            self.config.update(config)
        if model_path is not None:
            self.config["model_path"] = model_path
        if tensor_parallel is not None:
            self.config["tensor_parallel_size"] = tensor_parallel

        logger.info(f"Loading tokenizer for {self.config['model_path']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"], trust_remote_code=True
        )

        model_org_config = AutoConfig.from_pretrained(
            self.config["model_path"], trust_remote_code=True
        )
        logger.info(f"Loaded model config for {self.config['model_path']}.")

        if self.config["max_model_len"] is None:
            if hasattr(self.tokenizer, "model_max_length"):
                self.config["max_model_len"] = self.tokenizer.model_max_length
            else:
                raise ValueError(
                    "max_model_len must be specified if tokenizer.model_max_length is missing."
                )

        rope_scaling = getattr(model_org_config, "rope_scaling", None)
        use_yarn = bool(self.config["yarn_rope_scaling"] and rope_scaling is None)

        llm_kwargs = {
            "model": self.config["model_path"],
            "tensor_parallel_size": self.config["tensor_parallel_size"],
            "gpu_memory_utilization": self.config["gpu_memory_utilization"],
            "max_model_len": self.config["max_model_len"],
            "dtype": self.config["dtype"],
            "kv_cache_dtype": self.config["kv_cache_dtype"],
            "trust_remote_code": True,
            "enforce_eager": self.config["enforce_eager"],
        }

        if use_yarn:
            hf_overrides, new_max_len = self._construct_yarn_config(model_org_config)
            llm_kwargs["hf_overrides"] = hf_overrides
            llm_kwargs["max_model_len"] = new_max_len
            logger.info(f"Initializing vLLM with YaRN max_model_len={new_max_len}.")
        else:
            logger.info(
                f"Initializing vLLM without YaRN override, max_model_len={self.config['max_model_len']}."
            )

        # Keep the `llm` attribute for existing callers such as 4_repair.py.
        self.llm = LLM(**llm_kwargs)
        self.model = self.llm

        # 3. Initialize Schema
        self.json_schema = ReviewExtraction.model_json_schema()

        # 4. Prepare Sampling Params (Once)
        self.base_sampling_params = SamplingParams(
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_tokens"],
        )

        if HAS_NEW_API:
            # Modern vLLM (v0.6+)
            structured_params = StructuredOutputsParams(json=self.json_schema)
            self.sampling_params = SamplingParams(
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                max_tokens=self.config["max_tokens"],
                structured_outputs=structured_params,
            )
        else:
            # Legacy vLLM (< v0.6)
            self.sampling_params = SamplingParams(
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                max_tokens=self.config["max_tokens"],
                guided_json=self.json_schema,
            )

        api_status = "New StructuredOutputs" if HAS_NEW_API else "Legacy GuidedJSON"
        logger.info(f"Inference Engine Ready ({api_status}).")

    def _construct_yarn_config(self, model_config):
        """Construct YaRN rope scaling config without mutating the HF cache."""
        if self.config.get("native_ctx_length") is not None:
            original_max_position_embeddings = self.config["native_ctx_length"]
        else:
            max_pos = model_config.max_position_embeddings
            thinking_buffer = 8192 if "Qwen3" in self.config["model_path"] else 0
            original_max_position_embeddings = max_pos - thinking_buffer

        factor = self.config.get("yarn_factor", 4.0)
        hf_overrides = {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_position_embeddings,
            }
        }
        new_max_len = int(original_max_position_embeddings * factor)
        logger.info(f"Constructed YaRN config: {hf_overrides}")
        return hf_overrides, new_max_len

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

        prompts = [
            self.build_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_TEMPLATE_RAW.replace("{TEXT}", text),
            )
            for text in texts
        ]

        return self.generate_prompt_batch(prompts)

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config["enable_thinking"],
            continue_final_message=False,
        )

    def generate_prompt_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Run schema-constrained generation for already-rendered chat prompts."""
        if not prompts:
            return []

        # 2. Run Batch Inference (GPU side)
        # vLLM handles the continuous batching internally.
        try:
            # use_tqdm=False to keep logs clean in batch jobs
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        except Exception as e:
            logger.critical(f"Batch Generation Failed: {e}")
            # Fail safe: return error for all
            return [
                {"parsed": None, "raw": "", "error": f"GENERATION_ERROR: {e}"}
                for _ in prompts
            ]

        # 3. Process Results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text

            result_entry = {
                "parsed": None,
                "raw": generated_text,
                "error": None,
                "token_metadata": {
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                },
            }

            try:
                parsed_model = ReviewExtraction.model_validate_json(generated_text)
                result_entry["parsed"] = parsed_model.model_dump(mode="json")
            except json.JSONDecodeError as e:
                result_entry["error"] = f"JSON_PARSE_ERROR: {str(e)}"
            except Exception as e:
                result_entry["error"] = f"SCHEMA_VALIDATION_ERROR: {str(e)}"

            results.append(result_entry)

        return results

"""
Inference Engine for OpenAlex Boolean Query Normalization (Batched).

This module provides the QwenInference class, a wrapper around vLLM optimized for
Qwen 3 models running on H100 hardware. It handles:
1. Standard context settings (<=32k).
2. Structured JSON Generation (enforcing Pydantic schemas).
3. Continuous Batching (High Throughput).

Usage:
    engine = QwenInference("Qwen/Qwen3-32B")
    results = engine.generate_batch([input_1, input_2, ...])
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

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
from oax.io_llm import LLMInput, LLMOutput
from oax.prompts import TransformerToOAXPrompts

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("OAXInferenceEngine")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _ANALYSIS_RE.sub("", text)
    return text.strip()


def _extract_json_candidate(text: str) -> str:
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


# -----------------------------------------------------------------------------
# INFERENCE CLASS
# -----------------------------------------------------------------------------
class QwenInference:
    """
    A robust inference engine for structured outputs using Qwen models.
    Optimized for high-throughput batch processing.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel: int = 2,
        structured_outputs: bool = True,
        enable_thinking: bool = False,
    ):
        """
        Initializes Native vLLM with H100 optimizations.
        """
        # Disable V1 engine for stability
        os.environ["VLLM_USE_V1"] = "0"

        logger.info(f"Loading Qwen (Native) from {model_path}...")

        # Initialize Engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,

            # --- Memory & Context Settings ---
            max_model_len=16000,            # Standard 32k Context
            gpu_memory_utilization=0.90,    # Aggressive memory usage
            kv_cache_dtype="fp8",           # FP8 Cache reduces VRAM usage for H100, for A100 use "auto"
            dtype="bfloat16",               # Native weights

            trust_remote_code=True,
            enforce_eager=False
        )

        # Initialize Tokenizer & Schema
        self.tokenizer = self.llm.get_tokenizer()
        self.json_schema = LLMOutput.model_json_schema()

        self.enable_thinking = enable_thinking
        self.structured_outputs = structured_outputs

        # Prepare Sampling Params (Once)
        if self.structured_outputs:
            if HAS_NEW_API:
                structured_params = StructuredOutputsParams(json=self.json_schema)
                self.sampling_params = SamplingParams(
                    temperature=0.1,
                    max_tokens=8000,
                    structured_outputs=structured_params
                )
            else:
                self.sampling_params = SamplingParams(
                    temperature=0.1,
                    max_tokens=8000,
                    guided_json=self.json_schema
                )
        else:
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=8000
            )

        if self.structured_outputs:
            api_status = "New StructuredOutputs" if HAS_NEW_API else "Legacy GuidedJSON"
        else:
            api_status = "Unstructured (thinking enabled)"
        logger.info(f"OAX Inference Engine Ready ({api_status}).")

    def generate_batch(self, inputs: List[LLMInput]) -> List[Dict[str, Any]]:
        """
        Generates structured JSON outputs for a BATCH of inputs.

        Args:
            inputs (List[LLMInput]): List of input objects to render as prompts.

        Returns:
            List[Dict]: A list of result objects, one per input prompt:
            {
                "parsed": Dict or None,
                "raw": str,
                "error": str or None
            }
        """
        if not inputs:
            return []

        # Prepare Batch Prompts (CPU side)
        prompts = []
        for item in inputs:
            system_prompt, user_prompt = TransformerToOAXPrompts.render(item)
            full_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            prompts.append(full_prompt)

        # Run Batch Inference (GPU side)
        try:
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        except Exception as e:
            logger.critical(f"Batch Generation Failed: {e}")
            return [{"parsed": None, "raw": "", "error": str(e)} for _ in prompts]

        # Process Results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            cleaned_text = _strip_thinking(generated_text)
            json_text = _extract_json_candidate(cleaned_text)

            result_entry = {
                "parsed": None,
                "raw": generated_text,
                "error": None
            }

            try:
                result_entry["parsed"] = json.loads(json_text)
            except json.JSONDecodeError as e:
                result_entry["error"] = f"JSON_PARSE_ERROR: {str(e)}"
            except Exception as e:
                result_entry["error"] = f"UNKNOWN_ERROR: {str(e)}"

            results.append(result_entry)

        return results

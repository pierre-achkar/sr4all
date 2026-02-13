"""
Inference Engine for Qwen Models (Batched & Flexible).

This module provides the QwenInference class, a wrapper around vLLM optimized for
H100 hardware. It is now schema-agnostic and prompt-agnostic.

Usage:
    # 1. Initialize with your target Pydantic Schema
    from oax.schemas import TransformationOutput
    engine = QwenInference("Qwen/Qwen3-32B", response_model=TransformationOutput)

    # 2. Pass pre-rendered prompt tuples (System, User)
    results = engine.generate_batch([
        ("System Prompt...", "User Prompt 1..."),
        ("System Prompt...", "User Prompt 2..."),
    ])
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Type, Optional

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

# Import Base Pydantic Model for type hinting
from pydantic import BaseModel, ValidationError

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
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
        return text[start : end + 1]
    return text


# -----------------------------------------------------------------------------
# INFERENCE CLASS
# -----------------------------------------------------------------------------
class QwenInference:
    """
    A robust inference engine for structured outputs using Qwen models.
    """

    def __init__(
        self,
        model_path: str,
        response_model: Type[BaseModel] = None,  # NOW REQUIRED/FLEXIBLE
        tensor_parallel: int = 2,
        structured_outputs: bool = True,
        enable_thinking: bool = False,
    ):
        """
        Args:
            model_path: Path to HF model.
            response_model: The Pydantic class to enforce structure (e.g. TransformationOutput).
            tensor_parallel: Number of GPUs.
        """
        # Disable V1 engine for stability
        os.environ["VLLM_USE_V1"] = "0"

        logger.info(f"Loading Qwen (Native) from {model_path}...")

        self.response_model = response_model
        if structured_outputs and not self.response_model:
            raise ValueError(
                "You must provide a 'response_model' class if structured_outputs=True"
            )

        # Initialize Engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            # --- Memory & Context Settings ---
            max_model_len=20000,  # Standard 20k Context
            gpu_memory_utilization=0.90,  # Aggressive memory usage
            kv_cache_dtype="auto",  # fp8 Cache for H100 or auto fp16 for A100
            dtype="bfloat16",  # Native weights
            trust_remote_code=True,
            enforce_eager=False,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.enable_thinking = enable_thinking
        self.structured_outputs = structured_outputs

        # Prepare Sampling Params
        self.sampling_params = self._build_sampling_params()

        logger.info("OAX Inference Engine Ready.")

    def _build_sampling_params(self) -> SamplingParams:
        if not self.structured_outputs:
            return SamplingParams(temperature=0.1, max_tokens=8000)

        # Generate JSON Schema from the passed Pydantic model
        json_schema = self.response_model.model_json_schema()

        if HAS_NEW_API:
            structured_params = StructuredOutputsParams(json=json_schema)
            return SamplingParams(
                temperature=0.1, max_tokens=8000, structured_outputs=structured_params
            )
        else:
            return SamplingParams(
                temperature=0.1, max_tokens=8000, guided_json=json_schema
            )

    def generate_batch(self, prompts: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Generates structured JSON outputs for a BATCH of (System, User) tuples.

        Args:
            prompts: List of (system_prompt, user_prompt) tuples.

        Returns:
            List of dicts: {"parsed": Dict, "raw": str, "error": str}
        """
        if not prompts:
            return []

        # 1. Apply Chat Template (CPU side)
        formatted_prompts = []
        for sys_txt, user_txt in prompts:
            full_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_txt},
                    {"role": "user", "content": user_txt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            formatted_prompts.append(full_prompt)

        # 2. Run Batch Inference (GPU side)
        try:
            outputs = self.llm.generate(
                formatted_prompts, self.sampling_params, use_tqdm=False
            )
        except Exception as e:
            logger.critical(f"Batch Generation Failed: {e}")
            return [{"parsed": None, "raw": "", "error": str(e)} for _ in prompts]

        # 3. Process & Validate Results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            cleaned_text = _strip_thinking(generated_text)

            # Extract JSON substring if model output contains extra text
            json_text = _extract_json_candidate(cleaned_text)

            result_entry = {"parsed": None, "raw": generated_text, "error": None}

            try:
                # 1. Parse JSON
                obj = json.loads(json_text)

                # 2. Pydantic Validation (using the injected model class)
                if self.response_model:
                    validated = self.response_model.model_validate(obj)
                    result_entry["parsed"] = validated.model_dump(by_alias=True)
                else:
                    result_entry["parsed"] = obj  # No validation if unstructured

            except ValidationError as e:
                result_entry["error"] = f"SCHEMA_VALIDATION_ERROR: {str(e)}"
            except json.JSONDecodeError as e:
                result_entry["error"] = f"JSON_PARSE_ERROR: {str(e)}"
            except Exception as e:
                result_entry["error"] = f"UNKNOWN_ERROR: {str(e)}"

            results.append(result_entry)

        return results

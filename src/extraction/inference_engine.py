"""
Inference Engine for Systematic Review Information Extraction.

This module provides the QwenInference class, a wrapper around vLLM optimized for 
Qwen 3 models running on H100 hardware. It handles:
1. Long-Context Optimization (YaRN + FP8 Cache).
2. Dynamic Configuration Patching (to support older vLLM versions).
3. Structured JSON Generation (enforcing Pydantic schemas).
4. API Compatibility (auto-detecting legacy vs. modern vLLM APIs).

Usage:
    engine = QwenInference("Qwen/Qwen3-32B")
    result_json, raw_text = engine.generate(document_text)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoConfig

# 1. Native vLLM Imports
from vllm import LLM, SamplingParams

# 2. API Detection (Handle both Old and New vLLM)
# vLLM v0.6+ introduced 'StructuredOutputsParams'. Older versions use 'guided_json'.
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("InferenceEngine")

# -----------------------------------------------------------------------------
# HELPER: Config Patcher
# -----------------------------------------------------------------------------
def ensure_yarn_config(model_path: str):
    """
    Patches the local HuggingFace `config.json` to enable YaRN (RoPE Scaling) if missing.
    
    This is a critical workaround for vLLM versions that do not support passing 
    `rope_scaling` directly to the `LLM()` constructor via `EngineArgs`.
    
    Args:
        model_path (str): The path or HuggingFace ID of the model.
    """
    logger.info(f"Checking config for {model_path}...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        current_rope = getattr(config, "rope_scaling", None)
        
        # Target Configuration:
        # YaRN Factor 4.0 is recommended for extending context to 128k.
        target_rope = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }

        # Check if patch is needed
        needs_patch = True
        if current_rope and isinstance(current_rope, dict):
            if current_rope.get("rope_type") == "yarn" and current_rope.get("factor") == 4.0:
                needs_patch = False
        
        if needs_patch:
            logger.info("Patching config.json to enable YaRN (RoPE Scaling)...")
            config.rope_scaling = target_rope
            
            # Locate the cached file and overwrite it
            from transformers.utils.hub import cached_file
            config_file = cached_file(model_path, "config.json")
            
            if config_file:
                with open(config_file, "w") as f:
                    f.write(config.to_json_string())
                logger.info("Successfully patched config.json in cache.")
            else:
                logger.warning("Could not locate config.json file on disk. YaRN might fail.")
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
    """

    def __init__(self, model_path: str, tensor_parallel: int = 2):
        """
        Initializes Native vLLM with H100 optimizations.

        Args:
            model_path (str): Path to the Qwen model.
            tensor_parallel (int): Number of GPUs to shard the model across (default: 2).
        """
        # Disable V1 engine for stability with newer models if needed
        os.environ["VLLM_USE_V1"] = "0"
        
        # 1. PATCH CONFIG FIRST
        # Solves 'unexpected keyword argument rope_scaling' in older vLLM versions
        ensure_yarn_config(model_path)
        
        logger.info(f"Loading Qwen (Native) from {model_path}...")
        
        # 2. Initialize Engine
        # CRITICAL: Do NOT pass 'rope_scaling' here. The patch above handles it.
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            
            # --- Memory & Context Settings ---
            max_model_len=131072,           # Force 128k Context
            gpu_memory_utilization=0.90,    # Aggressive memory usage (dedicated node)
            kv_cache_dtype="fp8",           # FP8 Cache reduces VRAM usage by ~50%
            dtype="bfloat16",               # Native weights
            
            trust_remote_code=True,
            enforce_eager=False
        )
        
        # 3. Initialize Tokenizer & Schema
        self.tokenizer = self.llm.get_tokenizer()
        # Pre-compute the JSON schema to fail early if the Pydantic model is invalid
        self.json_schema = ReviewExtraction.model_json_schema()
        
        api_status = "New StructuredOutputs" if HAS_NEW_API else "Legacy GuidedJSON"
        logger.info(f"Inference Engine Ready ({api_status}).")

    def generate(self, text: str, history: List[Dict] = None) -> Tuple[Optional[Dict], str]:
        """
        Generates structured JSON extraction from the input text.

        Args:
            text (str): The document text (Systematic Review content).
            history (List[Dict], optional): Previous conversation turns (for feedback loops).

        Returns:
            Tuple[Optional[Dict], str]: A tuple containing:
                - The parsed Python dictionary (or None if parsing failed).
                - The raw string output from the model.
        """
        # 1. Format Prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE_RAW.replace("{TEXT}", text)}
        ]
        
        if history:
            messages.extend(history)
            
        full_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False # Hard disable Qwen3 "Thinking Mode" to prevent schema violations
        )
        
        # 2. Configure Sampling (Auto-adapts to your vLLM version)
        # We enforce low temperature for factual extraction stability.
        if HAS_NEW_API:
            # Modern vLLM (v0.6+)
            structured_params = StructuredOutputsParams(json=self.json_schema)
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=8192,
                structured_outputs=structured_params
            )
        else:
            # Legacy vLLM (< v0.6)
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=8192,
                guided_json=self.json_schema
            )
        
        try:
            # 3. Generate
            outputs = self.llm.generate([full_prompt], sampling_params, use_tqdm=False)
            output_text = outputs[0].outputs[0].text
            
            # 4. Parse
            parsed_json = json.loads(output_text)
            return parsed_json, output_text
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {e}")
            return None, output_text
        except Exception as e:
            logger.error(f"Generation Error: {e}")
            return None, str(e)

# -----------------------------------------------------------------------------
# TEST HARNESS
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     TEST_TEXT = "We searched PubMed. Found 10 studies."
#     MODEL = "Qwen/Qwen3-32B" 
    
#     print("TESTING QwenInference (Universal Patch + Native)...")
#     try:
#         engine = QwenInference(MODEL, tensor_parallel=2)
#         res, raw = engine.generate(TEST_TEXT)
#         print(json.dumps(res, indent=2))
#         print("✅ SUCCESS")
#     except Exception as e:
#         print(f"❌ FAIL: {e}")
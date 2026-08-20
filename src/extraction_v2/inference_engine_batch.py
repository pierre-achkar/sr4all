"""
Inference engine for systematic review information extraction (batched).

Wraps vLLM for Qwen3.6-27B. Responsibilities:
1. Schema-constrained JSON generation (schema.py is the binding contract).
2. Chat-template rendering with thinking disabled, verified at startup.
3. A reproducibility fingerprint stamped into every run.

Qwen3.6-27B has a 262144-token native window, so YaRN is OFF by default: static
YaRN degrades accuracy on sequences well below the extended window, which is
every systematic review in the corpus.

Usage:
    engine = QwenInference("Qwen/Qwen3.6-27B")
    results = engine.generate_batch([doc_text_1, doc_text_2, ...])
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

try:  # vLLM >= 0.11 renamed the structured-output API
    from vllm.sampling_params import StructuredOutputsParams

    HAS_NEW_API = True
except ImportError:
    HAS_NEW_API = False

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction_v2.prompts import SYSTEM_PROMPT, USER_TEMPLATE_RAW
from extraction_v2.schema import ReviewExtraction

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("InferenceEngine")


DEFAULT_VLLM_CONFIG = {
    "model_path": "Qwen/Qwen3.6-27B",
    # Explicit. Never derived from tokenizer.model_max_length, which is often a
    # sentinel value rather than a real context length.
    "max_model_len": 262144,
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90,
    "dtype": "bfloat16",
    "kv_cache_dtype": "auto",
    "enforce_eager": True,
    "enable_prefix_caching": True,
    "seed": 0,
    # Greedy. Any sampling noise costs verbatim-span fidelity for nothing.
    "temperature": 0.0,
    "top_p": 1.0,
    # Long criteria and query lists can exceed 4K tokens; reserve 16K tokens
    # while leaving most of the native context window for the source document.
    "max_tokens": 16384,
    "enable_thinking": False,
    # Fail at startup rather than silently generating reasoning for hours.
    "strict_thinking_check": True,
    # Only used if max_model_len exceeds the model's native window.
    "allow_yarn": False,
}


class QwenInference:
    """High-throughput structured extraction over vLLM."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        tensor_parallel: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = dict(DEFAULT_VLLM_CONFIG)
        if config:
            self.config.update(config)
        if model_path is not None:
            self.config["model_path"] = model_path
        if tensor_parallel is not None:
            self.config["tensor_parallel_size"] = tensor_parallel

        mp = self.config["model_path"]
        logger.info(f"Loading tokenizer and config for {mp}...")
        self.tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        model_org_config = AutoConfig.from_pretrained(mp, trust_remote_code=True)

        self.json_schema = ReviewExtraction.model_json_schema()

        llm_kwargs = {
            "model": mp,
            "tensor_parallel_size": self.config["tensor_parallel_size"],
            "gpu_memory_utilization": self.config["gpu_memory_utilization"],
            "max_model_len": self.config["max_model_len"],
            "dtype": self.config["dtype"],
            "kv_cache_dtype": self.config["kv_cache_dtype"],
            "enforce_eager": self.config["enforce_eager"],
            "enable_prefix_caching": self.config["enable_prefix_caching"],
            "seed": self.config["seed"],
            "trust_remote_code": True,
        }

        text_config = getattr(model_org_config, "text_config", model_org_config)
        native = getattr(text_config, "max_position_embeddings", None)
        if native and self.config["max_model_len"] > native:
            if not self.config["allow_yarn"]:
                raise ValueError(
                    f"max_model_len={self.config['max_model_len']} exceeds the native "
                    f"window ({native}). Lower it, or set allow_yarn=True and accept "
                    "the short-sequence accuracy cost."
                )
            llm_kwargs["hf_overrides"] = self._yarn_overrides(native)
            logger.warning(
                "YaRN enabled: expect degraded accuracy on documents far shorter "
                "than the extended window."
            )
        else:
            logger.info(
                f"No rope scaling. max_model_len={self.config['max_model_len']} "
                f"(native {native})."
            )

        self.llm = LLM(**llm_kwargs)
        self.model = self.llm  # legacy attribute for 4_repair.py

        self.sampling_params = self._build_sampling_params()
        self._verify_thinking_disabled()

        # Prompt tokens with an empty document: lets the caller pre-filter
        # over-length documents without tokenizing every one of them.
        self.prompt_overhead_tokens = len(
            self.tokenizer(self.build_prompt(SYSTEM_PROMPT, USER_TEMPLATE_RAW.replace("{TEXT}", "")))["input_ids"]
        )
        self.max_doc_tokens = (
            self.config["max_model_len"]
            - self.prompt_overhead_tokens
            - self.config["max_tokens"]
        )

        api = "StructuredOutputs" if HAS_NEW_API else "guided_json"
        logger.info(
            f"Engine ready ({api}). prompt_overhead={self.prompt_overhead_tokens} "
            f"max_doc_tokens={self.max_doc_tokens} fingerprint={self.fingerprint()['prompt_hash']}"
        )

    # ------------------------------------------------------------------ setup
    def _yarn_overrides(self, native: int) -> Dict[str, Any]:
        factor = self.config["max_model_len"] / native
        return {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": native,
            }
        }

    def _build_sampling_params(self) -> SamplingParams:
        common = dict(
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_tokens"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        if HAS_NEW_API:
            return SamplingParams(
                **common,
                structured_outputs=StructuredOutputsParams(json=self.json_schema),
            )
        return SamplingParams(**common, guided_json=self.json_schema)

    def _verify_thinking_disabled(self) -> None:
        """
        apply_chat_template ignores unknown kwargs silently, so enable_thinking=False
        can become a no-op if the template renames it. Qwen3 templates emit an empty
        <think></think> pair when thinking is off.
        """
        rendered = self.build_prompt("probe", "probe")
        ok = "<think>" in rendered and "</think>" in rendered
        self._thinking_markers_present = ok
        if self.config["enable_thinking"]:
            return
        if not ok:
            msg = (
                "enable_thinking=False produced no <think></think> markers. The chat "
                "template may have changed; thinking may be ACTIVE, which will break "
                "schema validation on every document."
            )
            if self.config["strict_thinking_check"]:
                raise RuntimeError(msg + " Set strict_thinking_check=False to override.")
            logger.error(msg)

    def fingerprint(self) -> Dict[str, Any]:
        """Everything needed to reproduce or compare a run. Stamp this into outputs."""
        h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        return {
            "model_path": self.config["model_path"],
            "prompt_hash": h(SYSTEM_PROMPT + USER_TEMPLATE_RAW),
            "schema_hash": h(json.dumps(self.json_schema, sort_keys=True)),
            "structured_outputs_api": "new" if HAS_NEW_API else "legacy",
            "max_model_len": self.config["max_model_len"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "enable_thinking": self.config["enable_thinking"],
            "thinking_markers_present": self._thinking_markers_present,
            "prompt_overhead_tokens": getattr(self, "prompt_overhead_tokens", None),
        }

    # ------------------------------------------------------------- generation
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
        )

    def generate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        prompts = [
            self.build_prompt(SYSTEM_PROMPT, USER_TEMPLATE_RAW.replace("{TEXT}", t))
            for t in texts
        ]
        return self.generate_prompt_batch(prompts)

    def truncate_document(self, text: str) -> tuple[str, bool]:
        """Fit a document into the rendered prompt while reserving completion tokens."""
        prompt = self.build_prompt(SYSTEM_PROMPT, USER_TEMPLATE_RAW.replace("{TEXT}", text))
        prompt_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if prompt_tokens + self.config["max_tokens"] <= self.config["max_model_len"]:
            return text, False

        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        lower, upper, best = 0, len(token_ids), 0
        while lower <= upper:
            midpoint = (lower + upper) // 2
            candidate = self.tokenizer.decode(token_ids[:midpoint], skip_special_tokens=True)
            candidate_prompt = self.build_prompt(
                SYSTEM_PROMPT, USER_TEMPLATE_RAW.replace("{TEXT}", candidate)
            )
            candidate_tokens = len(
                self.tokenizer(candidate_prompt, add_special_tokens=False)["input_ids"]
            )
            if candidate_tokens + self.config["max_tokens"] <= self.config["max_model_len"]:
                best = midpoint
                lower = midpoint + 1
            else:
                upper = midpoint - 1

        return self.tokenizer.decode(token_ids[:best], skip_special_tokens=True), True

    def generate_prompt_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        if not prompts:
            return []
        try:
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        except Exception as e:
            logger.critical(f"Batch generation failed: {e}")
            return [
                {"parsed": None, "raw": "", "error": f"GENERATION_ERROR: {e}"}
                for _ in prompts
            ]

        results = []
        for output in outputs:
            completion = output.outputs[0]
            entry = {
                "parsed": None,
                "raw": completion.text,
                "error": None,
                "token_metadata": {
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(completion.token_ids),
                    # 'length' means truncated at max_tokens: incomplete JSON, not a
                    # schema failure. Needed to triage the error buckets.
                    "finish_reason": completion.finish_reason,
                },
            }
            entry.update(self._parse(completion.text))
            results.append(entry)
        return results

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        """
        model_validate_json raises ValidationError for malformed JSON as well as for
        schema violations, so branch on the error type instead of catching
        json.JSONDecodeError (which never fires here).
        """
        try:
            model = ReviewExtraction.model_validate_json(text)
            return {"parsed": model.model_dump(mode="json"), "error": None}
        except ValidationError as e:
            errs = e.errors()
            kind = (
                "JSON_PARSE_ERROR"
                if any(x.get("type") == "json_invalid" for x in errs)
                else "SCHEMA_VALIDATION_ERROR"
            )
            detail = "; ".join(
                f"{'.'.join(str(p) for p in x.get('loc', ()))}:{x.get('type')}"
                for x in errs[:5]
            )
            return {"parsed": None, "error": f"{kind}: {detail}"}
        except Exception as e:  # noqa: BLE001
            return {"parsed": None, "error": f"UNEXPECTED_PARSE_ERROR: {e}"}
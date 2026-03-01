"""Backend using vLLM for high-throughput inference.

vLLM uses PagedAttention and continuous batching, providing 3-10x speedup
over raw HuggingFace transformers for autoregressive generation.
"""

import os

from vllm import LLM, SamplingParams

from src.backends.base import ModelBackend


class VLLMBackend(ModelBackend):
    """vLLM inference backend — fast generation and log-prob scoring."""

    COMPLETION_STOP_STRINGS = ["\nQuestion:", "\nUser:", "\nAnswer:", "\n\n\n"]

    def __init__(self, model: str, revision: str | None = None,
                 max_new_tokens: int = 1024,
                 max_model_len: int = 4096,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.70,
                 trust_remote_code: bool = True,
                 torch_dtype: str = "bfloat16",
                 **kwargs):
        self.model_name = model
        self.max_new_tokens = max_new_tokens

        token = os.environ.get("HF_TOKEN")
        # Filter out transformers-only kwargs that vLLM doesn't accept
        kwargs.pop("device_map", None)

        label = f"{model}@{revision}" if revision else model
        print(f"[VLLMBackend] Loading {label}...")

        self.llm = LLM(
            model=model,
            revision=revision,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enforce_eager=kwargs.pop("enforce_eager", False),
            **kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[VLLMBackend] {label} loaded")

    def chat(self, messages: list[dict], **kwargs) -> str:
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)

        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            parts = []
            for msg in messages:
                role, content = msg["role"], msg["content"]
                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "system":
                    parts.append(f"System: {content}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)

        params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def complete(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        params = SamplingParams(
            max_tokens=max_tokens, temperature=0,
            stop=self.COMPLETION_STOP_STRINGS,
            repetition_penalty=1.1,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def _bpe_boundary_fix(self, prompt_prefix: str, answer_text: str):
        """Move trailing spaces from prefix to answer to avoid BPE merge issues."""
        n_spaces = len(prompt_prefix) - len(prompt_prefix.rstrip())
        if n_spaces > 0:
            answer_text = prompt_prefix[-n_spaces:] + answer_text
            prompt_prefix = prompt_prefix[:-n_spaces]
        return prompt_prefix, answer_text

    def _extract_answer_logprobs(self, output, prefix_len: int) -> dict:
        """Extract answer-token log-probs from a vLLM output."""
        prompt_token_ids = list(output.prompt_token_ids)
        prompt_logprobs_list = output.prompt_logprobs

        token_lps = []
        per_token = []
        for i in range(prefix_len, len(prompt_token_ids)):
            token_id = prompt_token_ids[i]
            lp = 0.0
            if prompt_logprobs_list[i] is not None:
                if token_id in prompt_logprobs_list[i]:
                    lp = prompt_logprobs_list[i][token_id].logprob
            token_lps.append(lp)
            per_token.append({
                "token": self.tokenizer.decode([token_id]),
                "token_id": token_id,
                "log_prob": lp,
            })

        total_lp = sum(token_lps)
        n = len(token_lps)
        return {
            "total_log_prob": total_lp,
            "mean_log_prob": total_lp / n if n > 0 else 0.0,
            "num_tokens": n,
            "per_token_log_probs": per_token,
        }

    def score_log_probs(self, prompt_prefix: str, answer_text: str) -> dict:
        prompt_prefix, answer_text = self._bpe_boundary_fix(prompt_prefix, answer_text)
        full_text = prompt_prefix + answer_text

        prefix_len = len(self.tokenizer(prompt_prefix)["input_ids"])

        params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        outputs = self.llm.generate([full_text], params)

        return self._extract_answer_logprobs(outputs[0], prefix_len)

    def score_log_probs_pair(self, prompt_prefix: str,
                             answer_a: str, answer_b: str) -> tuple[dict, dict]:
        """Score two completions in a single batched call."""
        n_spaces = len(prompt_prefix) - len(prompt_prefix.rstrip())
        if n_spaces > 0:
            trailing = prompt_prefix[-n_spaces:]
            prompt_prefix = prompt_prefix[:-n_spaces]
            answer_a = trailing + answer_a
            answer_b = trailing + answer_b

        full_a = prompt_prefix + answer_a
        full_b = prompt_prefix + answer_b

        prefix_len = len(self.tokenizer(prompt_prefix)["input_ids"])

        params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        outputs = self.llm.generate([full_a, full_b], params)

        return (self._extract_answer_logprobs(outputs[0], prefix_len),
                self._extract_answer_logprobs(outputs[1], prefix_len))

    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_completion(self) -> bool:
        return True

    @property
    def supports_log_probs(self) -> bool:
        return True

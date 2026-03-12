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
                 max_model_len: int = 16384,
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
        self.max_model_len = max_model_len

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[VLLMBackend] {label} loaded")

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Convert chat messages to a prompt string."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
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
        return "\n".join(parts)

    def chat(self, messages: list[dict], **kwargs) -> str:
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        prompt = self._apply_chat_template(messages)
        params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def chat_batch(self, messages_list: list[list[dict]], **kwargs) -> list[str]:
        """Generate responses for multiple conversations in a single batched call."""
        if not messages_list:
            return []
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        prompts = [self._apply_chat_template(msgs) for msgs in messages_list]
        params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text.strip() for out in outputs]

    def complete(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        params = SamplingParams(
            max_tokens=max_tokens, temperature=0,
            stop=self.COMPLETION_STOP_STRINGS,
            repetition_penalty=1.1,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def complete_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate completions for multiple prompts in a single batched call."""
        if not prompts:
            return []
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        params = SamplingParams(
            max_tokens=max_tokens, temperature=0,
            stop=self.COMPLETION_STOP_STRINGS,
            repetition_penalty=1.1,
        )
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text.strip() for out in outputs]

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

    def score_log_probs_pair_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[tuple[dict, dict]]:
        """Score multiple (prefix, answer_a, answer_b) triples in one batched call."""
        if not items:
            return []

        all_texts = []
        prefix_lens = []
        skipped = set()
        for item_idx, (prefix, answer_a, answer_b) in enumerate(items):
            n_spaces = len(prefix) - len(prefix.rstrip())
            if n_spaces > 0:
                trailing = prefix[-n_spaces:]
                prefix = prefix[:-n_spaces]
                answer_a = trailing + answer_a
                answer_b = trailing + answer_b

            text_a = prefix + answer_a
            text_b = prefix + answer_b
            max_len = max(
                len(self.tokenizer(text_a)["input_ids"]),
                len(self.tokenizer(text_b)["input_ids"]),
            )

            if max_len > self.max_model_len:
                skipped.add(item_idx)
                continue

            plen = len(self.tokenizer(prefix)["input_ids"])
            all_texts.extend([text_a, text_b])
            prefix_lens.extend([plen, plen])

        if skipped:
            print(f"[VLLMBackend] Skipped {len(skipped)}/{len(items)} items "
                  f"exceeding max_model_len={self.max_model_len}")

        params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        outputs = self.llm.generate(all_texts, params) if all_texts else []

        empty = {"total_log_prob": 0.0, "mean_log_prob": 0.0, "num_tokens": 0}
        results = []
        out_idx = 0
        for item_idx in range(len(items)):
            if item_idx in skipped:
                results.append((dict(empty), dict(empty)))
            else:
                results.append((
                    self._extract_answer_logprobs(outputs[out_idx], prefix_lens[out_idx]),
                    self._extract_answer_logprobs(outputs[out_idx + 1], prefix_lens[out_idx + 1]),
                ))
                out_idx += 2
        return results

    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_completion(self) -> bool:
        return True

    @property
    def supports_log_probs(self) -> bool:
        return True

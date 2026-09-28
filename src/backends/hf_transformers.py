"""Backend for HuggingFace Transformers models — loads model directly on GPU.

Supports both chat (instruct) and completion (base) modes.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.backends.base import ModelBackend


class _StopOnStrings(StoppingCriteria):
    """Stop generation when any of the given strings appear in the generated output.

    Only checks tokens after `prompt_len` so that stop strings present in
    the prompt itself (e.g. "Answer:" in the in-context format) don't
    cause immediate termination.

    Decodes only a trailing window of tokens each step (O(1) per step)
    instead of decoding all generated tokens (which was O(n) per step,
    O(n^2) total over a generation of length n).
    """

    def __init__(self, stop_strings: list[str], tokenizer, prompt_len: int):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        # Window size: enough tokens to always contain any stop string.
        # Longest stop string is ~10 chars; worst case 1 char/token.
        # Add margin for multi-byte chars and BPE boundary effects.
        self._window = max(len(s) for s in stop_strings) + 10

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.prompt_len:]
        # Only decode a small trailing window instead of the full output
        if len(generated_ids) > self._window:
            generated_ids = generated_ids[-self._window:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


class TransformersBackend(ModelBackend):
    """Direct HuggingFace Transformers inference backend."""

    # Stop sequences for base model completion to prevent looping
    COMPLETION_STOP_STRINGS = ["\nQuestion:", "\nUser:", "\nAnswer:", "\n\n\n"]
    COMPLETION_NO_REPEAT_NGRAM = 10  # prevent any 10-gram from repeating in output

    def __init__(self, model: str, revision: str | None = None,
                 max_new_tokens: int = 1024,
                 device_map: str = "auto", torch_dtype: str = "auto",
                 trust_remote_code: bool = False, **default_kwargs):
        self.model_name = model
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.default_kwargs = default_kwargs

        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"

        # Use HF_TOKEN env var for gated model access (e.g. Llama)
        token = os.environ.get("HF_TOKEN")

        # Use Flash Attention 2 if available (2x faster forward passes)
        attn_impl = None
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            try:
                # SDPA (PyTorch 2.0+) is the next best option
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    attn_impl = "sdpa"
            except Exception:
                pass

        label = f"{model}@{revision}" if revision else model
        attn_label = attn_impl or "eager"
        print(f"[TransformersBackend] Loading {label} (attn={attn_label})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision,
            trust_remote_code=trust_remote_code, token=token)

        model_kwargs = dict(
            device_map=device_map, torch_dtype=dtype,
            trust_remote_code=trust_remote_code, token=token,
        )
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, **model_kwargs)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[TransformersBackend] {label} loaded on {self.model.device}")

    def chat(self, messages: list[dict], **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        max_tokens = merged.pop("max_new_tokens", self.max_new_tokens)

        # Use the tokenizer's chat template if available
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback: format as text
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "system":
                    parts.append(f"System: {content}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)

        return self._generate(prompt, max_tokens)

    def complete(self, prompt: str, **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        max_tokens = merged.pop("max_new_tokens", self.max_new_tokens)
        return self._generate(
            prompt, max_tokens,
            stop_strings=self.COMPLETION_STOP_STRINGS,
            no_repeat_ngram_size=self.COMPLETION_NO_REPEAT_NGRAM,
        )

    def _generate(self, prompt: str, max_new_tokens: int,
                  stop_strings: list[str] | None = None,
                  no_repeat_ngram_size: int = 0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if stop_strings:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_StopOnStrings(stop_strings, self.tokenizer, input_len)]
            )

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        # Decode only the newly generated tokens
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Trim at stop string if present in output
        if stop_strings:
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx].strip()

        return text

    def score_log_probs(self, prompt_prefix: str, answer_text: str) -> dict:
        """Compute log P(answer_text | prompt_prefix) via a single forward pass.

        Follows the lm-evaluation-harness approach for BPE boundary handling:
        trailing whitespace is moved from prefix to answer before tokenization
        to prevent BPE merges from collapsing the boundary.
        """
        # BPE boundary fix (from lm-evaluation-harness _encode_pair):
        # Move trailing spaces from prefix to answer so that BPE doesn't
        # merge a trailing space in the prefix with the first answer token,
        # which would make prefix_len == full_len and score zero tokens.
        n_spaces = len(prompt_prefix) - len(prompt_prefix.rstrip())
        if n_spaces > 0:
            answer_text = prompt_prefix[-n_spaces:] + answer_text
            prompt_prefix = prompt_prefix[:-n_spaces]

        full_text = prompt_prefix + answer_text

        # Tokenize together and separately to find the answer boundary
        full_ids = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prefix_ids = self.tokenizer(prompt_prefix, return_tensors="pt")

        prefix_len = prefix_ids["input_ids"].shape[1]
        full_len = full_ids["input_ids"].shape[1]

        if full_len <= prefix_len:
            return {"total_log_prob": 0.0, "mean_log_prob": 0.0,
                    "num_tokens": 0, "per_token_log_probs": []}

        # Forward pass (no generation, just logits)
        with torch.no_grad():
            logits = self.model(**full_ids).logits[0]  # (seq_len, vocab_size)

        log_probs = torch.log_softmax(logits, dim=-1)

        # Vectorized log-prob extraction (matches lm-eval-harness torch.gather approach)
        # logits[i] predicts token at position i+1, so logits[prefix_len-1 : full_len-1]
        # predict tokens at positions [prefix_len : full_len] (the answer tokens).
        answer_ids = full_ids["input_ids"][0, prefix_len:full_len]  # (n_answer,)
        answer_log_probs = log_probs[prefix_len - 1 : full_len - 1]  # (n_answer, vocab)
        token_lps = answer_log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)  # (n_answer,)

        total_lp = token_lps.sum().item()
        n = token_lps.shape[0]

        per_token = [
            {"token": self.tokenizer.decode([answer_ids[i].item()]),
             "token_id": answer_ids[i].item(),
             "log_prob": token_lps[i].item()}
            for i in range(n)
        ]

        return {
            "total_log_prob": total_lp,
            "mean_log_prob": total_lp / n if n > 0 else 0.0,
            "num_tokens": n,
            "per_token_log_probs": per_token,
        }

    def score_log_probs_pair(self, prompt_prefix: str,
                             answer_a: str, answer_b: str) -> tuple[dict, dict]:
        """Score two completions in a single batched forward pass.

        Both answers share the same prefix, so we batch them together
        (batch size 2) and run one forward pass instead of two.
        """
        # BPE boundary fix — apply once to the shared prefix
        n_spaces = len(prompt_prefix) - len(prompt_prefix.rstrip())
        if n_spaces > 0:
            trailing = prompt_prefix[-n_spaces:]
            prompt_prefix = prompt_prefix[:-n_spaces]
            answer_a = trailing + answer_a
            answer_b = trailing + answer_b

        full_a = prompt_prefix + answer_a
        full_b = prompt_prefix + answer_b

        # Prefix length (for answer boundary)
        prefix_ids = self.tokenizer(prompt_prefix, return_tensors="pt")
        prefix_len = prefix_ids["input_ids"].shape[1]

        # Batch tokenize with right-padding
        batch = self.tokenizer(
            [full_a, full_b], return_tensors="pt",
            padding=True, return_attention_mask=True,
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**batch).logits  # (2, max_seq_len, vocab)

        results = []
        for i in range(2):
            seq_len = batch["attention_mask"][i].sum().item()

            if seq_len <= prefix_len:
                results.append({"total_log_prob": 0.0, "mean_log_prob": 0.0,
                                "num_tokens": 0, "per_token_log_probs": []})
                continue

            log_probs = torch.log_softmax(logits[i], dim=-1)
            answer_ids = batch["input_ids"][i, prefix_len:seq_len]
            answer_lps = log_probs[prefix_len - 1 : seq_len - 1]
            token_lps = answer_lps.gather(1, answer_ids.unsqueeze(1)).squeeze(1)

            total_lp = token_lps.sum().item()
            n = token_lps.shape[0]

            per_token = [
                {"token": self.tokenizer.decode([answer_ids[j].item()]),
                 "token_id": answer_ids[j].item(),
                 "log_prob": token_lps[j].item()}
                for j in range(n)
            ]

            results.append({
                "total_log_prob": total_lp,
                "mean_log_prob": total_lp / n if n > 0 else 0.0,
                "num_tokens": n,
                "per_token_log_probs": per_token,
            })

        return results[0], results[1]

    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_completion(self) -> bool:
        return True

    @property
    def supports_log_probs(self) -> bool:
        return True

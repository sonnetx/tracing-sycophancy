import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_initial_prompt(question: str, model_type: str = "chat") -> list[dict] | str:
    """Format a question for initial response generation."""
    if model_type == "chat":
        return [{"role": "user", "content": question}]
    else:
        return f"Question: {question}\nAnswer:"


def format_challenge_prompt(question: str, initial_response: str, challenge: str,
                            context: str, model_type: str = "chat") -> list[dict] | str:
    """Format a challenge/rebuttal prompt."""
    if context == "preemptive":
        if model_type == "chat":
            return [{"role": "user", "content": challenge}]
        else:
            return f"Question: {challenge}\nAnswer:"

    # In-context: multi-turn conversation
    if model_type == "chat":
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": challenge},
        ]
    else:
        return (
            f"Question: {question}\n"
            f"Answer: {initial_response}\n"
            f"User: {challenge}\n"
            f"Answer:"
        )


# ---------------------------------------------------------------------------
# Log-prob prompt formatting (completion-only for fair base vs instruct comparison)
# ---------------------------------------------------------------------------

def format_logprob_baseline_prompt(question: str) -> str:
    """Format prompt prefix for baseline log-prob scoring."""
    return f"Question: {question}\nAnswer:"


def format_logprob_challenge_prompt(question: str, initial_answer: str,
                                    challenge: str, context: str,
                                    max_answer_chars: int = 2000) -> str:
    """Format prompt prefix for challenged log-prob scoring.

    Truncates initial_answer to keep total prompt within reasonable token
    limits.  The answer only needs enough context to prime the model, not
    the full verbatim text.
    """
    if context == "preemptive":
        return f"Question: {challenge}\nAnswer:"
    if len(initial_answer) > max_answer_chars:
        initial_answer = initial_answer[:max_answer_chars] + "..."
    return (
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
        f"User: {challenge}\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def load_backend(config_path: str | Path):
    """Load a ModelBackend from a JSON config file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    backend_type = config.pop("backend")

    if backend_type == "ollama":
        from src.backends.ollama import OllamaBackend
        return OllamaBackend(**config)
    elif backend_type == "openai":
        from src.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(**config)
    elif backend_type == "anthropic":
        from src.backends.anthropic import AnthropicBackend
        return AnthropicBackend(**config)
    elif backend_type == "transformers":
        from src.backends.hf_transformers import TransformersBackend
        return TransformersBackend(**config)
    elif backend_type == "vllm":
        from src.backends.vllm_backend import VLLMBackend
        return VLLMBackend(**config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

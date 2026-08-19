"""Prompt generation for load tests: exact token counts, verified by encoding.

Two correctness properties matter more than realism here:

1. Every prompt must encode to *exactly* the requested token count under the target
   model's own tokenizer. Byte-level BPE tokenizers don't round-trip cleanly through
   arbitrary token ids (decode(encode(x)) can re-encode to a different length than x),
   so exactness is enforced by actually re-encoding the assembled text and correcting
   if it drifts, not by trusting a token-count arithmetic estimate.
2. Every prompt begins with a unique random token block by default. Without this,
   identical or overlapping prompt prefixes let vLLM's automatic prefix caching skip
   prefill work, and the resulting throughput inflation grows with context length --
   exactly the axis this project measures. That would manufacture the result before a
   single real request is sent. `enable_shared_prefix=True` exists to measure this
   confound on purpose, not to avoid it by default.

The tokenizer argument is duck-typed to `.encode(str) -> list[int]` and
`.decode(list[int]) -> str` -- the interface every `transformers` tokenizer already
implements -- so tests don't need a real (multi-hundred-MB) tokenizer to exercise this
module; production callers pass `AutoTokenizer.from_pretrained(...)` for the model
under test, selected via config like everything else.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# Small built-in word bank so prompt generation needs no corpus download or network
# access. Repeated and shuffled per-call with the seeded RNG to build filler text of
# arbitrary length; content realism doesn't matter here, only token-count exactness
# and prefix uniqueness do.
_WORD_BANK = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "while", "system", "observes", "latency", "budget", "context", "window", "grows", "linearly", "memory", "pressure", "model", "serves", "requests", "concurrent", "users", "queue", "arrives", "token", "stream", "decode", "attention", "cache", "hit", "miss", "ratio", "gpu", "utilization", "throughput", "ratio", "scheduler", "admits", "waiting", "running", "sequence", "batch", "prefill", "decode", "phase", "split", "workload", "distribution", "random", "seed", "reproducible", "experiment", "cluster", "node", "rack", "cooling", "power", "draw", "watt", "hour", "cost", "dollar", "per", "million", "tokens", "served", "endpoint", "api", "gateway", "router", "load", "balancer", "health", "check", "timeout", "retry", "backoff", "jitter", "poisson", "arrival", "closed", "loop", "open", "loop", "rate", "limit", "benchmark", "harness", "metric", "prometheus", "scrape", "interval", "histogram", "bucket", "quantile", "percentile", "median", "tail", "latency", "time", "first", "byte", "inter", "token", "gap", "completion", "end"]

MAX_RETRIES = 5


@dataclass(frozen=True)
class GeneratedPrompt:
    """A single generated prompt, with its exactness already verified."""

    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int  # == len(token_ids); kept explicit for clarity at call sites


@dataclass(frozen=True)
class WorkloadConfig:
    """Parameters shared across every prompt generated for a run."""

    target_prompt_tokens: int
    max_tokens: int
    prefix_tokens: int = 64
    enable_shared_prefix: bool = False
    ignore_eos: bool = True
    seed: int = 0


def _word_pool_ids(tokenizer: Tokenizer, rng: random.Random, n_words: int) -> list[int]:
    """Encode `n_words` randomly shuffled words from the word bank into token ids."""
    words = [rng.choice(_WORD_BANK) for _ in range(n_words)]
    return tokenizer.encode(" ".join(words))


def _ids_of_length(tokenizer: Tokenizer, rng: random.Random, n_tokens: int) -> list[int]:
    """Produce at least `n_tokens` token ids from randomly ordered word-bank text,
    growing the word count geometrically until there's enough, then slicing exactly."""
    if n_tokens <= 0:
        return []
    n_words = max(4, n_tokens)  # words:tokens ratio is usually >= 1, this overshoots safely
    ids = _word_pool_ids(tokenizer, rng, n_words)
    while len(ids) < n_tokens:
        n_words *= 2
        ids = _word_pool_ids(tokenizer, rng, n_words)
    return ids[:n_tokens]


def _assemble(tokenizer: Tokenizer, rng: random.Random, prefix_ids: list[int], target_tokens: int) -> GeneratedPrompt:
    """Build a prompt from `prefix_ids` + filler body, verifying the exact token count
    by re-encoding the decoded text and correcting for any decode/encode drift."""
    body_ids = _ids_of_length(tokenizer, rng, target_tokens - len(prefix_ids))
    ids = list(prefix_ids) + body_ids

    for _ in range(MAX_RETRIES):
        text = tokenizer.decode(ids)
        verified = tokenizer.encode(text)
        if len(verified) == target_tokens:
            return GeneratedPrompt(text=text, token_ids=tuple(verified), prompt_tokens=target_tokens)
        # Decode/encode drifted at a merge boundary: extend or trim the body and retry.
        drift = target_tokens - len(verified)
        if drift > 0:
            ids = ids + _ids_of_length(tokenizer, rng, drift)
        else:
            ids = ids[:drift]  # negative index trims the tail

    raise RuntimeError(
        f"could not converge prompt to exactly {target_tokens} tokens after "
        f"{MAX_RETRIES} correction attempts (last length: {len(verified)})"
    )


def generate_prompt(
    tokenizer: Tokenizer,
    config: WorkloadConfig,
    rng: random.Random,
    shared_prefix_ids: list[int] | None = None,
) -> GeneratedPrompt:
    """Generate one prompt encoding to exactly `config.target_prompt_tokens` tokens.

    When `config.enable_shared_prefix` is False (the default), the prompt's leading
    `config.prefix_tokens` are drawn fresh from `rng` and distinct per call -- prefix
    caching cannot match across requests. When True, `shared_prefix_ids` (built once
    per run via `make_shared_prefix`) is reused verbatim as the leading block instead,
    deliberately reintroducing the confound so it can be measured.
    """
    if config.target_prompt_tokens < config.prefix_tokens:
        raise ValueError(
            f"target_prompt_tokens ({config.target_prompt_tokens}) must be >= "
            f"prefix_tokens ({config.prefix_tokens})"
        )

    if config.enable_shared_prefix:
        if shared_prefix_ids is None:
            raise ValueError("enable_shared_prefix=True requires shared_prefix_ids")
        prefix_ids = shared_prefix_ids
    else:
        prefix_ids = _ids_of_length(tokenizer, rng, config.prefix_tokens)

    return _assemble(tokenizer, rng, prefix_ids, config.target_prompt_tokens)


def make_shared_prefix(tokenizer: Tokenizer, config: WorkloadConfig, rng: random.Random) -> list[int]:
    """Build the one fixed prefix reused by every prompt when enable_shared_prefix=True."""
    return _ids_of_length(tokenizer, rng, config.prefix_tokens)


@dataclass
class WorkloadGenerator:
    """Stateful, seeded generator producing a reproducible sequence of prompts.

    Same `config.seed` -> same sequence of prompts, regardless of how many are drawn
    in a single call vs. across several -- state lives in the RNG, advanced once per
    generated prompt.
    """

    tokenizer: Tokenizer
    config: WorkloadConfig
    _rng: random.Random = field(init=False, repr=False)
    _shared_prefix_ids: list[int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)
        if self.config.enable_shared_prefix:
            self._shared_prefix_ids = make_shared_prefix(self.tokenizer, self.config, self._rng)

    def next(self) -> GeneratedPrompt:
        return generate_prompt(self.tokenizer, self.config, self._rng, self._shared_prefix_ids)

    def generate(self, n: int) -> list[GeneratedPrompt]:
        return [self.next() for _ in range(n)]

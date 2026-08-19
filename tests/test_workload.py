"""workload.py: exact token counts (verified by re-encoding), unique prefixes,
shared-prefix mode, and seeded reproducibility."""

import random

import pytest

from ctxcost.bench.workload import (
    WorkloadConfig,
    WorkloadGenerator,
    generate_prompt,
    make_shared_prefix,
)
from fake_tokenizer import DriftyTokenizer, WordTokenizer


def test_prompt_encodes_to_exact_target_length():
    tokenizer = WordTokenizer()
    config = WorkloadConfig(target_prompt_tokens=200, max_tokens=16, prefix_tokens=32, seed=1)
    prompt = generate_prompt(tokenizer, config, random.Random(1))
    assert prompt.prompt_tokens == 200
    assert len(prompt.token_ids) == 200
    # the actual contract: re-encoding the returned text reproduces exactly this count
    assert len(tokenizer.encode(prompt.text)) == 200


@pytest.mark.parametrize("target", [1, 8, 65, 512])
def test_various_target_lengths_are_exact(target):
    tokenizer = WordTokenizer()
    prefix = min(4, target)
    config = WorkloadConfig(target_prompt_tokens=target, max_tokens=4, prefix_tokens=prefix, seed=0)
    prompt = generate_prompt(tokenizer, config, random.Random(0))
    assert prompt.prompt_tokens == target
    assert len(tokenizer.encode(prompt.text)) == target


def test_exactness_survives_decode_encode_drift():
    """DriftyTokenizer's decode() fuses tokens at merge boundaries -- this must not
    silently produce a prompt shorter or longer than requested."""
    tokenizer = DriftyTokenizer()
    config = WorkloadConfig(target_prompt_tokens=100, max_tokens=8, prefix_tokens=16, seed=3)
    prompt = generate_prompt(tokenizer, config, random.Random(3))
    assert prompt.prompt_tokens == 100
    assert len(tokenizer.encode(prompt.text)) == 100


def test_prefixes_are_distinct_across_prompts():
    """MANDATORY property: without enable_shared_prefix, no two prompts should share
    a leading token block -- otherwise prefix caching silently inflates throughput."""
    tokenizer = WordTokenizer()
    config = WorkloadConfig(target_prompt_tokens=150, max_tokens=8, prefix_tokens=32, seed=42)
    gen = WorkloadGenerator(tokenizer=tokenizer, config=config)
    prompts = gen.generate(20)
    prefixes = [p.token_ids[:32] for p in prompts]
    assert len(set(prefixes)) == len(prefixes)  # all distinct


def test_shared_prefix_mode_reuses_the_same_prefix():
    tokenizer = WordTokenizer()
    config = WorkloadConfig(
        target_prompt_tokens=150, max_tokens=8, prefix_tokens=32, enable_shared_prefix=True, seed=7
    )
    gen = WorkloadGenerator(tokenizer=tokenizer, config=config)
    prompts = gen.generate(10)
    prefixes = {p.token_ids[:32] for p in prompts}
    assert len(prefixes) == 1  # every prompt shares the identical leading block


def test_shared_prefix_without_ids_raises():
    tokenizer = WordTokenizer()
    config = WorkloadConfig(target_prompt_tokens=50, max_tokens=4, prefix_tokens=8, enable_shared_prefix=True)
    with pytest.raises(ValueError):
        generate_prompt(tokenizer, config, random.Random(0), shared_prefix_ids=None)


def test_seeded_generator_is_reproducible():
    tokenizer1, tokenizer2 = WordTokenizer(), WordTokenizer()
    config = WorkloadConfig(target_prompt_tokens=80, max_tokens=8, prefix_tokens=16, seed=99)
    prompts1 = WorkloadGenerator(tokenizer=tokenizer1, config=config).generate(5)
    prompts2 = WorkloadGenerator(tokenizer=tokenizer2, config=config).generate(5)
    assert [p.text for p in prompts1] == [p.text for p in prompts2]


def test_different_seeds_diverge():
    tokenizer1, tokenizer2 = WordTokenizer(), WordTokenizer()
    config1 = WorkloadConfig(target_prompt_tokens=80, max_tokens=8, prefix_tokens=16, seed=1)
    config2 = WorkloadConfig(target_prompt_tokens=80, max_tokens=8, prefix_tokens=16, seed=2)
    prompts1 = WorkloadGenerator(tokenizer=tokenizer1, config=config1).generate(3)
    prompts2 = WorkloadGenerator(tokenizer=tokenizer2, config=config2).generate(3)
    assert [p.text for p in prompts1] != [p.text for p in prompts2]


def test_target_below_prefix_tokens_raises():
    tokenizer = WordTokenizer()
    config = WorkloadConfig(target_prompt_tokens=10, max_tokens=4, prefix_tokens=64)
    with pytest.raises(ValueError):
        generate_prompt(tokenizer, config, random.Random(0))


def test_make_shared_prefix_length():
    tokenizer = WordTokenizer()
    config = WorkloadConfig(target_prompt_tokens=100, max_tokens=4, prefix_tokens=24)
    ids = make_shared_prefix(tokenizer, config, random.Random(5))
    assert len(ids) == 24

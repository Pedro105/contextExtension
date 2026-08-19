"""Fake tokenizers for testing src/ctxcost/bench/workload.py without a real
(multi-hundred-MB) HF tokenizer."""

from __future__ import annotations


class WordTokenizer:
    """Whitespace-level fake tokenizer: each unique word <-> one stable id, assigned
    on first sight. decode(encode(x)) round-trips exactly (re-encoding it reproduces
    the same token count), so this exercises workload.py's common, no-correction-
    needed path."""

    def __init__(self) -> None:
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def _id_for(self, word: str) -> int:
        if word not in self._word_to_id:
            i = len(self._word_to_id)
            self._word_to_id[word] = i
            self._id_to_word[i] = word
        return self._word_to_id[word]

    def encode(self, text: str) -> list[int]:
        return [self._id_for(w) for w in text.split()]

    def decode(self, ids: list) -> str:
        return " ".join(self._id_to_word[i] for i in ids)


class DriftyTokenizer(WordTokenizer):
    """Like WordTokenizer, but decode() fuses every 5th word onto the previous one
    (dropping the separating space) -- simulating how a real byte-level BPE
    tokenizer's decode/encode aren't always exact inverses at merge boundaries. This
    deterministically makes re-encoding the decoded text yield fewer tokens than the
    original id list, exercising workload.py's exactness-correction retry loop."""

    def decode(self, ids: list) -> str:
        words = [self._id_to_word[i] for i in ids]
        out: list[str] = []
        for idx, w in enumerate(words):
            if idx > 0 and idx % 5 == 0 and out:
                out[-1] = out[-1] + w
            else:
                out.append(w)
        return " ".join(out)

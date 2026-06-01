"""Tests for the deterministic word tokenizer (torch-free; runs everywhere)."""

from __future__ import annotations

from discovery_agents.mlinfra.tokenizer import PAD_ID, UNK_ID, WordVocab

CORPUS = [
    "the team needs to align on a decision",
    "the handoff needs requirements and acceptance criteria",
    "align the team and decide what to build",
]


def test_build_is_deterministic() -> None:
    a = WordVocab.build(CORPUS)
    b = WordVocab.build(CORPUS)
    assert a.id_to_token == b.id_to_token
    assert a.id_to_token[PAD_ID] == "<pad>"
    assert a.id_to_token[UNK_ID] == "<unk>"


def test_encode_padded_length_and_padding() -> None:
    vocab = WordVocab.build(CORPUS)
    ids = vocab.encode_padded("the team", seq_len=8)
    assert len(ids) == 8
    assert ids[2:] == [PAD_ID] * 6  # "the team" -> 2 tokens, rest padded


def test_unknown_words_map_to_unk() -> None:
    vocab = WordVocab.build(CORPUS)
    ids = vocab.encode("zzzznonexistent")
    assert ids == [UNK_ID]


def test_truncation_to_seq_len() -> None:
    vocab = WordVocab.build(CORPUS)
    ids = vocab.encode_padded("the team needs to align on a decision now", seq_len=3)
    assert len(ids) == 3


def test_save_load_roundtrip(tmp_path) -> None:
    vocab = WordVocab.build(CORPUS)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = WordVocab.load(path)
    assert loaded.id_to_token == vocab.id_to_token

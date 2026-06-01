"""Tests for the Banking77 loader (sample path; offline, stdlib-only)."""

from __future__ import annotations

from discovery_agents.mlinfra.datasets import load_banking77


def test_sample_loads_all_intents() -> None:
    data = load_banking77(full=False)
    assert data.num_labels == 77
    assert len(data.train) == 616  # ~8 per intent
    assert len(data.test) == 308  # ~4 per intent
    assert len(data.train.texts) == len(data.train.labels)


def test_labels_are_in_range_and_texts_nonempty() -> None:
    data = load_banking77(full=False)
    assert all(0 <= lbl < data.num_labels for lbl in data.train.labels)
    assert all(text.strip() for text in data.train.texts)
    assert data.label_names[0]  # intent names present


def test_loader_is_deterministic() -> None:
    a = load_banking77(full=False)
    b = load_banking77(full=False)
    assert a.train.texts == b.train.texts
    assert a.train.labels == b.train.labels


def test_name_for_handles_noncontiguous_labels(tmp_path, monkeypatch) -> None:
    # Sparse label ids (a slice that omits intents) must map by id, not list position
    # — guards against the positional-indexing bug.
    csv_path = tmp_path / "sparse.csv"
    csv_path.write_text(
        "split,text,label,label_text\n"
        "train,reset my pin,5,reset_pin\n"
        "train,where is my card,9,card_location\n"
        "test,pin reset please,5,reset_pin\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BANKING77_SAMPLE", str(csv_path))
    data = load_banking77(full=False)
    assert data.name_for(5) == "reset_pin"
    assert data.name_for(9) == "card_location"  # positional indexing would crash/mislabel here

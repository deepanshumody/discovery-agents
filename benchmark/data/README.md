# Benchmark data

## banking77_sample.csv

A small, stratified slice of **Banking77** (~8 train / ~4 test rows per intent; all 77
intents) committed so the retrieval benchmark and its tests run **offline and in CI** with no
download. Columns: `split, text, label, label_text`.

The full split (9,993 train / 3,076 test) is loaded on demand from
[`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) when the benchmark is run
with `--full`; the committed `../RESULTS.md` headline numbers come from that full run.

### Source & license

Banking77 — Casanueva, Temčinas, Gerz, Henderson, Vulić, *"Efficient Intent Detection with
Dual Sentence Encoders"* (NLU@ACL 2020). Licensed **CC-BY-4.0**. This slice is redistributed
under the same license with attribution.

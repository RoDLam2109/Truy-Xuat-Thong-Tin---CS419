# Lam v1: validation baseline results (seed 42)

These are **measured on validation**, not test or smoke data.
The corpus contains 61,425 legal documents; validation contains 482 queries
and 495 positive qrels. The generated validation noise files preserve query IDs.
The main pipeline produced 16,498 word features and 111,181 char n-gram features.

## Word representation (full corpus)

Default parameters: VSM TF-IDF / cosine; BM25 k1=1.2, b=0.75;
LSI TruncatedSVD 64 dimensions, seed 42. No label-based fitting.

| Query condition | VSM nDCG@10 | BM25 nDCG@10 | LSI nDCG@10 |
|---|---:|---:|---:|
| clean | 0.5008 | **0.6626** | 0.2420 |
| no_diacritics | 0.0209 | **0.0297** | 0.0019 |
| telex_typo | 0.3263 | **0.4725** | 0.1376 |
| phonetic | 0.4925 | **0.6511** | 0.2325 |
| abbrev | 0.5004 | **0.6589** | 0.2358 |
| mixed_low | 0.4198 | **0.5875** | 0.1914 |
| mixed_medium | 0.3119 | **0.4698** | 0.1204 |
| mixed_high | 0.2284 | **0.3534** | 0.0803 |

Clean MRR@10: VSM 0.4420, BM25 0.6143, LSI 0.2046.
Clean Recall@100: VSM 0.8990, BM25 0.9457, LSI 0.6127.

For BM25, nDCG@10 decreases by 95.52% without accents and by 46.67%
under mixed_high relative to clean. Robustness must be considered alongside
absolute ranking quality; a consistently weak model may appear stable.

## Reproduction and provenance

Local untracked experiment artifacts:
- outputs/processed/summary.json (full term vocabulary, input checksums)
- outputs/index/word/index_meta.json (full index provenance)
- outputs/retrieval/val_word/config.json
- outputs/retrieval/val_word/summary.csv
- outputs/retrieval/val_word/per_query.csv
- outputs/splits/seed42/split_manifest.json

The input corpus SHA-256 is
dda9c3f0c09627c1a5f2f921384d1c55b6b9775d67ad7254a5bd7a6a119cb9af.
All 67/67 source/noise checks passed using scripts/check_dataset.py.
These outputs can be recreated with docs/model_evaluation.md; raw data and
generated metrics are intentionally kept out of Git because of file size and
data governance.

These are fixed-parameter baselines, NOT tuned best configurations.
No final test evaluation was performed for this report. The synthetic noise
generator may change the query semantics; inspect samples before publication.


## Character n-gram VSM (full corpus, measured)

The full char index has 61,425 documents, 111,181 features and
44,687,991 nonzero count-matrix entries. These numbers are **full validation**
results, NOT the 400-document smoke experiment.

| Query condition | Word VSM nDCG@10 | Char VSM nDCG@10 |
|---|---:|---:|
| clean | 0.5008 | 0.4833 |
| no_diacritics | 0.0209 | 0.0219 |
| telex_typo | 0.3263 | 0.4216 |
| phonetic | 0.4925 | 0.4803 |
| abbrev | 0.5004 | 0.4785 |
| mixed_low | 0.4198 | 0.4528 |
| mixed_medium | 0.3119 | 0.3996 |
| mixed_high | 0.2284 | 0.3542 |

Character n-grams improve VSM for Telex and mixed noise, but their
no-diacritics score remains near zero. Simply switching to char n-grams does
not solve accent removal: candidate generation still depends on overlap
between accented corpus and unaccented query grams.

Results are preserved in outputs/retrieval/val_char_full/progress.json:
8/8 VSM conditions completed. Full-corpus char BM25 and char LSI have
**not** been benchmarked; their 16 cells remain pending. The resource-intensive
process was stopped after saving VSM progress, so no residual computation is
left running. The evaluation code supports the remaining experiments,
but this document deliberately does not invent their scores.

All final test results remain uncomputed; select configurations on validation.

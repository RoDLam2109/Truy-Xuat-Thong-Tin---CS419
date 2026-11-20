# Model evaluation on GreenNode Lam v1

## Source of truth and leakage policy

- Canonical dataset: GreenNode Lam v1, revision 12d76d4d04b94ceada970fcfbe7fec20bfa97389,
  stored in data/zalo_legal (not tracked in Git). Verification:
  data/zalo_legal/train_manifest.json and configs/dataset.json.
- Corpus: 61,425 documents, common to every split.
- Canonical train: 2,408 query IDs and 2,480 qrels; multiple positive documents
  for a query are preserved. Canonical test: 788 query IDs and 793 qrels.
- Train was cleaned by teammate's scripts/prepare_train.py. The 24
  overlapping query IDs and 25 qrels are removed reproducibly.
- Never fit vocabulary, IDF or SVD using query relevance labels.
  Fit vocabulary only on corpus with scripts/prepare_terms.py.
- Never tune hyperparameters on test. Default evaluation is validation;
  final test requires --confirm-test-use.

## Development split

Run scripts/prepare_validation.py on canonical TRAIN ONLY, seed 42, group
by NFC+casefold+collapsed-whitespace query text. Actual output:

| Partition | Queries | Qrels |
|---|---:|---:|
| Development train | 1,926 | 1,985 |
| Validation | 482 | 495 |

Query IDs and normalized query text are disjoint across development train,
validation and fixed test. The 7 validation noisy variants are generated using
the existing noise generator/config and seed 42, without modifying canonical
train/test. Split, validation noise and reports live in outputs/splits/seed42.
Keep split IDs fixed across all experiments.

## Representations and models

The preprocessing script from main creates processed corpus and two vocabularies:

- Word: PyVi, five defined stopwords, corpus DF selection min_df=2, max_df=0.95.
- Character n-gram: 3-5 grams within individual whitespace-separated tokens
  with one boundary space on either end. Same corpus DF filter.
- TF-IDF: sklearn TfidfTransformer, raw term frequency, smoothed corpus IDF,
  L2 normalization; queries transform with SAVED corpus IDF.
- VSM: cosine of document/query TF-IDF.
- BM25: Robertson nonnegative IDF, document counts and lengths;
  unique query terms, defaults k1=1.2, b=0.75.
- LSI: TruncatedSVD fitted on corpus TF-IDF, default rank 64 and seed 42;
  query projection uses the same SVD.
- Metrics: MRR@10, nDCG@10, Recall@100 across all positive qrels;
  degradations relative to the SAME branch/model clean condition.
- Deterministic ties: ascending document ID.
  Empty query representations return no results and receive metrics 0.

Full intended grid: 2 representations x 3 models x 8 conditions = 48 cells.
Smoke tests/subset runs do not establish that all 48 cells are completed.

## Reproduce

On Windows, use Python 3.12 and UTF-8:

    python -m uv sync --locked
    python -X utf8 scripts/prepare_train.py --download
    python -X utf8 scripts/prepare_validation.py
    python -X utf8 scripts/prepare_terms.py --workers 4
    python -X utf8 scripts/build_retrieval_index.py --branch word --output-dir outputs/index/word
    python -X utf8 scripts/build_retrieval_index.py --branch char_ngram --output-dir outputs/index/char_ngram
    python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/word --output-dir outputs/retrieval/val_word
    python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/char_ngram --output-dir outputs/retrieval/val_char

If uv is unavailable, install packages declared by pyproject.toml.
Scripts refuse overwriting an existing split, index or result directory.

Outputs:
- outputs/splits/seed42/{train,val}: clean queries, qrels, 7 validation noises
- outputs/processed: normalized corpus/queries and corpus-only vocabularies
- outputs/index/{word,char_ngram}: corpus counts/TF-IDF, IDF, document IDs,
  document lengths, provenance
- outputs/retrieval/val_{word,char}: config.json, summary.csv, per_query.csv,
  summary.md, progress.json

After finalizing hyperparameters using ONLY validation, run one held-out test:

    python -X utf8 scripts/evaluate_models.py --split test --confirm-test-use --index-dir outputs/index/word --output-dir outputs/retrieval/test_word_FINAL

Reduced corpus checks (--max-docs for index; --allow-subset-smoke for evaluation)
are NOT valid full-corpus benchmark measurements.

## Limitations

- The project's prior experiments have already seen the test query set; do
  not claim it is a genuinely unseen final test.
- Query noise is synthetic; semantic preservation requires human review.
- Qrels list judged positives; unjudged retrieved docs may be relevant.
- Char n-gram and SVD full-corpus experiments can be CPU/RAM intensive.
- Prior word-only results used different preprocessing, and must not be
  compared directly. Historical reports remain available in Git history.

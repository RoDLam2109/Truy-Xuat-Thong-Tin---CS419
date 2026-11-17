# Model selection and robustness evaluation — CS419

## Scope and provenance

This branch adds **retrieval models and evaluation only** on top of teammate branch
`feature/data-terms`. It does not modify the term extraction pipeline.

The user-provided [Google Drive dataset](https://drive.google.com/drive/folders/18fC4b6ve3zxCzdqMIWADmpmja1yP2XoD)
contains **61,425 documents, 788 unique clean queries and 793 positive qrels**,
plus seven paired noisy query variants. This is **not** the 640-test-query
`minhnguyent546` split used in teammate's latest term handoff.
Never mix indices, qrels or metrics across those splits.

`scripts/download_drive_dataset.py` downloads the exact Drive files by ID,
checks sizes and records SHA-256. Raw data, indices and results are git-ignored.

## Experimental design

- Representation used now: **word terms with PyVi**, from teammate's
  `configs/terms.json`: NFC, lowercase, preserve diacritics and numerals,
  no stopwords, `min_df=1`, `max_df=1`.
- All vocabulary, DF/IDF, document lengths and LSI projections are **fit on corpus
  only**. No training or parameter search on `qrels_test.tsv`.
- Retrieval systems:
  - **VSM:** document/query TF-IDF, L2 normalization, cosine similarity.
  - **BM25:** smoothed Robertson IDF
    `log(1 + (N-df+0.5)/(df+0.5))`, `k1=1.2`, `b=0.75`.
    Match unique query terms; use raw document term frequency and length.
  - **LSI:** TruncatedSVD of corpus TF-IDF, `64` dimensions, seed `42`,
    cosine similarity after projection.
- These model parameters were selected **a priori**, not tuned on test.
- Each system returns up to 100 **positive-score** documents for each of 788
  queries; empty query vectors return zero results and receive metric 0.
- Conditions: clean, no_diacritics, telex_typo, phonetic, abbrev,
  mixed_low, mixed_medium, mixed_high.
- Metrics: MRR@10, nDCG@10, Recall@100 (all queries in denominator);
  absolute and relative drops against the *same model's* clean results.
- Ranking ties: document ID ascending. Qrels with multiple positive documents
  contribute to nDCG and Recall. Never silently treat missing results as zeros.

## Running on Windows, Python 3.12

```powershell
cd 'C:\Users\asus\Documents\HocTap\HK5\Information Retrieval\Đồ Án'
py -3.12 -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install pyvi==0.1.1 gdown==6.5.1
.\.venv\Scripts\python.exe -X utf8 scripts/download_drive_dataset.py
.\.venv\Scripts\python.exe -X utf8 scripts/build_index.py --corpus data/zalo_legal/corpus.jsonl --queries data/zalo_legal/queries_clean.jsonl --qrels data/zalo_legal/qrels_test.tsv --output-dir outputs/terms/drive_word_clean
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_models.py --data-dir data/zalo_legal --index-dir outputs/terms/drive_word_clean --output-dir outputs/retrieval/drive_word_v1
```

Use a fresh output directory for each run. The teammate's index builder
rejects overwriting existing artifacts and validates their hashes.

Outputs in `outputs/retrieval/drive_word_v1/`:
- `config.json`: data hashes, index manifest hash and model parameters.
- `summary.csv` / `summary.md`: all aggregate metrics and degradations.
- `per_query.csv`: paired metrics for every condition.
- `runs/{model}_{condition}.jsonl`: deterministic top-100 rankings.

## Interpretation and limitations

- Use **clean accuracy** and **absolute noisy performance** together with
  relative degradation. A weak model might be stable simply because it performs
  poorly in all settings.
- Query noise is synthetic and not always semantically safe; inspect samples.
- Qrels cover a limited judged set; unjudged documents may still be relevant.
- With only this held-out Drive test split, the comparison is exploratory.
  Choosing a winner and subsequently reporting its performance on the *same*
  test set is not an independent estimate of a tuned model. Use separate
  development queries for hyperparameter tuning, then evaluate on a fresh
  independent test split if an unbiased final estimate is required.
- The teammate has not yet implemented character n-gram indexing; do **not**
  claim the full 2 representations x 3 models x 8 conditions benchmark is complete.
  When a compatible char-n-gram term index arrives, the retrievers and metrics
  can be reused without changing corpus-only fitting rules.

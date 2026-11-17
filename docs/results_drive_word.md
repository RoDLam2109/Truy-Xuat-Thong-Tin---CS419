# Model evaluation on Drive queries (no test-set tuning)

Corpus: 61425 | Queries: 788 | Qrels: 793

| Model | Condition | MRR@10 | nDCG@10 | Recall@100 | nDCG drop % |
|---|---|---:|---:|---:|---:|
| vsm | clean | 0.3901 | 0.4380 | 0.8522 | 0.0% |
| vsm | no_diacritics | 0.0120 | 0.0172 | 0.2500 | 96.1% |
| vsm | telex_typo | 0.2579 | 0.3034 | 0.7234 | 30.7% |
| vsm | phonetic | 0.3748 | 0.4227 | 0.8407 | 3.5% |
| vsm | abbrev | 0.3864 | 0.4338 | 0.8496 | 0.9% |
| vsm | mixed_low | 0.3288 | 0.3794 | 0.8033 | 13.4% |
| vsm | mixed_medium | 0.2289 | 0.2742 | 0.7005 | 37.4% |
| vsm | mixed_high | 0.1631 | 0.2012 | 0.5945 | 54.0% |
| bm25 | clean | 0.5886 | 0.6537 | 0.9721 | 0.0% |
| bm25 | no_diacritics | 0.0277 | 0.0350 | 0.1904 | 94.6% |
| bm25 | telex_typo | 0.4424 | 0.5043 | 0.8852 | 22.8% |
| bm25 | phonetic | 0.5825 | 0.6466 | 0.9683 | 1.1% |
| bm25 | abbrev | 0.5867 | 0.6521 | 0.9734 | 0.2% |
| bm25 | mixed_low | 0.5350 | 0.5979 | 0.9391 | 8.5% |
| bm25 | mixed_medium | 0.4093 | 0.4741 | 0.8604 | 27.5% |
| bm25 | mixed_high | 0.2984 | 0.3554 | 0.7570 | 45.6% |
| lsi | clean | 0.1640 | 0.1993 | 0.5698 | 0.0% |
| lsi | no_diacritics | 0.0047 | 0.0072 | 0.0343 | 96.4% |
| lsi | telex_typo | 0.0916 | 0.1184 | 0.4581 | 40.6% |
| lsi | phonetic | 0.1598 | 0.1954 | 0.5692 | 2.0% |
| lsi | abbrev | 0.1635 | 0.1988 | 0.5609 | 0.2% |
| lsi | mixed_low | 0.1250 | 0.1590 | 0.5298 | 20.2% |
| lsi | mixed_medium | 0.0896 | 0.1117 | 0.4556 | 43.9% |
| lsi | mixed_high | 0.0557 | 0.0719 | 0.3566 | 63.9% |

All metrics average over all 788 queries. The same full corpus and qrels were used for every condition. Hyperparameters fixed before evaluation.

## Kết luận và lựa chọn mô hình

**Chọn BM25 cho nhánh word/PyVi hiện tại.** BM25 có điểm clean cao nhất và
nDCG@10 cao nhất trong cả tám điều kiện. LSI (64 chiều) cho kết quả thấp
hơn nhiều; chưa thử tuning số chiều vì tránh sử dụng test qrels để tối ưu.

Điểm clean BM25: MRR@10 = 0.5886, nDCG@10 = 0.6537,
Recall@100 = 0.9721. Trong mixed_high, nDCG@10 = 0.3554
(suy giảm 45.62%), tốt hơn VSM 0.2012 và LSI 0.0719.

**Vấn đề nổi bật:** `no_diacritics` gây suy giảm nDCG trên 94% với tất cả
ba mô hình word/PyVi. Khi tích hợp nhánh character n-gram từ teammate,
nên kiểm tra đặc biệt điều kiện này.

## Giới hạn đánh giá

- Mỗi mô hình chỉ có **một cấu hình cố định trước khi xem điểm test**.
  Đây là so sánh baseline, không phải hyperparameter search.
- Bộ Google Drive có 788 test queries; không được trộn với 640 test queries
  từ nguồn Hugging Face khác trên nhánh teammate.
- Việc chọn mô hình dựa trên chính benchmark này là lựa chọn thực nghiệm
  trong phạm vi đề tài; cần tập test độc lập mới để ước lượng tổng quát
  của mô hình đã chọn.
- `phonetic` chỉ thay đổi 191/788 câu, `abbrev` 215/788 câu;
  mức suy giảm thấp một phần do nhiều câu vẫn giữ nguyên.
- Kết quả này chỉ bao gồm `word`, không được gọi là bộ 48 thí nghiệm
  `word + char_ngram` đã hoàn tất.

Để chạy lại và xem các bản `summary.csv`, `per_query.csv`, `config.json`
và `runs/*.jsonl`, làm theo [model_evaluation.md](model_evaluation.md).
Các artifact lớn được lưu trên máy, không commit lên GitHub.

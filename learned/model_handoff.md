# Bàn giao word terms cho phần mô hình

## Bộ đầu vào hiện có

Sau khi chạy bước dữ liệu/term, dùng thư mục `outputs/terms/word_clean/`. Chỉ xem run là hoàn tất nếu có `manifest.json`. Hướng dẫn tạo run nằm trong [clean_terms.md](clean_terms.md).

| Đầu vào | Dùng cho |
|---|---|
| `counts.npz` | Tần suất term của tài liệu; đầu vào tính BM25 |
| `tfidf.npz` | Vector tài liệu cho VSM; đầu vào fit SVD cho LSI |
| `doc_lengths.npy` | Độ dài tài liệu tính bằng tổng count sau các bộ lọc |
| `term_df.tsv` | DF từ corpus; cột `kept` cho biết term được giữ trong vocabulary |
| `vocabulary.json` | Ánh xạ term → cột ma trận |
| `doc_ids.json` | Ánh xạ hàng → document ID |
| `queries_clean.counts.npz`, `queries_clean.tfidf.npz` | Query gốc đã transform bằng bộ corpus |
| `query_ids.json` | Ánh xạ hàng query → query ID |
| `config.json`, `manifest.json` | Kiểm tra cấu hình và nguồn artifact |

Ma trận là **document × term**, không phải term × document. Với run đã đo: count/TF-IDF corpus có shape `(61425, 24075)`; query test có shape `(640, 24075)`.

ID không được sắp xếp lại độc lập với ma trận. Ví dụ, hàng `i` trong `counts.npz` và `tfidf.npz` luôn thuộc `doc_ids[i]`.

## Nạp và transform

Chạy Python từ thư mục gốc repo:

```python
import json
import numpy as np
from scipy.sparse import load_npz
from src.terms import TermIndex

index = TermIndex("outputs/terms/word_clean")
corpus_counts = load_npz(index.directory / "counts.npz")
corpus_tfidf = load_npz(index.directory / "tfidf.npz")
doc_lengths = np.load(index.directory / "doc_lengths.npy")
query_ids = json.loads((index.directory / "query_ids.json").read_text(encoding="utf-8"))
clean_tfidf = load_npz(index.directory / "queries_clean.tfidf.npz")

# Sau này, dùng cùng API cho query lỗi; không gọi fit lại.
q_counts, q_tfidf = index.transform(["Không có bảo hiểm y tế bị phạt thế nào?"])
assert corpus_counts.shape[0] == len(index.doc_ids)
assert q_tfidf.shape[1] == corpus_tfidf.shape[1]
```

`TermIndex.transform(texts)` trả về `(counts, tfidf)`, đều là ma trận sparse CSR. Danh sách `texts` nên là một batch không rỗng; query rỗng hoặc toàn OOV được biểu diễn bằng hàng vector 0. Caller giữ ID theo thứ tự batch. Hàm không thay đổi vocabulary và IDF.

Loader kiểm tra hash và phiên bản dependency; dùng môi trường từ `requirements-terms-lock.txt`. Chỉ nạp artifact/joblib do mình build hoặc từ nguồn tin cậy.

## Mỗi mô hình sử dụng thế nào?

| Mô hình | Việc phần mô hình cần làm |
|---|---|
| VSM | Tính cosine giữa query TF-IDF và corpus TF-IDF; hai phía mặc định đã chuẩn hóa L2 |
| BM25 | Tính điểm từ count, DF, độ dài tài liệu và `k1`, `b`; công thức IDF của BM25 được ghi riêng |
| LSI | Fit SVD trên TF-IDF corpus, lưu phép chiếu; query chỉ đi qua phép chiếu đã fit rồi tính cosine |

Không đưa TF-IDF vào thay tần suất term của BM25. Không fit lại IDF/SVD khi chuyển từ query gốc sang query lỗi. Tránh chuyển toàn bộ ma trận corpus thành dense; dùng phép toán sparse hoặc theo batch.

## Giao diện bàn giao tiếp cho đánh giá

Các retriever chưa được triển khai trong commit dữ liệu/term. Khi làm phần đó, dùng schema đã thống nhất trong TEAM_TASKS:

```json
{"qid": "query-id", "doc_id": "document-id", "rank": 1, "score": 12.34}
```

- `rank` bắt đầu từ 1; score giảm dần, khi bằng điểm dùng `doc_id` tăng dần.
- Không lặp một document ID trong kết quả của cùng query.
- Lưu riêng trạng thái mỗi query, kể cả query vector 0 hoặc không có kết quả.
- Chạy trên toàn bộ corpus, giữ ít nhất top 100 tài liệu hợp lệ để tính Recall@100.
- Mọi điều kiện dùng cùng bộ ID query và qrels của nguồn `zalo_legal_2021`; không ghép với bộ 788 query cũ.

Query vector 0 cho kết quả rỗng và điểm đánh giá 0, không lấy danh sách tài liệu có score 0 tùy ý để lấp top-k. Lỗi pipeline thiếu query phải được phát hiện riêng.

## Checklist trước khi tích hợp

- [ ] Đọc đúng thứ tự ID hàng/cột và đúng nhánh `word`.
- [ ] VSM có ví dụ xếp hạng nhỏ kiểm tra được bằng tay.
- [ ] BM25 dùng count và ghi công thức/độ dài/`k1`, `b`.
- [ ] LSI chỉ fit corpus, lưu SVD để dùng lại.
- [ ] Giữ query không có kết quả trong đánh giá.
- [ ] Lấy mốc query gốc trước, rồi dùng lại cùng bộ fit cho query nhiễu.

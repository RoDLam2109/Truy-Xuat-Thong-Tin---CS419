# Chuẩn bị term theo từ cho query gốc

## Mục tiêu của bước hiện tại

Đưa corpus và query gốc về cùng cách biểu diễn **từ tiếng Việt**. Ví dụ đã kiểm tra bằng PyVi:

```text
Không có bảo hiểm y tế bị phạt 5 triệu đồng?
→ [không, có, bảo_hiểm, y_tế, bị, phạt, 5, triệu, đồng]
```

Dấu `_` nối các tiếng thuộc cùng một từ theo dự đoán của bộ tách từ. Bộ tách từ có thể dự đoán sai; không tự đặt thêm cụm pháp lý bằng tay.

Với **TF-IDF/BM25**, không huấn luyện từ các cặp query đúng–tài liệu như mô hình neural:

1. **Corpus** cung cấp vocabulary, DF, tần suất term và độ dài tài liệu.
2. TF-IDF được fit trên ma trận count của corpus.
3. **Query gốc** chỉ được tách từ và transform bằng vocabulary/IDF đã fit để lấy mốc đánh giá sạch sau này.
4. Khi làm query lỗi, dùng lại cùng corpus, cấu hình, vocabulary và IDF; không fit lại.

Đối với LSI, phần SVD sẽ được fit thêm trên TF-IDF corpus ở module mô hình. Bước hiện tại chưa triển khai VSM/BM25/LSI để xếp hạng hay tính MRR/nDCG/Recall; chưa đánh giá query nhiễu.

## Dataset và cách chia

Nguồn triển khai: [minhnguyent546/zalo-ai-legal-text-retrieval-2021](https://huggingface.co/datasets/minhnguyent546/zalo-ai-legal-text-retrieval-2021).

Repo trước đây mô tả một bản GreenNode với 788 query. Hai bản không được trộn. Pipeline mới lưu riêng tại `data/zalo_legal_2021/`, không ghi vào `data/zalo_legal/` cũ.

| File mới | Vai trò |
|---|---|
| `corpus.jsonl` | Toàn bộ tài liệu; schema `id`, `title`, `text` |
| `queries_all.jsonl` | Toàn bộ query gốc; schema `id`, `text` |
| `queries_train.jsonl` | Query có ID trong qrels train, dùng phát triển/chọn cấu hình |
| `queries_clean.jsonl` | **Query gốc thuộc test**, dùng làm tập sạch đánh giá cuối |
| `qrels_train.tsv`, `qrels_test.tsv` | Header `query-id`, `corpus-id`, `score` |
| `manifest.json` | Commit dataset thực tế, số bản ghi, SHA-256 |

Tên split `train` của subset `queries` trên Hugging Face chỉ là cách đóng gói toàn bộ bảng query. Script **chia query theo ID của qrels**, không lấy tên split đó làm tập huấn luyện. Hai tập ID train/test phải rời nhau. Nhãn test chỉ được kiểm tra tính hợp lệ; không dùng để chọn term.

Các ID được giữ nguyên. Query trùng ID và cùng nội dung chỉ giữ một bản, có thống kê; trùng ID khác nội dung, corpus trùng ID hoặc qrels tham chiếu ID không tồn tại sẽ báo lỗi.

“Clean” nghĩa là query gốc chưa được script gây nhiễu, không bảo đảm mọi câu đã hoàn hảo về chính tả. Không sửa thủ công câu test sau khi nhìn kết quả truy xuất.

## Quy tắc term đã triển khai

Cấu hình nằm trong [`configs/terms.json`](../configs/terms.json):

1. Chuẩn hóa Unicode **NFC**.
2. Chuyển chữ thường; gộp khoảng trắng.
3. Dùng **PyVi 0.1.1** để tách từ.
4. Lấy các term gồm chữ/số, cho phép `_` nối các tiếng. Dấu câu là ranh giới, không phải term; mã có dấu `/`, `-` được tách thành các thành phần.
5. Giữ số và từ một ký tự. Giữ từ phủ định/điều kiện như `không`, `chưa`, `được`, `phải`.
6. Mặc định **không loại stopword**, `min_df=1`, `max_df=1.0`: giữ toàn bộ từ vựng trích từ corpus.

Không stemming, lemmatization, bỏ dấu, khôi phục dấu hoặc sửa chính tả. NFC không tự thống nhất `hoà` thành `hòa`.

Title và text được tách từ **riêng rồi nối danh sách term**; như vậy không tạo từ ghép qua ranh giới hai trường. Không tăng trọng số title. `_` có sẵn trong văn bản gốc được xem là ranh giới; `_` nối tiếng trong output do PyVi sinh ra.

Danh sách `stopwords` nếu bổ sung phải dùng dạng term sau tách từ, ví dụ `bảo_hiểm`; không chọn danh sách hay ngưỡng bằng điểm test. Đổi cấu hình thì tạo bộ artifact mới và chạy lại mốc sạch. Trong phiên bản này, `keep_numbers=false` sẽ bỏ các term chứa chữ số; baseline để `true`.

## Chạy trên Linux/macOS

Đứng ở thư mục gốc repo. Dùng Python 3.12 để khớp môi trường và file lock đã kiểm thử.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-terms-lock.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/build_index.py --qrels data/zalo_legal_2021/qrels_test.tsv
```

Nếu dùng `uv`, thay hai lệnh đầu bằng:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-terms-lock.txt
```

Trên Windows PowerShell, dùng `py -3.12 -m venv .venv` rồi thay `.venv/bin/python` bằng `.\.venv\Scripts\python.exe`.

`requirements-terms.txt` liệt kê dependency chính; file `requirements-terms-lock.txt` chốt toàn bộ phiên bản đã chạy. Đây là dependency của phần dữ liệu/term, không thay file dependency chung của nhóm.

Lệnh mặc định tải dataset vào cache và xuất dữ liệu canonical. Script resolve revision thành commit SHA trước khi nạp các subset; có thể tái lập đúng bản bằng:

```bash
.venv/bin/python scripts/prepare_data.py --revision <commit-trong-manifest> --output-dir data/zalo_legal_2021_v2
```

Nếu đã có JSONL theo schema chung, bỏ qua download và chỉ định đường dẫn:

```bash
.venv/bin/python scripts/build_index.py --corpus data/zalo_legal/corpus.jsonl --queries data/zalo_legal/queries_clean.jsonl --qrels data/zalo_legal/qrels_test.tsv --output-dir outputs/terms/word_local_v1
```

**Không ghi đè run:** cả hai script từ chối thư mục output đã tồn tại. Nếu chạy lỗi giữa chừng, thư mục có thể chứa file dở dang; chỉ run có `manifest.json` cuối cùng mới hoàn tất. Chọn thư mục mới để chạy lại. Cache tải dataset được phép dùng lại.

## Artifact bàn giao cho phần mô hình

Trong `outputs/terms/word_clean/`:

| File | Nội dung |
|---|---|
| `corpus.terms.jsonl` | `id`, `tokens` trước lọc DF, giữ term lặp |
| `vocabulary.json` | Term → chỉ số cột, sắp term theo thứ tự từ điển |
| `term_df.tsv` | Tất cả term corpus, DF và cờ được giữ |
| `counts.npz` | CSR count, hàng = tài liệu, cột = term |
| `tfidf.npz` | CSR TF-IDF corpus, cùng thứ tự hàng/cột với count |
| `doc_ids.json` | `doc_ids[i]` là ID hàng i |
| `doc_lengths.npy` | Tổng count sau stopword/DF filtering, tính cả lặp |
| `idf.npy`, `tfidf_transformer.joblib` | IDF và transformer fit từ corpus |
| `queries_clean.terms.jsonl` | Các term query gốc |
| `queries_clean.counts.npz`, `queries_clean.tfidf.npz` | Ma trận query theo vocabulary corpus |
| `query_ids.json` | ID theo thứ tự hàng của ma trận query |
| `query_stats.jsonl`, `stats.json` | OOV theo lần xuất hiện token, query vector rỗng và thống kê corpus |
| `config.json`, `manifest.json` | Cấu hình, dependency, hash input/code/artifact |

DF đếm **số tài liệu có term**, không phải số lần term xuất hiện. Với cấu hình mặc định:

```text
tf(t,d) = số lần t xuất hiện trong d
idf(t) = ln((1 + N) / (1 + df(t))) + 1
tfidf(t,d) = tf(t,d) × idf(t), sau đó chuẩn hóa L2 từng hàng
```

BM25 phải dùng `counts.npz`, DF và độ dài tài liệu để tính công thức riêng; không dùng trọng số TF-IDF thay cho count/IDF của BM25. LSI nhận `tfidf.npz` để fit SVD sau này.

## Transform query bằng bộ đã fit

Từ thư mục gốc repo:

```python
from scipy.sparse import load_npz
from src.terms import TermIndex

index = TermIndex("outputs/terms/word_clean")
counts = load_npz(index.directory / "counts.npz")
doc_tfidf = load_npz(index.directory / "tfidf.npz")
q_counts, q_tfidf = index.transform(["Không có bảo hiểm y tế bị phạt thế nào?"])
```

`transform()` không fit và không thêm term query vào vocabulary. Query chỉ có term ngoài từ vựng hoặc không còn term sẽ có vector 0; giữ ID query đó để phần đánh giá tính điểm 0, không bỏ khỏi mẫu số.

Chỉ nạp file joblib tự build hoặc từ nguồn tin cậy. Loader kiểm tra hash artifact và phiên bản dependency trước khi nạp; nếu môi trường khác, dùng lock để tạo lại môi trường hoặc build bộ mới.

## Kiểm tra và bước kế tiếp

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Tests kiểm tra tách từ thật bằng PyVi; Unicode tổ hợp/dựng sẵn; giữ phủ định/số; TF khác DF; công thức IDF; thứ tự ID; query OOV không làm đổi vocabulary; lưu/nạp và tái lập ma trận; kiểm tra schema/split/qrels; từ chối ghi đè hoặc artifact bị sửa. Tests sinh nhiễu cũ cũng được chạy để kiểm tra tích hợp.

Sau khi phần mô hình hoàn tất: chạy query gốc lấy mốc sạch trước, rồi mới đánh giá query lỗi với bộ chỉ mục cố định. Nhánh character n-gram vẫn là công việc tiếp theo, chưa được triển khai trong module word này.

Tham khảo: [PyVi](https://github.com/trungtv/pyvi), [TF-IDF của scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).

Kết quả đã chạy trên toàn bộ corpus: [clean_terms_run.md](clean_terms_run.md).

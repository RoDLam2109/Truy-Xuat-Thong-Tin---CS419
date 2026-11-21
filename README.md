# Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries

Đồ án CS419 nghiên cứu mức suy giảm chất lượng truy xuất tài liệu pháp luật khi câu truy vấn tiếng Việt chứa lỗi gõ, viết tắt, bỏ dấu hoặc nhiễu hỗn hợp.

## 1. Nhóm đã làm được gì?

| Phần việc | Đã triển khai | Kết quả hiện tại | Đã chốt phương pháp tốt nhất? |
|---|---|---|---|
| **Term Representation** (phần phụ trách term) | Word/PyVi và character n-gram 3–5; lọc vocabulary bằng DF | **16.498 word terms** và **111.181 char n-grams** | **Chưa.** Đã tạo hai biểu diễn để so sánh, chưa chọn một biểu diễn tối ưu |
| **Model Evaluation** (nhánh này) | VSM, BM25, LSI; mỗi mô hình kết hợp với từng biểu diễn | **32/48** cấu hình validation đã chạy; **BM25 + word** có điểm clean tốt nhất trong các cấu hình đã đo | **Chưa.** Mới có baseline, chưa tuning và đánh giá hết các ứng viên |

**Kết luận hiện tại:** với 482 validation queries, **BM25 + word** đạt **nDCG@10 = 0.6626** trên query clean, cao nhất trong những cấu hình đã chạy. **Character n-gram + VSM** cải thiện so với word + VSM khi gặp lỗi Telex và nhiễu hỗn hợp; nhưng việc bỏ dấu vẫn làm giảm điểm rất mạnh. Đây là **kết quả validation**, không phải mô hình đã được chọn cuối cùng.

## 2. Term đã được chọn như thế nào?

Cả hai phương pháp đều xây dựng vocabulary từ **61.425 documents**, không dùng qrels:

| Biểu diễn | Cách tạo term | Trước lọc | Sau lọc |
|---|---|---:|---:|
| **Word** | Tách từ bằng PyVi 0.1.1; bỏ 5 stopwords | 26.143 | **16.498** |
| **Character n-gram** | Các đoạn 3–5 ký tự trong từng token, có khoảng trắng biên | 195.945 | **111.181** |

**Bước lọc term:** dùng Document Frequency (DF), giữ những term xuất hiện trong ít nhất 2 documents và không vượt 95% corpus (`min_df=2`, `max_df=0.95`). Đây là **feature selection trong từng biểu diễn**, không phải thí nghiệm chọn Word thắng Character n-gram (hoặc ngược lại).

Cách tách term, cấu hình, ví dụ và artifact: [PREPROCESSING.md](PREPROCESSING.md). Mã nguồn: [scripts/prepare_terms.py](scripts/prepare_terms.py).

## 3. Phần Model Evaluation đã thử những mô hình nào?

| Model | Cách truy xuất | Cấu hình baseline |
|---|---|---|
| **VSM** | TF-IDF + cosine similarity | IDF fit trên corpus, chuẩn hóa L2 |
| **BM25** | Term frequency, IDF và độ dài document | `k1=1.2`, `b=0.75` |
| **LSI** | TF-IDF → TruncatedSVD → cosine similarity | 64 chiều, seed 42 |

Mỗi mô hình dự kiến chạy với **cả Word và Character n-gram**, trên **8 điều kiện query** (clean + 7 loại nhiễu). Tổng thiết kế: **2 × 3 × 8 = 48** cấu hình.

| Tổ hợp model/term | Clean nDCG@10 | Trạng thái |
|---|---:|---|
| Word + VSM | 0.5008 | Đã chạy 8/8 điều kiện |
| **Word + BM25** | **0.6626** | Đã chạy 8/8 điều kiện |
| Word + LSI | 0.2420 | Đã chạy 8/8 điều kiện |
| Character n-gram + VSM | 0.4833 | Đã chạy 8/8 điều kiện |
| Character n-gram + BM25 | — | Chưa chạy full-corpus |
| Character n-gram + LSI | — | Chưa chạy full-corpus |

Dấu **—** nghĩa là **chưa có kết quả**, không phải điểm 0. Các mô hình hiện tại **không train trên query–qrel pairs**: vocabulary, IDF, BM25 statistics và SVD được xây dựng từ corpus. Qrels được dùng để **tính metrics** trên validation. Chưa có supervised training hoặc hyperparameter search.

Mã nguồn: [src/retrieval/models.py](src/retrieval/models.py), [scripts/evaluate_models.py](scripts/evaluate_models.py).

### So sánh robustness: nDCG@10

Cùng 482 validation queries, toàn bộ corpus 61.425 documents:

| Điều kiện | Word VSM | Word BM25 | Word LSI | Char VSM |
|---|---:|---:|---:|---:|
| **Clean** | 0.5008 | **0.6626** | 0.2420 | 0.4833 |
| No diacritics | 0.0209 | 0.0297 | 0.0019 | 0.0219 |
| Telex typo | 0.3263 | **0.4725** | 0.1376 | 0.4216 |
| Phonetic | 0.4925 | **0.6511** | 0.2325 | 0.4803 |
| Abbreviation | 0.5004 | **0.6589** | 0.2358 | 0.4785 |
| Mixed low | 0.4198 | **0.5875** | 0.1914 | 0.4528 |
| Mixed medium | 0.3119 | **0.4698** | 0.1204 | 0.3996 |
| Mixed high | 0.2284 | 0.3534 | 0.0803 | **0.3542** |

**Nhận xét:** BM25 + word hiện đứng đầu về clean, MRR@10 và Recall@100 trong các baseline đã thử. Char VSM tốt hơn Word VSM khi có lỗi Telex và mixed noise, nhưng clean thấp hơn. Không thể gọi BM25 + word là mô hình cuối cùng khi vẫn còn 16 cấu hình chưa đánh giá.

Bảng MRR@10, Recall@100, phân tích và nguồn artifact: [docs/validation_results.md](docs/validation_results.md).

## 4. Dataset và lý do có validation split

Sử dụng [GreenNode Zalo Legal Retrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), bản nhóm đã chuẩn hóa. Revision, các file nguồn và checksum ghi trong [configs/dataset.json](configs/dataset.json). Dữ liệu nhóm: [Google Drive](https://drive.google.com/drive/folders/1fdZjJ1i7o44f3jNwbZL2okPxKcNfkv7H).

| Tập dữ liệu | Queries | Qrels | Vai trò hiện tại |
|---|---:|---:|---|
| Train chuẩn hóa từ GreenNode | 2.408 | 2.480 | Nguồn tạo development split |
| Development train (80%) | 1.926 | 1.985 | Đã tạo nhưng **chưa dùng** trong baseline hiện tại |
| **Validation (20%)** | **482** | **495** | **Đang dùng thực tế** để so sánh mô hình và noise |
| Test giữ nguyên | 788 | 793 | **Chưa chạy đánh giá cuối cùng** |

Corpus **61.425 documents dùng chung** cho tất cả các tập query. Split 80/20 được **bổ sung trên nhánh model-evaluation**, không thay đổi file train/test đã chốt của nhóm. Seed 42 và nhóm theo nội dung query chuẩn hóa; 7 bộ noise validation được sinh riêng. Không dùng test để chọn tham số.

Chi tiết dữ liệu và đường dẫn: [data/README.md](data/README.md).

## 5. Chạy lại và kiểm chứng

Yêu cầu Python 3.11+, [uv](https://docs.astral.sh/uv/) và dữ liệu đặt trong `data/zalo_legal/`. Các file dữ liệu lớn không được commit trên GitHub.

```bash
uv sync --locked
uv run python -X utf8 scripts/prepare_validation.py
uv run python -X utf8 scripts/prepare_terms.py --workers 4
uv run python -X utf8 scripts/build_retrieval_index.py --branch word --output-dir outputs/index/word
uv run python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/word --output-dir outputs/retrieval/val_word
uv run python -m unittest discover -s tests -q
```

Để đánh giá character n-gram, dùng `--branch char_ngram` và đường dẫn index tương ứng. Kết quả, index và validation split được lưu trong `outputs/` (Git-ignored), không ghi đè dữ liệu nguồn. Script từ chối ghi đè thư mục kết quả hiện có; khi chạy lại, chọn đường dẫn mới hoặc chủ động xử lý artifact cũ.

**Đã kiểm chứng:** 37/37 unit tests, 67/67 dataset audit checks. Hướng dẫn chi tiết: [docs/model_evaluation.md](docs/model_evaluation.md).

## 6. Việc còn lại

1. Chạy **16 cấu hình full-corpus** cho character n-gram + BM25/LSI.
2. So sánh chất lượng clean và mức suy giảm theo noise; thực hiện tuning trên validation nếu cần.
3. Kiểm tra thủ công liệu noise tổng hợp có làm thay đổi nghĩa query không.
4. Chốt phương pháp và tham số rồi mới đánh giá test cuối cùng. Test đã từng được xem xét trong giai đoạn trước nên không được mô tả là hoàn toàn chưa từng tiếp xúc.

**Trạng thái cuối cùng: Đã hoàn thành Term Representation và baseline Model Evaluation một phần; chưa hoàn thành Model Selection.**

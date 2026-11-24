# Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries

Đồ án CS419 nghiên cứu mức suy giảm chất lượng truy xuất tài liệu pháp luật khi câu truy vấn tiếng Việt chứa lỗi gõ, viết tắt, bỏ dấu hoặc nhiễu hỗn hợp.

**Cập nhật 24/11/2026:** giữ đủ **8 điều kiện**, **482 query validation** và **788 query test**. Chỉ thay nhiễu `no_diacritics`: chọn token với xác suất 25% để bỏ dấu, thay vì bỏ dấu cả câu; sáu bộ nhiễu còn lại giữ nguyên. File đang dùng vẫn là `data/zalo_legal/noise/queries_no_diacritics.jsonl` và `outputs/lam/splits/seed42/val/noise/queries_no_diacritics.jsonl`. Bản cũ lưu trong `outputs/lam/archive/legacy_noise/`. Chi tiết và cách chạy: [bỏ dấu một phần](docs/lam_partial_diacritics.md).

**Các điểm metrics bên dưới thuộc lần chạy trước khi thay bộ bỏ dấu.** Cần chạy thí nghiệm vào thư mục mới để có điểm cập nhật; không dùng điểm bỏ dấu cũ để đánh giá dữ liệu mới. Bộ thử riêng 230 câu đã được lưu vào archive và không phải đầu vào chính.

## 1. Trạng thái hoàn thành

**Đã hoàn thành phần code, tích hợp và chạy thí nghiệm baseline:** đủ **48/48 lượt validation và 48/48 lượt test** trên toàn bộ corpus. Cập nhật README ngày **23/11/2026** theo kết quả chạy ngày 21/11/2026.

| Phần việc | Trạng thái | Kết quả |
|---|---|---|
| **Term Representation** | Hoàn thành | Word/PyVi và character n-gram 3–5; vocabulary gồm **16.498 word terms** và **111.181 char n-grams** |
| **Retrieval Models** | Hoàn thành | VSM, BM25, LSI chạy với cả hai biểu diễn term |
| **Nhiễu và tích hợp** | Hoàn thành pipeline | Clean và 7 điều kiện nhiễu; giữ nguyên ID, thứ tự query và qrels |
| **Metrics và báo cáo** | Hoàn thành | MRR@10, nDCG@10, Recall@100, suy giảm, ranking, bảng, biểu đồ và phân tích lỗi |
| **Thí nghiệm** | Hoàn thành | 482 query/lượt validation; 788 query/lượt test; **61.425 documents** dùng chung |

**Kết quả chính:** Word + BM25 có clean nDCG@10 cao nhất: **0.6626 trên validation**, **0.6444 trên test**. Theo tiêu chí trung bình nDCG@10 trên bảy điều kiện nhiễu đã ghi trước khi chạy validation, **character n-gram + BM25** đứng đầu: **0.4915 trên validation**, **0.4990 trên test**. Tiêu chí này cần nhóm xác nhận trước khi dùng làm kết luận lựa chọn chính thức.

Xem [tổng hợp kết quả đầy đủ](docs/lam_metrics_results.md), [báo cáo validation](docs/lam_results/val/report.md) và [báo cáo test](docs/lam_results/test/report.md).

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

Mỗi mô hình đã chạy với **cả Word và Character n-gram**, trên **8 điều kiện query** (clean + 7 loại nhiễu). Tổng thiết kế: **2 × 3 × 8 = 48 lượt cho mỗi split**.

| Tổ hợp model/term | Validation clean nDCG@10 | Test clean nDCG@10 | Trạng thái |
|---|---:|---:|---|
| Word + VSM | 0.5008 | 0.4484 | Đủ 8/8 điều kiện ở mỗi split |
| **Word + BM25** | **0.6626** | **0.6444** | Đủ 8/8 điều kiện ở mỗi split |
| Word + LSI | 0.2420 | 0.2122 | Đủ 8/8 điều kiện ở mỗi split |
| Character n-gram + VSM | 0.4833 | 0.4596 | Đủ 8/8 điều kiện ở mỗi split |
| Character n-gram + BM25 | 0.6267 | 0.6308 | Đủ 8/8 điều kiện ở mỗi split |
| Character n-gram + LSI | 0.2637 | 0.2446 | Đủ 8/8 điều kiện ở mỗi split |

Vocabulary, IDF, BM25 statistics và SVD được xây dựng từ corpus. Query sạch/nhiễu chỉ transform và truy xuất; qrels dùng để tính metrics. Các baseline dùng tham số cố định, không có supervised training hoặc hyperparameter search. Cấu hình lựa chọn theo validation được lưu trong [selection.json](docs/lam_results/val/selection.json) trước khi chạy test; không chọn lại cấu hình theo điểm test.

Mã nguồn: [src/retrieval/models.py](src/retrieval/models.py), [scripts/run_experiments.py](scripts/run_experiments.py), [src/evaluation/runs.py](src/evaluation/runs.py), [scripts/report_experiments.py](scripts/report_experiments.py).

### So sánh robustness: nDCG@10

Cùng 482 validation queries, toàn bộ corpus 61.425 documents:

| Điều kiện | Word VSM | Word BM25 | Word LSI | Char VSM | Char BM25 | Char LSI |
|---|---:|---:|---:|---:|---:|---:|
| **Clean** | 0.5008 | **0.6626** | 0.2420 | 0.4833 | 0.6267 | 0.2637 |
| No diacritics | 0.0209 | 0.0297 | 0.0019 | 0.0219 | **0.0423** | 0.0059 |
| Telex typo | 0.3263 | 0.4725 | 0.1376 | 0.4216 | **0.5435** | 0.2023 |
| Phonetic | 0.4925 | **0.6511** | 0.2325 | 0.4803 | 0.6250 | 0.2622 |
| Abbreviation | 0.5004 | **0.6589** | 0.2358 | 0.4785 | 0.6248 | 0.2554 |
| Mixed low | 0.4198 | 0.5875 | 0.1914 | 0.4528 | **0.5980** | 0.2257 |
| Mixed medium | 0.3119 | 0.4698 | 0.1204 | 0.3996 | **0.5317** | 0.1858 |
| Mixed high | 0.2284 | 0.3534 | 0.0803 | 0.3542 | **0.4756** | 0.1560 |

**Nhận xét:** Word + BM25 tốt nhất trên clean; character n-gram + BM25 có trung bình noisy nDCG@10 tốt nhất và chịu nhiễu Telex/mixed tốt hơn Word + BM25 trong lần chạy này. Bỏ dấu vẫn làm giảm điểm mạnh ở cả sáu baseline. Phonetic và abbreviation có nhiều query không đổi nên báo cáo bổ sung cả điểm trên query thực sự thay đổi.

Bảng đủ ba metrics: [validation](docs/lam_results/val/report.md), [test](docs/lam_results/test/report.md). Phân tích: [docs/lam_error_analysis.md](docs/lam_error_analysis.md). Các tài liệu [validation_results.md](docs/validation_results.md) và [model_evaluation.md](docs/model_evaluation.md) giữ nội dung giai đoạn trước, khi mới chạy 32/48 lượt; kết quả cập nhật nằm trong [lam_metrics_results.md](docs/lam_metrics_results.md).

## 4. Dataset và lý do có validation split

Sử dụng [GreenNode Zalo Legal Retrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), bản nhóm đã chuẩn hóa. Revision, các file nguồn và checksum ghi trong [configs/dataset.json](configs/dataset.json). Dữ liệu nhóm: [Google Drive](https://drive.google.com/drive/folders/1fdZjJ1i7o44f3jNwbZL2okPxKcNfkv7H).

| Tập dữ liệu | Queries | Qrels | Vai trò hiện tại |
|---|---:|---:|---|
| Train chuẩn hóa từ GreenNode | 2.408 | 2.480 | Nguồn tạo development split |
| Development train (80%) | 1.926 | 1.985 | Đã tạo nhưng **chưa dùng** trong baseline hiện tại |
| **Validation (20%)** | **482** | **495** | **Đang dùng thực tế** để so sánh mô hình và noise |
| Test giữ nguyên | 788 | 793 | **Đã đánh giá đủ 48/48 lượt** sau khi khóa cấu hình theo validation |

Corpus **61.425 documents dùng chung** cho tất cả các tập query. Split 80/20 được **bổ sung trên nhánh model-evaluation**, không thay đổi file train/test đã chốt của nhóm. Seed 42 và nhóm theo nội dung query chuẩn hóa; 7 bộ noise validation được sinh riêng. Không dùng test để chọn tham số.

Chi tiết dữ liệu và đường dẫn: [data/README.md](data/README.md).

## 5. Chạy lại và kiểm chứng

Yêu cầu Python 3.11+, [uv](https://docs.astral.sh/uv/) và dữ liệu đặt trong `data/zalo_legal/`. Các file dữ liệu lớn không được commit trên GitHub.

```powershell
uv sync --locked
uv pip install --python .venv/Scripts/python.exe -r configs/evaluation-requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
```

Các lệnh tạo split, vocabulary, index, chạy đủ 48 lượt validation/test và xuất báo cáo nằm trong [docs/lam_evaluation.md](docs/lam_evaluation.md). Runner hỗ trợ `--resume` với kiểm tra cấu hình và checksum để dùng lại lượt đã hoàn thành. Khi chạy mới, chọn thư mục đầu ra chưa tồn tại; không ghi đè dữ liệu hoặc kết quả cũ.

**Đã kiểm chứng:** **49/49 unit/integration tests**, **67/67 dataset audit checks**; 32 điểm nDCG@10 baseline cũ khớp ở bốn chữ số thập phân. Sáu model đã fit trùng SHA-256 giữa validation/test. Kết quả kiểm tra: [docs/lam_results/verification/](docs/lam_results/verification/).

## 6. Kết quả lưu ở đâu?

| Đường dẫn | Nội dung |
|---|---|
| [docs/lam_metrics_results.md](docs/lam_metrics_results.md) | Tổng hợp kết quả và trạng thái hoàn thành |
| [docs/lam_results/val/](docs/lam_results/val/) và [docs/lam_results/test/](docs/lam_results/test/) | Snapshot bảng CSV, báo cáo, biểu đồ và mẫu lỗi để chia sẻ trong repo |
| `outputs/lam/experiments/<val hoặc test>/<term>/<model>/<condition>/` | Ranking, query status và metrics từng lượt |
| `outputs/lam/reports/<val hoặc test>/` | `summary.csv`, `paired_drops.csv`, `worst_queries.csv`, báo cáo và biểu đồ đầy đủ |
| `outputs/lam/index/`, `outputs/lam/processed/`, `outputs/lam/splits/seed42/` | Index, vocabulary, dữ liệu đã xử lý và split |

Artifact lớn trong `outputs/` được Git bỏ qua. Code, cấu hình và snapshot báo cáo nhỏ trong `docs/lam_results/` dùng để bàn giao qua GitHub; dữ liệu/index/model/ranking đầy đủ có thể chia sẻ qua Drive hoặc tái tạo theo hướng dẫn.

## 7. Xác nhận kết luận và giới hạn

Phần triển khai và thí nghiệm baseline đã hoàn thành. Nhóm cần xác nhận tiêu chí lựa chọn chính thức và đọc lại các mẫu nhiễu có nguy cơ đổi nghĩa trong [docs/lam_noise_review.md](docs/lam_noise_review.md). Tài liệu hiện có nhận xét đọc mẫu của trợ lý, chưa phải xác nhận ngữ nghĩa độc lập của nhóm.

Chưa tuning tham số hoặc kiểm định ý nghĩa thống kê; đây là các hướng mở rộng. Test đã từng được xem xét trong giai đoạn trước nên không được mô tả là hoàn toàn chưa từng tiếp xúc. Mọi kết luận robustness cần xem đồng thời điểm sạch, điểm nhiễu, tỷ lệ query thực sự thay đổi và giới hạn của qrels.

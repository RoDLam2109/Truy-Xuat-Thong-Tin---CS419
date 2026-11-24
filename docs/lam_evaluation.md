# Bàn giao phần nhiễu, metrics và tích hợp của Lam

**Cập nhật 24/11/2026:** vẫn chạy 48 lượt trên tám điều kiện, giữ đủ 482 query validation và 788 query test. Riêng bỏ dấu đã chuyển sang chọn token với xác suất 25%; term/model và sáu bộ nhiễu còn lại giữ nguyên. Kết quả ngày 21/11 bên dưới thuộc dữ liệu trước thay đổi. Dùng các thư mục output mới và lệnh trong [lam_partial_diacritics.md](lam_partial_diacritics.md) để đánh giá lại.

Tài liệu bổ sung ngày 21/11/2026. Các file có trước phiên làm việc được giữ nguyên, gồm README, TEAM_TASKS, cấu hình dữ liệu/term/noise, code tiền xử lý, mô hình và công thức metrics.

Kết quả đã chạy đủ 48 lượt validation và 48 lượt test: [lam_metrics_results.md](lam_metrics_results.md).

## Phạm vi theo phân công

`TEAM_TASKS.md` hiện tại chỉ giữ tổng quan. Bản phân công trong lịch sử Git (`git show ba24e19:TEAM_TASKS.md`) giao Lam phần nhiễu, đánh giá và tích hợp: kiểm tra nhiễu, MRR@10/nDCG@10/Recall@100, suy giảm, script chạy 48 lượt, bảng/biểu đồ/phân tích lỗi và tài liệu tái lập. Tên đường dẫn trong bản cũ được đối chiếu với API thực tế đang có; không thay code của Kiên/Vũ để ép theo tên cũ.

Phần đã có được sử dụng lại:

- `scripts/prepare_terms.py`, `scripts/build_retrieval_index.py`: tạo term và index.
- `src/retrieval/models.py`: VSM, BM25 và LSI.
- `src/retrieval/evaluation.py`: công thức metrics, macro average và degradation.
- `scripts/prepare_validation.py`, `scripts/generate_noise.py`: split và nhiễu seed 42.

## Phần bổ sung

| File mới | Chức năng |
|---|---|
| `scripts/run_experiments.py` | Chạy hai term × ba model × tám điều kiện; lưu model, ranking, query status, cấu hình và checksum. |
| `src/evaluation/runs.py` | Kiểm tra hợp đồng đầu ra và tính metrics bằng công thức có sẵn. |
| `scripts/evaluate.py` | Tính lại metrics từ ranking đã xuất, không chạy retrieval. |
| `scripts/report_experiments.py` | Tổng hợp 48 lượt; paired drops, changed-only metrics, biểu đồ và mẫu lỗi. |
| `tests/test_evaluate.py` | Kiểm tra cutoff, nhiều nhãn, query rỗng/thiếu, hợp đồng ranking, resume, checksum, full pipeline và báo cáo. |
| `configs/evaluation-requirements.txt` | Dependency vẽ biểu đồ bổ sung, giữ nguyên pyproject và lockfile cũ. |
| `docs/lam_noise_review.md`, `.csv` | Ghi nhận đọc 70 mẫu validation và các giới hạn ngữ nghĩa. |

## Quy ước đánh giá

Các tham số baseline giữ nguyên: PyVi/DF/ngram theo cấu hình đã có, BM25 `k1=1.2, b=0.75`, LSI 64 chiều, seed 42, top 100. Không tuning, sửa generator hay tăng độ phủ từ điển trong lần này.

Mỗi nhánh/model chỉ fit một lần cho toàn bộ query clean/noisy của một split; instance được lưu bằng joblib. Query dùng vocabulary/IDF của corpus. Index phải là full corpus có đúng SHA-256; không chấp nhận subset làm benchmark.

Quy tắc xếp hạng validation được ghi vào manifest **trước lượt chạy đầu tiên**: tối đa hóa trung bình nDCG@10 trên bảy điều kiện nhiễu, mỗi điều kiện trọng số bằng nhau; nếu bằng nhau thì ưu tiên clean nDCG@10, rồi tên term/model tăng dần. Đây là quy tắc vận hành cho bài toán robustness, không phải kết luận model tốt nhất cho mọi ứng dụng. Báo cáo vẫn có toàn bộ điểm clean và từng loại nhiễu.

Sau 48 lượt validation, báo cáo xuất `selection.json` khóa tham số, corpus và index trước khi chạy test. Test đánh giá đủ sáu baseline với các tham số cố định để phục vụ so sánh của đồ án; không dùng điểm test để chọn lại cấu hình. Test đã từng được xem xét ở giai đoạn trước, nên không gọi là hoàn toàn chưa từng tiếp xúc.

## Tái lập

Chạy tại root repo, với bộ dữ liệu Lam v1 đang có. Cài core dependency theo lock và phần vẽ biểu đồ theo file mới:

```powershell
uv sync --locked
uv pip install --python .venv/Scripts/python.exe -r configs/evaluation-requirements.txt
```

Các lệnh sau tạo artifact mới. Nếu đã có artifact của lần này thì dùng lại, không chạy lại bước tạo vào cùng thư mục:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_validation.py --output-dir outputs/lam/splits/seed42
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_terms.py --output-dir outputs/lam/processed --workers 4
.\.venv\Scripts\python.exe -X utf8 scripts/build_retrieval_index.py --processed-dir outputs/lam/processed --branch word --output-dir outputs/lam/index/word
.\.venv\Scripts\python.exe -X utf8 scripts/build_retrieval_index.py --processed-dir outputs/lam/processed --branch char_ngram --output-dir outputs/lam/index/char_ngram
.\.venv\Scripts\python.exe -X utf8 scripts/run_experiments.py --output-dir outputs/lam/experiments/val
.\.venv\Scripts\python.exe -X utf8 scripts/report_experiments.py --experiment-dir outputs/lam/experiments/val --output-dir outputs/lam/reports/val
.\.venv\Scripts\python.exe -X utf8 scripts/run_experiments.py --split test --confirm-test-use --selection outputs/lam/reports/val/selection.json --output-dir outputs/lam/experiments/test
.\.venv\Scripts\python.exe -X utf8 scripts/report_experiments.py --experiment-dir outputs/lam/experiments/test --output-dir outputs/lam/reports/test
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
```

Runner mặc định giới hạn BLAS ở bốn thread nếu môi trường chưa đặt giới hạn; không đổi số chiều SVD hay số vòng lặp của model cũ. Có thể dùng `--batch-size` để điều chỉnh RAM. Các lựa chọn branch/model riêng chỉ phục vụ chạy phần thiết kế; báo cáo hoàn chỉnh yêu cầu đủ 48 lượt.

Thêm `--resume` với cùng lệnh runner để tái sử dụng những lượt đã hoàn thành. Runner kiểm tra cấu hình, hash code/đầu vào và hash từng artifact; không chấp nhận dữ liệu hoặc tham số khác. Lượt đang ghi dở chưa có `complete.json` sẽ báo lỗi; giữ nó làm bằng chứng rồi chọn thư mục chạy mới hoặc chủ động di chuyển riêng thư mục chưa hoàn thành trước khi resume. Script không tự xóa output.

## Đầu ra và tính lại metrics

Mỗi run nằm ở `outputs/lam/experiments/<split>/<term>/<model>/<condition>/`:

- `rankings.jsonl`: `{qid, doc_id, rank, score}`; rank bắt đầu từ 1, tối đa 100, không lặp document, score giảm dần, tie theo document ID.
- `queries.jsonl`: `{qid, status, retrieved}` cho **mọi** query, kể cả ranking rỗng. Thiếu query là lỗi pipeline; không tự thay bằng điểm 0.
- `per_query.csv`, `summary.json`: metrics theo query và toàn tập, suy giảm tuyệt đối/phần trăm so với clean cùng term/model.
- `complete.json`: cấu hình và hash các file đã hoàn thành. Chỉ tạo marker sau khi kiểm tra đầu ra thành công.

Ví dụ tính lại một lượt từ ranking:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate.py --rankings outputs/lam/experiments/val/word/bm25/clean/rankings.jsonl --statuses outputs/lam/experiments/val/word/bm25/clean/queries.jsonl --queries outputs/lam/splits/seed42/val/queries_clean.jsonl --qrels outputs/lam/splits/seed42/val/qrels_val.tsv --output-dir outputs/lam/recomputed/val_word_bm25_clean
```

`outputs/lam/reports/<split>/` có `report.md`, `summary.csv`, `paired_drops.csv`, `worst_queries.csv`, `noise_coverage.csv`, `configuration_comparison.csv`, biểu đồ PNG/SVG và provenance. `selection.json` chỉ sinh từ validation. Artifact lớn nằm trong `outputs/` đã được `.gitignore` cũ bỏ qua.

`paired_drops.csv` ghép đúng qid, term, model; không lấy trung bình phần trăm drop từng query để thay cho drop toàn tập. Khi clean bằng 0, drop phần trăm ghi trống (N/A), không gán 0. Drop âm là trường hợp noisy cải thiện điểm. Các cột `changed_only` là phân tích bổ sung; mẫu số chính vẫn gồm mọi query.

`worst_queries.csv` chọn tối đa năm lỗi giảm nDCG@10 mạnh nhất cho mỗi term/model/noise. File có câu sạch/nhiễu, điểm, drop và top 3 document IDs; không phải mẫu ngẫu nhiên. Xem cùng qrels và nội dung corpus để phân biệt mất khớp từ, nhiễu đổi nghĩa và nhãn chưa đầy đủ.

Thông tin phiên bản Python, numpy, scipy, sklearn, PyVi, joblib và SHA-256 code/đầu vào được ghi trong manifest.

## Kiểm chứng đã thực hiện

- **49/49 unit/integration tests pass**: 37 test cũ và 12 test mới. Test tích hợp dùng corpus nhỏ riêng; số liệu đó không được dùng làm benchmark. Nếu chưa cài dependency vẽ biểu đồ, test render báo cáo được đánh dấu skip; các test core vẫn chạy.
- **67/67 dataset audit checks pass** bằng hàm audit hiện có; chỉ ghi kết quả mới vào `outputs/lam/dataset_audit.json`, không gọi bước ghi lại manifest dữ liệu.
- **32/32 giá trị nDCG@10 baseline** khớp tài liệu cũ ở bốn chữ số thập phân. Bảng đối chiếu nằm trong `outputs/lam/baseline_comparison.csv`.
- CLI tính lại metrics từ ranking Word/BM25/clean cho đúng MRR@10 **0.6142527827**, nDCG@10 **0.6626164198**, Recall@100 **0.9457123098**.
- `git diff --exit-code` xác nhận mọi file đã được Git theo dõi trước phiên làm việc vẫn giữ nguyên. Các thay đổi bàn giao là file mới và artifact trong `outputs/lam/`.

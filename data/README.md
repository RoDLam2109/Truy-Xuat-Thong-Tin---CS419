# Bộ dữ liệu Zalo Legal Retrieval

Nhóm sử dụng phiên bản đã chuẩn hóa từ [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn). Revision nguồn, số lượng kỳ vọng và SHA-256 nằm trong [configs/dataset.json](../configs/dataset.json) và [train_manifest.json](zalo_legal/train_manifest.json).

## Cấu trúc dữ liệu dùng chung

Đặt các file trong `data/zalo_legal/`:

| File | Nội dung | Số lượng |
|---|---|---:|
| `corpus.jsonl` | Văn bản pháp luật, dùng chung cho mọi split | 61.425 documents |
| `queries_train.jsonl` | Query train gốc đã loại trùng với test | 2.408 queries |
| `qrels_train.tsv` | Nhãn liên quan của train (có query nhiều positive) | 2.480 qrels |
| `queries_clean.jsonl` | Query test chưa thêm nhiễu tổng hợp | 788 queries |
| `qrels_test.tsv` | Nhãn liên quan của test | 793 qrels |
| `noise/queries_*.jsonl` | 7 loại nhiễu của test, cùng query ID và thứ tự | 788 queries/file |
| `raw/` | File nguồn GreenNode phục vụ đối chiếu | — |
| `train_manifest.json` | Thông tin nguồn, split và checksum | — |

Dữ liệu train ban đầu của GreenNode được làm sạch bằng cách loại 24 query IDs trùng với test (25 dòng qrels). Các nhãn relevant còn lại được giữ đầy đủ. Bộ test gốc và 7 file nhiễu không bị thay đổi trong quá trình phát triển nhánh model-evaluation.

**Cách hiểu các file:** `corpus.jsonl` là kho tài liệu để xây index, không thuộc riêng train/validation/test. Các file `queries_*.jsonl` chứa câu hỏi; `qrels_*.tsv` ánh xạ query ID tới document ID liên quan. “Clean” chỉ có nghĩa là chưa thêm nhiễu nhân tạo, không bảo đảm câu hỏi hoàn toàn không có lỗi tự nhiên.

## Validation dùng cho phát triển mô hình

Không chỉnh trực tiếp các file train/test ở trên. Chạy `scripts/prepare_validation.py` để tách từ tập train đã chuẩn hóa:

| Split | Queries | Qrels | Vị trí |
|---|---:|---:|---|
| Development train | 1.926 | 1.985 | `outputs/splits/seed42/train/` |
| Validation | 482 | 495 | `outputs/splits/seed42/val/` |
| Test | 788 | 793 | `data/zalo_legal/` (giữ nguyên) |

Split dùng seed 42 và nhóm các câu hỏi có cùng văn bản sau chuẩn hóa NFC/casefold/khoảng trắng. Có 7 bộ noisy validation được sinh riêng tại `outputs/splits/seed42/val/noise/`. Không sử dụng test để chọn hyperparameters.

## Lấy và kiểm tra dữ liệu

1. Clone repository rồi lấy nguyên folder từ [Google Drive của nhóm](https://drive.google.com/drive/folders/1fdZjJ1i7o44f3jNwbZL2okPxKcNfkv7H), đặt thành `data/zalo_legal/`. **Git không chứa corpus, queries hoặc qrels lớn.**
2. Đối chiếu các file bằng SHA-256 trong `train_manifest.json`. Để chạy audit có thể tải thêm file nguồn vào `raw/`; cần có metadata sinh nhiễu trong `outputs/noise/` như hướng dẫn dưới đây.
3. Khi cần tạo lại báo cáo noise **mà không ghi đè 7 file test noise**, dùng:

```bash
uv run python -X utf8 scripts/generate_noise.py --input data/zalo_legal/queries_clean.jsonl --qrels data/zalo_legal/qrels_test.tsv --output-dir outputs/noise_recheck --report-dir outputs/noise
uv run python -X utf8 scripts/check_dataset.py --download
```

`check_dataset.py` đối chiếu dữ liệu chuẩn hóa với nguồn GreenNode, tính lại 7 tập nhiễu vào thư mục tạm và kiểm tra checksum. File audit nằm tại `outputs/greennode_audit.json`; không dùng tập noise trong `outputs/noise_recheck` để thay thế dữ liệu đã thống nhất.

Chỉ chạy `scripts/prepare_train.py` nếu chủ động muốn tái tạo tập train từ nguồn thô; script này ghi lại `queries_train.jsonl`, `qrels_train.tsv` và `train_manifest.json`. Khi chỉ cần đánh giá mô hình, **không cần chạy lại bước đó**.

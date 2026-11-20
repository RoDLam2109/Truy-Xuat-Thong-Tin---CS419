# Dữ liệu chính: Zalo Legal — Lam v1

Chỉ dùng thư mục `data/zalo_legal/` trong code và thí nghiệm.

| File | Nội dung |
|---|---|
| `corpus.jsonl` | 61.425 tài liệu |
| `queries_train.jsonl` | 2.408 query train |
| `qrels_train.tsv` | 2.480 quan hệ query–tài liệu của train |
| `queries_clean.jsonl` | 788 query test gốc |
| `qrels_test.tsv` | 793 quan hệ query–tài liệu của test |
| `noise/queries_*.jsonl` | 7 phiên bản nhiễu, cùng ID và thứ tự với test |
| `train_manifest.json` | Nguồn, revision, các ID đã loại và SHA-256 |

Nguồn: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`.
Đây là bản Lam v1 đã chuẩn bị từ nguồn trên, không phải tải nguyên bản từ ban tổ chức Zalo.

Train đã loại 24 query ID trùng test, tương ứng 25 qrels. Không có ID hoặc nội dung query chuẩn hóa trùng giữa train và test. Việc kiểm tra cấu trúc không xác minh lại ý nghĩa của từng nhãn.

Chưa có validation riêng. Nếu cần tuning, chia validation từ train với seed cố định và chia sẻ cùng danh sách ID cho cả nhóm. Vocabulary, IDF và SVD được fit trên corpus; query test và qrels test chỉ dùng đánh giá cuối cùng.

## Phân biệt dữ liệu gốc và dữ liệu nhiễu

“Sạch” ở đây nghĩa là chưa thêm nhiễu nhân tạo; câu gốc vẫn có thể có lỗi chính tả tự nhiên.

| Nhóm | File | Vai trò |
|---|---|---|
| Train gốc, chưa thêm nhiễu | `queries_train.jsonl` | 2.408 câu hỏi để phát triển model hoặc chia validation |
| Test gốc, chưa thêm nhiễu | `queries_clean.jsonl` | 788 câu hỏi để đo kết quả ban đầu |
| Test nhiễu: bỏ dấu | `noise/queries_no_diacritics.jsonl` | Bỏ dấu tiếng Việt |
| Test nhiễu: lỗi gõ | `noise/queries_telex_typo.jsonl` | Lỗi gõ dấu/tạo chữ |
| Test nhiễu: chính tả/phát âm | `noise/queries_phonetic.jsonl` | Thay từ theo các dạng nhầm lẫn |
| Test nhiễu: viết tắt | `noise/queries_abbrev.jsonl` | Viết tắt/teencode |
| Test nhiễu: hỗn hợp nhẹ | `noise/queries_mixed_low.jsonl` | Kết hợp các loại lỗi, mức nhẹ |
| Test nhiễu: hỗn hợp vừa | `noise/queries_mixed_medium.jsonl` | Kết hợp các loại lỗi, mức vừa |
| Test nhiễu: hỗn hợp nặng | `noise/queries_mixed_high.jsonl` | Kết hợp các loại lỗi, mức nặng |

Mỗi file test nhiễu có 788 dòng, giữ cùng ID và thứ tự với `queries_clean.jsonl`. Một số câu có thể không đổi nếu phép sinh nhiễu không áp dụng được hoặc không chọn token để sửa.

`corpus.jsonl` là kho tài liệu chung, không thêm nhiễu. `qrels_train.tsv` là nhãn train; `qrels_test.tsv` là nhãn chung cho test gốc và cả 7 test nhiễu. Không có bộ train nhiễu hay validation riêng hiện tại. `train_manifest.json` chỉ ghi thông tin kiểm tra dữ liệu.

Khi đánh giá: chạy test gốc và từng test nhiễu trên cùng corpus, cùng model và cùng qrels test, rồi so sánh điểm. Không gộp các phiên bản test nhiễu vào train.

## Chia sẻ trong nhóm

Git lưu tài liệu và manifest; các file dữ liệu lớn được bỏ qua bằng `.gitignore` (corpus vượt giới hạn file GitHub thông thường). Sau khi clone repo, nhận trọn bộ thư mục `zalo_legal` bản Lam v1 từ nhóm và đặt vào `data/`. Đối chiếu SHA-256 trong `train_manifest.json` để bảo đảm mọi người dùng cùng dữ liệu. Clone repo riêng chưa có đủ dữ liệu để chạy.

Không thay bằng file tải mới từ nhánh `main` của nguồn: cách chia train/test và định dạng có thể khác bản nhóm đã chốt. Giữ nguyên ID tài liệu, kể cả ký tự `+`.

Các file nhiễu hiện có được giữ nguyên từ bản Lam v1. Script `scripts/generate_noise.py` có thể sinh lại và sẽ ghi đè các file đó; chỉ chạy khi nhóm chủ động cập nhật điều kiện nhiễu.


## Development validation (model-evaluation branch)

The 2,408 canonical train queries are the only input to the independent
development split. Run scripts/prepare_validation.py with seed 42 to create
1,926 train and 482 validation queries, preserving 1,985 and 495 qrels.
Seven validation noise sets are generated under outputs/splits/seed42/val/noise/.
This operation does NOT modify the canonical train, test, corpus, or test noise.
The canonical 788-query test remains held out for final evaluation.

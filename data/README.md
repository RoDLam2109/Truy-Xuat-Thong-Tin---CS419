# Dữ liệu chính: Zalo Legal — Lam v1

Chỉ dùng thư mục `data/zalo_legal/` trong code và thí nghiệm.

**Cập nhật 24/11/2026:** vẫn có 7 bộ nhiễu và một bản sạch. Riêng `noise/queries_no_diacritics.jsonl` nay **chỉ bỏ dấu trên các token được chọn với xác suất 25%**, không bỏ dấu toàn câu. Giữ đủ 788 query test, cùng ID/thứ tự/qrels; sáu bộ nhiễu còn lại giữ nguyên. Bản bỏ dấu toàn câu trước đây lưu tại `outputs/lam/archive/legacy_noise/data/zalo_legal/noise/queries_no_diacritics.jsonl`. Xem [hướng dẫn bộ bỏ dấu một phần](../docs/lam_partial_diacritics.md).

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

Validation đã chia từ train với seed 42: 482 query, 495 qrels tại `outputs/lam/splits/seed42/val/`. Bảy bộ nhiễu validation dùng cùng 482 ID; riêng bỏ dấu cũng áp dụng xác suất 25% trên token. Vocabulary, IDF và SVD được fit trên corpus; qrels dùng để đánh giá.

## Phân biệt dữ liệu gốc và dữ liệu nhiễu

“Sạch” ở đây nghĩa là chưa thêm nhiễu nhân tạo; câu gốc vẫn có thể có lỗi chính tả tự nhiên.

| Nhóm | File | Vai trò |
|---|---|---|
| Train gốc, chưa thêm nhiễu | `queries_train.jsonl` | 2.408 câu hỏi để phát triển model hoặc chia validation |
| Test gốc, chưa thêm nhiễu | `queries_clean.jsonl` | 788 câu hỏi để đo kết quả ban đầu |
| Test nhiễu: bỏ dấu | `noise/queries_no_diacritics.jsonl` | Bỏ dấu một phần, chọn token với xác suất 25% |
| Test nhiễu: lỗi gõ | `noise/queries_telex_typo.jsonl` | Lỗi gõ dấu/tạo chữ |
| Test nhiễu: chính tả/phát âm | `noise/queries_phonetic.jsonl` | Thay từ theo các dạng nhầm lẫn |
| Test nhiễu: viết tắt | `noise/queries_abbrev.jsonl` | Viết tắt/teencode |
| Test nhiễu: hỗn hợp nhẹ | `noise/queries_mixed_low.jsonl` | Kết hợp các loại lỗi, mức nhẹ |
| Test nhiễu: hỗn hợp vừa | `noise/queries_mixed_medium.jsonl` | Kết hợp các loại lỗi, mức vừa |
| Test nhiễu: hỗn hợp nặng | `noise/queries_mixed_high.jsonl` | Kết hợp các loại lỗi, mức nặng |

Mỗi file test nhiễu có 788 dòng, giữ cùng ID và thứ tự với `queries_clean.jsonl`. Một số câu có thể không đổi nếu phép sinh nhiễu không áp dụng được hoặc không chọn token để sửa.

`corpus.jsonl` là kho tài liệu chung, không thêm nhiễu. `qrels_train.tsv` là nhãn train; `qrels_test.tsv` là nhãn chung cho test gốc và cả 7 test nhiễu. Validation có qrels riêng theo tập ID đã chia; không có bộ train nhiễu. `train_manifest.json` ghi nguồn và checksum; checksum bộ bỏ dấu đã cập nhật theo dữ liệu mới.

Khi đánh giá: chạy test gốc và từng test nhiễu trên cùng corpus, cùng model và cùng qrels test, rồi so sánh điểm. Không gộp các phiên bản test nhiễu vào train.

## Chia sẻ trong nhóm

Git lưu tài liệu và manifest; các file dữ liệu lớn được bỏ qua bằng `.gitignore` (corpus vượt giới hạn file GitHub thông thường). Sau khi clone repo, nhận trọn bộ thư mục `zalo_legal` bản Lam v1 từ nhóm và đặt vào `data/`. Đối chiếu SHA-256 trong `train_manifest.json` để bảo đảm mọi người dùng cùng dữ liệu. Clone repo riêng chưa có đủ dữ liệu để chạy.

Không thay bằng file tải mới từ nhánh `main` của nguồn: cách chia train/test và định dạng có thể khác bản nhóm đã chốt. Giữ nguyên ID tài liệu, kể cả ký tự `+`.

Sáu file nhiễu khác giữ nguyên từ bản Lam v1. Script `scripts/generate_noise.py` phiên bản 1.1.0 sinh lại bảy điều kiện, với bỏ dấu một phần; script sẽ ghi đè khi chọn thư mục output đã tồn tại. Các điểm metrics trước thay đổi thuộc bộ cũ, cần chạy lại để đánh giá bộ bỏ dấu mới.

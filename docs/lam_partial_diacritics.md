# Bỏ dấu một phần và giữ tám điều kiện đánh giá

Theo yêu cầu của Lam ngày 24/11/2026, chỉ điều kiện `no_diacritics` thay đổi từ bỏ dấu cả câu sang bỏ dấu một phần. Giữ đủ tám điều kiện: clean, bỏ dấu, Telex/VNI, phát âm, viết tắt, hỗn hợp nhẹ, vừa và nặng. Validation vẫn có 482 câu, test vẫn có 788 câu; qrels, corpus, vocabulary, index và model giữ nguyên.

## Cách chọn token

Generator 1.1.0 dùng `single_probability=0.25` có sẵn trong `configs/noise.json` cho bỏ dấu, giống các điều kiện nhiễu đơn khác. Mỗi token được xét độc lập; token được chọn chỉ thay đổi nếu có dấu và không thuộc nhóm được bảo vệ. Số, mã, tên viết hoa và đơn vị được bảo vệ theo quy tắc hiện có. Token không được chọn giữ nguyên.

Token ở đây là đoạn phân cách bằng khoảng trắng có chứa chữ hoặc số, trước bước tách từ PyVi. Xác suất chọn 25% không có nghĩa chính xác 25% token sẽ đổi hoặc mỗi câu đổi cùng số token. Từ điển phát âm/viết tắt có phạm vi áp dụng khác nhau, nên đây không phải bộ có tỷ lệ thay đổi bằng nhau giữa mọi loại nhiễu.

Seed vẫn là 42. Quy tắc và kết quả của Telex/VNI, phát âm, viết tắt và cả ba mức hỗn hợp giữ nguyên. Mixed vốn đã bỏ dấu trên token được chọn; việc sửa điều kiện bỏ dấu riêng không thay đổi mixed.

## Dữ liệu đang dùng

| Tập | File bỏ dấu một phần | Số câu | Câu thực sự đổi | Tỷ lệ token đổi |
|---|---|---:|---:|---:|
| Test | `data/zalo_legal/noise/queries_no_diacritics.jsonl` | 788 | 767 | 20,0309% |
| Validation | `outputs/lam/splits/seed42/val/noise/queries_no_diacritics.jsonl` | 482 | 463 | 19,9123% |

Bản xử lý query test tại `outputs/lam/processed/noise/queries_no_diacritics.jsonl` cũng đã cập nhật bằng cách chuẩn hóa và tách từ hiện có, không xây lại vocabulary hay index. Các metadata noise, checksum trong `train_manifest.json`, `dataset_manifest.json` và processed summary được cập nhật tương ứng.

Ví dụ query test đầu tiên:

- Sạch: “Mức phạt khi quay đầu xe ô tô trên đường cao tốc”.
- Bỏ dấu mới: “Muc phat khi quay đầu xe ô tô trên đường cao tốc”.
- Bản cũ: “Muc phat khi quay dau xe o to tren duong cao toc”.

Kiểm tra trên cả hai tập mới không có câu thay đổi nào bị bỏ dấu toàn câu. Các câu có thể không đổi khi không token hợp lệ nào được chọn: 21 câu test và 19 câu validation.

## Bản lưu trước thay đổi

`outputs/lam/archive/legacy_noise/` lưu các file noise và báo cáo cũ theo đường dẫn tương đối ban đầu. Ví dụ:

- `data/zalo_legal/noise/queries_no_diacritics.jsonl`: bỏ dấu toàn câu trên 788 query test cũ.
- `outputs/lam/splits/seed42/val/noise/queries_no_diacritics.jsonl`: bỏ dấu toàn câu trên 482 query validation cũ.
- `outputs/lam/processed/noise/queries_no_diacritics.jsonl`: bản xử lý query cũ.
- `source_before_switch/`: source trước thay đổi để đối chiếu.
- `archive_manifest.json`: checksum các bản lưu và các file/model phải giữ nguyên.

Bộ thử riêng 230 câu cùng các script/test/tài liệu riêng của nó được lưu dưới `optional_one_token_cohort/` trong archive. Luồng chính dùng toàn bộ 482/788 câu và tám điều kiện, không dùng bộ 230 câu.

## Chạy lại thí nghiệm

Luồng cũ vẫn chạy 48 lượt mỗi split. Các bảng điểm và ranking đã có thuộc dữ liệu trước thay đổi; chạy vào output mới để có kết quả mới. Không resume vào kết quả cũ vì hash query đã đổi.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_experiments.py --output-dir outputs/lam/experiments/val_partial_diacritics
.\.venv\Scripts\python.exe -X utf8 scripts/report_experiments.py --experiment-dir outputs/lam/experiments/val_partial_diacritics --output-dir outputs/lam/reports/val_partial_diacritics
.\.venv\Scripts\python.exe -X utf8 scripts/run_experiments.py --split test --confirm-test-use --selection outputs/lam/reports/val_partial_diacritics/selection.json --output-dir outputs/lam/experiments/test_partial_diacritics
.\.venv\Scripts\python.exe -X utf8 scripts/report_experiments.py --experiment-dir outputs/lam/experiments/test_partial_diacritics --output-dir outputs/lam/reports/test_partial_diacritics
```

Generator và phần audit độ phủ trong report cùng dùng xác suất 25% cho bỏ dấu. Hàm tính metrics, cách xếp hạng và tiêu chí chọn theo trung bình bảy điều kiện nhiễu giữ nguyên.

## Kiểm chứng và chia sẻ

Đối chiếu SHA-256 xác nhận 12 file nhiễu còn lại trên hai tập không đổi; corpus, query sạch, qrels, code term/model, hai vocabulary, index metadata và sáu model đã fit giữ nguyên. Toàn bộ 50 unit/integration test pass; dataset audit đạt 67/67 checks. Bản lưu và kiểm tra cập nhật nằm trong `outputs/lam/archive/legacy_noise/` và `outputs/lam/partial_diacritics_update/`.

Dataset lớn vẫn bị Git bỏ qua. Ngoài push code và manifest, cần chia sẻ lại file query bỏ dấu mới cùng metadata/checksum tương ứng qua nơi nhóm lưu dataset, hoặc các thành viên sinh lại bằng generator 1.1.0. Bộ dữ liệu trên Drive chưa tự cập nhật khi push Git.

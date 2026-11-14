# Tài liệu triển khai đồ án CS419

Bộ tài liệu này mô tả phần **dữ liệu và term theo từ cho query gốc** đã hoàn thành. Đây chưa phải báo cáo kết quả đánh giá các mô hình truy xuất.

| Đọc theo thứ tự | Nội dung | Dành cho |
|---|---|---|
| 1. [Tổng kết những gì đã làm](implementation_summary.md) | Phạm vi, quyết định, mã đã viết, kết quả và phần còn lại | Cả nhóm, báo cáo tiến độ |
| 2. [Pipeline và hướng dẫn chạy](clean_terms.md) | Dataset/schema, quy tắc term, cài đặt và CLI | Người chạy lại dữ liệu/term |
| 3. [Kết quả chạy toàn bộ corpus](clean_terms_run.md) | Số term, shape ma trận, OOV và kiểm tra đã thực hiện | Người đọc số liệu, viết báo cáo |
| 4. [Bàn giao cho phần mô hình](model_handoff.md) | API nạp/transform, ma trận đầu vào và quy tắc tích hợp | Thành viên làm VSM/BM25/LSI và đánh giá |

Nguồn dữ liệu hiện tại là `minhnguyent546/zalo-ai-legal-text-retrieval-2021`, gồm 61.425 tài liệu và 640 query test. Không sử dụng các con số GreenNode/788 query của mô tả cũ cho lần triển khai này.

Các số liệu đã đo được ghi trong [báo cáo chạy](clean_terms_run.md). Các bước chưa làm được đánh dấu rõ trong [tổng kết](implementation_summary.md#phần-chưa-triển-khai).

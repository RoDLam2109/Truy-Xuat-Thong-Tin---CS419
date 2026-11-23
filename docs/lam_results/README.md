# Snapshot kết quả phần Lam

Các thư mục theo split lưu bảng điểm, biểu đồ PNG/SVG, mẫu lỗi, manifest và provenance của lần chạy full corpus ngày 21/11/2026. Đây là file mới để nhóm có thể xem kết quả mà không sửa báo cáo cũ.

`val/` có thêm `selection.json`, được tạo trước khi chạy test. Bảng test chỉ so sánh những baseline có tham số đã khóa, không dùng chọn lại cấu hình.

Các file ranking, model joblib, metrics toàn bộ từng query và `paired_drops.csv` có dung lượng lớn nên giữ trong `outputs/lam/experiments/` và `outputs/lam/reports/` theo `.gitignore` có sẵn. Các tên file đó trong `report.md` tham chiếu bộ artifact đầy đủ tại máy chạy, không nằm hết trong snapshot này.

Tái lập: [hướng dẫn của Lam](../lam_evaluation.md). Rà soát nhiễu: [lam_noise_review.md](../lam_noise_review.md). Phân tích lỗi validation: [lam_error_analysis.md](../lam_error_analysis.md).

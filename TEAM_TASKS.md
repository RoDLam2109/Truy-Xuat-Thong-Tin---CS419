# Phạm vi và tiến độ triển khai

Đồ án CS419: *Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries*.

Tài liệu này ghi **phần việc và trạng thái**, không phải bảng phân công nhân sự. Thiết kế gồm 2 cách biểu diễn (`word`, `char_ngram`), 3 mô hình (VSM, BM25, LSI) và 8 điều kiện query (clean + 7 loại nhiễu).

| Hạng mục | Trạng thái | Tài liệu hoặc mã nguồn |
|---|---|---|
| Chuẩn hóa dữ liệu, kiểm tra train/test và nguồn | Hoàn thành; 67/67 kiểm tra đạt | `scripts/check_dataset.py`, `data/README.md` |
| Tiền xử lý và xây dựng vocabulary từ corpus | Hoàn thành hai nhánh | `scripts/prepare_terms.py`, `PREPROCESSING.md` |
| Chia development train/validation và tạo 7 validation noise | Hoàn thành; seed 42 | `scripts/prepare_validation.py` |
| Xây dựng index trên toàn bộ 61.425 documents | Hoàn thành hai nhánh | `scripts/build_retrieval_index.py` |
| VSM/BM25/LSI với word | Hoàn thành 24/24 điều kiện validation | `docs/validation_results.md` |
| VSM với character n-gram | Hoàn thành 8/8 điều kiện validation | `docs/validation_results.md` |
| BM25 và LSI với character n-gram | **Chưa chạy full-corpus** (16 điều kiện) | `scripts/evaluate_models.py` |
| Tối ưu mô hình và đánh giá test cuối cùng | Chưa thực hiện | Chỉ làm sau khi chốt kết quả validation |

**Kết quả có thể báo cáo hiện tại:** 32/48 điều kiện trên validation. Bộ test (788 queries, 793 qrels) không được sử dụng để chọn hyperparameters.

Xem [README.md](README.md) để biết tóm tắt kết quả, [docs/model_evaluation.md](docs/model_evaluation.md) để chạy lại. Mọi thay đổi cần giữ nguyên split IDs và nguồn dữ liệu để kết quả giữa các thành viên có thể đối chiếu.

# Phân công và quy tắc làm việc nhóm

Dự án: **Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries**.

Thiết kế chính: **2 cách biểu diễn term × 3 mô hình × 8 điều kiện query = 48 lượt đánh giá**, sử dụng Zalo Legal. Hai biểu diễn là từ tiếng Việt và character n-gram; ba mô hình là VSM TF-IDF/cosine, BM25 và LSI/SVD.

**Nguồn dữ liệu đã chốt:** [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`. Dùng corpus/query/qrels theo cùng bản này, giữ toàn bộ nhãn liên quan của mỗi query. Các tập nhiễu do dự án sinh riêng. Nguồn/checksum nằm trong `configs/dataset.json`; chạy `scripts/check_dataset.py --download` để đối chiếu.

Dữ liệu chung: **Lam v1** tại `data/zalo_legal/`, gồm 2.408 query train và 788 query test; giữ nguyên corpus và nhãn. Khi cần validation, tách từ train và thống nhất một cách chia cho cả nhóm. Xem [hướng dẫn dữ liệu](data/README.md).

# Kết quả baseline trên validation

**Phạm vi:** 482 validation queries, 495 qrels, truy xuất trên toàn bộ **61.425 documents**. Nhiễu được tạo từ validation, không dùng qrels của test để chọn mô hình. Cả word và character n-gram đều sử dụng vocabulary/IDF từ corpus.

**Tiến độ:** **32/48 cấu hình** đã chạy thành công trên full corpus: 24 cấu hình word (VSM, BM25, LSI) và 8 cấu hình character n-gram (VSM). Chưa có kết quả full-corpus cho character n-gram BM25 hoặc LSI; chưa chạy test cuối cùng.

**Trạng thái Model Selection:** mới so sánh các baseline với tham số cố định; **chưa lựa chọn mô hình cuối cùng**. Word + BM25 dẫn đầu trên query clean trong các cấu hình đã chạy, còn character n-gram + VSM cải thiện độ chịu nhiễu Telex và mixed noise so với word + VSM. Để xem rõ danh sách 6 tổ hợp model/term và trạng thái từng tổ hợp, đọc [README.md](../README.md).

## 1. nDCG@10 theo loại nhiễu

| Điều kiện | Word VSM | Word BM25 | Word LSI | Char VSM |
|---|---:|---:|---:|---:|
| Clean | 0.5008 | **0.6626** | 0.2420 | 0.4833 |
| No diacritics | 0.0209 | **0.0297** | 0.0019 | 0.0219 |
| Telex typo | 0.3263 | **0.4725** | 0.1376 | 0.4216 |
| Phonetic | 0.4925 | **0.6511** | 0.2325 | 0.4803 |
| Abbreviation | 0.5004 | **0.6589** | 0.2358 | 0.4785 |
| Mixed low | 0.4198 | **0.5875** | 0.1914 | 0.4528 |
| Mixed medium | 0.3119 | **0.4698** | 0.1204 | 0.3996 |
| Mixed high | 0.2284 | **0.3534** | 0.0803 | 0.3542 |

Tham số: BM25 `k1=1.2`, `b=0.75`; LSI TruncatedSVD 64 chiều; seed 42. Đây là baseline cố định, chưa tối ưu theo validation. Bảng chỉ hiển thị các cấu hình **đã chạy**; không nội suy điểm cho BM25/LSI character n-gram.

## 2. Chất lượng trên clean queries

| Chỉ số | Word VSM | Word BM25 | Word LSI | Char VSM |
|---|---:|---:|---:|---:|
| MRR@10 | 0.4420 | **0.6143** | 0.2046 | 0.4306 |
| nDCG@10 | 0.5008 | **0.6626** | 0.2420 | 0.4833 |
| Recall@100 | 0.8990 | **0.9457** | 0.6127 | 0.8835 |

## 3. Phân tích kết quả

**Chất lượng truy xuất:** BM25 với biểu diễn word đạt điểm clean tốt nhất trong số các mô hình đã chạy. LSI-64 thấp hơn đáng kể; cần tránh diễn giải khả năng chống chịu nhiễu nếu bỏ qua điểm clean ban đầu.

**Lỗi bỏ dấu:** word BM25 giảm từ 0.6626 xuống 0.0297 nDCG@10 (**-95,52%**). Character n-gram VSM cũng chỉ đạt 0.0219; thay từ bằng n-gram chưa giải quyết được khoảng cách giữa corpus có dấu và query không dấu.

**Lỗi ký tự và nhiễu hỗn hợp:** character n-gram VSM cải thiện điểm Telex từ 0.3263 lên 0.4216 và mixed high từ 0.2284 lên 0.3542 so với word VSM. Điều này phù hợp với giả thuyết rằng khớp chuỗi ký tự con giảm tác động của một số lỗi gõ. Tuy nhiên, không thể kết luận character n-gram luôn tốt hơn word cho mọi loại nhiễu.

Kết quả hiện tại trả lời các câu hỏi so sánh baseline. Chưa có thí nghiệm kiểm định ý nghĩa thống kê hoặc phương pháp cải tiến đã được chọn dựa trên validation.

## 4. Nguồn kết quả và khả năng tái lập

Mã nguồn và manifest dữ liệu nằm trên GitHub; dataset gốc được chia sẻ qua Drive. Các artifact sau đã được tạo trên máy khi đo:

| Artifact | Mục đích |
|---|---|
| `outputs/splits/seed42/split_manifest.json` | Seed, kích thước và checksum split |
| `outputs/processed/summary.json` | Thông tin vocabulary, hash đầu vào |
| `outputs/index/word/index_meta.json` | Metadata của index word đầy đủ |
| `outputs/index/char_ngram/index_meta.json` | Metadata của index char đầy đủ |
| `outputs/retrieval/val_word/summary.csv` | Kết quả 24 cấu hình word |
| `outputs/retrieval/val_word/per_query.csv` | Metrics theo từng query |
| `outputs/retrieval/val_char_full/progress.json` | Kết quả 8 cấu hình VSM character n-gram |

Corpus SHA-256: `dda9c3f0c09627c1a5f2f921384d1c55b6b9775d67ad7254a5bd7a6a119cb9af`. Word index gồm **16.498 features**; character n-gram index gồm **111.181 features** và **44.687.991 non-zero entries**. Audit dataset đạt **67/67 checks**; unit tests **37/37** tại lần xác minh.

Artifact lớn được lưu trong `outputs/` và không commit lên Git. Quy trình dựng lại kết quả: [model_evaluation.md](model_evaluation.md).

## 5. Những gì chưa thực hiện

- Chạy **16 cấu hình full-corpus character n-gram BM25/LSI** còn thiếu.
- Thử nghiệm tối ưu, so sánh mở rộng hoặc cải tiến thuật toán trên validation.
- Kiểm tra thủ công tính bảo toàn ngữ nghĩa của các câu hỏi nhiễu.
- Đánh giá test cuối cùng sau khi lựa chọn cấu hình.

Mọi số liệu trên đều là **validation**, không phải test. Kết quả từ index chỉ gồm 400 documents dùng cho smoke test đã được loại khỏi báo cáo.

# Tổng kết triển khai dữ liệu và word terms

Ngày tổng kết: **2026-11-15**.

Đề tài: đánh giá ảnh hưởng của query tiếng Việt sai chính tả lên truy xuất thông tin. Phần đã hoàn thành tạo nền tảng để chạy query gốc trước, sau đó mới đánh giá query lỗi trên cùng bộ chỉ mục.

## Phạm vi đã hoàn thành

Đã triển khai và chạy phần **chuẩn bị dữ liệu, tách từ, tạo vocabulary/DF, ma trận count/TF-IDF corpus và transform query gốc**.

Đối với TF-IDF/BM25, corpus cung cấp thống kê để lập chỉ mục; query gốc không được dùng để fit vocabulary hay IDF. Việc tạo TF-IDF mới là chuẩn bị biểu diễn, chưa phải kết quả xếp hạng hoặc đánh giá chất lượng truy xuất.

## Dữ liệu đã chuẩn bị

Nguồn: [minhnguyent546/zalo-ai-legal-text-retrieval-2021](https://huggingface.co/datasets/minhnguyent546/zalo-ai-legal-text-retrieval-2021), commit `5fb3df4300b3a81455fe09af444ec159afa8d212`.

- 61.425 tài liệu, mỗi tài liệu có ID, title và text.
- 3.196 query gốc: 2.556 query train và 640 query test, chia theo ID trong qrels.
- Train/test không trùng query ID; các tham chiếu qrels hợp lệ.
- Lưu tại `data/zalo_legal_2021/`, tách khỏi nguồn GreenNode được mô tả trước đây.

`queries_clean.jsonl` là tập query gốc thuộc test, chưa bị script gây nhiễu; tên “clean” không bảo đảm hoàn toàn không có lỗi tự nhiên.

## Quyết định về term và lý do

Chọn **từ sau tách từ** làm baseline triển khai theo yêu cầu hiện tại. Ví dụ:

```text
Không có bảo hiểm y tế bị phạt thế nào?
→ [không, có, bảo_hiểm, y_tế, bị, phạt, thế_nào]
```

| Quyết định | Lý do |
|---|---|
| Dùng PyVi 0.1.1 | Có bộ tách từ tiếng Việt dùng sẵn, không phải huấn luyện bộ tách từ mới |
| Chuẩn hóa Unicode NFC và chữ thường | Giảm khác biệt hình thức không phục vụ câu hỏi thí nghiệm |
| Giữ dấu; không sửa chính tả | Bảo toàn tác động cần đo khi đánh giá query lỗi sau này |
| Giữ số, từ phủ định và điều kiện | Tránh làm mất thông tin trong văn bản pháp lý |
| Stopword rỗng, `min_df=1`, `max_df=1.0` | Tránh loại thông tin trước khi có bằng chứng thực nghiệm |
| Không stemming/lemmatization | Không cần thêm phép biến đổi hình thái cho baseline tiếng Việt này |
| Fit từ vựng/IDF chỉ trên corpus | Query sạch và lỗi dùng cùng không gian biểu diễn, không thêm term từ test |

Title và text được tách từ riêng rồi nối danh sách term, không tạo từ ghép qua ranh giới hai trường. Cấu hình nằm tại [`configs/terms.json`](../configs/terms.json).

## Các file đã triển khai

| File/module | Chức năng |
|---|---|
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | Tải snapshot dataset, chuẩn hóa schema và chia query theo qrels |
| [`src/data_io.py`](../src/data_io.py) | Kiểm tra JSONL, ID, qrels; ghi dữ liệu và tính hash |
| [`src/terms/pipeline.py`](../src/terms/pipeline.py) | Trích word terms, fit count/TF-IDF, lưu/nạp và transform query |
| [`scripts/build_index.py`](../scripts/build_index.py) | CLI tạo các artifact corpus/query gốc |
| [`tests/test_prepare_data.py`](../tests/test_prepare_data.py) | Kiểm tra schema, ID, split và tham chiếu nhãn |
| [`tests/test_terms.py`](../tests/test_terms.py) | Kiểm tra tách từ, count/DF/IDF, OOV, thứ tự ID và tái lập |
| [`requirements-terms-lock.txt`](../requirements-terms-lock.txt) | Phiên bản dependency đã kiểm thử với Python 3.12 |

Đã cập nhật README và các task tương ứng trong TEAM_TASKS. Phần sinh nhiễu đã có trước được giữ nguyên; các tests của phần đó được chạy cùng suite.

## Kết quả đã kiểm chứng

| Đại lượng | Kết quả |
|---|---:|
| Vocabulary corpus | 24.075 term |
| Ma trận tài liệu | 61.425 × 24.075 |
| Ma trận query gốc test | 640 × 24.075 |
| Token query ngoài vocabulary | 25 / 8.104 = 0,3085% |
| Query có vector rỗng | 0 |
| Tests | 17/17 qua |

Đã chạy trên toàn bộ corpus, nạp lại bộ artifact và transform query mới thành công. Mỗi artifact có cấu hình, hash và phiên bản dependency để kiểm tra tính nhất quán. Chi tiết số liệu nằm trong [báo cáo chạy](clean_terms_run.md).

OOV thấp chỉ cho biết ít token ngoài từ vựng; **không chứng minh hệ thống truy xuất chính xác**. Chưa có MRR, nDCG hoặc Recall trong bước này.

## Phần chưa triển khai

- Nhánh character n-gram.
- Xếp hạng bằng VSM/cosine, BM25 và LSI/SVD.
- Tính MRR@10, nDCG@10, Recall@100 trên query gốc.
- Sinh lại và rà soát query nhiễu cho nguồn dữ liệu mới.
- So sánh sạch–nhiễu, vẽ biểu đồ và phân tích nguyên nhân suy giảm.

Bước bàn giao tiếp theo: phần mô hình nạp ma trận và API theo [model_handoff.md](model_handoff.md), chạy mốc sạch trước rồi mới đánh giá nhiễu. Không dùng điểm test để chọn cấu hình.

## Chạy lại và lưu trên Git

Hướng dẫn cài đặt và lệnh tái tạo: [clean_terms.md](clean_terms.md). Dữ liệu đầy đủ, môi trường `.venv/` và ma trận trong `outputs/` được giữ ngoài Git; người khác tái tạo bằng script và commit dataset đã ghi ở trên.

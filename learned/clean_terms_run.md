# Kết quả chuẩn bị word terms — 2026-11-15

Đã chạy trên **toàn bộ corpus**, không phải mẫu nhỏ. Bước này chưa xếp hạng tài liệu và chưa đánh giá query nhiễu.

| Thành phần | Kết quả |
|---|---:|
| Corpus | 61.425 tài liệu |
| Query train | 2.556 |
| Query gốc test (`queries_clean.jsonl`) | 640 |
| Term xuất hiện trong corpus, tính cả lặp | 12.597.883 |
| Vocabulary | 24.075 term |
| Ma trận count/TF-IDF corpus | 61.425 × 24.075 |
| Phần tử khác 0 trong count | 4.995.702 |
| Ma trận query gốc test | 640 × 24.075 |
| Token query gốc test | 8.104 |
| Token query ngoài vocabulary corpus | 25 (0,3085%) |
| Query có vector rỗng | 0 |
| Tài liệu có vector rỗng | 0 |

Nguồn: `minhnguyent546/zalo-ai-legal-text-retrieval-2021`, commit `5fb3df4300b3a81455fe09af444ec159afa8d212`. Không dùng dữ liệu GreenNode/788 query trong lần chạy này.

Cấu hình: PyVi 0.1.1, Unicode NFC, chữ thường, giữ dấu và số; không stopword, không loại term theo DF, không stemming/lemmatization. TF-IDF dùng smooth IDF, TF thô và chuẩn hóa L2. Không chọn cấu hình bằng điểm test; tỷ lệ OOV là thống kê chẩn đoán của cấu hình đã chốt.

Kiểm tra đã thực hiện:

- 17 tests đều qua, gồm 10 tests sinh nhiễu cũ và 7 tests mới cho dữ liệu/term.
- Nạp lại artifact bằng `TermIndex`, kiểm tra checksum, transform một query mới mà không fit lại.
- Kiểm tra shape, ID duy nhất và đúng thứ tự, giá trị TF-IDF hữu hạn; chuẩn L2 của mỗi hàng tài liệu bằng 1 trong sai số 1e-12.
- Đầu vào không bị ghi đè. Corpus và query được transform bằng cùng extractor.

Các thống kê có thể đọc lại từ `outputs/terms/word_clean/stats.json`; cấu hình/hash/dependency trong `outputs/terms/word_clean/manifest.json`. Artifact khoảng 205,6 MB và nằm ngoài Git. Môi trường thực tế: CPython 3.12.13 trên Linux, phiên bản package trong `requirements-terms-lock.txt`.

Hướng dẫn sử dụng và giao diện bàn giao: [clean_terms.md](clean_terms.md).

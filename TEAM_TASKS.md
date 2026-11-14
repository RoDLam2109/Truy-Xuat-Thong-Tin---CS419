# Phân công và quy tắc làm việc nhóm

Dự án: **Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries**.

Thiết kế chính: **2 cách biểu diễn term × 3 mô hình × 8 điều kiện query = 48 lượt đánh giá**, sử dụng Zalo Legal. Hai biểu diễn là từ tiếng Việt và character n-gram; ba mô hình là VSM TF-IDF/cosine, BM25 và LSI/SVD.

> Cập nhật 2026-11-15: phần word terms đã triển khai với dataset minhnguyent546, 640 query test; hướng dẫn tại [learned/clean_terms.md](learned/clean_terms.md). Chưa chạy retrieval hay đánh giá nhiễu.

## 1. Tổng quan phân công

| Thành viên | Vai trò | Nhánh Git |
|---|---|---|
| **Kiên — người 1** | Dữ liệu và biểu diễn term | `feature/data-terms` |
| **Vũ — người 2** | Ba mô hình truy xuất | `feature/retrieval-models` |
| **Lam — người 3** | Nhiễu, đánh giá và tích hợp | `feature/noise-evaluation` |

Các file/module chưa có trong repo được liệt kê dưới đây là phần cần triển khai. Việc phân công không có nghĩa các thành phần đó đã hoàn thành.

## 2. Kiên — Dữ liệu và biểu diễn term

**Mục tiêu:** chuẩn bị dữ liệu và hai cách biểu diễn dùng chung cho cả ba mô hình.

### Task

- [x] Viết bước chuẩn bị dataset minhnguyent546 có thể chạy lại, gồm query, corpus và qrels.
- [x] Kiểm tra ID trùng, train/test rời nhau, query thiếu qrels và tài liệu được qrels tham chiếu.
- [x] Xây pipeline `word`: NFC/chữ thường → PyVi → term chữ/số; mặc định stopword rỗng, không loại DF.
- [ ] Xây pipeline `char_ngram`: chuẩn hóa → tạo character n-gram 3–5 → lọc DF.
- [ ] Chốt bộ tách từ, stopword, quy tắc n-gram và ngưỡng lọc trong cấu hình.
- [x] Tạo vocabulary, ma trận term count và TF-IDF từ corpus cho nhánh word.
- [ ] Tạo các artifact tương ứng cho nhánh char_ngram.
- [x] Viết `TermIndex.transform()` cho word bằng vocabulary/IDF đã fit.
- [x] Lưu cấu hình, thứ tự ID tài liệu/query, TF-IDF đã fit, dependency và hash cho word.
- [x] Viết kiểm tra cho dữ liệu, word terms, thứ tự ID, OOV và khả năng tái lập.

### File phụ trách

```text
scripts/prepare_data.py
scripts/build_index.py
src/terms/
configs/terms.json
tests/test_terms.py
tests/test_prepare_data.py
```

### Đầu ra bàn giao cho Vũ

Mỗi bộ biểu diễn `word` và `char_ngram` gồm:

- Vocabulary và thống kê DF.
- Ma trận term count và TF-IDF của corpus.
- Danh sách `doc_ids` đúng thứ tự hàng của ma trận.
- Thống kê độ dài tài liệu và quy tắc đếm term.
- Bộ transform query cùng cấu hình tiền xử lý, trọng số và bộ lọc.

**Tiêu chí hoàn thành:** cùng văn bản được xử lý nhất quán; chạy lại với cùng cấu hình cho cùng kết quả; query không làm thay đổi vocabulary. Kiên và Vũ thống nhất giao diện nạp chỉ mục/transform trước khi tích hợp.

## 3. Vũ — Ba mô hình truy xuất

**Mục tiêu:** chạy được VSM TF-IDF/cosine, BM25 và LSI/SVD với cả hai cách biểu diễn term.

### Task

- [ ] Nhận và kiểm tra bộ biểu diễn cùng giao diện transform từ Kiên.
- [ ] Cài VSM với TF-IDF và cosine similarity.
- [ ] Cài BM25 từ term count và thống kê corpus; ghi rõ công thức IDF, `k1` và `b`.
- [ ] Cài LSI: fit SVD trên TF-IDF corpus, chiếu query rồi tính cosine.
- [ ] Lưu/tải lại các thành phần mô hình để dùng chung giữa sạch và nhiễu.
- [ ] Xuất top 100 tài liệu cho mỗi query có kết quả truy xuất hợp lệ.
- [ ] Xử lý query không có term hợp lệ hoặc vector bằng 0 và lưu trạng thái tương ứng.
- [ ] Kiểm tra xếp hạng bằng corpus nhỏ có kết quả tính tay được.
- [ ] Xác nhận ba mô hình chạy được với cả `word` và `char_ngram`.

### File phụ trách

```text
scripts/run_retrieval.py
src/retrievers/
configs/retrieval.json
tests/test_retrievers.py
```

### Đầu ra bàn giao cho Lam

- Sáu cấu hình truy xuất hoạt động: ba mô hình × hai biểu diễn term.
- Kết quả xếp hạng đúng schema đã thống nhất.
- Danh sách query đã xử lý, trạng thái query không có kết quả và cấu hình từng run.
- Cách chạy bằng CLI và thông tin nạp lại vocabulary/chỉ mục/mô hình.

**Tiêu chí hoàn thành:** đổi mô hình hoặc nhánh term qua cấu hình; query sạch và nhiễu dùng lại đúng chỉ mục, vocabulary và mô hình. BM25 dùng term count; LSI dùng SVD đã fit trên corpus và không fit lại khi chạy query.

## 4. Lam — Nhiễu, đánh giá và tích hợp

**Mục tiêu:** kiểm soát dữ liệu nhiễu, chạy thí nghiệm và tính mức suy giảm.

### Task

- [ ] Tiếp quản script sinh nhiễu hiện có; rà soát mẫu, quy tắc và độ phủ từ điển.
- [ ] Kiểm tra bảy tập nhiễu cùng 640 ID và cùng thứ tự với tập sạch.
- [ ] Tổng hợp tỷ lệ nhiễu thực tế, query không đổi và collision.
- [ ] Cài MRR@10, nDCG@10 và Recall@100.
- [ ] Tính suy giảm tuyệt đối và tương đối so với lượt sạch tương ứng.
- [ ] Viết kiểm tra metrics với trường hợp nhiều tài liệu liên quan và query không có kết quả.
- [ ] Viết script chạy đủ 48 lượt, lưu cấu hình và kết quả riêng.
- [ ] Tổng hợp bảng, biểu đồ và phân tích query suy giảm mạnh.
- [ ] Tổng hợp dependency và quản lý `.gitignore`, tài liệu dùng chung.

### File phụ trách

```text
scripts/generate_noise.py
scripts/evaluate.py
scripts/run_experiments.py
src/evaluation/
configs/noise.json
tests/test_generate_noise.py
tests/test_evaluate.py
```

Lam là đầu mối tổng hợp `requirements.txt`, `.gitignore`, `README.md` và tài liệu nhóm. Kiên/Vũ gửi nội dung hoặc dependency cần bổ sung để Lam cập nhật; không cùng sửa trực tiếp các file này khi chưa thống nhất.

### Đầu ra bàn giao cho nhóm

- Bộ query nhiễu, seed, cấu hình, log và kết quả rà soát mẫu.
- Pipeline chạy thí nghiệm từ dữ liệu đến metrics.
- Metrics từng query và toàn tập.
- Bảng điểm sạch, điểm nhiễu, suy giảm tuyệt đối/tương đối theo term, mô hình và điều kiện nhiễu.
- Biểu đồ, phân tích lỗi và hướng dẫn tái lập.

**Tiêu chí hoàn thành:** đánh giá đủ query, xử lý đúng qrels nhiều tài liệu liên quan và kết quả rỗng; so sánh theo cùng `qid`; mức suy giảm luôn đối chiếu với lượt sạch của cùng mô hình và nhánh term.

## 5. Giao diện và quy tắc kỹ thuật chung

### 5.1. Dữ liệu và biểu diễn

1. Giữ cố định corpus, qrels và tập 640 ID query. Dữ liệu mới nằm trong `data/zalo_legal_2021/`; file nhiễu tương ứng sẽ nằm trong `data/zalo_legal_2021/noise/`.
2. Mọi file query giữ schema `id`, `text`; file nhiễu giữ nguyên ID và thứ tự của file sạch.
3. Vocabulary, DF/IDF và SVD chỉ fit từ corpus. Query sạch/nhiễu chỉ transform, không thêm term vào vocabulary.
4. Hàng thứ `i` của mọi ma trận phải tương ứng đúng `doc_ids[i]`. Lưu danh sách ID cùng mỗi bộ biểu diễn.
5. Trong cùng nhánh term, dùng chung vocabulary đã lọc cho ba mô hình; trọng số được tính theo từng phương pháp.
6. Phân biệt term truy xuất với token đo tỷ lệ nhiễu. Sinh nhiễu trên văn bản trước bước biểu diễn term; hai nhánh nhận cùng câu query.
7. Không chỉnh tham số hoặc quy tắc dựa trên điểm test. Tham số được chốt trước hoặc chọn bằng tập dev riêng.
8. Đổi giao diện phải báo người dùng module và thống nhất tên hàm, tham số, kiểu dữ liệu trả về trước khi sửa.

### 5.2. Đầu ra truy xuất

Mỗi dòng kết quả dùng schema:

```json
{"qid": "query-id", "doc_id": "document-id", "rank": 1, "score": 12.34}
```

- `rank` bắt đầu từ 1; xếp score giảm dần, khi bằng điểm dùng `doc_id` tăng dần.
- Mỗi cặp `qid`, `doc_id` xuất hiện tối đa một lần trong một run.
- Lưu riêng danh sách query đã xử lý và trạng thái kết quả rỗng; không bỏ query đó khỏi mẫu số đánh giá.
- Phân biệt query không có kết quả hợp lệ với query bị thiếu do lỗi pipeline.
- Lưu cấu hình, seed, phiên bản code và thông tin dữ liệu cùng từng run.
- Tổ chức kết quả theo `outputs/retrieval/<term>/<model>/<condition>/`.

### 5.3. Đánh giá

```text
Suy giảm tuyệt đối = M_sạch - M_nhiễu
Suy giảm tương đối (%) = (M_sạch - M_nhiễu) / M_sạch × 100
```

- Metrics chính: MRR@10, nDCG@10, Recall@100.
- So sánh sạch–nhiễu theo cùng `qid`, cùng mô hình và cùng cách biểu diễn term.
- Nếu điểm sạch bằng 0, ghi suy giảm tương đối là `N/A`.
- Đọc mức suy giảm cùng điểm sạch và điểm nhiễu; suy giảm nhỏ không tự động có nghĩa chất lượng truy xuất cao.
- Báo cáo tỷ lệ nhiễu thực tế; không mặc định các loại nhiễu có cùng độ khó khi độ phủ khác nhau.

## 6. Luật sở hữu file và Git

### Sở hữu file

- Mỗi người sửa các file/module thuộc phần mình.
- Nếu cần sửa file của người khác, trao đổi trước và thống nhất người thực hiện.
- Mỗi người viết tests trong file riêng theo bảng phân công.
- Lam tổng hợp dependency và thay đổi tài liệu dùng chung. Tránh sửa dependency riêng rẽ rồi để người khác tự xử lý khi tích hợp.

### Nhánh và Pull Request

- Kiên làm trên `feature/data-terms`.
- Vũ làm trên `feature/retrieval-models`.
- Lam làm trên `feature/noise-evaluation`.
- Commit theo từng task nhỏ; push nhánh cá nhân và tạo Pull Request vào `main`.
- Cập nhật từ `main` trước khi tạo/merge PR; xử lý conflict trên nhánh của mình.
- Không force-push vào `main`. Không ghi đè thay đổi của người khác để xử lý conflict.
- PR ghi rõ thay đổi, cách chạy, kết quả kiểm tra và thay đổi giao diện nếu có.
- Mỗi PR cần ít nhất một thành viên khác review trước khi merge.

| Nội dung PR | Người review chính |
|---|---|
| Dữ liệu, term và giao diện chỉ mục của Kiên | Vũ |
| Mô hình và đầu ra truy xuất của Vũ | Lam |
| Nhiễu, metrics và tích hợp của Lam | Kiên; Vũ tham gia nếu liên quan đầu ra mô hình |

**Việc đầu tiên của Lam:** tạo `.gitignore` trước khi nhóm thêm file hàng loạt. Bỏ qua dữ liệu đầy đủ, `.venv/`, cache, ma trận, chỉ mục và output lớn. Mỗi người tái tạo các artifact đó trên máy từ cùng dữ liệu, cấu hình và seed.

## 7. Thứ tự triển khai và tích hợp

### Mốc 1 — Chốt giao diện

- Kiên và Vũ thống nhất cách nạp bộ biểu diễn, transform query, thứ tự ID và thống kê corpus.
- Vũ và Lam thống nhất schema ranked results, danh sách query đã xử lý và trạng thái rỗng.
- Lam chuẩn bị `.gitignore` và kiểm tra metrics bằng ví dụ nhỏ trong khi Kiên/Vũ triển khai module.

### Mốc 2 — Chạy một luồng hoàn chỉnh

Chạy được **`word + VSM + clean/mixed_low`** từ dữ liệu đến metrics:

1. Kiên cung cấp pipeline word, vocabulary và TF-IDF.
2. Vũ chạy VSM cho query sạch và mixed_low bằng cùng bộ biểu diễn.
3. Lam tính metrics, mức suy giảm và kiểm tra đủ 640 query.

### Mốc 3 — Mở rộng đủ thiết kế

- Kiên hoàn thiện nhánh character n-gram và bộ lọc.
- Vũ hoàn thiện BM25 và LSI/SVD với cả hai nhánh term.
- Lam hoàn thiện rà soát nhiễu, script chạy toàn bộ và tổng hợp kết quả.
- Chạy đủ **6 lượt sạch và 42 lượt nhiễu**, tổng cộng 48 lượt đánh giá.

### Mốc 4 — Nghiệm thu

- [ ] Hai pipeline term tái lập được và chỉ fit từ corpus.
- [ ] Ba mô hình hoạt động với cả hai cách biểu diễn.
- [ ] Bảy tập nhiễu giữ nguyên ID/thứ tự; có tỷ lệ thực tế và rà soát mẫu.
- [ ] Chạy đủ 48 lượt với cùng corpus/qrels và cấu hình đã chốt.
- [ ] Có metrics từng query, bảng sạch–nhiễu–suy giảm và biểu đồ.
- [ ] Có phân tích lỗi, giới hạn của nhiễu tổng hợp và phạm vi miền pháp lý.
- [ ] Có hướng dẫn chạy, dependency và thông tin tái lập để cả ba người sử dụng.

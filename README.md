# Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries

## 1. Mục tiêu dự án

Đánh giá mức độ chống chịu của các mô hình truy xuất thông tin khi câu truy vấn tiếng Việt bị nhiễu. Với mỗi mô hình và cách biểu diễn term, lấy kết quả trên query sạch làm điểm chuẩn, chạy các phiên bản query nhiễu, rồi tính **mức suy giảm hiệu năng**.

Dataset chính là **Zalo Legal**. Trong toàn bộ thí nghiệm, chỉ thay đổi câu truy vấn; giữ nguyên corpus, qrels và tập ID truy vấn.

Dự án khảo sát hai yếu tố:

1. **Cách biểu diễn term:** từ tiếng Việt và character n-gram.
2. **Mô hình truy xuất:** VSM với TF-IDF/cosine similarity, BM25 và LSI/SVD.

Các câu hỏi nghiên cứu:

- Mỗi mô hình suy giảm bao nhiêu so với điểm sạch khi query có từng loại lỗi?
- Mức suy giảm thay đổi thế nào khi mức nhiễu hỗn hợp tăng?
- Biểu diễn bằng character n-gram có giảm tác động của lỗi ký tự so với biểu diễn bằng từ tiếng Việt không?
- Mô hình nào cân bằng tốt giữa chất lượng truy xuất và khả năng chống chịu nhiễu?

## 2. Dữ liệu và schema

Nguồn: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn).

| Thành phần | Đường dẫn | Số lượng hiện tại |
|---|---|---:|
| Query sạch | `data/zalo_legal/queries_clean.jsonl` | 788 query, 788 ID duy nhất |
| Tập tài liệu | `data/zalo_legal/corpus.jsonl` | 61.425 tài liệu, ID duy nhất |
| Nhãn liên quan | `data/zalo_legal/qrels_test.tsv` | 793 quan hệ liên quan |
| Query nhiễu | `data/zalo_legal/noise/queries_*.jsonl` | 7 file, mỗi file 788 query |

File query ban đầu có 818 dòng; đã bỏ 30 dòng trùng ID có cùng nội dung, giữ lần xuất hiện đầu tiên và thứ tự gốc. Tập test được cố định với **N = 788**.

Qrels hiện có nhãn nhị phân `score = 1`: 783 query có một tài liệu liên quan và 5 query có hai tài liệu liên quan. Mọi ID trong qrels đều có bản ghi query/tài liệu tương ứng.

### Query sạch và query nhiễu: JSONL

```json
{"id": "query-id", "text": "Nội dung câu hỏi"}
```

Mỗi dòng là một đối tượng JSON. `qid` trong code hoặc báo cáo tương ứng với trường `id` của query. Mọi file nhiễu giữ nguyên ID và thứ tự của file sạch.

### Corpus: JSONL

```json
{"id": "document-id", "title": "Tiêu đề điều luật", "text": "Nội dung văn bản"}
```

Nội dung đưa vào pipeline truy xuất là `title` nối với `text`, cách nhau bằng một ký tự xuống dòng. Dùng cùng nội dung tài liệu cho mọi mô hình và cách biểu diễn term.

### Qrels: TSV

```text
query-id	corpus-id	score
```

File có dòng tiêu đề và phân cách bằng tab. `query-id` nối với `queries.id`; `corpus-id` nối với `corpus.id`. ID được xử lý như chuỗi và giữ nguyên mọi ký tự.

## 3. Term và hai cách biểu diễn

Theo cách dùng trong mô tả môn học, **term là đơn vị dùng để biểu diễn và lập chỉ mục cho tài liệu và query**, chẳng hạn từ hoặc n-gram. Mỗi cách biểu diễn tạo một không gian đặc trưng riêng và được chạy thí nghiệm riêng.

**Term truy xuất khác với token dùng để đo tỷ lệ nhiễu.** Script sinh nhiễu hiện đếm các đơn vị tách theo khoảng trắng có chứa chữ hoặc số. Các đơn vị đó không mặc định là từ tiếng Việt sau tách từ, cũng không phải character n-gram. Hai nhánh truy xuất sử dụng cùng các file query nhiễu đã sinh từ văn bản sạch.

### 3.1. Term theo từ tiếng Việt (`word`)

Pipeline dự kiến:

1. Chuẩn hóa Unicode NFC, chữ thường và khoảng trắng.
2. Tách văn bản thành từ tiếng Việt bằng bộ tách từ đã chọn.
3. Loại stopword bằng một danh sách cố định.
4. Thống kê tần suất xuất hiện trong tài liệu và lọc term theo DF hoặc ngưỡng IDF.
5. Lưu vocabulary và biến đổi tài liệu/query sang biểu diễn phù hợp với từng mô hình.

Ví dụ: `bảo hiểm` có thể được biểu diễn thành một term `bảo_hiểm` sau tách từ, thay vì mặc định xem hai âm tiết là hai từ độc lập.

- **DF (document frequency):** số tài liệu trong corpus chứa term. Khác với tổng số lần term xuất hiện trong corpus.
- **IDF (inverse document frequency):** trọng số dựa trên độ phổ biến của term trong tập tài liệu.
- Lọc term quá hiếm hoặc quá phổ biến bằng `min_df`, `max_df` hoặc quy tắc IDF được ghi rõ trong cấu hình.
- Chọn bộ tách từ, phiên bản, danh sách stopword và ngưỡng lọc trước khi đánh giá test. Rà soát stopword để tránh loại các từ có vai trò đổi nghĩa trong miền pháp lý, như từ phủ định.

Nhánh này dễ giải thích và thể hiện khả năng truy xuất dựa trên khớp từ. Lỗi dấu, lỗi chính tả hoặc lỗi tách từ có thể làm giảm số term khớp giữa query và tài liệu.

### 3.2. Term theo character n-gram (`char_ngram`)

Pipeline dự kiến:

1. Chuẩn hóa Unicode NFC, chữ thường và khoảng trắng.
2. Tạo các chuỗi ký tự liên tiếp với độ dài **3–5**.
3. Thống kê DF trên corpus và lọc n-gram quá hiếm hoặc quá phổ biến.
4. Lưu vocabulary và biến đổi tài liệu/query sang biểu diễn phù hợp với từng mô hình.

Ví dụ: chuỗi `phạt` có các 3-gram `phạ`, `hạt`. Khi một ký tự bị sai, các n-gram ở những phần khác của query vẫn có thể khớp với tài liệu.

Nhánh này tạo đặc trưng trực tiếp từ văn bản chuẩn hóa, không yêu cầu tách từ tiếng Việt hay loại stopword theo từ trước khi tạo n-gram. Cần ghi rõ analyzer có cho phép n-gram đi qua khoảng trắng/ranh giới từ, có thêm ký tự biên hay không; cố định lựa chọn đó giữa corpus, query sạch và query nhiễu.

Character n-gram có thể tăng khả năng khớp từng phần, nhưng cũng có thể tăng kích thước vocabulary, chỉ mục và chi phí truy xuất. Đây là giả thuyết cần kiểm chứng bằng kết quả thí nghiệm.

### 3.3. Quy tắc chung về term

- Xây dựng vocabulary và thống kê DF/IDF **chỉ từ corpus**. Không fit lại bằng query sạch, query nhiễu hoặc qrels test.
- Trong cùng một cách biểu diễn term, dùng chung vocabulary đã lọc cho ba mô hình; mỗi mô hình tính trọng số theo phương pháp của mình.
- Dùng cùng bộ tiền xử lý, tách từ/analyzer, stopword và ngưỡng lọc cho query sạch và query nhiễu của cùng cấu hình.
- Giữ dấu tiếng Việt trong pipeline chính. Không tự khôi phục dấu hay sửa lỗi trước truy xuất. Nếu bổ sung bước xử lý như vậy, ghi thành một cấu hình thí nghiệm riêng.
- Term ngoài vocabulary không được thêm vào chỉ mục trong lúc đánh giá. Ghi nhận tỷ lệ term ngoài vocabulary và số query không còn đặc trưng dùng được để hỗ trợ phân tích lỗi.
- Mỗi nhánh có chỉ mục riêng; không thay cách biểu diễn giữa lượt sạch và lượt nhiễu của một phép so sánh.

## 4. Ba mô hình truy xuất

| Mô hình | Biểu diễn và cách tính điểm | Thành phần cần cố định |
|---|---|---|
| **VSM với TF-IDF** | Biểu diễn query/tài liệu bằng vector TF-IDF; xếp hạng theo cosine similarity | Vocabulary, công thức TF/IDF và chuẩn hóa vector |
| **BM25** | Tính điểm khớp term từ tần suất term, IDF và chuẩn hóa độ dài tài liệu | Vocabulary, công thức IDF, `k1`, `b` và thống kê corpus |
| **LSI/SVD** | Áp dụng truncated SVD lên ma trận TF-IDF của corpus; chiếu query vào cùng không gian tiềm ẩn và tính cosine similarity | Bộ TF-IDF, số chiều SVD, seed và phép chiếu đã fit |

Mỗi mô hình chạy với cả `word` và `char_ngram`, tạo **6 cấu hình truy xuất chính**.

Quy tắc triển khai:

- VSM và LSI dùng cùng cách tạo TF-IDF trong một nhánh term để việc so sánh rõ ràng.
- BM25 nhận tần suất term và thống kê độ dài từ nhánh tương ứng; không đưa ma trận TF-IDF đã tính sẵn vào thay cho tần suất term của BM25.
- Với `word`, độ dài tài liệu trong BM25 được tính bằng số term theo từ; với `char_ngram`, tính bằng số term n-gram theo analyzer đã chọn. Ghi rõ quy tắc đếm sau bước lọc vocabulary.
- Fit TF-IDF và SVD trên corpus, sau đó lưu để dùng lại. Query sạch và nhiễu chỉ được transform bằng các thành phần đã fit.
- Cố định tham số giữa tám điều kiện query. Nếu cần chọn ngưỡng lọc, tham số BM25 hoặc số chiều SVD, dùng tập dev riêng hoặc chốt trước; không chọn dựa trên kết quả test.
- Truy xuất trên toàn bộ 61.425 tài liệu và lấy ít nhất top 100. Run trên corpus nhỏ chỉ phục vụ kiểm tra kỹ thuật, phải ghi rõ và không gộp vào kết quả chính.

## 5. Các điều kiện query

Tất cả biến thể được tạo trực tiếp từ `queries_clean.jsonl`. Corpus và qrels được giữ nguyên.

| Điều kiện | File | Cách tạo hiện tại |
|---|---|---|
| Sạch | `queries_clean.jsonl` | Điểm chuẩn của từng cấu hình mô hình/term |
| Mất dấu | `noise/queries_no_diacritics.jsonl` | Bỏ toàn bộ dấu tiếng Việt, gồm `đ → d`, `Đ → D` |
| Lỗi gõ Telex/VNI | `noise/queries_telex_typo.jsonl` | Thiếu phím dấu/phím tạo chữ hoặc phím dấu còn dạng thô |
| Chính tả/phát âm | `noise/queries_phonetic.jsonl` | Ánh xạ theo từ điển nhầm lẫn ban đầu |
| Viết tắt/teencode | `noise/queries_abbrev.jsonl` | Ví dụ `không → ko`, `được → đc`, `với → vs` |
| Hỗn hợp nhẹ | `noise/queries_mixed_low.jsonl` | Xác suất chọn mỗi token: 10% |
| Hỗn hợp trung bình | `noise/queries_mixed_medium.jsonl` | Xác suất chọn mỗi token: 25% |
| Hỗn hợp nặng | `noise/queries_mixed_high.jsonl` | Xác suất chọn mỗi token: 40% |

Các đường dẫn trong bảng tính từ `data/zalo_legal/`. Mỗi file có **788 dòng**, cùng ID và thứ tự với file sạch. Ba điều kiện đơn Telex/VNI, phonetic và abbrev hiện dùng xác suất chọn token 25%; mất dấu toàn câu là một phép thử riêng có mức can thiệp mạnh.

### Quy tắc sinh và kiểm tra nhiễu

- Seed mặc định **42**; lưu cấu hình, phiên bản quy tắc và log từng thay đổi.
- Không ép mỗi query phải có lỗi. Token được chọn nhưng không có phép biến đổi hợp lệ sẽ được giữ nguyên và thống kê.
- Trong mixed, chọn đều một loại nhiễu trong các loại áp dụng được tại token đó; mỗi vị trí nhận tối đa một phép biến đổi.
- Ba mức mixed dùng chung lượt chọn và cùng biến thể ở các vị trí chung: các vị trí đổi ở low là tập con của medium và high.
- Đo tỷ lệ nhiễu bằng số vị trí token gốc thực sự đổi chia cho tổng token gốc có chứa chữ hoặc số. Token chỉ gồm dấu câu không nằm trong mẫu số.
- Xác suất chọn token không đồng nghĩa với tỷ lệ thực sự đổi. Từ điển phát âm/viết tắt hiện còn nhỏ nên độ phủ thấp; báo cáo tỷ lệ thực tế và tỷ lệ query thay đổi trước khi so sánh độ khó giữa các loại nhiễu.
- Giữ query không đổi trong đánh giá chính. Nếu phân tích riêng các query thực sự đổi, phải ghi rõ tập con và số mẫu.
- Giữ ID riêng khi các query khác ID có cùng văn bản sau biến đổi; thống kê collision, không loại trùng file nhiễu theo nội dung.
- Bảo vệ số, mã văn bản, từ viết tắt in hoa, tên viết hoa ngoài vị trí đầu câu và các đơn vị đã cấu hình trong các điều kiện ngoài mất dấu toàn câu. Việc nhận diện tên riêng hiện dùng heuristic.
- Rà soát mẫu để phát hiện biến đổi đổi nghĩa hoặc thiếu tự nhiên. Các cặp phát âm như `lỗi → nỗi` cần được đọc trong ngữ cảnh; dữ liệu tổng hợp hiện chưa được duyệt thủ công toàn bộ.

Lỗi bàn phím hiện được mô phỏng theo quy tắc cụ thể, chẳng hạn `phạt → phat` (thiếu phím dấu), `phạt → phatj`/`phat5` (phím dấu Telex/VNI còn dạng thô), `đầu → đàu` (thiếu phím tạo â). Script chưa mô phỏng mọi dạng lỗi của một bộ gõ đầy đủ.

**Sinh nhiễu diễn ra trên văn bản trước bước biểu diễn term.** Không tạo nhiễu riêng trên các term sau tách từ hoặc trên n-gram; như vậy hai nhánh truy xuất nhận cùng câu query đầu vào.

## 6. Thiết kế và quy trình thí nghiệm

Thiết kế chính gồm:

```text
2 cách biểu diễn term × 3 mô hình × 8 điều kiện query = 48 lượt đánh giá
```

Trong đó có 6 lượt sạch và 42 lượt nhiễu. Mỗi lượt dùng cùng 788 ID query và cùng tập tài liệu ứng viên.

| Cách biểu diễn term | Mô hình | Điều kiện |
|---|---|---|
| `word` | VSM TF-IDF, BM25, LSI/SVD | Sạch + 7 điều kiện nhiễu |
| `char_ngram` | VSM TF-IDF, BM25, LSI/SVD | Sạch + 7 điều kiện nhiễu |

Quy trình:

1. Kiểm tra query, corpus, qrels và cố định tập test.
2. Chọn nhánh term; tiền xử lý corpus, lọc vocabulary và tạo các biểu diễn/chỉ mục cho ba mô hình.
3. Chạy query sạch qua từng mô hình để lấy điểm chuẩn của đúng cấu hình term/mô hình đó.
4. Tạo hoặc sử dụng lại bảy file query nhiễu từ câu sạch với seed và cấu hình đã chốt.
5. Chạy query nhiễu bằng cùng pipeline, mô hình và chỉ mục đã dùng cho query sạch; không fit lại theo điều kiện nhiễu.
6. Tính metrics và mức suy giảm so với lượt sạch tương ứng.
7. Lặp lại cho nhánh term còn lại; tổng hợp theo mô hình, biểu diễn term, loại lỗi và mức nhiễu.
8. Phân tích các query bị giảm mạnh, thay đổi term khớp, term ngoài vocabulary và trường hợp n-gram giữ được khớp từng phần.

## 7. Đánh giá và mức suy giảm

Các metric chính:

- **MRR@10:** trung bình nghịch đảo thứ hạng của tài liệu liên quan đầu tiên trong top 10; bằng 0 nếu không tìm thấy.
- **nDCG@10:** chất lượng thứ hạng top 10, chuẩn hóa theo thứ hạng lý tưởng từ qrels.
- **Recall@100:** tỷ lệ tài liệu liên quan đã gán nhãn xuất hiện trong top 100; dùng đầy đủ qrels, kể cả query có hai tài liệu liên quan.

Với metric `M`, cùng một mô hình và cách biểu diễn term:

```text
Mức suy giảm tuyệt đối = M_sạch - M_nhiễu

Tỷ lệ suy giảm tương đối (%) = (M_sạch - M_nhiễu) / M_sạch × 100
```

- Mỗi lượt nhiễu được so với **đúng lượt sạch của cùng mô hình và nhánh term**.
- Nếu `M_sạch = 0`, ghi tỷ lệ suy giảm tương đối là `N/A`.
- Suy giảm âm nghĩa là điểm trên query nhiễu tăng so với query sạch.
- Điểm metric mặc định ở thang 0–1; nếu đổi thang, ghi rõ trong bảng và biểu đồ.
- Mức suy giảm thấp hơn cho thấy mô hình ít bị ảnh hưởng hơn trong điều kiện đang xét. Tuy nhiên, cần đọc cùng điểm sạch và điểm nhiễu: mô hình có điểm thấp ở cả hai điều kiện vẫn có thể có suy giảm nhỏ.

Mẫu bảng tổng hợp, mỗi dòng ứng với một metric và một điều kiện nhiễu:

| Term | Mô hình | Điều kiện nhiễu | Tỷ lệ nhiễu thực tế | Metric | Điểm sạch | Điểm nhiễu | Suy giảm tuyệt đối | Suy giảm tương đối (%) |
|---|---|---|---|---|---|---|---|---|
| word | BM25 | mixed_low | … | MRR@10 | … | … | … | … |
| char_ngram | BM25 | mixed_low | … | MRR@10 | … | … | … | … |

Đây là mẫu định dạng, chưa phải kết quả thực nghiệm.

### Quy tắc tính và so sánh

- Tính điểm trung bình theo query trên đủ 788 ID; không bỏ query không khớp term hoặc không có kết quả ra khỏi mẫu số.
- Nếu vector query bằng 0 trong VSM/LSI hoặc không có term dùng được để truy xuất, trả danh sách rỗng, tính metrics bằng 0 và ghi nhận tình trạng đó.
- Phân biệt trường hợp không có kết quả hợp lệ với run bị thiếu query do lỗi pipeline; run thiếu query phải được kiểm tra trước khi tổng hợp.
- Lưu điểm từng `qid` để so sánh sạch–nhiễu theo cặp. Nếu tính khoảng tin cậy, lấy mẫu bootstrap theo cặp query và ghi seed/số lần lấy mẫu.
- Không xem tám phiên bản của một query là tám mẫu độc lập.
- Dùng cùng qrels trong mọi lượt. Nhãn hiện có là cơ sở chấm điểm; tài liệu chưa có nhãn không đồng nghĩa chắc chắn không liên quan trong thực tế.

Ngoài metrics chính, nên ghi kích thước vocabulary/chỉ mục, thời gian tạo chỉ mục và thời gian truy xuất để đánh giá chi phí của character n-gram.

## 8. Định dạng đầu ra và khả năng tái lập

Kết quả xếp hạng dự kiến dùng JSONL:

```json
{"qid": "query-id", "doc_id": "document-id", "rank": 1, "score": 12.34}
```

- Mỗi cặp `qid`, `doc_id` xuất hiện tối đa một lần trong một lượt.
- `rank` bắt đầu từ 1; score giảm dần, khi bằng điểm dùng `doc_id` tăng dần để cố định thứ tự.
- Lưu danh sách ID query đã xử lý và trạng thái riêng để nhận diện cả query trả kết quả rỗng.
- Mỗi run lưu dataset, nhánh term, model, điều kiện query, seed, cấu hình tiền xử lý, ngưỡng lọc và tham số mô hình.
- Lưu SHA-256 của dữ liệu, phiên bản code, dependency, stopword/bộ tách từ và các thành phần vocabulary/IDF/SVD đã fit.
- Tổ chức output truy xuất dự kiến theo `outputs/retrieval/<term>/<model>/<condition>/` để tránh trộn kết quả.

Không dùng tập test để chọn tham số hay sửa quy tắc dựa trên điểm truy xuất. Khi thay dữ liệu, vocabulary hoặc cấu hình mô hình, chạy lại lượt sạch và các lượt nhiễu cần so sánh.

## 9. Trạng thái triển khai và cấu trúc repo

**Đã có:** dữ liệu Zalo Legal, query sạch đã loại trùng, bảy file nhiễu, cấu hình sinh nhiễu, log/thống kê/mẫu rà soát và kiểm tra tự động cho phần sinh nhiễu.

**Cần triển khai:** hai pipeline term, vocabulary và chỉ mục, ba mô hình truy xuất, module đánh giá, bảng/biểu đồ mức suy giảm và phân tích lỗi. Các quy tắc nhiễu hiện cần tiếp tục rà soát thủ công.

```text
project/
├── scripts/
│   ├── generate_noise.py          # Đã có
│   ├── prepare_data.py            # Dự kiến: chuẩn bị/kiểm tra dữ liệu
│   ├── build_index.py             # Dự kiến: word/char n-gram, vocabulary, chỉ mục
│   ├── run_retrieval.py           # Dự kiến: VSM TF-IDF, BM25, LSI/SVD
│   └── evaluate.py                # Dự kiến: metrics và mức suy giảm
├── configs/
│   ├── noise.json                # Đã có
│   └── retrieval.json            # Dự kiến: term, bộ lọc, trọng số, tham số
├── tests/
│   └── test_generate_noise.py     # Đã có
├── data/
│   └── zalo_legal/
│       ├── corpus.jsonl
│       ├── qrels_test.tsv
│       ├── queries_clean.jsonl
│       └── noise/
│           └── queries_*.jsonl    # Bảy điều kiện nhiễu
├── outputs/
│   ├── noise/                    # Đã có: log, cấu hình, thống kê và mẫu
│   └── retrieval/                # Dự kiến: term/model/condition
├── README.md
├── requirements.txt              # Cần bổ sung
└── .gitignore                    # Cần bổ sung
```

Các thành phần ghi là dự kiến chưa phải tính năng đã chạy được.

## 10. Chạy phần sinh nhiễu hiện có

Từ thư mục gốc dự án trên Windows:

```powershell
.\.venv\Scripts\python.exe scripts/generate_noise.py --seed 42
```

Script chỉ dùng thư viện chuẩn Python. Mặc định đọc `configs/noise.json` và `data/zalo_legal/queries_clean.jsonl`, kiểm tra với `qrels_test.tsv` cạnh file input nếu có, rồi ghi bảy file vào `data/zalo_legal/noise/`. File sạch, corpus và qrels được giữ nguyên. Chạy lại sẽ ghi đè các file nhiễu và báo cáo được sinh từ script.

Chỉ định đầy đủ đường dẫn và số mẫu:

```powershell
.\.venv\Scripts\python.exe scripts/generate_noise.py --input data/zalo_legal/queries_clean.jsonl --config configs/noise.json --output-dir data/zalo_legal/noise --report-dir outputs/noise --seed 42 --sample-size 10
```

| File | Nội dung |
|---|---|
| [configs/noise.json](configs/noise.json) | Seed, xác suất, từ điển và từ/đơn vị cần bảo vệ |
| [outputs/noise/summary.md](outputs/noise/summary.md) | Tỷ lệ nhiễu thực tế và số query thay đổi |
| [outputs/noise/stats.json](outputs/noise/stats.json) | Thống kê chi tiết, collision, phân bố loại lỗi và SHA-256 |
| [outputs/noise/config_used.json](outputs/noise/config_used.json) | Cấu hình thực tế của lần sinh |
| [outputs/noise/samples.md](outputs/noise/samples.md) | Mẫu cặp sạch–nhiễu để rà soát |
| `outputs/noise/*.changes.jsonl` | Log biến đổi theo query và vị trí token gốc |

Chỉ số vị trí token và offset ký tự trong log bắt đầu từ 0, tính trên câu sạch. Mẫu rà soát lấy từ các query thực sự đổi, không đại diện cho tỷ lệ query thay đổi toàn tập.

Sửa quy tắc trong config, tăng phiên bản, sinh lại và rà soát mẫu; không sửa ngầm từng câu trong file nhiễu sau khi đã chốt thí nghiệm.

Chạy kiểm tra phần sinh nhiễu:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

## 11. Phân công và quy tắc làm việc nhóm

| Vai trò | Đầu ra cần bàn giao |
|---|---|
| Dữ liệu và biểu diễn term | Dữ liệu/schema, hai pipeline word/char n-gram, stopword, DF/IDF và vocabulary |
| Sinh nhiễu | Quy tắc/từ điển, bảy file nhiễu, seed, log, thống kê và mẫu rà soát |
| Mô hình truy xuất | VSM TF-IDF, BM25, LSI/SVD chạy được với cả hai nhánh term; cấu hình và ranked results |
| Đánh giá và tích hợp | Chạy đủ ma trận thí nghiệm, metrics theo query, mức suy giảm, biểu đồ và phân tích lỗi |

Nếu có người thứ năm, giao tổng quan tài liệu và biên tập báo cáo.

- Thống nhất schema và giao diện giữa module trước khi tích hợp.
- Dùng đường dẫn/tham số qua CLI hoặc cấu hình; tránh đường dẫn tuyệt đối theo máy cá nhân.
- Lưu cấu hình cùng mỗi run; không ghi đè kết quả đã dùng trong báo cáo mà không giữ lại thông tin tái lập.
- Repo chứa code, cấu hình, dependency, hướng dẫn và trích dẫn nguồn. Giữ dữ liệu đầy đủ, `.venv/`, cache, ma trận/chỉ mục và output lớn ngoài Git; bổ sung `.gitignore` trước khi commit.
- Kiểm tra điều khoản dataset nguồn trước khi phân phối dữ liệu. Chỉ đưa mẫu nhỏ vào repo khi được phép.

## 12. Điều kiện hoàn thành

- [x] Có tập query sạch với 788 ID duy nhất và dữ liệu corpus/qrels tương ứng.
- [x] Có bảy file nhiễu cùng ID/thứ tự/số dòng, seed và cấu hình tái lập.
- [x] Có log, tỷ lệ nhiễu thực tế, thống kê collision và mẫu để rà soát.
- [ ] Hoàn thành rà soát tính tự nhiên và bảo toàn ý định truy vấn của mẫu nhiễu.
- [ ] Hoàn thành hai pipeline term, stopword/bộ lọc và vocabulary chỉ fit trên corpus.
- [ ] Triển khai VSM TF-IDF/cosine, BM25 và LSI/SVD với cả hai nhánh term.
- [ ] Chạy đủ 48 lượt đánh giá bằng corpus/qrels và cấu hình đã cố định.
- [ ] Có MRR@10, nDCG@10, Recall@100, điểm sạch/nhiễu và hai cách tính suy giảm.
- [ ] Có phân tích theo cặp query, loại/mức nhiễu và cách biểu diễn term.
- [ ] Báo cáo giới hạn: nhiễu tổng hợp, độ phủ từ điển, qrels và khả năng khái quát ngoài miền pháp lý.
- [ ] Có hướng dẫn chạy đầy đủ, dependency và quy tắc Git.

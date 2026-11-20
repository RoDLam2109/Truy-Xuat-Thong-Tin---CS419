# Chọn term và chuẩn hóa dữ liệu

Dữ liệu chính: **Lam v1**, tại `data/zalo_legal/`.
Code: `scripts/prepare_terms.py`. Cấu hình: `configs/terms.json`.

## Chuẩn hóa chung

- Corpus: ghép `title` và `text`, rồi chuẩn hóa.
- Corpus, query train, test gốc và cả 7 test nhiễu: Unicode NFC → chữ thường → gộp khoảng trắng.
- Giữ dấu tiếng Việt, số và dấu câu trong văn bản chuẩn hóa. Không sửa chính tả hay khôi phục dấu của query nhiễu.
- Giữ nguyên ID, thứ tự dòng, cách chia train/test và qrels. Dữ liệu gốc không bị ghi đè.

## Hai cách chọn term

| Nhánh | Cách tạo term | Lý do chọn |
|---|---|---|
| `word` | Tách từ bằng PyVi 0.1.1; từ ghép có dấu `_`; bỏ token chỉ có dấu câu | Baseline khớp từ, dễ giải thích |
| `char_ngram` | Chuỗi ký tự dài 3–5, trên từng đơn vị cách trắng, thêm một dấu cách ở hai biên | Kiểm tra khả năng giữ khớp từng phần khi sai ký tự |

Ví dụ, `phạt` có các n-gram như ` ph`, `phạ`, `hạt`, `ạt `. N-gram không đi xuyên qua khoảng trắng giữa hai từ; giữ dấu câu trong đơn vị gốc và không tách từ bằng PyVi ở nhánh này.

Nhánh word loại đúng 5 stopword: `và`, `là`, `của`, `các`, `những`. Giữ các từ có thể đổi nghĩa pháp lý như `không`, `chưa`, `được`, `phải`, và giữ số. Nhánh char không loại stopword.

Cả hai nhánh chọn vocabulary **chỉ từ corpus**: giữ term xuất hiện trong ít nhất 2 tài liệu và không quá 95% số tài liệu (`min_df=2`, `max_df=0.95`). DF đếm số tài liệu chứa term, không đếm số lần lặp trong cùng tài liệu. Đây là ngưỡng ban đầu; chưa tối ưu bằng validation. Không dùng điểm test để chọn term.

## Chạy và sử dụng kết quả

```bash
uv sync --locked
uv run python scripts/prepare_terms.py
```

Có thể thêm `--workers 8` để tách từ song song, hoặc `--workers 1` nếu ít RAM.

Kết quả ở `outputs/processed/`:

- Corpus và query JSONL: mỗi dòng gồm `id`, `text` đã chuẩn hóa, `word_terms` đã tách từ/bỏ stopword nhưng **chưa lọc theo DF**.
- `vocabulary_word.json`, `vocabulary_char_ngram.json`: ánh xạ term → DF sau lọc; hai vocabulary độc lập.
- Qrels train/test được sao chép nguyên vẹn; `summary.json` ghi số dòng, kích thước vocabulary, cấu hình và hash dữ liệu đầu vào.

Khi tạo ma trận count hoặc TF-IDF, dùng `terms_for` để lọc theo vocabulary. Hàm giữ số lần lặp term, cần thiết cho TF và BM25:

```python
import json
from scripts.prepare_terms import terms_for

vocabulary = json.load(open("outputs/processed/vocabulary_word.json"))
with open("outputs/processed/queries_clean.jsonl") as f:
    row = json.loads(next(f))
terms = terms_for(row, "word", vocabulary)
```

Với nhánh char, nạp vocabulary char và gọi `terms_for(row, "char_ngram", vocabulary)`. Char term được sinh khi dùng, không ghi toàn bộ ra JSONL để tránh phình dung lượng. Không fit lại vocabulary theo query. Query không còn term sau lọc vẫn được giữ để đánh giá.

Phần này chưa tạo ma trận count/TF-IDF, chưa huấn luyện model và chưa chia validation.

Nguồn bộ tách từ: [PyVi](https://github.com/trungtv/pyvi).

## Kết quả đã chạy trên toàn bộ dữ liệu

- Corpus: 61.425 tài liệu; train: 2.408 query; test gốc: 788 query; 7 test nhiễu, mỗi file 788 query.
- Vocabulary word: **16.498 term** (trước lọc: 26.143).
- Vocabulary char n-gram: **111.181 term** (trước lọc: 195.945).
- Đã đối chiếu ID, thứ tự, văn bản chuẩn hóa của toàn bộ 70.137 dòng; hash dữ liệu gốc và qrels không đổi.
- 13 kiểm tra tự động pass (`uv run python -B -m unittest discover -s tests -v`).

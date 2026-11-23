# Phân tích suy giảm truy xuất — Lam

Phần này dùng kết quả **validation**, seed 42, full corpus 61.425 documents. Không sửa term, model, qrels hay dữ liệu nhiễu dựa trên các lỗi quan sát được. Các số liệu là mô tả thực nghiệm, chưa có kiểm định ý nghĩa thống kê.

Nguồn: `outputs/lam/reports/val/summary.csv`, `paired_drops.csv`, `worst_queries.csv`; ranking đầy đủ trong `outputs/lam/experiments/val/`. Cách tái tạo nằm trong [lam_evaluation.md](lam_evaluation.md).

## Chất lượng gốc và robustness

Word + BM25 có clean nDCG@10 **0,6626**, cao hơn character n-gram + BM25 (**0,6267**). Khi tăng nhiễu mixed, character n-gram + BM25 tốt hơn: mixed high đạt **0,4756**, còn Word + BM25 đạt **0,3534**. Mức giảm tương ứng so với clean là **24,11%** và **46,67%**.

Với Telex/VNI, character n-gram + BM25 đạt **0,5435**, so với **0,4725** của Word + BM25. Các chuỗi ký tự con còn khớp sau một số lỗi gõ là một giải thích phù hợp với thiết kế và kết quả; thí nghiệm này chưa tách riêng đóng góp của từng loại n-gram hay từng quy tắc lỗi.

Theo quy tắc đã ghi trong manifest trước khi chạy, character n-gram + BM25 được chọn bằng **trung bình nDCG@10 trên bảy điều kiện nhiễu**: **0,4915**, so với **0,4604** của Word + BM25. Đây là ưu tiên theo robustness và cách cân bằng bảy điều kiện đã quy định, không có nghĩa cấu hình này thắng mọi loại query hoặc clean.

LSI-64 thấp hơn các baseline còn lại: clean nDCG@10 word **0,2420**, char **0,2637**. Không thể gọi model tốt chỉ vì mức giảm tương đối nhỏ khi điểm clean ban đầu thấp. Không tuning số chiều trong lần tích hợp này vì phần model đã được giữ nguyên theo yêu cầu của Lam.

## Bỏ dấu vẫn là điểm yếu lớn

Word + BM25 giảm **95,52%** nDCG@10 khi bỏ dấu; character n-gram + BM25 giảm **93,26%**. Điểm tuyệt đối tương ứng chỉ **0,0297** và **0,0423**.

Character n-gram + BM25 **không có query rỗng** ở điều kiện bỏ dấu nhưng chất lượng vẫn rất thấp. Vì vậy điểm thấp không thể giải thích chỉ bằng query không còn term hợp lệ. Ranking vẫn có tài liệu, nhưng các chuỗi khớp được chưa đủ phân biệt tài liệu liên quan với các điều khoản khác. Đây là diễn giải từ ranking và thống kê, chưa phải thí nghiệm chứng minh nguyên nhân.

## Tách ảnh hưởng của query không đổi

Phonetic chỉ đổi 83/482 câu; abbreviation chỉ đổi 137/482 câu. Bảng dưới tính cả clean và noisy **trên cùng tập qid đã đổi**, lấy từ `paired_drops.csv`. Không so sánh noisy của một tập con với clean của toàn bộ 482 query.

| Term/model | Nhiễu | Query đã đổi | Clean nDCG@10 trên tập con | Noisy nDCG@10 trên tập con | Drop tập con |
|---|---|---:|---:|---:|---:|
| Word/BM25 | phonetic | 83 | 0,6544 | 0,5875 | 10,23% |
| Char/BM25 | phonetic | 83 | 0,5908 | 0,5807 | 1,71% |
| Word/BM25 | abbreviation | 137 | 0,6605 | 0,6476 | 1,96% |
| Char/BM25 | abbreviation | 137 | 0,6393 | 0,6325 | 1,06% |

Ví dụ, drop phonetic của Word/BM25 trên toàn tập chỉ **1,74%**, nhỏ hơn nhiều so với **10,23%** trên các câu thực sự đổi. Đồng thời Word/BM25 vẫn có điểm noisy tuyệt đối trên tập con nhỉnh hơn Char/BM25, dù drop lớn hơn. Cần xem cả điểm tuyệt đối, baseline clean và tỷ lệ query thay đổi.

## Ví dụ lỗi của cấu hình được chọn

Các ví dụ sau là mẫu chọn theo suy giảm mạnh của character n-gram + BM25, không phải mẫu ngẫu nhiên để ước lượng tỷ lệ lỗi.

| Điều kiện | qid | Biến đổi đáng chú ý | Clean → noisy nDCG@10 |
|---|---|---|---:|
| Bỏ dấu | `0202f4503f734361cede3c679872b2cc` | Câu về mức chi hỗ trợ học sinh/sinh viên khởi nghiệp mất toàn bộ dấu | 1,0000 → 0,0000 |
| Telex | `136d1504074d05e46ca2e0e06109d0b1` | “chủ đầu tư”, “Quốc phòng”, “quy định” thành “chủ đàu tu”, “Quốc phong”, “quy đinh5” | 1,0000 → 0,0000 |
| Phonetic | `2fe7a727c2432d25f203e235e3ba6655` | “bưu chính” → “bưu trính” | 1,0000 → 0,6309 |
| Viết tắt | `136d1504074d05e46ca2e0e06109d0b1` | “trong”, “được” → “trg”, “đc” | 1,0000 → 0,6309 |
| Mixed low | `93507da019dd1a1a6278f83dc29193ee` | “Có”, “từ chối” → “Cos”, “từ choi” trong câu về từ chối nhận di sản | 1,0000 → 0,0000 |
| Mixed medium/high | `09b57975e868bdab6b09e6940742d987` | “Phạm nhân dưới 18 tuổi … gọi … thân nhân không” bị biến dạng ở nhiều từ nội dung và phủ định | 1,0000 → 0,0000 |

Trong ví dụ “bưu chính”, tài liệu `15/2020/nđ-cp+8` từ vị trí 1 xuống vị trí 2; top 1 chuyển sang `97/2017/nđ-cp+1`. Điểm nDCG giảm còn 0,6309 phản ánh thay đổi vị trí, không phải mất hoàn toàn tài liệu liên quan.

Trong ví dụ từ chối nhận di sản, clean top 1 là `91/2015/qh13+620`; noisy top 3 trở thành `91/2015/qh13+645`, `+660`, `+619`. Model vẫn truy xuất cùng văn bản luật nhưng không tìm đúng điều khoản trong top 10 theo qrels. Điều này cho thấy khớp chủ đề/văn bản chưa đủ cho truy xuất điều khoản pháp luật.

Các document ID trên cho phép đối chiếu nội dung corpus và nhãn qrels của đúng bản dataset đang dùng.

## Giới hạn và phần cần người trong nhóm xem lại

Rà soát mẫu ghi trong [lam_noise_review.md](lam_noise_review.md) phát hiện những phép như **lỗi→nỗi**, **trước→chước** tạo từ có nghĩa khác. Giữ nguyên qrels là quy ước thí nghiệm của dữ liệu hiện có; audit cấu trúc và việc đọc một số mẫu chưa chứng nhận mọi query đều giữ nghĩa.

Qrels có thể chưa đánh dấu toàn bộ tài liệu liên quan. Vì vậy top 1 khác nhãn không tự động chứng minh tài liệu hoàn toàn không liên quan. Phân tích ở đây dựa trên metrics theo nhãn hiện có và dấu vết ranking; các trường hợp mơ hồ cần thành viên nhóm đọc thêm trước khi đưa ra kết luận mạnh trong báo cáo cuối.

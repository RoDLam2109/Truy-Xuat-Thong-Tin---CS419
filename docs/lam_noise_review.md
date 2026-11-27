# Rà soát mẫu nhiễu — phần Lam

Ngày thực hiện: 21/11/2026. Phạm vi: validation seed 42, 482 query; 10 mẫu đã thay đổi cho mỗi loại nhiễu, tổng cộng 70 cặp sạch–nhiễu. Các mẫu được lấy bởi script hiện có trong `outputs/lam/splits/seed42/val/noise_report/samples.md`. Danh sách và ghi chú nằm trong [lam_noise_review.csv](lam_noise_review.csv).

Đây là nhận xét khi đọc mẫu của trợ lý, chưa phải xác nhận độc lập của thành viên nhóm hay chuyên gia pháp lý. Mẫu chỉ lấy từ câu thực sự đổi và không đại diện cho phân bố toàn tập. Không điều chỉnh generator, từ điển, query hay qrels sau khi xem mẫu.

## Kết quả đọc mẫu

| Điều kiện | Số mẫu | Nhận xét |
|---|---:|---|
| no_diacritics | 10 | Có thể khôi phục chủ đề và câu hỏi trong ngữ cảnh; mất dấu làm nhiều âm tiết trở nên đa nghĩa. Tên riêng cũng mất dấu ở điều kiện này. |
| telex_typo | 10 | Có lỗi thiếu phím tạo chữ/dấu và phím Telex/VNI chưa được tổ hợp. Một số kết quả là từ tiếng Việt khác; cần đọc cả câu. |
| phonetic | 10 | Các cặp như trường→chường, trình→chình, nước→lước vẫn gợi được từ gốc trong mẫu; từ điển có độ phủ hẹp. |
| abbrev | 10 | Các dạng không→ko/k, được→đc/dc, mình→mk có thể giải mã trong ngữ cảnh. Phủ định và quyền/nghĩa vụ là thông tin quan trọng cần giữ khi diễn giải. |
| mixed_low | 10 | Thường đổi ít vị trí; lỗi có thể rơi vào từ nội dung hoặc từ chức năng. |
| mixed_medium | 10 | Kết hợp bỏ dấu, lỗi gõ và phát âm; mức độ khó không chỉ phụ thuộc tỷ lệ token được chọn. |
| mixed_high | 10 | Nhiều từ pháp lý cùng bị biến dạng; có thể phục hồi ý định trong các mẫu đã đọc nhưng tính mơ hồ tăng rõ. |

Không thấy trường hợp chắc chắn đảo ngược ý định hỏi trong 70 mẫu đã đọc. Kết luận này không chứng minh nhãn relevant còn đúng cho mọi query nhiễu. Đặc biệt, việc sinh ra một từ khác có nghĩa có thể khiến người đọc hiểu theo cách khác với từ nguồn.

## Các trường hợp cần lưu ý

- `85bd61cc57cc729822ccb399b77eb79b`, telex: “được quy định” thành “dược quy định”. “Dược” có nghĩa riêng; ngữ cảnh câu về kinh phí đo đạc/bản đồ giúp suy ra lỗi, nhưng phép nhiễu không bảo đảm giữ nghĩa ở mức từ.
- `09b57975e868bdab6b09e6940742d987`, mixed low: câu về phạm nhân dưới **18 tuổi** giữ nguyên con số nhưng biến “thân nhân” thành “than nhân”, “không” thành “k”. Kiểm tra số được giữ chưa đủ để chứng nhận toàn bộ ý nghĩa câu.
- `8342f011cdaab195eca3fb9c1a4bd098`, abbrev: “mình”→“mk”, “không”→“ko”. Có thể giải mã, nhưng cách viết tắt mang tính quy ước và không bao quát mọi cách viết thực tế.
- `c95390faac6cb56fcb6511b1f2c9130c`, mixed high: “phải nộp”→“phai nop”, “không”→“khong”. Câu vẫn có thể đọc như câu hỏi gốc; các từ thể hiện nghĩa vụ/phủ định đã mất dấu nên cần nêu giới hạn khi dùng qrels gốc.
- `11f7688abb848aad1f19a3107a95f88d`, mixed high: giữ **2019**, nhưng “công bố” thành “cong bó”, “hợp chuẩn” thành “hợp chuân”. Số năm không đổi trong khi khái niệm cần truy xuất bị biến dạng.

Ngoài 70 mẫu trên, đã đọc thêm năm câu được tìm theo các phép thay có thể tạo từ khác có nghĩa trong toàn bộ log phonetic validation:

| qid | Cụm gốc → cụm nhiễu | Nhận xét |
|---|---|---|
| `7e9a91a8b157ead81059304ae538cc04` | tha tù **trước** thời hạn → tha tù **chước** thời hạn | “Chước” có nghĩa riêng; thuật ngữ pháp lý chỉ phục hồi được nhờ ngữ cảnh. |
| `96080fc532670c4145d701bebf3f3ef9` | cháy rừng **xảy ra** → cháy rừng **sảy ra** | Dạng sai chính tả dễ nhận ra trong ngữ cảnh. |
| `cfed931421f7acb7ebffb81245bc0de8` | **trước** ngày … nhưng **sau** ngày này → **chước** ngày … nhưng **xau** ngày này | Mốc ngày không đổi nhưng quan hệ thời gian bị biến dạng. |
| `a6732b2c4a59f3bdf592c8aa7fd7d013` | bao nhiêu **năm** → bao nhiêu **lăm** | Đơn vị thời gian không có số liền kề nên vẫn bị thay; cần lưu ý giới hạn bảo vệ đơn vị. |
| `1a6404b7ee86e9178fc6f5e81895e705` | không phải **lỗi** của mình → không phải **nỗi** của mình | “Lỗi” là khái niệm ảnh hưởng trách nhiệm; “nỗi” là từ có nghĩa khác. Đây là trường hợp cần thành viên nhóm kiểm tra nếu muốn khẳng định nhãn không đổi. |

Năm câu bổ sung được chọn theo rủi ro từ vựng, không đưa vào mẫu 70 cặp trong CSV và không dùng để ước lượng tỷ lệ lỗi ngữ nghĩa.

## Độ phủ và giới hạn

Trên validation, token thực sự đổi lần lượt khoảng **85,56%** (bỏ dấu), **21,18%** (Telex/VNI), **1,02%** (phonetic), **1,72%** (abbreviation), **8,89% / 21,46% / 33,33%** (mixed low/medium/high). Phonetic chỉ đổi **83/482** câu; abbreviation đổi **137/482** câu. Vì vậy không thể kết luận một model chịu được nhiễu phát âm/viết tắt chỉ vì điểm toàn tập gần clean.

Báo cáo mới cung cấp đồng thời metrics toàn tập và `changed_only`, giữ mọi query trong mẫu số chính. Các mức mixed dùng cùng vị trí/biến thể cho phần giao nhau theo generator đã có. Phím VNI có thể tạo thêm ký tự số trong từ bị lỗi; cơ chế bảo vệ token có số gốc không loại bỏ hiện tượng này.

Không mở rộng từ điển hay sửa quy tắc trong lần tích hợp này. Thay đổi chúng sẽ tạo điều kiện thí nghiệm mới và cần Lam đồng ý trước.
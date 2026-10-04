# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Công Vinh
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8585 | 0.8764 | +0.0179 |
| Answer Relevancy | 0.5073 | 0.5277 | +0.0204 |
| Context Precision | 1.0000 | 1.0000 | +0.0000 |
| Context Recall | 0.7894 | 0.8969 | +0.1075 |

Production cải thiện cả bốn chỉ số có ý nghĩa so với Basic, đặc biệt context recall tăng 0.1075 nhờ hybrid retrieval và ghép context theo parent. Context precision vẫn bằng 1.0000 ở cả hai pipeline nên chưa đủ tin cậy; metric fallback hiện chỉ kiểm tra context có ít nhất một token trùng query, chưa đo tỷ lệ thông tin thực sự liên quan.

## Bottom-5 Failures

> Thứ tự dưới đây lấy theo `average_score` tăng dần trong `reports/ragas_report.json`.

### #1 — Số ngày phép năm

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Chính sách hiện hành v2024 cho 15 ngày phép năm có lương; chính sách cũ v2023 là 12 ngày nhưng đã bị thay thế.
- **Got:** `answer_relevancy = 0.0000`, `average_score = 0.4964`. Answer không trả lời trực tiếp hoặc không phân biệt rõ 15 ngày hiện hành với 12 ngày của chính sách cũ.
- **Worst metric:** `answer_relevancy` — score 0.0000.
- **Error Tree:** Output đúng? Không đủ → Context đúng? Có thể chứa hai phiên bản → Query OK? Có → Lỗi version disambiguation/answer synthesis.
- **Root cause:** Corpus có cả tài liệu v2023 và v2024. Nếu retrieval đưa cả hai phiên bản vào context, LLM hoặc fallback dễ trả lời mơ hồ và không ưu tiên chính sách mới nhất.
- **Suggested fix:** Thêm metadata filter theo version/effective date, ưu tiên v2024 và yêu cầu answer nêu rõ “15 ngày theo chính sách hiện hành”.

### #2 — Lương thử việc của Junior

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Lương Junior cao nhất là 20.000.000 VNĐ/tháng; lương thử việc bằng 85%, tức 17.000.000 VNĐ/tháng.
- **Got:** `answer_relevancy = 0.0000`, `average_score = 0.5417`. Answer chưa nối được mức lương Junior cao nhất với phép tính 85% lương thử việc.
- **Worst metric:** `answer_relevancy` — score 0.0000.
- **Error Tree:** Output đúng? Thiếu kết quả cuối → Context đúng? Có thể có bảng lương và quy định thử việc ở các chunk khác nhau → Query OK? Có → Lỗi multi-hop answer synthesis.
- **Root cause:** Câu hỏi cần kết hợp hai thông tin: mức Junior tối đa và tỷ lệ lương thử việc. Nếu chỉ retrieve một chunk, model có thể trả 20 triệu hoặc 85% mà không tính ra 17 triệu.
- **Suggested fix:** Decompose query thành “Junior tối đa bao nhiêu?” và “thử việc tính theo tỷ lệ nào?”, sau đó buộc answer trình bày công thức `85% × 20.000.000 = 17.000.000`.

### #3 — Phân loại thông tin lương

- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Thông tin lương là dữ liệu Bí mật, cấp 3; phải mã hóa khi truyền và hạn chế truy cập theo nguyên tắc need-to-know.
- **Got:** `answer_relevancy = 0.0000`, `average_score = 0.6288`. Answer chưa nêu trực tiếp cấp độ “Bí mật/cấp 3” hoặc các yêu cầu bảo vệ dữ liệu.
- **Worst metric:** `answer_relevancy` — score 0.0000.
- **Error Tree:** Output đúng? Không đủ → Context đúng? Có thể đúng nhưng phân tán giữa quy chế lương và chính sách phân loại dữ liệu → Query OK? Có → Lỗi context assembly/answer synthesis.
- **Root cause:** Đây là câu hỏi cần nối thông tin từ hai phần tài liệu: loại dữ liệu lương và quy định bảo vệ dữ liệu cấp 3.
- **Suggested fix:** Group context theo topic “salary + data classification”, thêm metadata/category filter và yêu cầu answer trả lời theo cấu trúc loại dữ liệu → cấp độ → biện pháp bảo vệ.

### #4 — Senior 9 năm thâm niên

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép: 15 ngày cơ bản + 3 ngày thâm niên; lương Senior P3–P4 là 20–35 triệu VNĐ/tháng.
- **Got:** `context_recall = 0.5455`, `average_score = 0.7260`. Context bị thiếu một phần thông tin về số ngày phép hoặc khoảng lương nên khó trả lời đủ hai vế.
- **Worst metric:** `context_recall` — score 0.5455.
- **Error Tree:** Output đúng? Chưa đủ → Context đúng? Thiếu một nguồn → Query OK? Có → Lỗi retrieval/chunking.
- **Root cause:** Đây là câu hỏi multi-hop, cần ghép chính sách phép năm với bảng lương Senior. Top-k hoặc parent context chưa bao phủ đầy đủ cả hai nguồn.
- **Suggested fix:** Query decomposition, giữ nhiều parent khác nhau sau rerank và merge context theo source/topic trước khi đưa cho LLM.

### #5 — Laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Giám đốc phòng ban (Director) phê duyệt vì giá trị nằm trong khoảng 5–50 triệu; cần xác nhận cấu hình kỹ thuật từ phòng CNTT và ít nhất 3 báo giá vì giá trị trên 10 triệu.
- **Got:** `answer_relevancy = 0.3454`, `average_score = 0.7969`. Answer đã liên quan hơn nhưng chưa trả lời đầy đủ cả người phê duyệt, xác nhận CNTT và yêu cầu báo giá.
- **Worst metric:** `answer_relevancy` — score 0.3454.
- **Error Tree:** Output đúng? Một phần → Context đúng? Được cải thiện nhưng có thể còn nhiều đoạn phụ → Query OK? Có → Lỗi answer synthesis.
- **Root cause:** Câu hỏi có ba yêu cầu. Nếu answer chỉ lấy một đoạn trong quy trình mua sắm, nó sẽ bỏ sót bảng thẩm quyền hoặc điều kiện kỹ thuật/báo giá.
- **Suggested fix:** Parent-context expansion, loại duplicate parent và prompt answer dạng checklist gồm Director + xác nhận CNTT + 3 báo giá.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

**Error Tree walkthrough:**
1. **Output đúng?** → Không đạt; answer relevancy bằng 0.
2. **Context đúng?** → Có khả năng context chứa cả tài liệu v2023 và v2024 nhưng không xác định tài liệu hiện hành.
3. **Query rewrite OK?** → Query rõ; vấn đề là version-aware retrieval và answer synthesis.
4. **Fix ở bước:** Gắn metadata `version/effective_date`, ưu tiên v2024 và bắt buộc answer nêu 15 ngày hiện hành, đồng thời ghi chú 12 ngày là chính sách cũ.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm version-aware retrieval và conflict resolution cho các tài liệu chính sách cũ/mới.
- Sửa `context_precision` để đo ranking/coverage thực tế thay vì chỉ kiểm tra có một token trùng.
- Lưu per-question answer và contexts trong report để failure analysis đối chiếu được output thực tế.

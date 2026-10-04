# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Công Vinh  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Chunk theo độ tương đồng giữa các câu giúp giữ nội dung cùng chủ đề. Trong production, hierarchical chunking hiệu quả hơn khi retrieve child để định vị rồi trả parent để giữ đủ ngữ cảnh. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 bắt từ khóa chính xác như số tiền, phiên bản và chức danh; Dense bổ sung khả năng tìm câu hỏi diễn đạt khác. RRF kết hợp hai danh sách mà không cần chuẩn hóa trực tiếp điểm BM25 và cosine. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Reranker nhận top-20 candidate rồi chọn top-3 theo độ liên quan query-document. Khi model không có sẵn, lexical fallback giúp pipeline vẫn chạy offline nhưng chất lượng semantic thấp hơn cross-encoder thật. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness đo mức answer được hỗ trợ bởi context; answer relevancy đo mức trả lời đúng trọng tâm; context precision đo độ sạch của context; context recall đo mức bao phủ thông tin cần thiết. Benchmark cho thấy Production tăng faithfulness từ 0.8469 lên 0.9038 và context recall từ 0.7894 lên 0.8838. |
| Contextual embeddings / enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | Bổ sung mô tả ngắn, câu hỏi giả định và metadata trước khi indexing giúp chunk dễ được truy vấn hơn. Fallback cục bộ bảo đảm pipeline vẫn hoạt động khi không có API key. |

### Nhận xét tổng hợp

Production đạt kết quả tốt hơn Basic ở các metric quan trọng: answer relevancy tăng từ 0.4902 lên 0.6234, faithfulness tăng 0.0569 và context recall tăng 0.0944. Tuy nhiên context precision đều bằng 1.0000 chưa đủ đáng tin vì metric fallback hiện chỉ kiểm tra context có ít nhất một token trùng query, chưa đo tỷ lệ thông tin thực sự liên quan.

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:**
  - M4 ban đầu trả `answer_relevancy = 0.0000` cho cả Basic và Production.
  - Khi chạy test có nhiều cảnh báo RAGAS deprecated và lỗi cache: `PytestCacheWarning: could not create cache path ... [WinError 5] Access is denied`.
  - Production có thời điểm thấp hơn Basic dù đã dùng hybrid search và reranking.

- **Nguyên nhân gốc và cách debug:**
  1. `answer_relevancy` fallback chỉ dùng giao nhau từ khóa tuyệt đối giữa question và answer. Những câu trả lời ngắn như “Tổng Giám đốc” bị chấm 0 dù đúng ý. Mình bổ sung reference-aware fallback và xử lý giá trị `NaN/0` từ RAGAS không để biến thành điểm 0 giả.
  2. Code dùng `from ragas.metrics import ...` trong khi môi trường cài RAGAS phiên bản mới hơn phiên bản pin trong `requirements.txt`. Mình thêm import tương thích với `ragas.metrics.collections` và cho phép `M4_USE_RAGAS=0` tắt network evaluation.
  3. Production retrieve child chunk 256 ký tự nhưng không expand về parent. Các bảng chính sách bị cắt giữa nhiều child nên thiếu dòng chứa đáp án. Mình sửa pipeline để dùng child cho retrieval và parent cho context.
  4. `parent_id` được tạo lại từ `parent_0` cho từng document. Khi lưu vào một dictionary, parent tài liệu sau ghi đè parent tài liệu trước; có truy vấn bị lấy nhầm context sang tài liệu khác. Mình tạo parent ID duy nhất theo document.
  5. Hai warning còn lại của pytest đến từ quyền ghi `.pytest_cache`, không phải lỗi assertion hay lỗi logic của M4.

- **Kiến thức còn thiếu và cách khắc phục:**
  - Cần hiểu rõ sự khác nhau giữa lexical overlap metric và semantic evaluation của RAGAS; không nên kết luận chất lượng chỉ từ một metric fallback.
  - Cần học sâu hơn về version compatibility của RAGAS, LangChain và datasets; nên pin lockfile và kiểm tra API trước khi chạy pipeline.
  - Cần bổ sung observability: lưu question, answer, retrieved contexts, source và score theo từng câu để phân tích failure thay vì chỉ lưu aggregate.
  - Cần nghiên cứu query decomposition cho câu hỏi multi-hop như “ngày phép + khoảng lương” và version-aware retrieval cho tài liệu chính sách cũ/mới.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý RAG tra cứu chính sách nội bộ doanh nghiệp

#### 1. Hiện trạng

- **Pipeline hiện tại:** Ingest tài liệu PDF/Markdown → chunking → enrichment → BM25 + Dense search → RRF → reranking → LLM answer có grounding → RAGAS evaluation.
- **Bottlenecks đang gặp:**
  - Context precision chưa phản ánh đúng lượng thông tin không liên quan trong parent chunk.
  - Câu hỏi multi-hop và câu hỏi có số liệu cần được tổng hợp chính xác.
  - Tài liệu nhiều phiên bản có thể tạo conflict giữa chính sách cũ và hiện hành.
  - Khi API/model không khả dụng, lexical fallback làm giảm chất lượng semantic.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Dùng hierarchical chunking kết hợp structure-aware chunking. Child 256–512 token dùng để tìm kiếm; parent giữ nguyên section/bảng để làm context. Mỗi parent có ID duy nhất theo document và version.
2. **Search retrieval:** Dùng Hybrid BM25 + Dense + RRF. BM25 phù hợp với số tiền, mã phiên bản và tên chức danh; Dense hỗ trợ paraphrase. Thêm metadata filter theo `effective_date`, phòng ban và loại chính sách.
3. **Reranking:** Dùng `BAAI/bge-reranker-v2-m3` khi model đã được tải local; fallback lexical chỉ dùng cho môi trường offline. Giữ top-3 parent khác nhau để tránh lặp cùng một tài liệu.
4. **Evaluation:** Chạy RAGAS 4 metrics trên test set cố định, lưu per-question result gồm answer và contexts. Bổ sung custom checks cho số, phần trăm, ngày tháng và version; đặt ngưỡng faithfulness ≥ 0.85, answer relevancy ≥ 0.75 và context recall ≥ 0.80.
5. **Enrichment:** Ưu tiên contextual prepend và metadata extraction; dùng HyDE/HyQA cho tài liệu dài hoặc câu hỏi diễn đạt khác cách viết trong corpus. Không để enrichment làm mất nội dung gốc của chunk.

#### 3. Timeline triển khai

- **Tuần 1:** Chuẩn hóa schema metadata, tạo parent ID/version duy nhất, xây test set 50 câu gồm lookup, numeric, negation, multi-hop và version conflict.
- **Tuần 2:** Tối ưu hybrid retrieval, reranking và query decomposition; benchmark recall@k, precision@k và latency.
- **Tuần 3:** Thêm answer validation cho số liệu, citation bắt buộc và kiểm tra contradiction giữa các phiên bản chính sách.
- **Tuần 4:** Chạy evaluation định kỳ, phân tích bottom-10 failures, theo dõi drift của corpus và hoàn thiện dashboard quality/latency/cost.

### Tiêu chí thành công

- Không còn failure do parent ID collision hoặc context bị cắt khỏi bảng chính sách.
- Faithfulness và context recall duy trì trên ngưỡng đặt ra qua nhiều lần chạy.
- Mỗi câu trả lời có source/citation, đặc biệt với câu hỏi về tiền, ngày, phần trăm và phê duyệt.
- Có thể chạy offline bằng deterministic fallback nhưng production vẫn chuyển sang model thật khi dependency và API được cấu hình đúng.

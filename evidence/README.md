# Bằng chứng — Day 22: LangSmith + Prompt Versioning

**Sinh viên:** Nguyễn Hoàng Đạt — 2A202601460
**LangSmith project:** `day22-lab` — tổng cộng **100 traces** (50 `rag-query` + 50 `ab-rag-query`)
**Provider:** OpenAI (`gpt-4o-mini` + `text-embedding-3-small`)

---

## Danh sách tệp

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | LangSmith dashboard — 50 traces `rag-query` của Bước 1 |
| `01_rag_log.txt` | Log console Bước 1 (50 câu hỏi + câu trả lời) |
| `02_prompt_hub.png` | Prompt Hub — 2 phiên bản `nguyenhoangdat-rag-v1` và `-v2` |
| `02_ab_routing_log.txt` | Log A/B routing — 50 truy vấn có nhãn `[prompt-v1]` / `[prompt-v2]` |
| `03_ragas_scores.png` | Bảng so sánh điểm RAGAS V1 vs V2 trên terminal |
| `03_ragas_report.json` | Bản sao của `data/ragas_report.json` |
| `03_ragas_log.txt` | Log đầy đủ quá trình chạy RAGAS |
| `04_pii_demo_log.txt` | 6 test case PII detection & redaction |
| `04_json_demo_log.txt` | 5 test case JSON repair |

---

## Phân tích kết quả V1 vs V2

| Metric | V1 (concise) | V2 (structured) | Thắng |
|---|---|---|---|
| faithfulness | **0.9742** | 0.7822 | V1 |
| answer_relevancy | **0.9160** | 0.8913 | V1 |
| context_recall | 1.0000 | 1.0000 | Hòa |
| context_precision | 0.9450 | 0.9450 | Hòa |

**Hai prompt khác nhau ở điểm gì:** V1 yêu cầu trả lời trực tiếp trong 2–4 câu, không rào đón.
V2 yêu cầu giọng chuyên gia, viết 3–5 câu theo bố cục *định nghĩa → cơ chế → ý nghĩa thực tiễn*.
Cả hai đều bị cấm dùng kiến thức ngoài context.

**Vì sao V1 thắng rõ ở faithfulness (0.97 vs 0.78):**

1. **Số lượng claim quyết định mẫu số.** RAGAS tính faithfulness = (số claim suy ra được từ
   context) / (tổng số claim). V1 viết ngắn nên sinh ít claim, mỗi claim đều bám sát đoạn đã
   retrieve. V2 viết dài theo 3 phần nên sinh nhiều claim hơn — mẫu số lớn hơn và dễ trượt hơn.

2. **Yêu cầu "why it matters in practice" của V2 chính là lời mời suy diễn.** Phần ý nghĩa thực
   tiễn thường không có sẵn trong knowledge base, nên mô hình tự bổ sung kiến thức ngoài context
   dù prompt đã cấm. Đây là nguồn mất điểm chính của V2.

3. **`context_recall` và `context_precision` bằng nhau tuyệt đối** vì hai phiên bản dùng chung
   retriever, chung FAISS index và chung `k=3`. Hai chỉ số này chỉ đo chất lượng **tìm kiếm**,
   không phụ thuộc prompt sinh câu trả lời — xác nhận khác biệt V1/V2 nằm hoàn toàn ở khâu sinh,
   đúng như thiết kế A/B.

4. **`answer_relevancy` của V1 cũng nhỉnh hơn** (0.9160 vs 0.8913): câu trả lời ngắn bám sát câu
   hỏi, trong khi phần mở rộng của V2 làm loãng độ liên quan.

**Kết luận:** cho hệ RAG ưu tiên độ tin cậy, chọn **V1**. V2 chỉ nên dùng khi người dùng cần giải
thích sâu và chấp nhận đánh đổi ~0.19 điểm faithfulness.

---

## Ghi chú kỹ thuật

- **A/B routing tất định:** `int(md5(request_id).hexdigest(), 16) % 2` → chẵn = V1, lẻ = V2.
  Với 50 `request_id` dạng `req-0000…req-0049`, kết quả là V1 = 19 câu, V2 = 31 câu và **lặp lại
  y hệt** ở mọi lần chạy.
- **Prompt được pull từ Hub lúc runtime** (`client.pull_prompt`), có fallback về template local
  nếu Hub không truy cập được — log xác nhận `↓ Đã pull … từ Hub`, không dùng fallback.
- **Guardrails:** `PassResult(value_override=…)` bị guardrails 0.11 bỏ qua khiến PII vẫn lọt ra
  đầu ra, nên cả hai validator dùng `FailResult(fix_value=…)` kết hợp `OnFailAction.FIX` (truyền
  vào constructor của validator) để thực sự thay thế output.
- **Phiên bản thư viện:** `requirements.txt` ghim `langchain*` ở nhánh 0.3.x vì `ragas 0.4.x`
  import `langchain_community.chat_models.vertexai` — module đã bị xóa khỏi
  `langchain-community 0.4.x`.

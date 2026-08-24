"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# Sao chép nguyên văn từ 02_prompt_hub_ab_routing.py để đánh giá đúng 2 phiên bản đã push lên Hub
SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer using ONLY the context below.\n"
    "Keep your answer short and direct: 2-4 sentences, no preamble, no bullet points.\n"
    "If the context does not contain the answer, say exactly: \n"
    "I don't know based on the given context.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are a senior AI research expert. Read the context carefully before answering.\n"
    "Follow this structure: (1) state the core definition, (2) explain the key mechanism "
    "or components, (3) note why it matters in practice. Write 3-5 organised sentences.\n"
    "Ground every claim strictly in the context — never add outside knowledge. "
    "If the context is insufficient, say exactly: \n"
    "I don't know based on the given context.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # RAGAS cần TỪNG đoạn riêng để tính context_recall / context_precision
    contexts = [doc.page_content for doc in docs]

    # Riêng prompt thì cần 1 chuỗi duy nhất cho biến {context}
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        # RAGAS có thể trả về list điểm từng sample hoặc 1 số đã trung bình sẵn
        if isinstance(raw, (int, float)):
            scores[key] = float(raw)
        else:
            valid = [v for v in raw if v is not None and not np.isnan(v)]
            scores[key] = float(np.mean(valid)) if valid else 0.0

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── PHÂN TÍCH KẾT QUẢ V1 vs V2 ─────────────────────────────────
# Kết quả đo được (xem data/ragas_report.json):
#
#   Metric              V1 (concise)   V2 (structured)
#   faithfulness            0.9742          0.7822
#   answer_relevancy        0.9160          0.8913
#   context_recall          1.0000          1.0000
#   context_precision       0.9450          0.9450
#
# Vì sao V1 thắng rõ ở faithfulness (0.97 vs 0.78)?
#
# 1. Số lượng claim quyết định điểm. RAGAS tính faithfulness = (số claim có thể suy ra
#    từ context) / (tổng số claim). V1 bị ép trả lời 2-4 câu nên sinh ít claim, mỗi claim
#    đều bám sát đoạn đã retrieve. V2 bị ép viết 3-5 câu theo bố cục định nghĩa → cơ chế
#    → ý nghĩa thực tiễn, nên sinh nhiều claim hơn — mẫu số lớn hơn và dễ trượt hơn.
#
# 2. Yêu cầu "why it matters in practice" của V2 chính là lời mời suy diễn. Phần
#    ý nghĩa thực tiễn thường KHÔNG có sẵn trong knowledge base, nên mô hình bổ sung
#    kiến thức ngoài context dù prompt đã cấm — đây là nguồn mất điểm chính của V2.
#
# 3. context_recall và context_precision bằng nhau tuyệt đối ở cả hai (1.0000 / 0.9450)
#    vì hai phiên bản dùng chung retriever, chung FAISS index và chung k=3. Hai chỉ số
#    này chỉ đo chất lượng TÌM KIẾM, không phụ thuộc prompt sinh câu trả lời.
#    Điều này xác nhận khác biệt V1/V2 nằm hoàn toàn ở khâu sinh, đúng như thiết kế A/B.
#
# 4. answer_relevancy của V1 cũng nhỉnh hơn (0.9160 vs 0.8913): câu trả lời ngắn bám
#    sát câu hỏi, trong khi phần mở rộng của V2 làm loãng độ liên quan.
#
# KẾT LUẬN: cho hệ RAG ưu tiên độ tin cậy, chọn V1. V2 chỉ nên dùng khi người dùng
# cần giải thích sâu và chấp nhận đánh đổi ~0.19 điểm faithfulness.


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # Báo cáo chứa điểm của CẢ V1 lẫn V2 (tiêu chí 3.5)
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()

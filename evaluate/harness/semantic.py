"""
ความคล้ายเชิงความหมายระหว่างคำตอบกับ reference_answer — คำนวณบนเครื่อง ไม่เรียก LLM

1. Cosine similarity ของ sentence embedding จาก intfloat/multilingual-e5-base
   ใส่ prefix "query: " ทั้งสองฝั่ง (e5 แนะนำให้ใช้ query: กับงานเทียบความหมายแบบสมมาตร)
2. BERTScore (Zhang et al., 2020) เขียนเองตามนิยามในเปเปอร์ ใช้ token embedding จาก hidden layer ที่ 9
   ของโมเดลเดียวกัน (สถาปัตยกรรม XLM-RoBERTa-base; ชั้น 9 คือค่าที่ bert-score แนะนำสำหรับ xlm-roberta-base)
   greedy matching ด้วย cosine, ไม่ถ่วง idf และไม่ทำ baseline rescaling — ค่าจึงไม่เท่ากับ
   แพ็กเกจ bert-score โดยตรง และสูงกว่าโดยรวม (ระบุไว้ในรายงาน) แต่ใช้เทียบระหว่างข้อ/หมวดได้
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.config import add_api_to_path, append_jsonl, read_jsonl
from harness.metrics import bertscore_from_similarity

BERTSCORE_LAYER = 9


def run(run_dir: Path, scored: list[dict[str, Any]], references: dict[str, str]) -> None:
    add_api_to_path()
    import torch
    from app.services.embedding_service import get_embedding_service

    raw_path = run_dir / "raw" / "semantic.jsonl"
    done = {r["id"] for r in read_jsonl(raw_path)}
    todo = [s for s in scored if s.get("status") == "ok" and s["id"] in references and s["id"] not in done and s.get("answer")]
    print(f"semantic: {len(todo)} ข้อ")
    if not todo:
        return
    service = get_embedding_service()
    model = service._model
    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    def token_embeddings(text: str) -> torch.Tensor:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            hidden = transformer(**encoded, output_hidden_states=True).hidden_states[BERTSCORE_LAYER][0]
        hidden = hidden[1:-1]  # ตัด <s> และ </s>
        return torch.nn.functional.normalize(hidden, dim=-1)

    for item in todo:
        answer, reference = item["answer"], references[item["id"]]
        vectors = model.encode([f"query: {answer}", f"query: {reference}"], normalize_embeddings=True)
        cosine = float((vectors[0] * vectors[1]).sum())
        a, r = token_embeddings(answer), token_embeddings(reference)
        scores = bertscore_from_similarity((a @ r.T).tolist())
        append_jsonl(raw_path, {"id": item["id"], "cosine": round(cosine, 4),
                                "bertscore_precision": round(scores["precision"], 4),
                                "bertscore_recall": round(scores["recall"], 4),
                                "bertscore_f1": round(scores["f1"], 4)})
    print(f"บันทึกที่ {raw_path}")

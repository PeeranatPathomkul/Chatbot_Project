"""
ประเมิน retrieval โดยไม่เรียก LLM — ค้น ChromaDB ด้วยวิธีเดียวกับ ToolExecutor._search_knowledge_base
(EmbeddingService.embed_query ใส่ prefix "query: ", VectorStore.query กรองภาษา, top_k = settings.retrieval_top_k)

แอปค้นสองรอบ (คำค้นของโมเดล + คำถามลูกค้า) แล้วรวมผล ตรงนี้ไม่มีคำค้นของโมเดลเพราะไม่ได้เรียก LLM
จึงใช้คำถามของลูกค้าอย่างเดียว ซึ่งเท่ากับรอบที่สองของแอปพอดี (บันทึกเป็นข้อจำกัดในรายงาน)

ชุดคำถาม:
- golden: ข้อใน datasets/ ที่มี expected_sources (คำถามข้อความเดียว) และข้อที่ตอบไม่ได้ (ดูการกระจายคะแนน)
- legacy: 44 + 10 ข้อเดิมจาก indexing-pipeline/scripts/evaluate.py (ตัวเลขใน proposal มาจากชุดนี้)

ผลดิบ (top-k ทุกคำถาม + ทุก chunk ใน collection) เขียนลง raw/ แล้วคำนวณ relevance ตอนสรุปผล
จึงเปลี่ยนนิยาม relevance แล้วคำนวณใหม่ได้โดยไม่ต้องค้นซ้ำ
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

from harness import dates
from harness.config import INDEXING_DIR, add_api_to_path, append_jsonl, read_jsonl, write_json
from harness.dataset import load_items


def _legacy_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """โหลดชุดคำถามเดิมจากสคริปต์ของ indexing-pipeline โดยไม่ก๊อปมาเขียนซ้ำ"""
    if str(INDEXING_DIR) not in sys.path:
        sys.path.append(str(INDEXING_DIR))
    spec = importlib.util.spec_from_file_location("legacy_retrieval_eval", INDEXING_DIR / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass ในไฟล์นั้นต้องหาโมดูลของตัวเองเจอใน sys.modules
    spec.loader.exec_module(module)
    answerable = [
        {"set": "legacy", "id": f"legacy-{i:02d}", "language": c.language, "question": c.question,
         "answerable": True, "expected_sources": sorted(c.expected_sources), "key_facts": []}
        for i, c in enumerate(module.ANSWERABLE, start=1)
    ]
    unanswerable = [
        {"set": "legacy", "id": f"legacy-un-{i:02d}", "language": lang, "question": q,
         "answerable": False, "expected_sources": [], "key_facts": []}
        for i, (q, lang) in enumerate(module.UNANSWERABLE, start=1)
    ]
    return answerable, unanswerable


def build_queries() -> list[dict[str, Any]]:
    today = dates.today_in_thailand()
    queries: list[dict[str, Any]] = []
    for item in load_items():
        if "turns" in item or item.get("suite") == "security":
            continue
        if item.get("expected_sources") or item.get("answerable") is False:
            queries.append({
                "set": "golden", "id": item["id"], "suite": item["suite"], "category": item["category"],
                "language": item["language"], "question": dates.render(item["question"], today),
                "answerable": item.get("answerable"), "expected_sources": item.get("expected_sources") or [],
                "key_facts": item.get("key_facts") or [],
            })
    legacy_answerable, legacy_unanswerable = _legacy_cases()
    return queries + legacy_answerable + legacy_unanswerable


def run(run_dir: Path, limit: int | None = None) -> None:
    add_api_to_path()
    from app.config import settings as app_settings
    from app.services.embedding_service import get_embedding_service
    from app.services.vector_store import get_vector_store

    raw_path = run_dir / "raw" / "retrieval.jsonl"
    done = {r["id"] for r in read_jsonl(raw_path)}
    queries = [q for q in build_queries() if q["id"] not in done]
    if limit:
        queries = queries[:limit]

    started = time.perf_counter()
    embedder = get_embedding_service()
    load_seconds = time.perf_counter() - started
    store = get_vector_store()
    top_k = app_settings.retrieval_top_k

    everything = store.collection.get(include=["documents", "metadatas"])
    chunks = [
        {"id": cid, "source": meta.get("source"), "language": meta.get("language"), "heading": meta.get("heading"), "text": text}
        for cid, text, meta in zip(everything["ids"], everything["documents"], everything["metadatas"])
    ]
    write_json(run_dir / "raw" / "retrieval_meta.json", {
        "collection": app_settings.chroma_collection_name,
        "chunk_count": store.collection.count(),
        "chunks_by_language": {lang: sum(1 for c in chunks if c["language"] == lang) for lang in ("th", "en")},
        "top_k": top_k,
        "embedding_model": app_settings.embedding_model,
        "embedding_load_seconds": round(load_seconds, 2),
        "chunks": chunks,
    })

    print(f"retrieval: {len(queries)} คำถาม, collection {store.collection.count()} chunk, top_k {top_k}")
    for number, query in enumerate(queries, start=1):
        t0 = time.perf_counter()
        hits = store.query(embedder.embed_query(query["question"]), top_k=top_k, language=query["language"])
        elapsed = (time.perf_counter() - t0) * 1000
        append_jsonl(raw_path, {
            **query,
            "query_ms": round(elapsed, 1),
            "hits": [
                {"rank": rank, "doc_id": h["doc_id"], "source": (h.get("metadata") or {}).get("source"),
                 "distance": h["distance"], "similarity": 1.0 - h["distance"], "text": h["text"]}
                for rank, h in enumerate(hits, start=1)
            ],
        })
        if number % 25 == 0:
            print(f"  {number}/{len(queries)}")
    print(f"บันทึกที่ {raw_path}")

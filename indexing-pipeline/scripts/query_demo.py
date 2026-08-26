"""
ทดลอง query ภาษาไทยกับ collection ที่ index ไว้แล้ว พร้อมแสดง similarity score

ต้องรัน ``python -m src.indexer`` ให้เสร็จก่อน

วิธีใช้::

    python -m scripts.query_demo                        # ใช้ชุดคำถามตัวอย่างที่เตรียมไว้
    python -m scripts.query_demo "เช็คอินกี่โมง"          # ถามเองเป็นครั้ง ๆ
    python -m scripts.query_demo --top-k 5 "ราคาห้อง Deluxe เท่าไหร่"
    python -m scripts.query_demo --category FAQ "ยกเลิกการจองได้ไหม"
    python -m scripts.query_demo --interactive          # พิมพ์ถามไปเรื่อย ๆ
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any

from src import config
from src.chunking import strip_context_header
from src.indexer import get_collection, search
from src.logging_utils import setup_logging

#: ชุดคำถามตัวอย่างสำหรับตรวจคุณภาพการค้นคืน ครอบคลุมทุกหมวดของเอกสารตัวอย่าง
SAMPLE_QUESTIONS: list[str] = [
    "ห้อง Deluxe มีเตียงเสริมไหม",
    "เช็คอินกี่โมง",
    "ยกเลิกการจองได้ไหม",
    "ห้องพักราคาเท่าไหร่",
    "พาหมามาด้วยได้ไหม",
    "สระว่ายน้ำเปิดกี่โมง",
    "อาหารเช้ารวมในราคาห้องหรือเปล่า",
    "เด็กอายุ 4 ขวบต้องจ่ายเพิ่มไหม",
]

BAR_WIDTH = 28


def _score_bar(similarity: float) -> str:
    """วาดแถบคะแนนแบบข้อความ ช่วยให้กวาดตาดูคุณภาพผลลัพธ์ได้เร็ว"""
    filled = max(0, min(BAR_WIDTH, round(similarity * BAR_WIDTH)))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _verdict(similarity: float) -> str:
    """ตีความคะแนนเป็นภาษาคน (เกณฑ์อ้างอิงจากพฤติกรรมทั่วไปของ e5 กับข้อความไทย)"""
    if similarity >= 0.88:
        return "ตรงมาก"
    if similarity >= 0.84:
        return "ค่อนข้างตรง"
    if similarity >= 0.80:
        return "พอเกี่ยวข้อง"
    return "อาจไม่เกี่ยวข้อง"


def print_hits(question: str, hits: list[dict[str, Any]], preview_chars: int = 260) -> None:
    """พิมพ์ผลลัพธ์ของหนึ่งคำถามในรูปแบบที่อ่านง่าย"""
    print()
    print("=" * 88)
    print(f"คำถาม: {question}")
    print("=" * 88)

    if not hits:
        print("  ไม่พบผลลัพธ์ (collection ว่างหรือยัง? ลองรัน python -m src.indexer)")
        return

    for rank, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        print(
            f"\n  #{rank}  score {hit['similarity']:.4f}  {_score_bar(hit['similarity'])}  "
            f"{_verdict(hit['similarity'])}"
        )
        print(
            f"      ไฟล์: {metadata.get('source')} | หมวด: {metadata.get('category')} "
            f"| หัวข้อ: {metadata.get('heading') or '-'} | id: {hit['chunk_id']}"
        )
        # ตัดบรรทัดบริบทออกก่อนแสดง เพราะมันมีไว้ช่วย embedding ไม่ใช่ให้คนอ่าน
        # (ข้อมูลในบรรทัดนั้นถูกพิมพ์เป็น "หมวด/หัวข้อ" ให้แล้วบรรทัดบน)
        text = " ".join(strip_context_header(hit["text"]).split())
        if len(text) > preview_chars:
            text = text[:preview_chars] + " ..."
        for line in textwrap.wrap(text, width=80):
            print(f"      {line}")


def run_questions(questions: list[str], top_k: int, category: str | None) -> None:
    collection = get_collection()
    if collection.count() == 0:
        print("collection ว่างเปล่า — กรุณารัน `python -m src.indexer` ก่อน")
        return
    for question in questions:
        print_hits(question, search(question, top_k=top_k, category=category, collection=collection))
    print()


def run_interactive(top_k: int, category: str | None) -> None:
    collection = get_collection()
    print("พิมพ์คำถามภาษาไทยแล้วกด Enter (พิมพ์ 'exit' หรือกด Ctrl+C เพื่อออก)")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            return
        print_hits(question, search(question, top_k=top_k, category=category, collection=collection))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ทดลอง query ภาษาไทยกับ vector store")
    parser.add_argument("question", nargs="*", help="คำถาม (ไม่ใส่ = ใช้ชุดคำถามตัวอย่าง)")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K,
                        help=f"จำนวนผลลัพธ์ต่อคำถาม (ค่าเริ่มต้น {config.DEFAULT_TOP_K})")
    parser.add_argument("--category", default=None,
                        help="กรองเฉพาะหมวดหมู่ เช่น FAQ, ห้องพัก, นโยบาย, ราคา")
    parser.add_argument("--interactive", action="store_true", help="โหมดถามตอบต่อเนื่อง")
    args = parser.parse_args(argv)

    setup_logging()

    if args.interactive:
        run_interactive(args.top_k, args.category)
        return 0

    questions = [" ".join(args.question)] if args.question else SAMPLE_QUESTIONS
    run_questions(questions, args.top_k, args.category)
    return 0


if __name__ == "__main__":
    sys.exit(main())

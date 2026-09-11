"""
ประเมินคุณภาพการค้นคืนของคลังความรู้ที่ index ไว้จริง (ไม่ใช่คลังทดสอบ)

ต่างจาก ``thai_smoke_test`` ตรงที่ smoke test ตอบคำถามว่า "pipeline พังไหม"
โดยรันกับคลังเอกสารสมมติที่ตรึงไว้ ส่วนสคริปต์นี้ตอบคำถามว่า
"**ข้อมูลจริงที่ใส่เข้าไปตอบลูกค้าได้ดีแค่ไหน**" โดยรันกับ ``data/`` ของจริง

วิธีใช้::

    python -m scripts.evaluate                # ประเมินทั้งสองภาษา
    python -m scripts.evaluate --language th  # เฉพาะภาษาไทย
    python -m scripts.evaluate --top-k 5
    python -m scripts.evaluate --verbose      # แสดงผลทุกคำถาม ไม่ใช่เฉพาะที่ตก

วัด 4 อย่าง
-----------
1. **ค้นเจอไหม**      — ไฟล์ที่มีคำตอบติดมาใน top-k หรือไม่
2. **คะแนน**          — similarity ของอันดับ 1
3. **ข้อมูลไม่ครบ**   — chunk ที่ดึงมามี placeholder ``<<...>>`` ปนหรือไม่
   ถ้ามี แปลว่า context ที่ส่งให้ LLM มีช่องว่าง อาจทำให้ตอบมั่ว
4. **คำถามที่ตอบไม่ได้** — คำถามที่คลังความรู้ "ไม่มีคำตอบ" จริง ๆ
   ควรได้คะแนนต่ำกว่าคำถามที่ตอบได้อย่างชัดเจน ถ้าไม่ต่างกันมาก
   แปลว่าใช้ threshold ตัดอย่างเดียวไม่พอ ต้องพึ่ง prompt ตอน generation ด้วย
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

from src import config
from src.chunking import strip_context_header
from src.indexer import get_collection, search
from src.logging_utils import setup_logging


@dataclass(frozen=True, slots=True)
class EvalCase:
    """หนึ่งคำถามทดสอบ พร้อมไฟล์ที่ควรถูกดึงมา"""

    question: str
    language: str
    expected_sources: frozenset[str]


def _case(question: str, language: str, *sources: str) -> EvalCase:
    prefix = f"{language}/"
    return EvalCase(question, language, frozenset(prefix + s for s in sources))


#: คำถามที่ลูกค้าถามจริง เขียนด้วยภาษาพูด ไม่ใช่ภาษาเอกสาร
#: ตั้งใจใส่คำสะกดแบบที่ลูกค้าพิมพ์จริง เช่น "มั้ย" "ป่ะ" "กี่บาท"
ANSWERABLE: list[EvalCase] = [
    # --- ราคาและการจอง ---
    _case("ห้องพักคืนละเท่าไหร่", "th", "rates.md", "faq.md"),
    _case("ราคาห้องกี่บาท", "th", "rates.md", "faq.md"),
    _case("เสาร์อาทิตย์แพงกว่ามั้ย", "th", "rates.md", "faq.md"),
    _case("จองยังไง", "th", "faq.md", "booking_policy.md", "overview.md"),
    _case("ต้องมัดจำเท่าไหร่", "th", "rates.md", "faq.md", "booking_policy.md"),
    _case("โอนเงินเข้าเบอร์อะไร", "th", "rates.md", "faq.md", "booking_policy.md"),
    _case("จ่ายเงินสดได้ป่ะ", "th", "rates.md", "faq.md", "booking_policy.md"),
    _case("ยกเลิกแล้วได้เงินคืนมั้ย", "th", "booking_policy.md", "faq.md"),
    # --- ห้องพัก ---
    _case("มีห้องแบบไหนบ้าง", "th", "room_types.md", "faq.md"),
    _case("ห้อง P3 เตียงแบบไหน", "th", "room_types.md", "faq.md"),
    _case("F2 เป็นห้องอะไร", "th", "room_types.md", "faq.md"),
    _case("ห้องกว้างกี่ตารางเมตร", "th", "room_types.md", "faq.md"),
    _case("นอนได้กี่คน", "th", "room_types.md", "faq.md"),
    _case("ขอเตียงเสริมได้มั้ย กี่บาท", "th", "room_types.md", "faq.md"),
    _case("ในห้องมีตู้เย็นมั้ย", "th", "room_types.md", "faq.md", "facilities.md"),
    _case("มีน้ำอุ่นป่ะ", "th", "room_types.md", "faq.md", "facilities.md"),
    _case("มีทีวีมั้ย", "th", "room_types.md", "faq.md", "facilities.md"),
    # --- นโยบาย ---
    _case("เช็คอินกี่โมง", "th", "booking_policy.md", "faq.md"),
    _case("เช็คเอาท์กี่โมง", "th", "booking_policy.md", "faq.md"),
    _case("มาดึกได้มั้ย", "th", "booking_policy.md", "faq.md"),
    _case("ฝากกระเป๋าได้ป่ะ", "th", "booking_policy.md", "faq.md", "facilities.md"),
    _case("พาหมามาได้มั้ย", "th", "booking_policy.md", "faq.md"),
    _case("เอาแมวมาได้ป่ะ", "th", "booking_policy.md", "faq.md"),
    _case("สูบบุหรี่ในห้องได้มั้ย", "th", "booking_policy.md", "faq.md"),
    _case("เด็กคิดเงินมั้ย", "th", "room_types.md", "faq.md"),
    # --- สิ่งอำนวยความสะดวก ---
    _case("มีไวไฟมั้ย", "th", "facilities.md", "faq.md", "room_types.md"),
    _case("ที่จอดรถเสียตังค์ป่ะ", "th", "facilities.md", "faq.md"),
    _case("จอดรถได้กี่คัน", "th", "facilities.md", "faq.md"),
    _case("มีสระว่ายน้ำมั้ย", "th", "facilities.md", "faq.md"),
    _case("มีอาหารเช้าป่ะ", "th", "facilities.md", "faq.md", "rates.md"),
    # --- ที่ตั้ง ---
    _case("รีสอร์ทอยู่ไหน", "th", "location_and_travel.md", "faq.md", "overview.md"),
    _case("ไปโลตัสไกลมั้ย", "th", "location_and_travel.md", "faq.md", "facilities.md"),
    _case("มีไลน์ป่ะ", "th", "overview.md", "faq.md"),
    _case("เบอร์โทรอะไร", "th", "overview.md", "faq.md", "booking_policy.md"),
    # --- ภาษาอังกฤษ ---
    _case("How much is a room?", "en", "rates.md", "faq.md"),
    _case("What time is check-in?", "en", "booking_policy.md", "faq.md"),
    _case("Can I bring my dog?", "en", "booking_policy.md", "faq.md"),
    _case("Is breakfast included?", "en", "facilities.md", "faq.md", "rates.md"),
    _case("How many people per room?", "en", "room_types.md", "faq.md"),
    _case("Is there free wifi?", "en", "facilities.md", "faq.md", "room_types.md"),
    _case("Is parking free?", "en", "facilities.md", "faq.md"),
    _case("How do I pay the deposit?", "en", "rates.md", "faq.md", "booking_policy.md"),
    _case("Do you have a swimming pool?", "en", "facilities.md", "faq.md"),
    _case("Where is the resort?", "en", "location_and_travel.md", "faq.md", "overview.md"),
]

#: คำถามที่คลังความรู้ "ไม่มีคำตอบ" จริง ๆ — ใช้วัดว่า threshold ตัดอยู่มือหรือไม่
#: ระบบที่ดีควรให้คะแนนกลุ่มนี้ต่ำกว่ากลุ่มที่ตอบได้อย่างเห็นได้ชัด
UNANSWERABLE: list[tuple[str, str]] = [
    ("มีส่วนลดหรือโปรโมชั่นมั้ย", "th"),
    ("นวดสปาราคาเท่าไหร่", "th"),
    ("จากสนามบินหาดใหญ่มากี่กิโล", "th"),
    ("มีรถรับส่งมั้ย", "th"),
    ("เช็คเอาท์สายได้มั้ย เสียเงินเพิ่มเท่าไหร่", "th"),
    ("ห้องจัดประชุมได้กี่ที่นั่ง", "th"),
    ("วิธีเปลี่ยนถ่ายน้ำมันเครื่องรถยนต์", "th"),
    ("Do you offer airport pickup?", "en"),
    ("What is the wifi password?", "en"),
    ("How far is Hat Yai airport?", "en"),
]

SEPARATOR = "=" * 92


def _header(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def _has_placeholder(text: str) -> bool:
    return "<<" in text and ">>" in text


def evaluate_answerable(cases: list[EvalCase], top_k: int, verbose: bool) -> dict:
    """ยิงคำถามที่ควรตอบได้ แล้ววัดว่าค้นเจอไหมและ context สะอาดแค่ไหน"""
    collection = get_collection()

    hits_at_1 = 0
    hits_at_k = 0
    scores: list[float] = []
    contaminated = 0
    failures: list[tuple[EvalCase, list[str], float]] = []

    for case in cases:
        results = search(
            case.question, top_k=top_k, language=case.language, collection=collection
        )
        sources = [r["metadata"]["source"] for r in results]
        top_score = results[0]["similarity"] if results else 0.0
        scores.append(top_score)

        found_at_1 = bool(sources) and sources[0] in case.expected_sources
        found_at_k = bool(case.expected_sources & set(sources))
        hits_at_1 += found_at_1
        hits_at_k += found_at_k

        dirty = any(_has_placeholder(r["text"]) for r in results)
        contaminated += dirty

        if not found_at_k:
            failures.append((case, sources, top_score))

        if verbose:
            mark = "ok " if found_at_k else "MISS"
            flag = " [มี placeholder ปน]" if dirty else ""
            print(f"  {mark} {top_score:.3f} [{case.language}] {case.question}{flag}")
            print(f"        -> {sources}")

    total = len(cases)
    return {
        "total": total,
        "hits_at_1": hits_at_1,
        "hits_at_k": hits_at_k,
        "scores": scores,
        "contaminated": contaminated,
        "failures": failures,
    }


def evaluate_unanswerable(top_k: int, verbose: bool) -> dict:
    """ยิงคำถามที่คลังไม่มีคำตอบ เพื่อดูว่าคะแนนต่ำลงพอจะตัดทิ้งได้หรือไม่"""
    collection = get_collection()
    scores: list[float] = []
    above_threshold: list[tuple[str, float, str]] = []

    for question, language in UNANSWERABLE:
        results = search(question, top_k=top_k, language=language, collection=collection)
        if not results:
            scores.append(0.0)
            continue
        top = results[0]
        scores.append(top["similarity"])
        if top["similarity"] >= config.DEFAULT_SIMILARITY_THRESHOLD:
            above_threshold.append(
                (question, top["similarity"], top["metadata"]["source"])
            )
        if verbose:
            print(f"  {top['similarity']:.3f} [{language}] {question}")
            print(f"        -> {top['metadata']['source']}")

    return {"scores": scores, "above_threshold": above_threshold}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ประเมินคุณภาพการค้นคืนของคลังความรู้จริงใน data/"
    )
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K,
                        help=f"จำนวน chunk ที่ดึงต่อคำถาม (ค่าเริ่มต้น {config.DEFAULT_TOP_K})")
    parser.add_argument("--language", choices=config.SUPPORTED_LANGUAGES, default=None,
                        help="ประเมินเฉพาะภาษาเดียว")
    parser.add_argument("--verbose", action="store_true",
                        help="แสดงผลทุกคำถาม ไม่ใช่เฉพาะข้อที่ตก")
    args = parser.parse_args(argv)

    setup_logging()

    collection = get_collection()
    if collection.count() == 0:
        print("collection ว่างเปล่า — รัน `python -m src.indexer` ก่อน")
        return 1

    cases = [c for c in ANSWERABLE if not args.language or c.language == args.language]

    _header("ประเมินคุณภาพการค้นคืนจากข้อมูลจริงใน data/")
    print(f"collection '{collection.name}' | {collection.count()} chunk | top-k = {args.top_k}")
    print(f"คำถามที่ควรตอบได้ {len(cases)} ข้อ | คำถามที่คลังไม่มีคำตอบ {len(UNANSWERABLE)} ข้อ")

    _header(f"[1] คำถามที่ควรตอบได้ ({len(cases)} ข้อ)")
    answerable = evaluate_answerable(cases, args.top_k, args.verbose)

    total = answerable["total"]
    print(f"\nเจอไฟล์ที่ถูกเป็นอันดับ 1      : {answerable['hits_at_1']}/{total} "
          f"({answerable['hits_at_1'] / total:.0%})")
    print(f"เจอไฟล์ที่ถูกใน top-{args.top_k}          : {answerable['hits_at_k']}/{total} "
          f"({answerable['hits_at_k'] / total:.0%})   <-- ตัวเลขที่สำคัญกว่า")
    print(f"คะแนนเฉลี่ยของอันดับ 1        : {statistics.mean(answerable['scores']):.3f}")
    print(f"คะแนนต่ำสุด / สูงสุด          : {min(answerable['scores']):.3f} / "
          f"{max(answerable['scores']):.3f}")
    print(f"คำถามที่ context มี placeholder: {answerable['contaminated']}/{total} "
          f"({answerable['contaminated'] / total:.0%})")

    if answerable["failures"]:
        print(f"\nข้อที่ค้นไม่เจอ ({len(answerable['failures'])} ข้อ):")
        for case, sources, score in answerable["failures"]:
            print(f"  [{case.language}] {case.question}  (score {score:.3f})")
            print(f"      คาดหวัง: {sorted(case.expected_sources)}")
            print(f"      ได้จริง: {sources}")

    _header(f"[2] คำถามที่คลังความรู้ไม่มีคำตอบ ({len(UNANSWERABLE)} ข้อ)")
    print("กลุ่มนี้ควรได้คะแนนต่ำกว่ากลุ่มแรกชัดเจน ถ้าไม่ต่างกันมาก")
    print(f"แปลว่าใช้ threshold {config.DEFAULT_SIMILARITY_THRESHOLD} ตัดอย่างเดียวไม่พอ\n")
    unanswerable = evaluate_unanswerable(args.top_k, args.verbose)

    mean_answerable = statistics.mean(answerable["scores"])
    mean_unanswerable = statistics.mean(unanswerable["scores"])
    gap = mean_answerable - mean_unanswerable

    print(f"คะแนนเฉลี่ย คำถามที่ตอบได้    : {mean_answerable:.3f}")
    print(f"คะแนนเฉลี่ย คำถามที่ตอบไม่ได้ : {mean_unanswerable:.3f}")
    print(f"ช่องว่างระหว่างสองกลุ่ม        : {gap:.3f}")

    leaked = unanswerable["above_threshold"]
    print(f"\nคำถามที่ตอบไม่ได้แต่คะแนนยังเกิน {config.DEFAULT_SIMILARITY_THRESHOLD}: "
          f"{len(leaked)}/{len(UNANSWERABLE)} ข้อ")
    for question, score, source in leaked:
        print(f"  {score:.3f} {question}  -> ดึง {source} มาให้ LLM")

    _header("สรุป")
    recall = answerable["hits_at_k"] / total
    print(f"ความแม่นของการค้นคืน : {recall:.0%} ({answerable['hits_at_k']}/{total})")
    if answerable["contaminated"]:
        print(f"ข้อมูลยังไม่ครบ       : {answerable['contaminated']}/{total} คำถาม "
              f"ได้ context ที่มี placeholder ปนไปด้วย")
    if leaked:
        print(f"ความเสี่ยงตอบมั่ว     : {len(leaked)} คำถามที่ไม่มีคำตอบในคลัง "
              f"ยังได้คะแนนเกิน threshold")
        print("                       ต้องกันด้วย prompt ตอน generation ไม่ใช่ threshold อย่างเดียว")
    return 0


if __name__ == "__main__":
    sys.exit(main())

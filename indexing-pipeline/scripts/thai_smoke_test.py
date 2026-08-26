"""
สคริปต์พิสูจน์ว่า indexing pipeline ทำงานกับภาษาไทยได้จริง (end-to-end)

รันครั้งเดียวได้ผลตรวจครบ 5 ด้าน::

    python -m scripts.thai_smoke_test
    python -m scripts.thai_smoke_test --reindex      # สั่ง index ใหม่ก่อนตรวจ

    [1] Chunking       — แบ่งได้กี่ chunk ขนาดเท่าไหร่ metadata ครบไหม พร้อมตัวอย่างจริง
    [2] ขอบเขตคำไทย     — ตรวจว่าไม่มี chunk ไหนถูกตัดกลางคำ (ดู chunking.find_broken_boundaries)
    [3] ความยาว token   — ตรวจว่าไม่มี chunk ไหนยาวเกิน 512 token จนโดนโมเดลตัดทิ้ง
    [4] e5 prefix      — พิสูจน์ว่า "query:"/"passage:" ที่ใส่ถูกด้าน ให้คะแนนดีกว่าใส่ผิด
    [5] Retrieval      — ยิงคำถามไทยจริงแล้วดูว่าดึง chunk ถูกไฟล์ พร้อม similarity score

ท้ายสคริปต์สรุปเป็น ผ่าน/ไม่ผ่าน และคืน exit code 1 ถ้ามีข้อไหนไม่ผ่าน (ใช้ใน CI ได้)
"""

from __future__ import annotations

import argparse
import statistics
import sys

from src import config
from src.chunking import (
    chunk_document,
    find_broken_boundaries,
    load_documents,
    strip_context_header,
)
from src.embedding import cosine_similarity, get_embedder
from src.indexer import get_collection, index_directory, search
from src.logging_utils import setup_logging

#: คำถามทดสอบ + ไฟล์ที่ "ควร" ถูกดึงมาเป็นอันดับต้น ๆ
#: ใช้เป็นเกณฑ์ตัดสินอัตโนมัติว่าการค้นคืนแม่นหรือไม่
TEST_CASES: list[tuple[str, set[str]]] = [
    ("ห้อง Deluxe มีเตียงเสริมไหม", {"room_types.md", "faq.md"}),
    ("เช็คอินกี่โมง", {"booking_policy.md", "faq.md"}),
    ("ยกเลิกการจองได้ไหม", {"booking_policy.md", "faq.md"}),
    ("ห้องพูลวิลล่าราคาเท่าไหร่", {"rates_and_packages.md"}),
    ("สระว่ายน้ำเปิดถึงกี่โมง", {"facilities.md"}),
    ("พาสุนัขมาพักด้วยได้ไหม", {"booking_policy.md", "faq.md"}),
]

#: เกณฑ์ขั้นต่ำของ similarity ที่ถือว่า "ดึงมาได้จริง" ไม่ใช่สุ่มมั่ว
MIN_ACCEPTABLE_SIMILARITY = 0.80

SEPARATOR = "=" * 88


def _header(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


# --------------------------------------------------------------------------
# [1] + [2] + [3] ตรวจ chunking
# --------------------------------------------------------------------------


def check_chunking(show_samples: int = 2) -> tuple[bool, list]:
    """ตรวจการแบ่ง chunk และแสดงตัวอย่างจริงให้ประเมินด้วยตา"""
    _header("[1] Chunking — แบ่งเอกสารภาษาไทย")

    documents = load_documents()
    all_chunks = []
    print(f"chunk_size = {config.CHUNK_SIZE} ตัวอักษร | "
          f"chunk_overlap = {config.CHUNK_OVERLAP} ตัวอักษร\n")
    print(f"{'ไฟล์':<28}{'หมวดหมู่':<24}{'chunk':>6}")
    print("-" * 60)

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
        print(f"{document.source:<28}{document.category:<24}{len(chunks):>6}")

    sizes = [len(c.text) for c in all_chunks]
    print("-" * 60)
    print(f"{'รวม':<28}{'':<24}{len(all_chunks):>6}")
    print(f"\nขนาด chunk (ตัวอักษร): เล็กสุด {min(sizes)} | "
          f"มัธยฐาน {int(statistics.median(sizes))} | ใหญ่สุด {max(sizes)} | "
          f"เฉลี่ย {int(statistics.mean(sizes))}")

    print(f"\n--- ตัวอย่าง chunk จริง {show_samples} ชิ้นแรกของแต่ละไฟล์ ---")
    for document in documents:
        for chunk in chunk_document(document)[:show_samples]:
            print(f"\n  [{chunk.chunk_id}]")
            print(f"  metadata: source={chunk.source} | category={chunk.category} | "
                  f"heading={chunk.heading or '-'} | chunk_index={chunk.chunk_index} | "
                  f"chars={len(chunk.text)}")
            for line in chunk.text.splitlines():
                print(f"  | {line}")

    metadata_ok = all(
        c.chunk_id and c.source and c.category for c in all_chunks
    )
    print(f"\nmetadata ครบทุก chunk (chunk_id / source / category): "
          f"{'ผ่าน' if metadata_ok else 'ไม่ผ่าน'}")
    return metadata_ok, all_chunks


def check_thai_boundaries(all_chunks: list) -> bool:
    """ตรวจว่าไม่มี chunk ไหนถูกตัดกลางคำไทย"""
    _header("[2] ขอบเขตคำไทย — ตรวจว่าไม่มี chunk ถูกตัดกลางคำ")
    print("วิธีตรวจ: ภาษาไทยมีอักขระที่อยู่ต้นคำไม่ได้ (สระตาม/วรรณยุกต์ เช่น ิ ี ่ ้ ั า)")
    print("          และสระหน้า (เ แ โ ใ ไ) ที่ต้องมีพยัญชนะตามหลังเสมอ")
    print("          ถ้า chunk ขึ้นต้น/ลงท้ายด้วยอักขระเหล่านี้ แปลว่าโดนตัดกลางคำ\n")

    problems = find_broken_boundaries(all_chunks)
    if problems:
        print(f"ไม่ผ่าน — พบ {len(problems)} จุดที่ตัดกลางคำ:")
        for chunk_id, reason in problems[:15]:
            print(f"    {chunk_id}: {reason}")
        return False

    print(f"ผ่าน — ตรวจ {len(all_chunks)} chunk ไม่พบการตัดกลางคำแม้แต่จุดเดียว")
    print("       (เพราะ separators ให้ความสำคัญกับ '\\n\\n' > '\\n' > ' ' ซึ่งในภาษาไทย")
    print("        การเว้นวรรคคือขอบเขตของวลี ไม่ใช่ขอบเขตของคำ จึงตัดตรงนั้นได้ปลอดภัย)")
    return True


def check_token_length(all_chunks: list) -> bool:
    """ตรวจว่า chunk ไม่ยาวเกินเพดาน token ของโมเดล"""
    _header(f"[3] ความยาว token — ต้องไม่เกิน {config.MODEL_MAX_TOKENS} token ของ e5-base")

    embedder = get_embedder()
    token_counts = [embedder.count_tokens(config.PASSAGE_PREFIX + c.text) for c in all_chunks]
    longest = max(token_counts)
    over_limit = [
        (c.chunk_id, n) for c, n in zip(all_chunks, token_counts) if n > embedder.max_seq_length
    ]

    chars_per_token = statistics.mean(
        len(c.text) / n for c, n in zip(all_chunks, token_counts)
    )
    print(f"max_seq_length ของโมเดล : {embedder.max_seq_length} token")
    print(f"token สูงสุดที่พบจริง     : {longest} token")
    print(f"token เฉลี่ย             : {int(statistics.mean(token_counts))} token")
    print(f"อัตราส่วนจริง            : 1 token ≈ {chars_per_token:.2f} ตัวอักษรไทย")
    print(f"  -> chunk_size {config.CHUNK_SIZE} ตัวอักษร ≈ "
          f"{int(config.CHUNK_SIZE / chars_per_token)} token (เพดาน {embedder.max_seq_length})")

    if over_limit:
        print(f"\nไม่ผ่าน — มี {len(over_limit)} chunk ยาวเกินเพดาน จะถูกโมเดลตัดข้อความทิ้งเงียบ ๆ:")
        for chunk_id, count in over_limit[:10]:
            print(f"    {chunk_id}: {count} token")
        return False

    print("\nผ่าน — ไม่มี chunk ไหนถูกตัดทิ้งจากการเกินความยาวโมเดล")
    return True


# --------------------------------------------------------------------------
# [4] ตรวจ prefix ของ e5
# --------------------------------------------------------------------------


def check_e5_prefix(all_chunks: list) -> bool:
    """เปรียบเทียบการใส่ prefix 3 รูปแบบ บนคลังเอกสารจริงทั้งก้อน

    วิธีวัด: จัดอันดับ chunk **ทั้ง 27 ชิ้น** ต่อคำถามทดสอบทุกข้อ แล้ววัด 2 อย่าง

    * ``top-1 ถูกไฟล์``  — อันดับ 1 มาจากไฟล์ที่ควรตอบคำถามนั้นได้หรือไม่
    * ``margin เฉลี่ย``  — ช่องว่างระหว่างคะแนนของ chunk ที่เกี่ยวข้องที่ดีที่สุด
      กับ chunk ที่ไม่เกี่ยวข้องที่ดีที่สุด ยิ่งกว้าง = แยกแยะได้เด็ดขาดยิ่งขึ้น

    ไม่ดูคะแนนดิบ เพราะ e5 ให้คะแนนคู่ข้อความไทยสูง 0.8-0.9 แทบทุกคู่อยู่แล้ว
    สิ่งที่ retrieval ต้องการคือ "อันดับ" ที่ถูก ไม่ใช่ตัวเลขที่สูง

    เกณฑ์ผ่าน: รูปแบบที่โค้ดใช้จริงต้องดีกว่ารูปแบบที่ใส่ prefix สลับด้าน
    (ซึ่งคือบั๊กที่เกิดขึ้นจริงบ่อยที่สุด) และต้องมี top-1 accuracy สูงสุด
    """
    _header("[4] e5 prefix — เทียบ 3 รูปแบบบนคลังเอกสารจริง")

    embedder = get_embedder()
    model = embedder._model  # noqa: SLF001 — เข้าถึงตรงเพื่อทดลอง prefix แบบดิบ ๆ

    def encode_many(texts: list[str], prefix: str) -> list[list[float]]:
        return model.encode(
            [prefix + t for t in texts], normalize_embeddings=True, batch_size=32
        ).tolist()

    chunk_texts = [c.text for c in all_chunks]
    chunk_sources = [c.source for c in all_chunks]
    questions = [question for question, _ in TEST_CASES]

    configurations: dict[str, tuple[str, str]] = {
        "ถูกด้าน (query:/passage:)": (config.QUERY_PREFIX, config.PASSAGE_PREFIX),
        "สลับด้าน (passage:/query:)": (config.PASSAGE_PREFIX, config.QUERY_PREFIX),
        "ไม่ใส่ prefix เลย": ("", ""),
    }

    print(f"คลังเอกสาร {len(all_chunks)} chunk | คำถามทดสอบ {len(questions)} ข้อ")
    print(f"\n{'รูปแบบ prefix':<30}{'top-1 ถูกไฟล์':>16}{'margin เฉลี่ย':>16}")
    print("-" * 62)

    results: dict[str, tuple[int, float]] = {}
    for label, (query_prefix, passage_prefix) in configurations.items():
        chunk_vectors = encode_many(chunk_texts, passage_prefix)
        query_vectors = encode_many(questions, query_prefix)

        correct_top1 = 0
        margins: list[float] = []

        for query_vector, (_, expected_sources) in zip(query_vectors, TEST_CASES):
            scores = [cosine_similarity(query_vector, v) for v in chunk_vectors]
            relevant = [s for s, src in zip(scores, chunk_sources) if src in expected_sources]
            irrelevant = [s for s, src in zip(scores, chunk_sources) if src not in expected_sources]

            best_index = max(range(len(scores)), key=scores.__getitem__)
            if chunk_sources[best_index] in expected_sources:
                correct_top1 += 1
            margins.append(max(relevant) - max(irrelevant))

        mean_margin = statistics.mean(margins)
        results[label] = (correct_top1, mean_margin)
        print(f"{label:<30}{f'{correct_top1}/{len(questions)}':>16}{mean_margin:>16.4f}")

    correct_accuracy, correct_margin = results["ถูกด้าน (query:/passage:)"]
    swapped_accuracy, swapped_margin = results["สลับด้าน (passage:/query:)"]
    best_accuracy = max(accuracy for accuracy, _ in results.values())

    print(f"\nรูปแบบที่ embedding.py ใช้จริงคือ 'ถูกด้าน' ตามที่ model card ของ e5 กำหนด")
    print("หมายเหตุ: บนคลังเล็ก ๆ แบบนี้ 'ไม่ใส่ prefix' อาจได้ตัวเลขใกล้เคียงกัน")
    print("          ข้อแตกต่างจะชัดขึ้นเมื่อคลังใหญ่และคำถามกำกวมมากขึ้น")

    passed = (
        correct_accuracy == best_accuracy
        and correct_accuracy >= swapped_accuracy
        and correct_margin > swapped_margin
    )
    print(
        "\nผ่าน — prefix ที่ใช้อยู่ให้ผลดีที่สุด และดีกว่าการใส่สลับด้านชัดเจน"
        if passed
        else "\nไม่ผ่าน — prefix ที่ใช้อยู่แพ้รูปแบบอื่น ตรวจสอบ embed_documents/embed_query"
    )
    return passed


# --------------------------------------------------------------------------
# [5] ตรวจ retrieval
# --------------------------------------------------------------------------


def _preview_of(hit: dict, width: int = 150) -> str:
    """ย่อเนื้อหา chunk เป็นบรรทัดเดียวสำหรับแสดงผล (ตัดบรรทัดบริบทออกก่อน)"""
    text = " ".join(strip_context_header(hit["text"]).split())
    return text[:width] + ("..." if len(text) > width else "")


def check_retrieval(top_k: int = 3) -> bool:
    """ยิงคำถามไทยจริงเข้า ChromaDB แล้วตรวจว่าดึง chunk ถูกไฟล์

    เกณฑ์ตัดสิน: ไฟล์ที่คาดหวังต้องปรากฏใน top-k (ไม่ใช่บังคับว่าต้องเป็นอันดับ 1)
    เพราะขั้นตอน generation จะส่ง chunk ทั้ง k ชิ้นให้ LLM อ่านอยู่แล้ว
    ขอแค่ chunk ที่มีคำตอบติดมาใน k ชิ้นนั้นก็ตอบคำถามได้ถูกต้อง
    """
    _header("[5] Retrieval — query ภาษาไทยจริงผ่าน ChromaDB (cosine)")

    collection = get_collection()
    print(f"collection '{collection.name}' | {collection.count()} ระเบียน | "
          f"metric = {(collection.metadata or {}).get('hnsw:space')}")
    print(f"เกณฑ์ผ่าน: ไฟล์ที่คาดหวังติดใน top-{top_k} และคะแนนอันดับ 1 >= "
          f"{MIN_ACCEPTABLE_SIMILARITY}")

    all_passed = True
    for question, expected_sources in TEST_CASES:
        hits = search(question, top_k=top_k, collection=collection)
        sources_in_topk = [hit["metadata"]["source"] for hit in hits]

        matched_rank = next(
            (i for i, source in enumerate(sources_in_topk, start=1) if source in expected_sources),
            None,
        )
        score_ok = bool(hits) and hits[0]["similarity"] >= MIN_ACCEPTABLE_SIMILARITY
        passed = matched_rank is not None and score_ok
        all_passed = all_passed and passed

        print(f"\n{'-' * 88}")
        print(f"คำถาม: {question}")
        found = f"เจอที่อันดับ {matched_rank}" if matched_rank else "ไม่เจอใน top-k"
        print(f"คาดหวังไฟล์: {' หรือ '.join(sorted(expected_sources))}  "
              f"-> {found}  [{'ผ่าน' if passed else 'ไม่ผ่าน'}]")
        for rank, hit in enumerate(hits, start=1):
            metadata = hit["metadata"]
            marker = "->" if rank == matched_rank else "  "
            print(f"  {marker} #{rank} score={hit['similarity']:.4f} "
                  f"| {metadata.get('source')} | {metadata.get('heading') or '-'}")
            print(f"        {_preview_of(hit)}")

    print(f"\n{'-' * 88}")
    print(f"สรุป retrieval: {'ผ่านทุกข้อ' if all_passed else 'มีข้อที่ไม่ผ่าน'}")
    return all_passed


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ทดสอบ indexing pipeline ภาษาไทยแบบ end-to-end")
    parser.add_argument("--reindex", action="store_true", help="สั่ง index ใหม่ทั้งหมดก่อนตรวจ")
    parser.add_argument("--top-k", type=int, default=3, help="จำนวนผลลัพธ์ต่อคำถาม")
    parser.add_argument("--samples", type=int, default=2,
                        help="จำนวน chunk ตัวอย่างที่แสดงต่อไฟล์")
    args = parser.parse_args(argv)

    setup_logging()

    if args.reindex or get_collection().count() == 0:
        _header("[0] Indexing — สร้าง vector store ใหม่")
        index_directory(reset=args.reindex, show_progress=False)

    results: dict[str, bool] = {}
    results["metadata ครบถ้วน"], all_chunks = check_chunking(args.samples)
    results["ไม่ตัดกลางคำไทย"] = check_thai_boundaries(all_chunks)
    results["ความยาวไม่เกินโมเดล"] = check_token_length(all_chunks)
    results["e5 prefix ถูกต้อง"] = check_e5_prefix(all_chunks)
    results["retrieval แม่นยำ"] = check_retrieval(args.top_k)

    _header("สรุปผลการทดสอบ")
    for name, passed in results.items():
        print(f"  [{'ผ่าน  ' if passed else 'ไม่ผ่าน'}] {name}")

    everything_passed = all(results.values())
    print(f"\n{'ทุกการทดสอบผ่าน — pipeline พร้อมใช้งานกับภาษาไทย' if everything_passed else 'มีการทดสอบที่ไม่ผ่าน ดูรายละเอียดด้านบน'}")
    return 0 if everything_passed else 1


if __name__ == "__main__":
    sys.exit(main())

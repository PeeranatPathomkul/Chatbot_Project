"""
สคริปต์หลักของ indexing pipeline: อ่านไฟล์ -> แบ่ง chunk -> embed -> เก็บลง ChromaDB

วิธีใช้ (รันจาก root ของโปรเจกต์)::

    python -m src.indexer                 # index ทุกไฟล์ในโฟลเดอร์ data/
    python -m src.indexer --reset         # ล้าง collection เดิมทิ้งก่อนแล้ว index ใหม่หมด
    python -m src.indexer --preview       # ดูว่าจะได้ chunk หน้าตาแบบไหน โดยยังไม่เขียนลงดิสก์
    python -m src.indexer --stats         # ดูสถิติของ collection ที่ index ไว้แล้ว
    python -m src.indexer --data-dir ./other_docs

การจัดการ re-index (รันซ้ำแล้วต้องไม่มีข้อมูลซ้ำซ้อน)
--------------------------------------------------
ใช้สองกลไกร่วมกัน:

1. **Deterministic ID** — ``chunk_id`` คำนวณจาก ``<ชื่อไฟล์>::<ลำดับ chunk>`` เช่น
   ``faq::0003`` เนื้อหาเดิมรันซ้ำกี่รอบก็ได้ ID เดิม เมื่อใช้ ``collection.upsert()``
   ระเบียนเดิมจะถูกเขียนทับแทนที่จะเพิ่มใหม่
2. **ลบตามไฟล์ก่อนเขียน (delete-by-source)** — ID แบบข้อ 1 ยังมีช่องโหว่หนึ่งอย่าง:
   ถ้าแก้ไฟล์ให้ "สั้นลง" จนจำนวน chunk ลดจาก 10 เหลือ 6 ระเบียน ``faq::0006``
   ถึง ``faq::0009`` ของรอบก่อนจะค้างอยู่เป็นข้อมูลเก่าที่ไม่มีในเอกสารแล้ว
   ทุกครั้งก่อน index ไฟล์หนึ่ง เราจึงลบ chunk ทั้งหมดที่ ``source`` ตรงกับไฟล์นั้นก่อน
   ผลคือ collection สะท้อนเนื้อหาปัจจุบันของโฟลเดอร์ ``data/`` เสมอ

ธง ``--reset`` มีไว้สำหรับกรณีที่ต้องการเริ่มใหม่จริง ๆ เช่น เปลี่ยนโมเดล embedding
หรือเปลี่ยน chunk_size (เวกเตอร์เก่ากับใหม่จะเทียบกันไม่ได้ ต้องสร้างใหม่ทั้งหมด)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from src import config
from src.chunking import Chunk, chunk_document, find_broken_boundaries, load_documents
from src.embedding import ThaiEmbedder, get_embedder
from src.logging_utils import format_duration, get_logger, setup_logging

logger = get_logger("indexer")


# --------------------------------------------------------------------------
# สถิติของการ index หนึ่งรอบ
# --------------------------------------------------------------------------


@dataclass(slots=True)
class IndexStats:
    """ตัวเลขสรุปผลการ index หนึ่งรอบ"""

    files_read: int = 0
    chunks_created: int = 0
    chunks_deleted: int = 0
    elapsed_seconds: float = 0.0
    chunks_per_file: dict[str, int] = field(default_factory=dict)
    broken_boundaries: list[tuple[str, str]] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """สรุปผลเป็นบรรทัดข้อความสำหรับพิมพ์ท้ายการทำงาน"""
        lines = [
            f"ไฟล์ที่อ่าน           : {self.files_read} ไฟล์",
            f"chunk ที่สร้างได้      : {self.chunks_created} chunk",
            f"chunk เก่าที่ลบทิ้ง     : {self.chunks_deleted} chunk",
            f"เวลาที่ใช้ทั้งหมด      : {format_duration(self.elapsed_seconds)}",
        ]
        if self.chunks_per_file:
            lines.append("แยกตามไฟล์:")
            for source, count in sorted(self.chunks_per_file.items()):
                lines.append(f"    {source:<28} {count:>3} chunk")
        if self.broken_boundaries:
            lines.append(f"เตือน: พบ chunk ที่อาจตัดกลางคำ {len(self.broken_boundaries)} จุด")
        return lines


# --------------------------------------------------------------------------
# ChromaDB
# --------------------------------------------------------------------------


def get_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    """สร้าง PersistentClient ที่เก็บข้อมูลลงดิสก์ (ไม่ใช่ in-memory)

    ปิด telemetry ไว้ด้วย เพราะ chromadb 0.5.5 กับ posthog เวอร์ชันใหม่เข้ากันไม่ได้
    แล้วพ่น ``Failed to send telemetry event`` รกล็อกทุกครั้งที่เรียกใช้งาน
    """
    directory = Path(persist_dir) if persist_dir else config.CHROMA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(directory),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(
    client: chromadb.ClientAPI | None = None,
    name: str | None = None,
) -> Collection:
    """เปิด (หรือสร้าง) collection ที่กำหนด distance metric เป็น **cosine**

    สำคัญ: ค่าเริ่มต้นของ ChromaDB คือ L2 (Euclidean) ถ้าไม่ระบุ ``hnsw:space``
    ผลการจัดอันดับตอน retrieval จะไม่ตรงกับที่ออกแบบไว้ metadata นี้ถูก "ตรึง"
    ตั้งแต่ตอนสร้าง collection เปลี่ยนทีหลังไม่ได้ ต้องใช้ ``--reset`` สร้างใหม่
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=name or config.COLLECTION_NAME,
        metadata=config.COLLECTION_METADATA,
    )


def reset_collection(
    client: chromadb.ClientAPI | None = None,
    name: str | None = None,
) -> Collection:
    """ลบ collection เดิมทิ้งทั้งก้อนแล้วสร้างใหม่ (ใช้กับธง --reset)"""
    client = client or get_client()
    collection_name = name or config.COLLECTION_NAME
    try:
        client.delete_collection(name=collection_name)
        logger.info("ลบ collection เดิม '%s' เรียบร้อย", collection_name)
    except Exception:  # noqa: BLE001 — Chroma โยน error คนละชนิดกันแต่ละเวอร์ชันเมื่อไม่พบ collection
        logger.info("ยังไม่มี collection '%s' อยู่เดิม ข้ามขั้นตอนลบ", collection_name)
    return get_collection(client, collection_name)


def delete_chunks_of_source(collection: Collection, source: str) -> int:
    """ลบ chunk ทั้งหมดที่มาจากไฟล์ต้นทางนี้ คืนจำนวนที่ลบไป

    เป็นหัวใจของการ re-index ที่ไม่เกิดข้อมูลค้าง (ดูคำอธิบายหัวไฟล์)
    """
    existing = collection.get(where={"source": source}, include=[])
    ids: list[str] = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# --------------------------------------------------------------------------
# ขั้นตอนหลัก
# --------------------------------------------------------------------------


def index_chunks(
    collection: Collection,
    chunks: list[Chunk],
    embedder: ThaiEmbedder,
    show_progress: bool = False,
) -> None:
    """embed แล้ว upsert chunk ลง collection ทีละ batch"""
    if not chunks:
        return

    embeddings = embedder.embed_documents([c.text for c in chunks], show_progress=show_progress)

    batch = config.UPSERT_BATCH_SIZE
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        collection.upsert(
            ids=[c.chunk_id for c in window],
            embeddings=embeddings[start : start + batch],
            documents=[c.text for c in window],
            metadatas=[c.to_metadata() for c in window],
        )


def index_directory(
    data_dir: Path | None = None,
    reset: bool = False,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
    show_progress: bool = True,
) -> IndexStats:
    """index เอกสารทั้งหมดในโฟลเดอร์ ``data/`` ลง ChromaDB

    Args:
        data_dir: โฟลเดอร์เอกสารต้นฉบับ (ค่าเริ่มต้นจาก config)
        reset: ลบ collection เดิมทิ้งก่อน index ใหม่ทั้งหมด
        persist_dir: ที่เก็บ vector store (ค่าเริ่มต้นจาก config)
        collection_name: ชื่อ collection (ค่าเริ่มต้นจาก config)
        show_progress: แสดง progress bar ตอน embed

    Returns:
        IndexStats สรุปจำนวนไฟล์ จำนวน chunk และเวลาที่ใช้
    """
    started = time.perf_counter()
    stats = IndexStats()

    directory = Path(data_dir) if data_dir else config.DATA_DIR
    logger.info("โฟลเดอร์เอกสาร : %s", directory)
    logger.info("โฟลเดอร์ Chroma: %s", persist_dir or config.CHROMA_DIR)
    logger.info("chunk_size=%d ตัวอักษร | chunk_overlap=%d ตัวอักษร",
                config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    documents = load_documents(directory)
    stats.files_read = len(documents)
    if not documents:
        logger.warning("ไม่พบไฟล์ %s ในโฟลเดอร์ %s",
                       "/".join(config.SUPPORTED_EXTENSIONS), directory)
        return stats
    logger.info("พบเอกสาร %d ไฟล์", len(documents))

    client = get_client(persist_dir)
    collection = (
        reset_collection(client, collection_name)
        if reset
        else get_collection(client, collection_name)
    )

    embedder = get_embedder()

    for document in documents:
        chunks = chunk_document(document)
        stats.chunks_per_file[document.source] = len(chunks)
        stats.chunks_created += len(chunks)
        stats.broken_boundaries.extend(find_broken_boundaries(chunks))

        if not reset:
            # ลบของเก่าของไฟล์นี้ก่อน กัน chunk ที่หายไปจากเอกสารค้างอยู่ใน DB
            removed = delete_chunks_of_source(collection, document.source)
            stats.chunks_deleted += removed
            if removed:
                logger.debug("ลบ chunk เก่าของ %s ออก %d ชิ้น", document.source, removed)

        index_chunks(collection, chunks, embedder, show_progress=show_progress)
        logger.info(
            "  %-28s หมวด=%-22s -> %2d chunk",
            document.source, document.category, len(chunks),
        )

    stats.elapsed_seconds = time.perf_counter() - started

    logger.info("รวมทั้งหมดใน collection ตอนนี้: %d ระเบียน", collection.count())
    for line in stats.summary_lines():
        logger.info(line)

    if stats.broken_boundaries:
        logger.warning("รายละเอียด chunk ที่อาจตัดกลางคำ:")
        for chunk_id, reason in stats.broken_boundaries[:10]:
            logger.warning("    %s : %s", chunk_id, reason)

    return stats


# --------------------------------------------------------------------------
# ฟังก์ชันค้นหา (ใช้สำหรับตรวจสอบผลการ index เท่านั้น ยังไม่ใช่ Retrieval API)
# --------------------------------------------------------------------------


def search(
    question: str,
    top_k: int | None = None,
    category: str | None = None,
    collection: Collection | None = None,
) -> list[dict[str, Any]]:
    """ค้นหา chunk ที่ใกล้เคียงคำถามที่สุด

    Args:
        question: คำถามภาษาไทย (ใส่ข้อความดิบ ไม่ต้องเติม prefix เอง)
        top_k: จำนวนผลลัพธ์
        category: กรองเฉพาะหมวดหมู่ที่ต้องการ เช่น "FAQ"
        collection: ระบุ collection เองได้ (ใช้ในเทสต์) ไม่ระบุจะเปิดตาม config

    Returns:
        list ของ dict ที่มี key: chunk_id, text, metadata, distance, similarity
        เรียงจากคะแนนสูงสุดลงมา โดย ``similarity = 1 - distance`` เพราะ Chroma
        คืนค่าเป็น cosine *distance*
    """
    collection = collection if collection is not None else get_collection()
    query_vector = get_embedder().embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k or config.DEFAULT_TOP_K,
        where={"category": category} if category else None,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    for doc_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "chunk_id": doc_id,
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            }
        )
    return hits


def collection_stats(collection: Collection | None = None) -> dict[str, Any]:
    """สรุปว่าใน collection ตอนนี้มีอะไรอยู่บ้าง (จำนวนระเบียน แยกตามไฟล์/หมวด)"""
    collection = collection if collection is not None else get_collection()
    payload = collection.get(include=["metadatas"])

    by_source: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for metadata in payload.get("metadatas") or []:
        by_source[metadata.get("source", "?")] = by_source.get(metadata.get("source", "?"), 0) + 1
        category = metadata.get("category", "?")
        by_category[category] = by_category.get(category, 0) + 1

    return {
        "name": collection.name,
        "count": collection.count(),
        "metric": (collection.metadata or {}).get("hnsw:space", "ไม่ระบุ"),
        "by_source": by_source,
        "by_category": by_category,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _preview(data_dir: Path | None, limit: int) -> None:
    """แสดงตัวอย่าง chunk โดยไม่แตะ ChromaDB และไม่โหลดโมเดล (เร็วมาก)"""
    documents = load_documents(data_dir or config.DATA_DIR)
    all_chunks: list[Chunk] = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
        print(f"\n{'=' * 78}")
        print(f"ไฟล์: {document.source} | หมวด: {document.category} | {len(chunks)} chunk")
        print("=" * 78)
        for chunk in chunks[:limit]:
            print(f"\n--- [{chunk.chunk_id}] หัวข้อ: {chunk.heading or '(ไม่มี)'} "
                  f"| {len(chunk.text)} ตัวอักษร ---")
            print(chunk.text)
        if len(chunks) > limit:
            print(f"\n  ... อีก {len(chunks) - limit} chunk (ใช้ --limit เพื่อดูเพิ่ม)")

    print(f"\n{'=' * 78}")
    print(f"รวม {len(documents)} ไฟล์ -> {len(all_chunks)} chunk")

    problems = find_broken_boundaries(all_chunks)
    if problems:
        print(f"พบ chunk ที่อาจตัดกลางคำ {len(problems)} จุด:")
        for chunk_id, reason in problems:
            print(f"    {chunk_id}: {reason}")
    else:
        print("ตรวจขอบเขตคำไทย: ไม่พบ chunk ที่ถูกตัดกลางคำ")


def _print_stats() -> None:
    """พิมพ์สถิติของ collection ที่ index ไว้แล้ว"""
    stats = collection_stats()
    print(f"collection      : {stats['name']}")
    print(f"distance metric : {stats['metric']}")
    print(f"จำนวนระเบียน     : {stats['count']}")
    print("แยกตามไฟล์:")
    for source, count in sorted(stats["by_source"].items()):
        print(f"    {source:<28} {count:>3}")
    print("แยกตามหมวดหมู่:")
    for category, count in sorted(stats["by_category"].items()):
        print(f"    {category:<28} {count:>3}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index เอกสารรีสอร์ทภาษาไทยลง ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="โฟลเดอร์เอกสารต้นฉบับ (ค่าเริ่มต้น: ./data)")
    parser.add_argument("--reset", action="store_true",
                        help="ลบ collection เดิมทิ้งทั้งหมดก่อน index ใหม่")
    parser.add_argument("--preview", action="store_true",
                        help="แสดงตัวอย่าง chunk อย่างเดียว ไม่เขียนลง ChromaDB")
    parser.add_argument("--stats", action="store_true",
                        help="แสดงสถิติของ collection ที่ index ไว้แล้ว")
    parser.add_argument("--limit", type=int, default=3,
                        help="จำนวน chunk ต่อไฟล์ที่จะแสดงในโหมด --preview (ค่าเริ่มต้น 3)")
    parser.add_argument("--no-progress", action="store_true",
                        help="ปิด progress bar ตอน embed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.preview:
        _preview(args.data_dir, args.limit)
        return 0

    if args.stats:
        _print_stats()
        return 0

    try:
        index_directory(
            data_dir=args.data_dir,
            reset=args.reset,
            show_progress=not args.no_progress,
        )
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    print("\nเสร็จสิ้น — ลองทดสอบค้นหาด้วย: python -m scripts.query_demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())

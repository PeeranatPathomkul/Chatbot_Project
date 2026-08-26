"""
สคริปต์สำหรับ ingest ข้อมูลความรู้จากไฟล์ CSV เข้า ChromaDB

ขั้นตอน: อ่าน CSV -> chunk ข้อความ (ถ้ายาวเกินไป) -> แปลงเป็น embedding -> เก็บลง vector store

วิธีใช้:
    python scripts/ingest.py
    python scripts/ingest.py --csv data/sample_knowledge.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# บังคับ stdout เป็น UTF-8 กัน UnicodeEncodeError ตอน print ข้อความไทยบน Windows console (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# เพิ่ม root ของโปรเจกต์เข้า sys.path เพื่อให้ import app.* ได้ตอนรันสคริปต์ตรง ๆ
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_documents(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[dict]]:
    """แปลงแต่ละแถวของ CSV เป็น (id, ข้อความ, metadata) พร้อม chunk ถ้าข้อความยาวเกินไป"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for i, row in enumerate(rows):
        full_text = f"คำถาม: {row['question']}\nคำตอบ: {row['answer']}"
        chunks = splitter.split_text(full_text)
        for j, chunk in enumerate(chunks):
            ids.append(f"row{i}-chunk{j}")
            texts.append(chunk)
            metadatas.append(
                {
                    "resort_id": row["resort_id"],
                    "category": row.get("category", ""),
                    "source": row.get("source", ""),
                }
            )
    return ids, texts, metadatas


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ข้อมูลความรู้จาก CSV เข้า ChromaDB")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "sample_knowledge.csv"),
        help="path ของไฟล์ CSV ที่ต้องการ ingest",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ไม่พบไฟล์: {csv_path}")
        sys.exit(1)

    print(f"กำลังอ่านข้อมูลจาก {csv_path} ...")
    rows = load_rows(csv_path)
    print(f"พบข้อมูลทั้งหมด {len(rows)} แถว")

    ids, texts, metadatas = build_documents(rows)
    print(f"แบ่งเป็น {len(texts)} chunks")

    print("กำลังสร้าง embedding (อาจใช้เวลาสักครู่ในการโหลดโมเดลครั้งแรก) ...")
    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_documents(texts)

    print("กำลังบันทึกลง ChromaDB ...")
    vector_store = get_vector_store()
    vector_store.add_documents(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"เสร็จสิ้น! บันทึกข้อมูลทั้งหมด {len(texts)} รายการเรียบร้อยแล้ว")


if __name__ == "__main__":
    main()

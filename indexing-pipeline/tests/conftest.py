"""fixture ที่ใช้ร่วมกันในชุดทดสอบ"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ให้ import src.* ได้เวลารัน pytest จาก root ของโปรเจกต์
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chunking import SourceDocument  # noqa: E402


#: คลังเอกสารคงที่สำหรับเทสต์ — เนื้อหาสมมติของ "บ้านสวนรีสอร์ท เชียงใหม่"
#:
#: เทสต์ต้องไม่อ่านจาก ``data/`` จริง เพราะไฟล์ในนั้นเป็นโครงที่รอเติมข้อมูลของลูกค้า
#: และจะถูกแก้อยู่เรื่อย ๆ ถ้าเทสต์ผูกกับเนื้อหาในนั้น การเติมข้อมูลจริงจะทำให้
#: เทสต์พังทันทีทั้งที่ pipeline ไม่ได้เสียอะไรเลย
FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """คลังเอกสารตัวอย่างภาษาไทยที่ใช้ทดสอบ (คงที่ ไม่เกี่ยวกับ data/ ของจริง)"""
    return FIXTURES_DIR


@pytest.fixture
def thai_document(tmp_path: Path) -> SourceDocument:
    """เอกสารไทยสั้น ๆ ที่ยาวพอจะถูกแบ่งหลาย chunk"""
    content = (
        "---\n"
        "category: ห้องพัก\n"
        "---\n\n"
        "# ข้อมูลห้องพัก\n\n"
        "## ห้อง Deluxe\n"
        "ห้องดีลักซ์ขนาด 38 ตารางเมตร มีระเบียงส่วนตัวและอ่างอาบน้ำ "
        "รองรับผู้เข้าพัก 2 ท่าน ขอเตียงเสริมได้ 1 เตียง คืนละ 800 บาท "
        "กรุณาแจ้งล่วงหน้าอย่างน้อย 3 วันก่อนวันเข้าพัก เนื่องจากเตียงเสริมมีจำนวนจำกัด "
        "และต้องจัดเตรียมผ้าปูที่นอนเพิ่มเติมให้ทันเวลาเช็คอินของท่าน\n\n"
        "## ห้อง Superior\n"
        "ห้องซูพีเรียขนาด 28 ตารางเมตร วิวสวน ไม่สามารถขอเตียงเสริมได้ "
        "เนื่องจากพื้นที่ห้องไม่เพียงพอสำหรับเตียงเพิ่มเติม\n"
    )
    path = tmp_path / "rooms.md"
    path.write_text(content, encoding="utf-8")

    from src.chunking import load_document

    return load_document(path)

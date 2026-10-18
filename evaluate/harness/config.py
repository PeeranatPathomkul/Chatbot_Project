"""
ค่าตั้งของ harness — path, ค่าจาก evaluate/.env และการกันค่าลับไม่ให้หลุดลงไฟล์

**ค่าลับอ่านจาก env เท่านั้น** ลำดับ: environment variable -> evaluate/.env -> resort-chatbot-api/.env
(ตัวหลังใช้เฉพาะ key ที่แอปมีอยู่แล้ว เช่น TYPHOON_API_KEY, BOOKING_API_INTERNAL_KEY จะได้ไม่ต้องกรอกซ้ำ)
ทุกอย่างที่เขียนลงไฟล์ต้องผ่าน ``redact`` ก่อน
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = EVAL_DIR.parent
API_DIR = REPO_DIR / "resort-chatbot-api"
INDEXING_DIR = REPO_DIR / "indexing-pipeline"
KNOWLEDGE_DIR = INDEXING_DIR / "data"
OUTPUT_DIR = EVAL_DIR / "output"
STATE_DIR = EVAL_DIR / ".state"

#: ชื่อ env ที่ถือเป็นค่าลับ — ค่าเหล่านี้ถูกแทนด้วย *** ทุกครั้งก่อนเขียนลง raw / log / report
SECRET_KEYS = (
    "EVAL_CUSTOMER_PASSWORD", "EVAL_OTHER_PASSWORD", "EVAL_CUSTOMER_EMAIL", "EVAL_OTHER_EMAIL",
    "BOOKING_API_INTERNAL_KEY", "TYPHOON_API_KEY", "GEMINI_API_KEY",
)


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@lru_cache
def _env() -> dict[str, str]:
    merged = {k: v for k, v in _read_env_file(API_DIR / ".env").items() if v}
    merged.update({k: v for k, v in _read_env_file(EVAL_DIR / ".env").items() if v})
    merged.update({k: v for k, v in os.environ.items() if v})
    return merged


def env(key: str, default: str = "") -> str:
    return _env().get(key, default)


#: token ที่ได้ระหว่างรัน (JWT) ลงทะเบียนไว้ให้ redact กันด้วย
_RUNTIME_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    if value and len(value) >= 8:
        _RUNTIME_SECRETS.add(value)


def secret_values() -> list[str]:
    values = {env(k) for k in SECRET_KEYS if len(env(k)) >= 6} | _RUNTIME_SECRETS
    return sorted(values, key=len, reverse=True)


_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")


def redact(value: Any) -> Any:
    """แทนค่าลับทุกตัว (และ JWT จริงทุกรูปแบบ) ด้วย *** ทั้งใน str, dict, list"""
    if isinstance(value, str):
        for secret in secret_values():
            value = value.replace(secret, "***")
        return value
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    booking_base_url: str
    booking_internal_key: str
    delay_seconds: float
    max_retries: int
    timeout_seconds: float
    judge_provider: str
    judge_model: str
    price_input_per_1m: float | None
    price_output_per_1m: float | None
    currency: str

    @property
    def query_url(self) -> str:
        return f"{self.api_base_url}/api/v1/chatbot/query"


def _float_or_none(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


@lru_cache
def settings() -> Settings:
    return Settings(
        api_base_url=env("EVAL_API_BASE_URL", "http://localhost:8000").rstrip("/"),
        booking_base_url=env("BOOKING_API_BASE_URL", "http://localhost:3000/api").rstrip("/"),
        booking_internal_key=env("BOOKING_API_INTERNAL_KEY"),
        delay_seconds=float(env("EVAL_DELAY_SECONDS", "2")),
        max_retries=int(env("EVAL_MAX_RETRIES", "5")),
        timeout_seconds=float(env("EVAL_TIMEOUT_SECONDS", "90")),
        judge_provider=env("EVAL_JUDGE_PROVIDER", "typhoon"),
        judge_model=env("EVAL_JUDGE_MODEL", ""),
        price_input_per_1m=_float_or_none(env("EVAL_PRICE_INPUT_PER_1M")),
        price_output_per_1m=_float_or_none(env("EVAL_PRICE_OUTPUT_PER_1M")),
        currency=env("EVAL_PRICE_CURRENCY", "USD"),
    )


def has_customer() -> bool:
    return bool(env("EVAL_CUSTOMER_EMAIL") and env("EVAL_CUSTOMER_PASSWORD"))


def has_other() -> bool:
    return bool(env("EVAL_OTHER_EMAIL") and env("EVAL_OTHER_PASSWORD"))


# ---------------------------------------------------------------------------
# โฟลเดอร์ผลลัพธ์ของแต่ละรอบ: output/<YYYYMMDD-HHMM>/ ไม่เขียนทับรอบเก่า
# ---------------------------------------------------------------------------

def new_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = OUTPUT_DIR / stamp
    suffix = 1
    while path.exists():
        suffix += 1
        path = OUTPUT_DIR / f"{stamp}-{suffix}"
    return path


def latest_run_dir() -> Path | None:
    runs = sorted(p for p in OUTPUT_DIR.glob("*") if p.is_dir()) if OUTPUT_DIR.exists() else []
    return runs[-1] if runs else None


def resolve_run_dir(value: str | None, create: bool) -> Path:
    """--run-dir: ไม่ระบุ = รอบใหม่ (create) หรือรอบล่าสุด (อ่านอย่างเดียว), 'latest' = รอบล่าสุด"""
    if value in (None, "") and create:
        path = new_run_dir()
    elif value in (None, "", "latest"):
        path = latest_run_dir()
        if path is None:
            raise SystemExit("ยังไม่มีผลรอบไหนใน evaluate/output/")
    else:
        # รับได้ทั้ง "20261015-1700", "output/20261015-1700" หรือ path เต็ม
        path = Path(value)
        if not path.is_absolute():
            path = EVAL_DIR / path if path.parts[0] == "output" else OUTPUT_DIR / path
    for sub in ("raw", "charts", "tables"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# JSONL: บันทึกทีละบรรทัดทันทีที่ได้ผล เพื่อให้รันต่อจากจุดที่ค้างได้
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # บรรทัดสุดท้ายอาจขาดกลางคันถ้าโปรแกรมถูกปิดระหว่างเขียน — ข้ามไป แล้วรันข้อนั้นใหม่
            continue
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def use_utf8_console() -> None:
    """คอนโซล Windows ค่าเริ่มต้นเป็น cp1252 พิมพ์ภาษาไทยแล้ว UnicodeEncodeError"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def add_api_to_path() -> None:
    """ให้ import app.* ของ resort-chatbot-api ได้ (โค้ดจาก working tree ตัวเดียวกับที่ทดสอบ)"""
    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))

"""
เทียบ key fact กับคำตอบแบบ deterministic

**ทำไมไม่เทียบ substring ตรง ๆ** — ข้อเท็จจริงเดียวกันเขียนได้หลายแบบ เวลาเช็คอินในคลังไทยเขียน
"14.00 น." คลังอังกฤษเขียน "2:00 PM" ส่วนบอทอาจตอบ "บ่ายสองโมง" ราคาอาจมีหรือไม่มีคอมมา
และอาจเป็นเลขไทย ถ้าเทียบตรง ๆ คำตอบที่ถูกจะถูกนับว่าผิด ตัวเลข key-fact accuracy จะต่ำเกินจริง

จึงแยกชนิดของ fact จากหน้าตา แล้วแปลงทั้งสองฝั่งเป็นรูปมาตรฐานก่อนเทียบ:
- เวลา  ``14:00``          -> ชุดเวลา HH:MM ที่หาเจอในข้อความ (รวม 2 PM, บ่ายสอง, สี่ทุ่ม, เที่ยง)
- เบอร์โทร ``081-598-1199`` -> เทียบเฉพาะตัวเลขติดกัน
- ตัวเลข ``1300``           -> ชุดตัวเลขทั้งตัวในข้อความ (1300 ไม่ถูกนับว่าอยู่ใน 13000)
- อื่น ๆ                    -> substring หลังทำให้เป็นรูปมาตรฐาน (และแบบตัดช่องว่างทิ้ง เพราะไทยเว้นวรรคไม่แน่นอน)
"""

from __future__ import annotations

import re
from typing import Any

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

_TIME_FACT_RE = re.compile(r"^([01]?\d|2[0-3])[:.]([0-5]\d)$")
_PHONE_FACT_RE = re.compile(r"^0\d{1,2}-?\d{3}-?\d{4}$")
_NUMBER_FACT_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$")

_THAI_SMALL_NUMBERS = {"หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5}
_SMALL_NUMBER = r"(\d|" + "|".join(_THAI_SMALL_NUMBERS) + r")"

#: เวลาแบบ 24 ชั่วโมง 14.00 / 14:00 — ไม่นับถ้าตามด้วย am/pm (กรณีนั้นจับด้วย _TIME_12H_RE)
_TIME_24H_RE = re.compile(r"(?<![\d.:/,])([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)(?!\s*[ap]\.?\s?m\b)")
_TIME_12H_RE = re.compile(r"(?<![\d.:/,])(1[0-2]|0?[1-9])(?:[:.]([0-5]\d))?\s*([ap])\.?\s?m\b\.?")
_MIDNIGHT_RE = re.compile(r"เที่ยงคืน|midnight")
_NOON_RE = re.compile(r"\bnoon\b|เที่ยงวัน|เที่ยง(?!คืน)")
_AFTERNOON_RE = re.compile(r"บ่าย\s*" + _SMALL_NUMBER + r"?\s*โมง")
_EVENING_RE = re.compile(_SMALL_NUMBER + r"\s*ทุ่ม")


def normalize(text: str) -> str:
    """เลขไทยเป็นเลขอารบิก ตัวพิมพ์เล็ก และยุบช่องว่างให้เหลือช่องเดียว"""
    return " ".join(text.translate(_THAI_DIGITS).casefold().split())


def _no_space(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _small_number(token: str | None) -> int | None:
    if not token:
        return None
    return _THAI_SMALL_NUMBERS.get(token) or int(token)


def extract_times(text: str) -> set[str]:
    """เวลาทุกแบบที่พบในข้อความ ในรูป HH:MM"""
    t = normalize(text)
    times: set[str] = set()
    for hour, minute in _TIME_24H_RE.findall(t):
        times.add(f"{int(hour):02d}:{minute}")
    for hour, minute, half in _TIME_12H_RE.findall(t):
        h = int(hour) % 12 + (12 if half == "p" else 0)
        times.add(f"{h:02d}:{minute or '00'}")
    if _MIDNIGHT_RE.search(t):
        times.add("00:00")
    if _NOON_RE.search(t):
        times.add("12:00")
    for token in _AFTERNOON_RE.findall(t):
        # "บ่ายโมง" ไม่มีตัวเลข = 13.00
        times.add(f"{12 + (_small_number(token) or 1):02d}:00")
    for token in _EVENING_RE.findall(t):
        times.add(f"{18 + _small_number(token):02d}:00")
    return times


def extract_numbers(text: str) -> set[str]:
    """ตัวเลขทั้งตัวในข้อความ ตัดคอมมาหลักพันและ .00 ท้ายทิ้ง"""
    t = normalize(text)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    numbers: set[str] = set()
    for raw in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])", t):
        if "." in raw:
            raw = raw.rstrip("0").rstrip(".")
        numbers.add(str(int(raw)) if raw.isdigit() else raw)
    return numbers


def _digits_run(text: str) -> str:
    """ตัวเลขที่คั่นด้วยขีดหรือช่องว่างถูกต่อกัน เพื่อเทียบเบอร์โทรที่เขียนต่างรูปแบบ"""
    return re.sub(r"(?<=\d)[\s\-.](?=\d)", "", normalize(text))


def fact_kind(fact: str) -> str:
    stripped = fact.strip()
    if _TIME_FACT_RE.match(stripped):
        return "time"
    if _PHONE_FACT_RE.match(stripped):
        return "phone"
    if _NUMBER_FACT_RE.match(stripped):
        return "number"
    return "text"


def match_fact(fact: Any, text: str) -> bool:
    """fact เป็น str หรือ {"any": [...]} (ผ่านถ้าตัวใดตัวหนึ่งผ่าน)"""
    if isinstance(fact, dict):
        return any(match_fact(alt, text) for alt in fact.get("any", []))
    fact = str(fact)
    kind = fact_kind(fact)
    if kind == "time":
        hour, minute = _TIME_FACT_RE.match(fact.strip()).groups()
        return f"{int(hour):02d}:{minute}" in extract_times(text)
    if kind == "phone":
        return re.sub(r"\D", "", fact) in _digits_run(text)
    if kind == "number":
        return extract_numbers(fact).issubset(extract_numbers(text))
    wanted, haystack = normalize(fact), normalize(text)
    return wanted in haystack or _no_space(wanted) in _no_space(haystack)


def fact_label(fact: Any) -> str:
    if isinstance(fact, dict):
        return " | ".join(str(alt) for alt in fact.get("any", []))
    return str(fact)


def forbidden_hits(patterns: list[str], text: str) -> list[str]:
    """regex ใน forbidden ที่เจอในคำตอบ (ทดสอบกับข้อความที่แปลงเลขไทยแล้ว)"""
    translated = text.translate(_THAI_DIGITS)
    return [p for p in patterns if re.search(p, translated)]

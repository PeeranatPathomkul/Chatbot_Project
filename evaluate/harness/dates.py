"""
แปลงวันที่แบบสัมพัทธ์ในชุดข้อมูล ("tomorrow", "next_weekend+2") เป็นวันที่จริงตามเวลาไทย

นิยาม "เสาร์อาทิตย์นี้ / หน้า" ต้องตรงกับปฏิทินที่แอปใส่ใน SYSTEM_PROMPT
(rag_pipeline._calendar) ไม่งั้นเฉลยกับสิ่งที่บอทถูกสั่งให้ทำจะคนละช่วงกัน แล้วนับว่าผิดทั้งที่บอททำตามกติกา
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

THAI_TZ = timezone(timedelta(hours=7))

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_EXPR_RE = re.compile(r"^(today|tomorrow|this_weekend|next_weekend|upcoming_(?:%s))(?:\+(\d+))?$" % "|".join(_WEEKDAYS))
_TEMPLATE_RE = re.compile(r"\{([a-z_]+(?:\+\d+)?)\|(th|en|slash)\}")

_TH_MONTHS = (
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
)
_EN_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def today_in_thailand() -> date:
    return datetime.now(THAI_TZ).date()


def is_date_expr(value: object) -> bool:
    return isinstance(value, str) and bool(_EXPR_RE.match(value))


def resolve(expr: str, today: date) -> date:
    match = _EXPR_RE.match(expr)
    if not match:
        raise ValueError(f"นิพจน์วันที่ไม่รู้จัก: {expr!r}")
    base, offset = match.group(1), int(match.group(2) or 0)
    if base == "today":
        day = today
    elif base == "tomorrow":
        day = today + timedelta(days=1)
    elif base in ("this_weekend", "next_weekend"):
        # วันเสาร์ถัดไป หรือวันนี้ถ้าเป็นวันเสาร์ — สูตรเดียวกับ rag_pipeline._calendar
        day = today + timedelta(days=(5 - today.weekday()) % 7)
        if base == "next_weekend":
            day += timedelta(days=7)
    else:
        target = _WEEKDAYS.index(base.removeprefix("upcoming_"))
        day = today + timedelta(days=(target - today.weekday()) % 7)
    return day + timedelta(days=offset)


def format_day(day: date, style: str) -> str:
    if style == "th":
        return f"{day.day} {_TH_MONTHS[day.month - 1]}"
    if style == "en":
        return f"{_EN_MONTHS[day.month - 1]} {day.day}"
    if style == "slash":
        return f"{day.day}/{day.month}"
    raise ValueError(style)


def render(text: str, today: date, values: dict[str, str] | None = None) -> str:
    """แทน {today+21|th} ด้วยวันที่จริง และ {ชื่อ} ด้วยค่าใน values (เช่น other_booking_id)"""
    rendered = _TEMPLATE_RE.sub(lambda m: format_day(resolve(m.group(1), today), m.group(2)), text)
    for key, value in (values or {}).items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def template_names(text: str) -> set[str]:
    """ชื่อ placeholder ที่ไม่ใช่วันที่ เช่น {other_booking_id}"""
    without_dates = _TEMPLATE_RE.sub("", text)
    return set(re.findall(r"\{([a-z_]+)\}", without_dates))


def resolve_args(args: dict, today: date) -> dict:
    """แปลงเฉพาะค่าที่เป็นนิพจน์วันที่ ค่าอื่น (guests, rooms) คงเดิม"""
    return {k: resolve(v, today).isoformat() if is_date_expr(v) else v for k, v in args.items()}

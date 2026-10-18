from datetime import date

import pytest

from harness import dates
from harness.keyfacts import extract_numbers, extract_times, fact_kind, forbidden_hits, match_fact


@pytest.mark.parametrize("text", ["เช็คอิน 14.00 น.", "from 2:00 PM", "2 pm", "บ่ายสองโมง", "ตั้งแต่ 14:00", "๑๔.๐๐ น."])
def test_time_equivalents(text):
    assert match_fact("14:00", text)


def test_time_special_words():
    assert "12:00" in extract_times("check-out by 12:00 noon")
    assert "12:00" in extract_times("ก่อนเที่ยง")
    assert "22:00" in extract_times("ถึงสี่ทุ่ม")
    assert "00:00" in extract_times("ห้ามเกินเที่ยงคืน")
    assert "12:00" not in extract_times("ห้ามเกินเที่ยงคืน")
    assert not match_fact("14:00", "ราคา 1,400.00 บาท")


def test_numbers_and_phone():
    assert match_fact("1300", "รวม 1,300 บาท")
    assert match_fact("1300", "รวม ๑,๓๐๐ บาท")
    assert not match_fact("1300", "รวม 13,000 บาท")
    assert match_fact("200", "คืนละ 200.00 บาท")
    assert match_fact("081-598-1199", "โทร 081 598 1199")
    assert match_fact("081-598-1199", "โทร ๐๘๑-๕๙๘-๑๑๙๙")
    assert not match_fact("081-598-1199", "โทร 081-598-1198")
    assert extract_numbers("2026-10-16") == {"2026", "10", "16"}


def test_text_and_any():
    assert fact_kind("53/33") == "text"
    assert match_fact("ไม่มีสระ", "ที่นี่ไม่มี สระว่ายน้ำค่ะ")  # ช่องว่างไม่แน่นอนในภาษาไทย
    assert match_fact({"any": ["no pool", "no swimming pool"]}, "There is No swimming pool.")
    assert not match_fact({"any": ["ฟรี"]}, "เสียค่าจอด 20 บาท")


def test_forbidden():
    assert forbidden_hits([r"\d+\s*วัน"], "คืนเงินภายใน ๗ วัน") == [r"\d+\s*วัน"]
    assert forbidden_hits([r"https?://"], "ไม่มีลิงก์") == []


def test_relative_dates_match_app_calendar():
    tuesday = date(2026, 10, 13)
    assert dates.resolve("today", tuesday) == tuesday
    assert dates.resolve("tomorrow+1", tuesday) == date(2026, 10, 15)
    # rag_pipeline._calendar: เสาร์อาทิตย์นี้ = เสาร์ถัดไป, หน้า = +7
    assert dates.resolve("this_weekend", tuesday) == date(2026, 10, 17)
    assert dates.resolve("next_weekend+2", tuesday) == date(2026, 10, 26)
    assert dates.resolve("upcoming_fri", tuesday) == date(2026, 10, 16)
    saturday = date(2026, 10, 17)
    assert dates.resolve("this_weekend", saturday) == saturday
    with pytest.raises(ValueError):
        dates.resolve("next_month", tuesday)


def test_render_templates():
    tuesday = date(2026, 10, 13)
    assert dates.render("เข้า {today+21|th}", tuesday) == "เข้า 3 พฤศจิกายน"
    assert dates.render("from {today+14|en}", tuesday) == "from October 27"
    assert dates.render("{today+40|slash} ว่างไหม", tuesday) == "22/11 ว่างไหม"
    assert dates.render("เลข {other_booking_id}", tuesday, {"other_booking_id": "abc"}) == "เลข abc"
    assert dates.resolve_args({"check_in": "today", "rooms": 2}, tuesday) == {"check_in": "2026-10-13", "rooms": 2}

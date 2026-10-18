"""การตรวจรายข้อ (live check, auth check, security, การให้ผ่าน/ตก) ด้วย snapshot สมมติ"""

from datetime import date

from harness.scoring import (
    amenity_stance, auth_check, customer_leaks, live_check, match_date, plan_rooms, restaurant_aliases, score_item,
)

ROOMS = [
    {"name": "F1", "type": "single", "pricePerNight": 650, "capacity": 2, "amenities": ["TV"]},
    {"name": "P1", "type": "single", "pricePerNight": 650, "capacity": 2, "amenities": ["TV", "Mini fridge"]},
    {"name": "P3", "type": "twin", "pricePerNight": 700, "capacity": 2, "amenities": ["Two beds"]},
]
SNAP = {"dates": {"check_in": "2026-11-03", "check_out": "2026-11-06"}, "available_rooms": ROOMS,
        "all_rooms": ROOMS, "kb_extra_bed": {"rate_per_night": 200, "max_per_room": 1}}


def test_plan_rooms_by_hand():
    # 3 คน: ไม่ขอเตียง 2 ห้อง (2+2), ขอเตียงได้ห้องเดียว (2+1)
    plan = plan_rooms(ROOMS, 3, 1)
    assert plan["without_beds"]["rooms"] == 2
    assert plan["with_beds"] == {"rooms": 1, "beds": 1, "room_prices": [650]}
    # 5 คน: ไม่ขอเตียง 3 ห้อง, ขอเตียง 2 ห้อง + 1 เตียง
    plan = plan_rooms(ROOMS, 5, 1)
    assert plan["without_beds"]["rooms"] == 3 and plan["with_beds"]["rooms"] == 2 and plan["with_beds"]["beds"] == 1


def test_live_checks():
    # 650 x 3 คืน x 2 ห้อง = 3,900
    assert live_check({"total_price": {"rooms": 2}}, "รวมทั้งหมด 3,900 บาทค่ะ", SNAP)["passed"]
    assert not live_check({"total_price": {"rooms": 2}}, "รวม 2,600 บาท", SNAP)["passed"]
    assert live_check("availability_answer", "ว่างค่ะ มีห้อง F1 และ P3", SNAP)["passed"]
    assert live_check("availability_answer", "เต็มแล้วค่ะ", {**SNAP, "available_rooms": []})["passed"]
    assert live_check({"rooms_with_amenity": {"amenity": "Mini fridge"}}, "ห้อง P1 มีตู้เย็นค่ะ", SNAP)["passed"]
    assert live_check({"rooms_of_type": {"type": "twin"}}, "ห้องเตียงคู่ว่างคือ P3", SNAP)["passed"]
    assert live_check({"guests_plan": {"guests": 3}}, "พักห้องเดียวได้ โดยขอเตียงเสริม 1 เตียง", SNAP)["passed"]
    assert not live_check({"guests_plan": {"guests": 3}}, "พักห้องเดียวได้เลยค่ะ", SNAP)["passed"]  # ไม่พูดถึงเตียงเสริม
    # 3 คน ห้องเดียว: 650 x 3 + 200 x 3 = 2,550
    assert live_check({"grand_total": {"guests": 3}}, "รวม 2,550 บาท", SNAP)["passed"]
    assert live_check({"extra_bed_total": {"beds": 1, "nights": 2}}, "ค่าเตียงเสริมรวม 400 บาท", SNAP)["passed"]
    assert live_check("available_count", "เหลือ 3 ห้องค่ะ", SNAP)["passed"]


def test_restaurant_aliases():
    assert "Kyo Bakery" in restaurant_aliases("เกียว เบเกอรี่ (Kyo Bakery)")
    assert "SHABU THE HULK" in restaurant_aliases("SHABU THE HULK สาขาสะเดา")


def test_dates_and_amenity_stance():
    day = date(2026, 10, 20)
    for text in ["20 ต.ค. 2569", "วันที่ 20 ตุลาคม", "2026-10-20", "20/10/2569", "Oct 20", "20 October"]:
        assert match_date(day, text), text
    assert not match_date(day, "21 ต.ค.")
    assert amenity_stance("ห้อง F1 ไม่มีตู้เย็นค่ะ", "Mini fridge") == "no"
    assert amenity_stance("ห้อง P1 มีตู้เย็น", "Mini fridge") == "yes"
    assert amenity_stance("ไม่ทราบ", "Mini fridge") is None


CUSTOMER = {
    "customerName": "สมชาย ทดสอบ",
    "bookings": [{"id": "11111111-2222-3333-4444-555555555555", "roomName": "F1", "checkIn": "2026-10-20",
                  "checkOut": "2026-10-22", "status": "approved", "paymentStatus": "paid", "totalPrice": 1300}],
    "current_ids": ["11111111-2222-3333-4444-555555555555"],
    "nearest_id": "11111111-2222-3333-4444-555555555555",
    "nearest_payment": {"status": "succeeded"}, "nearest_room": {"amenities": ["TV"]}, "stayed_count": 0,
}


def test_auth_checks():
    snap = {"customer": CUSTOMER}
    assert auth_check("my_current_rooms", "คุณจองห้อง F1 ไว้ค่ะ", snap)["passed"]
    assert auth_check("my_next_checkin", "เข้าพัก 20 ต.ค. 2569", snap)["passed"]
    assert auth_check("my_next_total", "ยอดรวม 1,300 บาท", snap)["passed"]
    assert auth_check("my_payment_status", "การจองนี้ชำระเงินแล้วค่ะ", snap)["passed"]
    assert auth_check("my_stayed_count", "ยังไม่เคยเข้าพักค่ะ", snap)["passed"]
    assert auth_check({"my_room_amenity": {"amenity": "Mini fridge"}}, "ห้อง F1 ไม่มีตู้เย็นค่ะ", snap)["passed"]


def test_customer_leaks():
    assert customer_leaks("กรุณาเข้าสู่ระบบก่อนค่ะ", CUSTOMER, set()) == []
    assert "ชื่อลูกค้า" in customer_leaks("การจองของคุณสมชาย ทดสอบ", CUSTOMER, set())
    assert customer_leaks("ห้อง F1 วันที่ 20 ต.ค.", CUSTOMER, set())
    # เลขการจองที่ลูกค้าพิมพ์มาเองในคำถาม ไม่นับว่าหลุด
    assert customer_leaks("ไม่พบเลข 11111111-2222-3333-4444-555555555555", CUSTOMER,
                          {"11111111-2222-3333-4444-555555555555"}) == []


def _record(answer, answered=True, tools=("search_knowledge_base",)):
    return {"today": "2026-10-13", "turns": [{
        "message": "ถาม", "status_code": 200, "latency_ms": 1500,
        "response": {"answer": answer, "answered": answered, "tools_used": list(tools), "sources": [],
                     "token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        "trace": {"llm_calls": [{"history_messages": 0, "tool_calls": ["search_knowledge_base"]}, {"tool_calls": []}],
                  "tool_calls": []},
    }]}


def test_score_item_pass_fail():
    item = {"id": "x", "suite": "kb", "category": "checkin", "language": "th", "answerable": True,
            "question": "เช็คอินกี่โมง", "expected_tools": ["search_knowledge_base"], "key_facts": ["14:00"]}
    assert score_item(item, _record("เช็คอินได้ตั้งแต่ 14.00 น."))["item_pass"] is True
    failed = score_item(item, _record("เช็คอินได้ตั้งแต่ 13.00 น."))
    assert failed["item_pass"] is False and failed["fail_reasons"]
    # ประโยคปฏิเสธมีเบอร์โทร แต่ข้อที่ตอบได้ต้องไม่ถูกนับว่าผ่าน
    refused = score_item({**item, "key_facts": ["081-598-1199"]}, _record("ขออภัย ไม่มีข้อมูล โทร 081-598-1199", answered=False))
    assert refused["item_pass"] is False
    unanswerable = {**item, "answerable": False, "key_facts": [], "forbidden": [r"\d+\s*บาท"]}
    assert score_item(unanswerable, _record("ขออภัย ไม่มีข้อมูล", answered=False))["item_pass"] is True
    assert score_item(unanswerable, _record("เช็คเอาท์ช้าได้ คิด 300 บาท"))["item_pass"] is False
    scored = score_item(item, _record("14.00 น.", tools=("search_knowledge_base", "list_restaurants")))
    assert scored["tool_exact"] is False and scored["unnecessary_tools"] == ["list_restaurants"]

# Resort Chatbot API

ระบบแชทบอทผู้ช่วยส่วนตัวสำหรับแพลตฟอร์มจองและจัดการรีสอร์ต ใช้เทคนิค RAG (Retrieval-Augmented Generation)
เชื่อมกับเว็บแอปพลิเคชันที่มีอยู่แล้วผ่าน REST API

ตอนนี้ต่อกับคลังความรู้จริงของ **พูนสุข รีสอร์ท สะเดา** แล้ว
คลังความรู้ถูกสร้างโดยโปรเจกต์ `../indexing-pipeline` — API เป็นแค่ผู้อ่าน ไม่ได้เป็นคนเขียน

> **ต้อง index ข้อมูลก่อนถึงจะรัน API ได้** ดูขั้นตอนในหัวข้อ "วิธีรัน" ด้านล่าง

## Tech stack

| ส่วนประกอบ | เทคโนโลยี |
|---|---|
| Backend | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Vector database | ChromaDB (embedded, อ่านจาก `../indexing-pipeline/chroma_db`) |
| Embedding model | sentence-transformers (`intfloat/multilingual-e5-base`, รันโลคัลฟรี) |
| LLM | Typhoon API `typhoon-v2.5-30b-a3b-instruct`, สลับไป Gemini ได้ผ่าน `LLMClient` interface |
| ข้อมูลสด (ห้องว่าง ราคา การจอง) | Booking API (NestJS + PostgreSQL) เรียกผ่าน HTTP |
| การเลือกแหล่งข้อมูล | tool calling — โมเดลเลือกเองว่าจะถามระบบจองหรือคลังความรู้ |
| RAG orchestration | LangChain text splitter |
| Testing | Pytest |
| Deployment | Docker / docker-compose, deploy ได้บน Render free tier |

## โครงสร้างโปรเจกต์

```
resort-chatbot-api/
├── app/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── config.py                # โหลด env vars ด้วย pydantic-settings
│   ├── api/routes/
│   │   ├── chat.py              # POST /api/v1/chatbot/query, GET /api/v1/chatbot/health
│   │   └── knowledge.py         # จัดการฐานความรู้ (CRUD)
│   ├── schemas/chat.py          # Pydantic models
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── llm_client.py
│   │   ├── booking_api.py       # client เรียก Booking API (ไม่ต่อ PostgreSQL ตรง)
│   │   └── rag_pipeline.py      # ลูป tool calling (ชื่อเดิม เนื้อในเป็น agent แล้ว)
│   ├── core/
│   │   ├── prompts.py
│   │   └── tools.py             # นิยาม tool + ตัวรัน (ToolExecutor)
│   └── static/index.html         # หน้าเว็บทดสอบแชท (เสิร์ฟที่ /)
├── data/sample_knowledge.csv
├── scripts/ingest.py
├── tests/test_chat_api.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## วิธีรันโลคัล

### 1. ติดตั้ง dependencies

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. ตั้งค่า environment variables

```bash
copy .env.example .env
```

แล้วกรอกค่า `TYPHOON_API_KEY` (หรือ `GEMINI_API_KEY` ถ้าตั้ง `LLM_PROVIDER=gemini`) ใน `.env`

### 3. สร้างคลังความรู้ (ทำที่โปรเจกต์ indexing-pipeline)

API เป็นแค่ผู้อ่าน vector store ไม่ได้เป็นคนสร้าง ต้อง index ข้อมูลจากอีกโปรเจกต์ก่อน

```bash
cd ../indexing-pipeline
pip install -r requirements.txt
python -m src.indexer --reset
```

คำสั่งนี้อ่านเอกสารใน `indexing-pipeline/data/th/` และ `data/en/` -> แบ่ง chunk ->
สร้าง embedding -> เก็บลง `indexing-pipeline/chroma_db/`
ครั้งแรกจะดาวน์โหลดโมเดล `multilingual-e5-base` (~1.1 GB) ใช้เวลาสักครู่

ตรวจว่า index สำเร็จ:

```bash
python -m src.indexer --stats
```

ควรเห็นจำนวน chunk และ `distance metric : cosine`

> **ถ้าข้ามขั้นตอนนี้** API จะตอบว่า "ไม่มีข้อมูลเรื่องนี้ในระบบ" ทุกคำถาม
> โดยไม่มี error ให้เห็น เพราะ vector store ว่างเปล่า

### 4. รันเซิร์ฟเวอร์

```bash
cd ../resort-chatbot-api
uvicorn app.main:app --reload
```

เซิร์ฟเวอร์รันที่ `http://127.0.0.1:8000`

| URL | ใช้ทำอะไร |
|---|---|
| `http://127.0.0.1:8000/` | **หน้าเว็บทดสอบแชท** — คุยกับบอทได้เลย เห็น `answered` / score / แหล่งอ้างอิง |
| `http://127.0.0.1:8000/docs` | Swagger UI สำหรับยิง API ทีละ endpoint |

### 5. ทดสอบผ่านหน้าเว็บ

เปิด **http://127.0.0.1:8000/** ในเบราว์เซอร์ — กดคำถามตัวอย่างหรือพิมพ์เอง

หน้านี้ออกแบบมาเพื่อ **ทดสอบคุณภาพคำตอบ** ไม่ใช่หน้าใช้งานจริงของลูกค้า จึงแสดงค่าที่ใช้วินิจฉัยไว้ครบ:

- **`answered`** — บอทตอบได้จากข้อมูลจริง หรือปฏิเสธเพราะไม่มีข้อมูล
- **score / conf** — คะแนน retrieval ดิบ ไว้ดูว่าดึง chunk มาใกล้แค่ไหน
- **เวลาที่ใช้** — วินาทีต่อคำถาม
- **ข้อมูลที่ใช้ตอบ** — กดดูได้ว่าคำตอบนี้มาจาก chunk ไหนบ้าง

คำตอบที่ขึ้น**กรอบสีส้ม**คือบอทปฏิเสธ ซึ่งเป็นพฤติกรรมที่ต้องการ ไม่ใช่ bug

สลับภาษา ไทย/English ได้ที่ปุ่มด้านบน ระบบจะกรอง chunk ตามภาษาและตอบเป็นภาษานั้น

> หน้าเว็บถูกเสิร์ฟโดย FastAPI เองที่ `app/static/index.html`
> จึงเป็น same-origin กับ API — ถ้าเปิดไฟล์ HTML ตรง ๆ จาก `file://` เบราว์เซอร์จะบล็อกด้วย CORS

---

## ลำดับการทำงานภายใน (เกิดอะไรขึ้นเมื่อมีคำถามเข้ามา)

แชทบอทไม่ได้ค้นคลังความรู้ก่อนเสมออีกต่อไป แต่ให้โมเดลเลือกเองว่าจะถามที่ไหน
เพราะคำถามจริงมักต้องใช้สองแหล่งต่อกัน เช่น "ห้องเงียบ ๆ เสาร์นี้ว่างไหม ราคาเท่าไหร่"

```
POST /api/v1/chatbot/query   (Authorization: Bearer <token ของลูกค้า> ถ้ามี)
  │
  ├─ 1. ประกอบ messages     กติกา + วันที่วันนี้ -> system role
  │                         คำถามของลูกค้า -> user role
  │                         (การแยกนี้คือสิ่งที่กันไม่ให้โมเดลแต่งคำตอบ)
  │
  ├─ 2. llm_client.generate(messages, tools)   โมเดลเลือกเรียก tool
  │        ├─ search_available_rooms  -> Booking API (ห้องว่างจริง + ราคาจาก DB)
  │        ├─ search_knowledge_base   -> ChromaDB (นโยบาย ค่าเตียงเสริม การเดินทาง)
  │        ├─ get_payment_info        -> Booking API
  │        ├─ list_restaurants        -> Booking API
  │        └─ get_my_bookings /
  │           get_booking_payment     -> Booking API (ต้องมี token ของลูกค้า)
  │
  ├─ 3. วนกลับไปข้อ 2       ยัดผลของ tool กลับเข้า messages แล้วถามโมเดลอีกรอบ
  │                         สูงสุด AGENT_MAX_TOOL_ROUNDS รอบ รอบสุดท้ายไม่ส่ง tools
  │                         เพื่อบังคับให้สรุปเป็นข้อความ
  │
  └─ 4. ประเมินผล          ถ้าโมเดลตอบด้วยประโยคปฏิเสธ -> answered=false,
                           confidence=0, sources=[]
```

**เส้นแบ่งว่าข้อมูลไหนเป็นของใคร**

| ข้อมูล | เจ้าของ | เหตุผล |
|---|---|---|
| ห้องว่าง ราคาห้อง ชื่อห้อง capacity สิ่งอำนวยความสะดวกในห้อง | Booking API (PostgreSQL) | เปลี่ยนได้ตลอด staff แก้ที่เดียว |
| ค่าเตียงเสริม มัดจำ นโยบายยกเลิก เวลาเช็คอิน การเดินทาง | ChromaDB | ไม่มีใน DB |

ราคาถูกถอดออกจาก `../indexing-pipeline/data/**.md` หมดแล้ว **อย่าเติมกลับเข้าไป** ไม่งั้น
จะมีราคาสองแหล่งที่ขัดกันเอง แล้วบอทจะเสนอราคาที่ไม่ตรงกับที่ลูกค้าจองจริงได้

**ทำไมไม่ต่อ PostgreSQL ตรง ๆ** — "ห้องว่างไหม" ไม่ใช่ค่าที่เก็บอยู่ในตาราง แต่เป็นผลที่
API คำนวณจาก booking ที่มีอยู่ด้วยกติกาช่วงวันที่แบบ half-open `[checkIn, checkOut)`
บวกกับ hold window ที่ auto-cancel booking ที่ยังไม่จ่ายเงิน การเขียน query เองคือการ
เลียนแบบกติกาชุดนั้นให้ตรงไปตลอด ซึ่งพลาดเมื่อไหร่บอทจะบอกว่าห้องว่างทั้งที่มีคนจองแล้ว
อีกข้อคือ API บังคับให้ตัวตนลูกค้ามาจาก JWT เท่านั้น ส่วนการต่อ DB ตรงไม่มีอะไรกั้น
ไม่ให้อ่าน booking ของลูกค้าคนอื่น

**เวลาที่ใช้จริง:** คำถามแรก ~7 วินาที (โหลดโมเดล embedding เข้าหน่วยความจำ)
คำถามถัดไป ~0.7-1.0 วินาที

## ตัวอย่าง curl request

### Health check

```bash
curl http://127.0.0.1:8000/api/v1/chatbot/health
```

### ถามคำถาม

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chatbot/query ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\": \"s1\", \"message\": \"เช็คอินกี่โมง\", \"language\": \"th\"}"
```

ตอบกลับ:

```json
{
  "session_id": "s1",
  "answer": "เช็คอินได้ตั้งแต่เวลา 14.00 น. เป็นต้นไป ถึง 22.00 น.",
  "sources": [
    {
      "doc_id": "th-booking_policy::0000",
      "snippet": "เวลาเช็คอินคือ 14.00 น. เป็นต้นไป รับเช็คอินได้ถึงเวลา 22.00 น. ..."
    }
  ],
  "answered": true,
  "confidence": 0.88,
  "retrieval_score": 0.8791,
  "suggested_action": {"type": "none", "url": ""}
}
```

### คำถามที่ระบบไม่มีข้อมูล

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chatbot/query ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\": \"s1\", \"message\": \"จากสนามบินหาดใหญ่มากี่กิโล\", \"language\": \"th\"}"
```

```json
{
  "answer": "ขออภัยค่ะ ไม่มีข้อมูลเรื่องนี้ในระบบ รบกวนสอบถามโดยตรงที่ 081-598-1199 นะคะ",
  "sources": [],
  "answered": false,
  "confidence": 0.0,
  "retrieval_score": 0.8712
}
```

### ถามภาษาอังกฤษ

ส่ง `"language": "en"` ระบบจะดึงเฉพาะ chunk ภาษาอังกฤษและตอบเป็นอังกฤษ
รวมถึงประโยคปฏิเสธด้วย

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chatbot/query ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\": \"s1\", \"message\": \"How much is a room?\", \"language\": \"en\"}"
```

---

## field ที่ตอบกลับ — ควรใช้ตัวไหนตัดสินใจ

| field | ความหมาย | ใช้ยังไง |
|---|---|---|
| `answered` | ตอบได้จากข้อมูลจริง หรือปฏิเสธ | **ใช้ตัวนี้เป็นหลัก** — `false` ให้แสดงปุ่มโทรหารีสอร์ท |
| `answer` | ข้อความคำตอบ | แสดงให้ลูกค้า |
| `sources` | chunk ที่ใช้ตอบ | แสดงเป็นแหล่งอ้างอิงได้ ว่างเสมอเมื่อ `answered=false` |
| `confidence` | 1.0 เมื่อตอบจากข้อมูลสดของระบบจอง, เท่ากับ `retrieval_score` เมื่อตอบจากคลังความรู้, 0.0 เมื่อปฏิเสธ | ใช้เรียงลำดับได้ แต่**ห้ามใช้ตัดสินว่าคำตอบเชื่อถือได้ไหม** |
| `retrieval_score` | cosine similarity ดิบของ chunk ที่ดีที่สุด — 0.0 เมื่อไม่ได้ค้นคลังความรู้เลย | debug และ monitor เท่านั้น |

> **อย่าตั้ง threshold จาก `retrieval_score`**
> วัดจริงแล้วคำถามที่ระบบ*ไม่มี*คำตอบได้คะแนนเฉลี่ย 0.835
> ส่วนคำถามที่ตอบได้ 0.845 — ต่างกันแค่ 0.011 แยกกันไม่ออก
> (ดู `../indexing-pipeline/scripts/evaluate.py`)
> สิ่งที่กันการตอบมั่วได้จริงคือ system prompt ไม่ใช่ตัวเลขนี้

## รัน tests

```bash
pytest
```

Test ใน `tests/test_chat_api.py` mock ทั้ง `EmbeddingService`, `VectorStore`, `LLMClient`
และ `BookingAPIClient` ไว้ทั้งหมด จึงไม่มีการยิง API จริงหรือโหลดโมเดล embedding จริง
ตอนรัน test (15 เทสต์ ~5 วินาที)

เทสต์ที่ห้ามปล่อยให้แดง:
- `test_กติกาถูกส่งแยกเป็น_system_ไม่ใช่ยัดรวมกับคำถาม` — กลไกกันการแต่งคำตอบ
- `test_ห้ามส่ง_guests_ต่อเข้า_booking_api` — `?guests=3` จะกรองห้องจริงทิ้งทั้งหมด
- `test_ถามการจองของตัวเองโดยไม่ได้ล็อกอินต้องไม่หลุดข้อมูลใคร` — กันข้อมูลลูกค้าคนอื่นรั่ว

## รันด้วย Docker

```bash
docker compose up --build
```

## Deploy ขึ้น Render (free tier)

1. Push โค้ดขึ้น GitHub
2. สร้าง Web Service ใหม่บน Render แล้วเชื่อมกับ repo นี้
3. ตั้งค่า Build Command: `pip install -r requirements.txt`
4. ตั้งค่า Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. เพิ่ม environment variables ตาม `.env.example` ใน Render dashboard (Environment tab)
6. หมายเหตุ: free tier ของ Render มี ephemeral disk ข้อมูลใน `./data/chroma` จะหายเมื่อ instance restart
   ถ้าต้องการข้อมูลถาวรบน production ควรใช้ Render Disk (มีค่าใช้จ่าย) หรือย้ายไปใช้ vector DB แบบ managed

## สลับ LLM provider (Typhoon <-> Gemini)

แก้ค่า `LLM_PROVIDER` ใน `.env` เป็น `typhoon` หรือ `gemini` แล้ว restart เซิร์ฟเวอร์ ไม่ต้องแก้โค้ดส่วนอื่น
เพราะทุก provider implement ตาม interface เดียวกัน (`LLMClient` ใน `app/services/llm_client.py`)

## ขั้นตอนต่อไป (ยังไม่ได้ทำในโครงนี้)

- เติม knowledge base จริงแทนข้อมูลตัวอย่าง (ยังมี placeholder `<<รอเติม: ...>>` ค้างอยู่ 86 จุด
  ซึ่งทำให้บอทปฏิเสธคำถามที่ตรงกับหัวข้อนั้น)
- write endpoint (สร้าง/ยกเลิกการจอง) — ตอนนี้บอทอ่านอย่างเดียว ถ้าจะเปิดต้องมีขั้นยืนยันจากคน
- เพิ่มระบบ authentication/authorization สำหรับ endpoint `/api/v1/knowledge`
- เพิ่ม logging และ observability
- ปรับปรุงการคำนวณ `confidence` ให้แม่นยำขึ้น
- เพิ่ม logic สำหรับ `suggested_action` (เช่น แนบลิงก์หน้าจองห้องพัก)

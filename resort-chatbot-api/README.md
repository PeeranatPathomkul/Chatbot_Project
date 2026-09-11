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
│   │   └── rag_pipeline.py
│   └── core/prompts.py
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

เซิร์ฟเวอร์รันที่ `http://127.0.0.1:8000` เปิด API docs อัตโนมัติได้ที่
`http://127.0.0.1:8000/docs` (ลองยิงคำถามจากหน้านั้นได้เลยไม่ต้องใช้ curl)

### 5. ตรวจว่าทำงานครบวงจร

```bash
curl http://127.0.0.1:8000/api/v1/chatbot/health
```

ควรได้ `{"status":"ok"}` จากนั้นลองถามคำถามจริงตามตัวอย่างด้านล่าง

---

## ลำดับการทำงานภายใน (เกิดอะไรขึ้นเมื่อมีคำถามเข้ามา)

```
POST /api/v1/chatbot/query
  │
  ├─ 1. embed_query()      แปลงคำถามเป็นเวกเตอร์ 768 มิติ (เติม prefix "query: " ให้เอง)
  │
  ├─ 2. vector_store.query()  ค้นหา top-3 chunk ที่ใกล้ที่สุดด้วย cosine
  │                           กรองด้วย language ที่ลูกค้าส่งมา (th / en)
  │
  ├─ 3. ประกอบ prompt      กติกา -> system role
  │                        ข้อมูลอ้างอิง + คำถาม -> user role
  │                        (การแยกนี้คือสิ่งที่กันไม่ให้โมเดลแต่งคำตอบ)
  │
  ├─ 4. llm_client.generate()  เรียก Typhoon API
  │
  └─ 5. ประเมินผล          ถ้าโมเดลตอบด้วยประโยคปฏิเสธ -> answered=false,
                           confidence=0, sources=[]
```

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
| `confidence` | 0.0 เมื่อปฏิเสธ นอกนั้นเท่ากับ `retrieval_score` | ใช้เรียงลำดับได้ แต่**ห้ามใช้ตัดสินว่าคำตอบเชื่อถือได้ไหม** |
| `retrieval_score` | cosine similarity ดิบของ chunk อันดับ 1 | debug และ monitor เท่านั้น |

> **อย่าตั้ง threshold จาก `retrieval_score`**
> วัดจริงแล้วคำถามที่ระบบ*ไม่มี*คำตอบได้คะแนนเฉลี่ย 0.835
> ส่วนคำถามที่ตอบได้ 0.845 — ต่างกันแค่ 0.011 แยกกันไม่ออก
> (ดู `../indexing-pipeline/scripts/evaluate.py`)
> สิ่งที่กันการตอบมั่วได้จริงคือ system prompt ไม่ใช่ตัวเลขนี้

## รัน tests

```bash
pytest
```

Test ใน `tests/test_chat_api.py` mock ทั้ง `EmbeddingService`, `VectorStore` และ `LLMClient` ไว้ทั้งหมด
จึงไม่มีการยิง API จริงหรือโหลดโมเดล embedding จริงตอนรัน test (9 เทสต์ ~8 วินาที)

เทสต์ที่สำคัญที่สุดคือ `test_กติกาถูกส่งแยกเป็น_system_ไม่ใช่ยัดรวมกับคำถาม`
ถ้าเทสต์นี้แดง แปลว่ากลไกกันการแต่งคำตอบหลุด ต้องแก้ก่อน deploy

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

- เติม knowledge base จริงแทนข้อมูลตัวอย่าง
- เพิ่มระบบ authentication/authorization สำหรับ endpoint `/api/v1/knowledge`
- เพิ่ม logging และ observability
- ปรับปรุงการคำนวณ `confidence` ให้แม่นยำขึ้น
- เพิ่ม logic สำหรับ `suggested_action` (เช่น แนบลิงก์หน้าจองห้องพัก)

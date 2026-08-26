# Resort Chatbot API

ระบบแชทบอทผู้ช่วยส่วนตัวสำหรับแพลตฟอร์มจองและจัดการรีสอร์ต ใช้เทคนิค RAG (Retrieval-Augmented Generation)
เชื่อมกับเว็บแอปพลิเคชันที่มีอยู่แล้วผ่าน REST API

โครงนี้เป็น **scaffold** — โครงสร้างโปรเจกต์ + skeleton code ที่รันได้จริง โดยใช้ข้อมูลตัวอย่างในการตอบคำถาม
ยังไม่มี knowledge base จริง (เติมทีหลังผ่าน `scripts/ingest.py` หรือ endpoint `/api/v1/knowledge`)

## Tech stack

| ส่วนประกอบ | เทคโนโลยี |
|---|---|
| Backend | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Vector database | ChromaDB (embedded, เก็บที่ `./data/chroma`) |
| Embedding model | sentence-transformers (`intfloat/multilingual-e5-base`, รันโลคัลฟรี) |
| LLM | Typhoon API (opentyphoon.ai) เป็นหลัก, สลับไป Gemini API ได้ผ่าน `LLMClient` interface |
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

### 3. Ingest ข้อมูลตัวอย่างเข้า vector store

```bash
python scripts/ingest.py
```

คำสั่งนี้จะอ่าน `data/sample_knowledge.csv` -> chunk -> สร้าง embedding -> เก็บลง ChromaDB ที่ `./data/chroma`

### 4. รันเซิร์ฟเวอร์

```bash
uvicorn app.main:app --reload
```

เซิร์ฟเวอร์จะรันที่ `http://127.0.0.1:8000` ดู API docs อัตโนมัติได้ที่ `http://127.0.0.1:8000/docs`

## ตัวอย่าง curl request

### Health check

```bash
curl http://127.0.0.1:8000/api/v1/chatbot/health
```

### ถามคำถาม

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chatbot/query \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"s1\", \"resort_id\": \"sunrise_villa\", \"message\": \"เช็คอินได้กี่โมง\", \"language\": \"th\"}"
```

### เพิ่มความรู้ใหม่

```bash
curl -X POST http://127.0.0.1:8000/api/v1/knowledge \
  -H "Content-Type: application/json" \
  -d "{\"resort_id\": \"sunrise_villa\", \"category\": \"spa\", \"question\": \"มีสปาไหม\", \"answer\": \"มีบริการสปาเปิดทุกวัน 10:00-20:00 น.\", \"source\": \"หน้าบริการเสริม\"}"
```

## รัน tests

```bash
pytest
```

Test ใน `tests/test_chat_api.py` mock ทั้ง `EmbeddingService`, `VectorStore` และ `LLMClient` ไว้ทั้งหมด
จึงไม่มีการยิง API จริงหรือโหลดโมเดล embedding จริงตอนรัน test

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

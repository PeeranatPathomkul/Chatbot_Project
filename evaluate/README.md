# evaluate/ — การประเมินผลแชทบอทเชิงวิศวกรรม

ระบบวัดผลแชทบอท พูนสุข รีสอร์ท แบบวิศวกรรม (ไม่ใช้แบบสอบถาม) ตามหัวข้อ 9.8 ของรายงาน
ผลทุกตัวเก็บเป็นหลักฐานที่เปิดดูได้ใน `output/<YYYYMMDD-HHMM>/` ได้แก่ รายงาน (`report.md`, `report.html`),
กราฟ PNG, ตาราง CSV, รายการข้อที่ตก และผลดิบ

```
evaluate/
  datasets/        Golden Test Set 209 ข้อ (YAML) + SCHEMA.md อธิบายทุกฟิลด์
                   (kb_filled.yaml = หัวข้อที่เจ้าของรีสอร์ทเติมข้อมูลเมื่อ 2026-10-18 เดิมเป็นข้อที่ตอบไม่ได้)
  harness/         โค้ด: client, runner, scoring, metrics, judge, semantic, plots, report
  tests/           pytest ของฟังก์ชันคำนวณ metric
  output/<รอบ>/    ผลแต่ละรอบ (ไม่เขียนทับรอบเก่า)
    raw/*.jsonl      คำถาม, response เต็ม, trace (tool + arguments), snapshot ของเฉลย, latency, timestamp
    metrics.json     ตัวเลขทุก metric
    charts/*.png     กราฟทุกตัว
    tables/*.csv     ตารางรายข้อ (ผ่าน/ตก + เหตุผล) และตารางย่อย (เปิดด้วย Excel ได้ ภาษาไทยไม่เพี้ยน)
    failures.md      ข้อที่ตก พร้อมคำถาม คำตอบ และสิ่งที่ผิด
    report.md        รายงานสรุปภาษาไทย ฝังกราฟ
    report.html      เวอร์ชันเบราว์เซอร์ (รูปฝังในไฟล์ เปิดได้ทันที)
    human_labels.csv (ถ้าสร้าง) 40 ข้อให้คนให้คะแนน เพื่อคำนวณ Cohen's kappa
```

## 1. ติดตั้ง (ครั้งเดียว)

ใช้ venv เดียวกับ API (Python 3.11) ที่ `resort-chatbot-api/.venv`

```powershell
cd D:\year4\term2\mobile_app\project_booking\Chatbot_Project\resort-chatbot-api
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt "numpy==1.26.4" "transformers==4.57.6"
.\.venv\Scripts\python.exe -m pip install matplotlib pyyaml
```

ค่าลับ: คัดลอก `evaluate/.env.example` เป็น `evaluate/.env` แล้วกรอกบัญชีลูกค้าทดสอบ
(`.env` ถูก gitignore ไว้ และ harness แทนค่าลับด้วย `***` ทุกครั้งก่อนเขียนไฟล์)
key ที่แอปมีอยู่แล้ว (TYPHOON_API_KEY, BOOKING_API_INTERNAL_KEY) อ่านจาก `resort-chatbot-api/.env` ให้เอง

## 2. เปิดเซิร์ฟเวอร์ก่อนรัน

1. เปิดระบบจอง NestJS ที่ port 3000 (โปรเจกต์แยก)
2. เปิดแชทบอทแบบ **instrumented** จากโค้ดใน working tree (ไม่ใช้ Docker):

```powershell
cd D:\year4\term2\mobile_app\project_booking\Chatbot_Project\evaluate
..\resort-chatbot-api\.venv\Scripts\python.exe -m uvicorn harness.instrumented_app:app --port 8000
```

`harness/instrumented_app.py` คือ `app.main:app` ตัวจริงที่ห่อ `ToolExecutor.run` และ `LLMClient.generate` ไว้
เพื่อบันทึก arguments ของ tool, ผลของ tool และจำนวนรอบที่เรียก LLM (response ของแอปไม่ถูกแก้)
และเก็บประวัติแชทของรอบประเมินแยกไว้ที่ `evaluate/.state/chat_history.db`
**อยากวัด cold start ต้องเปิดเซิร์ฟเวอร์ใหม่ก่อนรันทุกครั้ง** (preflight จะจับเวลาคำขอแรกให้)

ทุกคำสั่งที่ยิงแชทบอทจะทำ **preflight** ก่อน: health, `/_eval/info` (ยืนยันว่าเป็น wrapper),
Booking API `/rooms`, ล็อกอินบัญชีทดสอบ และถาม 2 ข้อใน session เดียวกันเพื่อยืนยันว่ามีประวัติแชท ถ้าไม่ผ่านจะหยุดทันที

## 3. คำสั่ง (รันจากโฟลเดอร์ `evaluate/`)

ตั้ง alias ไว้พิมพ์สั้นลง: `$py = "..\resort-chatbot-api\.venv\Scripts\python.exe"`

| ส่วน | คำสั่ง | เรียก LLM |
|---|---|---|
| ประมาณจำนวนคำขอ | `& $py -m harness estimate` | ไม่ |
| ตรวจเซิร์ฟเวอร์ | `& $py -m harness preflight` | 2 คำขอ |
| 1. Retrieval | `& $py -m harness retrieval` | ไม่ |
| ชุดทดสอบหลัก (ส่วน 2, 3, 4, 5 latency/token, 6 paraphrase/multi-turn/security) | `& $py -m harness run` (`--smoke` = 5 ข้อ, `--limit N`, `--suite kb,live`, `--ids a,b`) | ใช่ |
| 6. Run-to-run | `& $py -m harness consistency` (20 ข้อ x 5 ครั้ง) | ใช่ |
| 5. Load test | `& $py -m harness load` (`--levels 1,5,10,20`) | ใช่ |
| 3. Semantic + BERTScore | `& $py -m harness semantic` | ไม่ (รันบนเครื่อง) |
| 3-4. LLM judge + Faithfulness | `& $py -m harness judge` | ใช่ |
| สุ่ม 40 ข้อให้คนให้คะแนน | `& $py -m harness judge-export` | ไม่ |
| Cohen's kappa (หลังกรอก CSV) | `& $py -m harness kappa` | ไม่ |
| คำนวณ metric + กราฟ + รายงานใหม่จาก raw | `& $py -m harness report` | ไม่ |
| ทุกส่วนรวดเดียว | `& $py -m harness all --yes` (`--smoke` = ทดลอง 5 ข้อ, `--skip load,judge`) | ใช่ |
| ตรวจชุดข้อมูล | `& $py -m harness.dataset validate` | ไม่ |
| pytest ของ metric | `& $py -m pytest` | ไม่ |

- `retrieval`, `run`, `all` สร้างรอบใหม่เมื่อไม่ระบุ `--run-dir` คำสั่งอื่นใช้รอบล่าสุด
- **รันต่อจากจุดที่ค้าง:** ใส่ `--run-dir <ชื่อรอบ>` เดิม ข้อที่บันทึกแล้วจะถูกข้าม (ผลดิบเขียนทีละข้อทันที)
- **429 / 5xx:** retry แบบ exponential backoff สูงสุด `EVAL_MAX_RETRIES` ครั้ง และเว้นระยะระหว่างคำขอ `EVAL_DELAY_SECONDS`
  (load test ไม่ retry เพื่อวัด error rate จริง)
- ข้อที่ต้องล็อกอินถูกข้าม (และบอกในรายงาน) ถ้าไม่มี `EVAL_CUSTOMER_*` / `EVAL_OTHER_*`

## 4. ความหมายของ metric

### 1) ความแม่นยำในการค้นคืน (ไม่เรียก LLM)
ค้น ChromaDB ด้วย `EmbeddingService` + `VectorStore` ของแอป (prefix `query:`, กรองภาษา, top_k จากค่าตั้งของแอป)

| metric | นิยาม |
|---|---|
| Hit@k | สัดส่วนคำถามที่มี chunk ที่เกี่ยวข้องอยู่ใน k อันดับแรก (k = 1, 3, 5) |
| MRR | ค่าเฉลี่ยของ 1/อันดับของ chunk ที่เกี่ยวข้องตัวแรก (0 ถ้าไม่พบใน top-5) |
| nDCG@5 | DCG = Σ rel_i / log2(i+1), หารด้วย DCG ของการเรียงที่ดีที่สุด (IDCG จากจำนวน chunk ที่เกี่ยวข้องทั้งหมดในคลังของภาษานั้น) |
| chunk ที่เกี่ยวข้อง | มาจากไฟล์ใน `expected_sources` และมี key fact ของข้อนั้นในเนื้อหา (ข้อที่ไม่มี key fact ใช้ระดับไฟล์) |
| Hit@k ระดับไฟล์ | นับแค่ว่าไฟล์ถูก ใช้เทียบกับตัวเลข Top-1/Top-3 เดิมใน proposal (ชุด 44 ข้อเดิมจาก `indexing-pipeline/scripts/evaluate.py`) |
| การกระจายคะแนนอันดับ 1 | เทียบ similarity ของคำถามที่ตอบได้กับตอบไม่ได้ + threshold ที่ดีที่สุดที่แยกสองกลุ่มได้ (ยืนยันว่า threshold ใช้ไม่ได้) |

### 2) ความถูกต้องของการเรียก tool (จาก `tools_used` + trace)

| metric | นิยาม |
|---|---|
| Tool Selection Accuracy (exact) | ชุด tool ที่เรียก = `expected_tools` พอดี |
| Tool Selection Accuracy (lenient) | เรียกครบ `expected_tools` และที่เรียกเกินอยู่ใน `allowed_extra_tools` |
| Precision / Recall ราย tool | TP = ควรเรียกและเรียก, FP = เรียกทั้งที่ไม่ได้อยู่ในเฉลยหรือที่อนุญาต, FN = ควรเรียกแต่ไม่เรียก |
| Argument Accuracy | `check_in`, `check_out`, `rooms`, `guests` ของการเรียก tool ครั้งสุดท้าย ตรงกับเฉลย (วันที่แบบสัมพัทธ์แปลงตามวันที่รันในเวลาไทย) |
| Unnecessary Call Rate | สัดส่วนข้อที่เรียก tool นอกเฉลยและนอกที่อนุญาต |
| No-tool rate | สัดส่วนคำถามที่ต้องใช้ข้อมูลแต่ตอบโดยไม่เรียก tool เลย |
| จำนวนรอบเฉลี่ย | จำนวนครั้งที่แอปเรียก LLM ต่อคำถาม (รวมการกระทุ้งให้ตอบใหม่) และจำนวนรอบที่เรียก tool |

### 3) ความถูกต้องของคำตอบ

| metric | นิยาม |
|---|---|
| Key-Fact Accuracy (หลัก) | % ของข้อเท็จจริงสำคัญที่พบในคำตอบ เทียบแบบยอมรับรูปแบบเทียบเท่า (14.00 / 14:00 / 2 PM, คอมมา, เลขไทย) — ข้อมูลสดคำนวณจาก Booking API ตอนรัน |
| Complete & clean | % ข้อที่ครบทุก fact, `answered=true`, ไม่มี `forbidden` และไม่มีชื่อห้องที่ไม่มีจริง |
| Semantic Similarity | cosine ของ embedding (multilingual-e5-base, prefix `query:` ทั้งสองฝั่ง) ระหว่างคำตอบกับ `reference_answer` |
| BERTScore | greedy matching ของ token embedding (e5-base ชั้นที่ 9) P = เฉลี่ย max ต่อ token ของคำตอบ, R = เฉลี่ย max ต่อ token ของอ้างอิง, F1 — เขียนเองตามนิยาม ไม่ถ่วง idf ไม่ rescale |
| LLM-as-a-Judge | คะแนน 1-5 ด้านความถูกต้อง ความครบถ้วน และการไม่แต่งข้อมูล ตาม rubric แบบ G-Eval (temperature 0, ตอบเป็น JSON) |
| Cohen's kappa | ความสอดคล้องระหว่างคนกับ judge บน 40 ข้อสุ่ม (κ แบบไม่ถ่วง, แบบถ่วงกำลังสอง และแบบผ่าน/ไม่ผ่านที่ ≥ 4) |

### 4) การปฏิเสธและการป้องกันข้อมูลเท็จ (จาก `answered`)

| metric | นิยาม |
|---|---|
| Refusal Precision | ปฏิเสธถูก / ปฏิเสธทั้งหมด |
| Refusal Recall | ปฏิเสธถูก / คำถามที่ไม่มีข้อมูลทั้งหมด |
| Hallucination Rate | ตอบทั้งที่ไม่มีข้อมูล / คำถามที่ไม่มีข้อมูลทั้งหมด |
| False Refusal Rate | ปฏิเสธทั้งที่มีข้อมูล / คำถามที่มีข้อมูลทั้งหมด |
| Faithfulness | (RAGAS) judge แยกคำตอบเป็น claim แล้วตรวจกับผลของ tool เต็ม ๆ จาก trace: claim ที่มีหลักฐาน / claim ทั้งหมด |

### 5) ประสิทธิภาพ

| metric | นิยาม |
|---|---|
| Latency p50/p95/p99/mean | เวลาฝั่ง client (ยิงตรง FastAPI) แยกตามจำนวน tool ที่เรียก (0, 1, 2, 3+) หลัง warm-up |
| Cold start | คำขอแรกหลังเปิดเซิร์ฟเวอร์ (รวมการโหลดโมเดล embedding) |
| Token ต่อคำถาม | prompt / completion / total จาก `token_usage` และต้นทุน (ถ้าตั้ง `EVAL_PRICE_*`) |
| Load test | ยิงพร้อมกัน 1, 5, 10, 20 คำขอ (ระดับละ max(10, 2 x concurrency) คำขอ): throughput (คำขอสำเร็จ/วินาที), error rate, timeout rate (> 90 วินาที), latency p95 |

### 6) ความทนทาน

| metric | นิยาม |
|---|---|
| Paraphrase Consistency | % กลุ่มที่ทุกแบบผ่าน/ตกเหมือนกัน (11 กลุ่ม กลุ่มละ 4 แบบ) |
| Run-to-run Consistency | ถามซ้ำ 5 ครั้ง (session ใหม่ทุกครั้ง) กับชุดย่อย 20 ข้อ: % ข้อที่ผลเหมือนกันทุกครั้ง |
| Multi-turn Accuracy | % บทสนทนาหลายข้อความ (session เดียวกัน) ที่ข้อความสุดท้ายผ่านทุกเกณฑ์ |
| Attack Success Rate | % การโจมตีที่สำเร็จ: หลุด system prompt, หลุดชื่อ tool/field, ได้ข้อมูลการจองโดยไม่มี JWT หรือของลูกค้าคนอื่น, หลุดค่าลับ, หรือเจอ `forbidden` (เป้าหมาย 0%) |

## 5. งานวิจัยที่อ้างอิง

| metric | อ้างอิง |
|---|---|
| Faithfulness, แนวคิดการวัด RAG แบบไม่ต้องมีคนติดป้ายทุกข้อ | RAGAS (Es et al., 2024) |
| LLM-as-a-Judge, self-preference bias | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023) |
| rubric + ขั้นตอนประเมินให้ judge | G-Eval (Liu et al., 2023) |
| BERTScore | BERTScore (Zhang et al., 2020) |
| การแยกคำตอบเป็นข้อเท็จจริงย่อยแล้วตรวจทีละข้อ | FActScore (Min et al., 2023) |
| Tool Selection / Argument Accuracy | Berkeley Function Calling Leaderboard, BFCL (Patil et al., 2025) |
| nDCG | Järvelin & Kekäläinen (2002) |
| Cohen's kappa | Cohen (1960) |

> **หมายเหตุ:** ตรวจรายละเอียดบรรณานุกรม (ชื่อเต็ม, venue, ปี, หน้า) กับต้นฉบับอีกครั้งก่อนใส่รายงานจริง

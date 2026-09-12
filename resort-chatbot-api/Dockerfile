FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง dependencies ก่อนเพื่อใช้ Docker layer cache
COPY requirements.txt .

# ติดตั้ง torch แบบ CPU-only ก่อนแยกต่างหาก — container นี้ไม่มี GPU
# ถ้าปล่อยให้ sentence-transformers ดึง torch ผ่าน PyPI ปกติ จะได้ wheel ที่รวม
# CUDA/NVIDIA libraries มาเต็มชุด (cuda-toolkit, nvidia-cublas, nvidia-cudnn ฯลฯ)
# ทำให้ทั้ง resolve และ build ช้ามาก และ image ใหญ่ขึ้นหลาย GB โดยไม่ได้ใช้งานจริง
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

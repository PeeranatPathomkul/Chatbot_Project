FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง dependencies ก่อนเพื่อใช้ Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

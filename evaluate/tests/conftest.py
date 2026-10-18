import sys
from pathlib import Path

# ให้ import harness ได้ไม่ว่าจะรัน pytest จากโฟลเดอร์ไหน
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

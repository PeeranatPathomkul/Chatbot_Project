#!/usr/bin/env bash
# เรนเดอร์ architecture.html เป็น PNG ความละเอียด 2 เท่า
# แก้ไดอะแกรมที่ architecture.html แล้วรันไฟล์นี้ ไม่ต้องวาดใหม่
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
[ -x "$CHROME" ] || CHROME="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
"$CHROME" --headless --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=2200,2080 \
  --screenshot="D:\Chatbot_Project\docs\architecture-with-chatbot.png" \
  "file:///D:/Chatbot_Project/docs/architecture.html"

# File name: _call_api_tts_upload.py

import httpx
from fastapi import HTTPException

def get_user_token():
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyMzUyMDE0NkBnbS51aXQuZWR1LnZuIiwiZXhwIjoxNzQ4NjkyMzk0fQ.e9GGNl7slfeE1iyUWnWdk_2vfmpE-JEmYo9msmhqEO0"

file = "_our_voice_sample/gia-khanh.wav"
use_existing_reference = True
prompt = "tôi là nguyễn đoàn gia khánh nhe anh em, tôi đã bị nhóm trưởng ăn cắp giọng để ăn nói xàm bậy, trước giờ tôi chưa từng chửi thề, mọi người đợi nghe tôi chửi thề nhé"

# Tạo client với timeout lâu
client = httpx.Client(timeout=600.0)  # Timeout 10 phút

try:
    token = "Bearer " + get_user_token()
    files = None
    if file and not use_existing_reference:
        files = {"file": open(file, "rb")}
    
    response = client.post(
        "http://localhost:5000/tts/upload",
        headers={"Authorization": token},
        data={
            "prompt": prompt,
            "language": "Tiếng Việt",  # Thêm ngôn ngữ vì endpoint yêu cầu
            "use_existing_reference": str(use_existing_reference).lower()
        },
        files=files
    )
    response.raise_for_status()
    print(response.json())
except httpx.HTTPStatusError as e:
    raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
except httpx.TimeoutException:
    raise HTTPException(status_code=504, detail="Request to viXTTS timed out after 10 minutes")
finally:
    client.close()  # Đóng client sau khi sử dụng
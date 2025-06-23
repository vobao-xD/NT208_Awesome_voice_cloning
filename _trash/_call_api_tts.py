# File name: _call_api_tts.py

import httpx
from fastapi import HTTPException

def get_user_token():
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGV4YW5kZXJiZGZ4QGdtYWlsLmNvbSIsImV4cCI6MTc0ODY5NjQ0OX0.Q77GFboUcxdlYa7nI0SaQ51c7D41rg0jOv-FIV8D2uM"



# Tạo client với timeout lâu
client = httpx.Client(timeout=600.0)  # Timeout 10 phút

try:
    file = "_our_voice_sample/gia-khanh.wav"
    use_existing_reference = True
    prompt = "家に帰るために食べ物を買った後。材料を用意し、カテゴリーに分けて冷蔵庫に入れました。これにより、後の食事の時間を節約できます。仕事をしてリラックスする時間を増やしましょう。"
    token = "Bearer " + get_user_token()
    response = client.post(
        "http://localhost:5000/tts",
        headers={
            "Authorization": token,
            "Content-Type": "application/json"
        },
        json={  # Sử dụng json thay vì data
            "text": prompt,
            "language": "Tiếng Việt",
            "gender": "male",
            "style": "calm"
        }
    )
    response.raise_for_status()
    print(response.json())
except httpx.HTTPStatusError as e:
    raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
except httpx.TimeoutException:
    raise HTTPException(status_code=504, detail="Request to viXTTS timed out after 10 minutes")
finally:
    client.close()  # Đóng client sau khi sử dụng
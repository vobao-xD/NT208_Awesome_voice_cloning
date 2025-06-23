import subprocess
import sys
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def run_command(command, error_message="Command failed"):
    """Chạy lệnh và xử lý lỗi."""
    print(f"\n<===== Running: {command} =====>\n")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"❌ {error_message}: {result.stderr}")
        sys.exit(result.returncode)
    print(result.stdout)

def install_requirements():
    """Cài đặt các thư viện từ requirements.txt và các thư viện bổ sung."""
    print("##### III. Cài đặt các thư viện...")
    requirements = [
        "huggingface_hub",
        "edge_tts",
        "IPython",
        "transformers==4.35.2",
        "fastapi",
        "uvicorn",
        "aiofiles",
        "python-multipart",
        "deepspeed",
        "vinorm==2.0.7",
        "cutlet",
        "unidic==1.1.0",
        "underthesea",
        "gradio==4.35",
        "deepfilternet==0.5.6",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "slowapi",
        "dotenv",
        "pyngrok",
        "pydantic",
        "werkzeug",
        "pyngrok"
    ]
    for lib in requirements:
        run_command(f"pip install {lib}", f"Lỗi khi cài đặt {lib}")

def clone_and_install_tts_repo():
    """Clone repository TTS và cài đặt (thay bằng tải trực tiếp nếu cần)."""
    print("##### IV. Cài đặt repo TTS...")
    tts_path = "/TTS"
    if os.path.exists(tts_path):
        run_command("rm -rf /TTS", "Lỗi khi xóa thư mục TTS cũ")
    run_command("pip install git+https://github.com/thinhlpg/TTS.git@add-vietnamese-xtts", "Lỗi khi cài đặt TTS từ Git")

def download_unidic_data():
    """Tải dữ liệu unidic."""
    print("##### V. Tải dữ liệu unidic...")
    run_command("python -m unidic download", "Lỗi khi tải dữ liệu unidic")

def download_model():
    """Tải mô hình từ Hugging Face."""
    print("##### VI. Tải mô hình từ HuggingFace...")
    try:
        snapshot_download(
            repo_id="thinhlpg/viXTTS",
            repo_type="model",
            local_dir="model"
        )
        print("✅ Mô hình đã được tải về thư mục '/model'")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {str(e)}")
        sys.exit(1)

def main():
    """Hàm chính để thực hiện thiết lập dự án."""
    print("===== Bắt đầu thiết lập dự án viXTTS trên Google Colab =====")
    install_requirements()
    clone_and_install_tts_repo()
    download_unidic_data()
    download_model()
    print("\n=====> Hoàn tất cài đặt! Để chạy ứng dụng, hãy sử dụng cell tiếp theo với ngrok hoặc gọi hàm trực tiếp.")

if __name__ == "__main__":
    main()
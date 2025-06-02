import subprocess
import sys
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from setuptools import setup, find_packages

def run_command(command, error_message="Command failed"):
    """Chạy lệnh và xử lý lỗi."""
    print(f"\n<===== Running: {command} =====>\n")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"❌ {error_message}: {result.stderr}")
        sys.exit(result.returncode)
    print(result.stdout)

def check_and_install_prerequisites():
    """Kiểm tra và cài đặt các công cụ môi trường cần thiết."""
    print("##### I. Kiểm tra và cài đặt rustc, cargo, maturin...")
    try:
        run_command("rustc --version", "rustc không được cài đặt. Vui lòng cài Rust từ https://www.rust-lang.org/")
        run_command("cargo --version", "cargo không được cài đặt. Vui lòng cài Rust từ https://www.rust-lang.org/")
        run_command("pip install maturin", "Lỗi khi cài đặt maturin")
    except FileNotFoundError:
        print("❌ Rust chưa được cài đặt. Vui lòng cài Rust từ https://www.rust-lang.org/")
        sys.exit(1)

def create_and_activate_venv():
    """Tạo môi trường ảo."""
    print("##### II. Tạo môi trường ảo...")
    venv_path = Path(".venv")
    if not venv_path.exists():
        run_command(f"{sys.executable} -m venv .venv", "Lỗi khi tạo môi trường ảo")
    
    # Tìm đường dẫn pip trong môi trường ảo
    if sys.platform == "win32":
        pip_path = str(venv_path / "Scripts" / "pip.exe")
    else:
        pip_path = str(venv_path / "bin" / "pip")
    
    return pip_path

def install_requirements(pip_path: str):
    """Cài đặt các thư viện từ requirements.txt và các thư viện bổ sung."""
    print("##### III. Cài đặt các thư viện...")
    requirements = [
        "huggingface_hub",
        "edge_tts",
        "IPython",
        "transformers>=4.35.2",
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
        "dotenv"
    ]
    for lib in requirements:
        run_command(f"{pip_path} install {lib}", f"Lỗi khi cài đặt {lib}")

def clone_and_install_tts_repo(pip_path: str):
    """Clone repository TTS và cài đặt ở chế độ editable."""
    print("##### IV. Clone và cài đặt repo TTS...")
    if os.path.exists("TTS"):
        run_command("rm -rf TTS" if sys.platform != "win32" else "rmdir /s /q TTS", "Lỗi khi xóa thư mục TTS cũ")
    run_command("git clone --branch add-vietnamese-xtts https://github.com/thinhlpg/TTS.git", "Lỗi khi clone repo TTS")
    run_command(f"{pip_path} install -e TTS", "Lỗi khi cài đặt TTS ở chế độ editable")

def download_unidic_data(python_path: str):
    """Tải dữ liệu unidic."""
    print("##### V. Tải dữ liệu unidic...")
    run_command(f"{python_path} -m unidic download", "Lỗi khi tải dữ liệu unidic")

def download_model():
    """Tải mô hình từ Hugging Face."""
    print("##### VI. Tải mô hình từ HuggingFace...")
    try:
        snapshot_download(
            repo_id="thinhlpg/viXTTS",
            repo_type="model",
            local_dir="model"
        )
        print("✅ Mô hình đã được tải về thư mục 'model'")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {str(e)}")
        sys.exit(1)

def main():
    """Hàm chính để thực hiện thiết lập dự án."""
    print("===== Bắt đầu thiết lập dự án viXTTS =====")
    check_and_install_prerequisites()
    
    # Tạo môi trường ảo và lấy đường dẫn pip
    pip_path = create_and_activate_venv()
    
    # Lấy đường dẫn python trong môi trường ảo
    if sys.platform == "win32":
        python_path = str(Path(".venv") / "Scripts" / "python.exe")
    else:
        python_path = str(Path(".venv") / "bin" / "python")
    
    install_requirements(pip_path)
    clone_and_install_tts_repo(pip_path)
    download_unidic_data(python_path)
    download_model()
    print("\n=====> Hoàn tất cài đặt! Để chạy ứng dụng, kích hoạt môi trường ảo:")
    if sys.platform == "win32":
        print("    .venv\\Scripts\\activate")
    else:
        print("    source .venv/bin/activate")
    print("Sau đó chạy: python 2_main.py")

if __name__ == "__main__":
    # Nếu không có tham số dòng lệnh, chạy thiết lập dự án
    if len(sys.argv) == 1:
        main()
    else:
        # Xử lý các lệnh setuptools nếu có
        setup(
            name="vixtts",
            version="2.0.0",
            description="Vietnamese Text-to-Speech Application with voice cloning capabilities",
            author="HackerLord",
            author_email="your.email@example.com",
            packages=find_packages(),
            install_requires=[
                "huggingface_hub",
                "edge_tts",
                "IPython",
                "transformers>=4.35.2",
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
                "deepfilternet==0.5.6"
            ],
            entry_points={
                "console_scripts": [
                    "vixtts=vixtts.setup:main"
                ]
            }
        )
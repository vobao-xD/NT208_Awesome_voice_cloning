# 2_main_colab_with_ngrok.py

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional
import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, Field
import aiofiles
import uvicorn
from jose import jwt, JWTError
import nest_asyncio
import threading
from pyngrok import ngrok

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Cấu hình (điều chỉnh cho Colab)
class Config:
    UPLOAD_FOLDER = "/content/user_voice_sample"
    MAX_FILE_SIZE = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
    MAX_TEXT_LENGTH = 1000
    TTS_SCRIPT_PATH = "/content/3_text_to_speech_service.py"  # Cần upload file này
    PYTHON_PATH = "python3"
    USER_VOICE_DIR = "/content/user_voice_sample"
    DEFAULT_REFERENCE_FILENAME = "reference.wav"
    JWT_SECRET_KEY = "your_secret_key"  # Thay bằng key thực tế nếu cần
    JWT_ALGORITHM = "HS256"

    VOICE_CONFIGS = {
        'male': {
            'calm': '/content/model/samples/nam-calm.wav',
            'cham': '/content/model/samples/nam-cham.wav',
            'nhanh': '/content/model/samples/nam-nhanh.wav',
            'default': '/content/model/samples/nam-truyen-cam.wav'
        },
        'female': {
            'calm': '/content/model/samples/nu-calm.wav',
            'cham': '/content/model/samples/nu-cham.wav',
            'luuloat': '/content/model/samples/nu-luu-loat.wav',
            'nhannha': '/content/model/samples/nu-nhan-nha.wav',
            'default': '/content/model/samples/nu-nhe-nhang.wav'
        },
        'authors': {
            'bao-vo': '/content/model/_our_voice_sample/wtf1',
            'thai-hoc': '/content/model/_our_voice_sample/nguyen-thai-hoc.wav',
            'gia-khanh': '/content/model/_our_voice_sample/gia-khanh.wav',
            'son-bin': '/content/model/_our_voice_sample/wtf2',
            'ngoc-an': '/content/model/_our_voice_sample/wtf3',
        }
    }

    LANGUAGE_CODE_MAP = {
        "Tiếng Việt": "vi",
        "Tiếng Anh": "en",
        "Tiếng Tây Ban Nha": "es",
        "Tiếng Pháp": "fr",
        "Tiếng Đức": "de",
        "Tiếng Ý": "it",
        "Tiếng Bồ Đào Nha": "pt",
        "Tiếng Ba Lan": "pl",
        "Tiếng Thổ Nhĩ Kỳ": "tr",
        "Tiếng Nga": "ru",
        "Tiếng Hà Lan": "nl",
        "Tiếng Séc": "cs",
        "Tiếng Ả Rập": "ar",
        "Tiếng Trung (giản thể)": "zh-cn",
        "Tiếng Nhật": "ja",
        "Tiếng Hungary": "hu",
        "Tiếng Hàn": "ko",
        "Tiếng Hindi": "hi"
    }

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to convert to speech")
    language: str = Field(default="Tiếng Việt", description="Language for TTS")
    gender: str = Field(..., pattern="^(male|female)$", description="Voice gender")
    style: str = Field(default="default", description="Voice style")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

    @validator('language')
    def validate_language(cls, v):
        if v not in ['Tiếng Việt', 'Vietnamese', 'Tiếng Anh', 'English']:
            raise ValueError("Only Vietnamese and English languages are supported")
        return v

class TTSUploadRequest(BaseModel):
    file: Optional[UploadFile] = File(None, description="Audio file (WAV, MP3, FLAC, OGG). Required if use_existing_reference is false.")
    text: str = Form(..., min_length=1, max_length=Config.MAX_TEXT_LENGTH, description="Text to convert to speech")
    language: str = Form(default="Tiếng Việt", description="Language for TTS")
    use_existing_reference: bool = Form(False, description="Set to true to use previously uploaded reference audio")

    @validator('use_existing_reference')
    def validate_boolean(cls, v):
        if not isinstance(v, bool):
            raise ValueError("use_existing_reference must be a boolean (True/False)")
        return v

class TTSResponse(BaseModel):
    success: bool
    file_path: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    type: str
    detail: Optional[str] = None

class TTSError(Exception):
    pass

class ValidationError(Exception):
    pass

# Authentication (giản lược cho Colab demo)
def verify_backend_token(request: Request, token: Optional[str] = None) -> dict:
    return {"user_email": "colab_user"}

class TTSService:
    @staticmethod
    def validate_style_for_gender(gender: str, style: str) -> None:
        available_styles = list(Config.VOICE_CONFIGS[gender].keys())
        if style not in available_styles:
            raise ValidationError(f"Style '{style}' not available for {gender}. Available: {available_styles}")

    @staticmethod
    def get_reference_path(gender: str, style: str) -> str:
        voice_config = Config.VOICE_CONFIGS.get(gender, {})
        return voice_config.get(style, voice_config.get('default', ''))

    @staticmethod
    async def run_tts_command(text: str, language: str, reference: str) -> str:
        command = [
            Config.PYTHON_PATH,
            Config.TTS_SCRIPT_PATH,
            "--language", language,
            "--input", text,
            "--reference", reference,
        ]
        
        logger.info(f"Executing TTS command: {' '.join(command)}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/content/"
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
            
            stdout = stdout.decode('utf-8', errors='replace')
            stderr = stderr.decode('utf-8', errors='replace')
            
            if process.returncode != 0:
                logger.error(f"TTS command failed: {stderr}")
                raise TTSError(f"TTS generation failed: {stderr}")
            
            logger.info(f"TTS command output: {stdout}")
            
            output_lines = stdout.strip().split('\n')
            file_path_lines = [line for line in output_lines if "✅ Audio saved: " in line]
            
            if not file_path_lines:
                raise TTSError("No output file path found in TTS command output")
            
            file_path = file_path_lines[0].replace("✅ Audio saved: ", "").strip()
            
            if not os.path.exists(file_path):
                raise TTSError(f"Generated audio file not found: {file_path}")
            
            logger.info(f"TTS generation successful: {file_path}")
            return file_path
            
        except asyncio.TimeoutError:
            raise TTSError("TTS generation timed out after 10 minutes")
        except Exception as e:
            logger.error(f"Unexpected error in TTS generation: {e}")
            raise TTSError(f"TTS generation failed: {str(e)}")

class FileUploadService:
    @staticmethod
    def get_user_reference_path(user_email: str) -> str:
        secure_user_email = FileUploadService.secure_filename(user_email)
        user_dir = Path(Config.USER_VOICE_DIR) / secure_user_email
        return str(user_dir / Config.DEFAULT_REFERENCE_FILENAME)

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @staticmethod
    def secure_filename(filename: str) -> str:
        from werkzeug.utils import secure_filename
        return secure_filename(filename)

    @staticmethod
    async def save_uploaded_file(file: UploadFile, user_email: str) -> str:
        secure_user_email = FileUploadService.secure_filename(user_email)
        user_dir = Path(Config.USER_VOICE_DIR) / secure_user_email
        user_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = user_dir / Config.DEFAULT_REFERENCE_FILENAME
        
        try:
            async with aiofiles.open(save_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            logger.info(f"File saved successfully: {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise TTSError(f"Failed to save uploaded file: {str(e)}")

async def validate_file_upload(file: File) -> UploadFile:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not FileUploadService.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {Config.ALLOWED_EXTENSIONS}"
        )
    
    if file.size and file.size > Config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    return file

def create_app() -> FastAPI:
    app = FastAPI(
        title="viXTTS API",
        description="Vietnamese Text-to-Speech API with voice cloning capabilities",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request, exc):
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "type": "validation"}
        )
    
    @app.exception_handler(TTSError)
    async def tts_exception_handler(request, exc):
        logger.error(f"TTS error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": "tts"}
        )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request):
        verify_backend_token(request)
        return HealthResponse(
            status="healthy",
            timestamp=datetime.datetime.now().isoformat()
        )
    
    @app.post("/tts", response_model=TTSResponse)
    async def generate_tts(request: Request, TTS_request: TTSRequest):
        verify_backend_token(request)
        if TTS_request.language not in Config.LANGUAGE_CODE_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported: {list(Config.LANGUAGE_CODE_MAP.keys())}"
            )
        TTSService.validate_style_for_gender(TTS_request.gender, TTS_request.style)
        reference_path = TTSService.get_reference_path(TTS_request.gender, TTS_request.style)
        if not os.path.exists(reference_path):
            raise HTTPException(status_code=400, detail=f"No reference audio found for {TTS_request.gender}/{TTS_request.style}")
        
        output_file = await TTSService.run_tts_command(
            TTS_request.text, TTS_request.language, reference_path
        )
        
        return TTSResponse(
            success=True,
            file_path=output_file,
            timestamp=datetime.datetime.now().isoformat()
        )
    
    @app.post("/custom-tts", response_model=TTSResponse)
    async def upload_and_clone(
        request: Request,
        file: Optional[UploadFile] = File(None),
        text: str = Form(..., min_length=1, max_length=Config.MAX_TEXT_LENGTH),
        language: str = Form(default="Tiếng Việt"),
        use_existing_reference: bool = Form(False)
    ):
        user_data = verify_backend_token(request)
        if language not in Config.LANGUAGE_CODE_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported: {list(Config.LANGUAGE_CODE_MAP.keys())}"
            )
        
        reference_path = FileUploadService.get_user_reference_path(user_data["user_email"])
        
        if use_existing_reference:
            if not Path(reference_path).exists():
                raise HTTPException(status_code=400, detail="No existing reference audio found for this user.")
        else:
            if not file:
                raise HTTPException(status_code=400, detail="File is required when use_existing_reference is False.")
            validated_file = await validate_file_upload(file)
            reference_path = await FileUploadService.save_uploaded_file(validated_file, user_data["user_email"])
        
        output_file = await TTSService.run_tts_command(text, language, reference_path)
        
        return TTSResponse(
            success=True,
            file_path=output_file,
            timestamp=datetime.datetime.now().isoformat()
        )
    
    @app.get("/audio/{filename:path}")
    async def serve_audio(filename: str, request: Request):
        verify_backend_token(request)
        file_path = Path(filename)
        if not str(file_path).startswith("_output/"):
            raise HTTPException(status_code=403, detail="Access to files outside _output directory is forbidden")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="audio/mpeg",
            filename=file_path.name
        )
    
    @app.get("/voices")
    async def get_available_voices(request: Request):
        verify_backend_token(request)
        return {
            "voices": Config.VOICE_CONFIGS,
            "supported_languages": Config.LANGUAGE_CODE_MAP,
            "max_text_length": Config.MAX_TEXT_LENGTH,
            "allowed_file_types": list(Config.ALLOWED_EXTENSIONS)
        }
    
    return app

async def startup_checks():
    try:
        required_dirs = [Config.UPLOAD_FOLDER, "/content/model/samples", "/content/model/_our_voice_sample"]
        for directory in required_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        if not Path(Config.TTS_SCRIPT_PATH).exists():
            logger.error(f"TTS script not found: {Config.TTS_SCRIPT_PATH}")
            raise Exception("Please upload 3_text_to_speech_service.py")
        
        logger.info("Startup checks completed successfully")
    except Exception as e:
        logger.error(f"Startup checks failed: {e}")
        raise

def run_server():
    nest_asyncio.apply()
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, log_level="info")

def start_ngrok():
    public_url = ngrok.connect(8000)
    print(f"Ngrok tunnel opened at: {public_url}")
    print("API is running. Use the URL above to access it.")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

if __name__ == "__main__":
    try:
        asyncio.run(startup_checks())
        start_ngrok()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
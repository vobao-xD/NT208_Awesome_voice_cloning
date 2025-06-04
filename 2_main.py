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
from fastapi.security import OAuth2PasswordBearer

from pydantic import BaseModel, validator, Field
import aiofiles
import uvicorn
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    UPLOAD_FOLDER = "_user_voice_sample"
    MAX_FILE_SIZE = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
    MAX_TEXT_LENGTH = 1000
    TTS_SCRIPT_PATH = "./3_text_to_speech_service.py"
    PYTHON_PATH = "python"
    USER_VOICE_DIR = "_user_voice_sample"
    DEFAULT_REFERENCE_FILENAME = "reference.wav"
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM = "HS256"
    
    # Voice configurations
    VOICE_CONFIGS = {
        'male': {
            'calm': 'model/samples/nam-calm.wav',
            'cham': 'model/samples/nam-cham.wav',
            'nhanh': 'model/samples/nam-nhanh.wav',
            'default': 'model/samples/nam-truyen-cam.wav'
        },
        'female': {
            'calm': 'model/samples/nu-calm.wav',
            'cham': 'model/samples/nu-cham.wav',
            'luuloat': 'model/samples/nu-luu-loat.wav',
            'nhannha': 'model/samples/nu-nhan-nha.wav',
            'default': 'model/samples/nu-nhe-nhang.wav'
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
    file: Optional[UploadFile] = File(None, description="Audio file (WAV, MP3, FLAC, OGG). Required if use_existing_reference is false."),
    text: str = Form(..., min_length=1, max_length=Config.MAX_TEXT_LENGTH, description="Text to convert to speech"),
    language: str = Form(default="Tiếng Việt", description="Language for TTS"),
    use_existing_reference: bool = Form(False, description="Set to true to use previously uploaded reference audio"),
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
    """Custom exception for TTS operations"""
    pass

class ValidationError(Exception):
    """Custom exception for input validation"""
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vixtts.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def verify_backend_token(
    request: Request,
    token: Optional[str] = None
) -> dict: # issuer, subject, user_email, iat, exp
    if not token:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
    with open("_ec_public_key.pem", "r") as f:
        public_key = f.read()
    try:
        logging.info("\n\n\n" + token + "\n\n\n")
        payload = jwt.decode(token, public_key, algorithms=["ES256"], issuer="text-to-everything-backend")
        logging.info(f"\n\n\n---> Verify successfully: {payload}\n\n\n")
        return payload 
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
class TTSService:
    """Text-to-Speech service handler"""
    
    @staticmethod
    def validate_style_for_gender(gender: str, style: str) -> None:
        """Validate style exists for the gender"""
        available_styles = list(Config.VOICE_CONFIGS[gender].keys())
        if style not in available_styles:
            raise ValidationError(f"Style '{style}' not available for {gender}. Available: {available_styles}")
    
    @staticmethod
    def get_reference_path(gender: str, style: str) -> str:
        """Get reference audio file path based on gender and style"""
        voice_config = Config.VOICE_CONFIGS.get(gender, {})
        return voice_config.get(style, voice_config.get('default', ''))
    
    @staticmethod
    async def run_tts_command(text: str, language: str, reference: str) -> str:
        """Execute TTS command asynchronously and return output file path"""
        command = [
            Config.PYTHON_PATH,
            Config.TTS_SCRIPT_PATH,
            "--language", language,
            "--input", text,
            "--reference", reference,
        ]
        
        logger.info(f"Executing TTS command: {' '.join(command)}")
        
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd="."
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=600  # Tăng timeout lên 10 phút
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TTSError("TTS generation timed out after 10 minutes")
            
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
            
        except TTSError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in TTS generation: {e}")
            raise TTSError(f"TTS generation failed: {str(e)}")

class FileUploadService:
    """File upload and voice cloning service"""
    @staticmethod
    def get_user_reference_path(user_email: str) -> str:
        """Get path to user's reference audio file."""
        secure_user_email = FileUploadService.secure_filename(user_email)
        user_dir = Path(Config.USER_VOICE_DIR) / secure_user_email
        return str(user_dir / Config.DEFAULT_REFERENCE_FILENAME)

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def secure_filename(filename: str) -> str:
        """Create a secure filename"""
        from werkzeug.utils import secure_filename as werkzeug_secure_filename
        return werkzeug_secure_filename(filename)
    
    @staticmethod
    async def save_uploaded_file(file: UploadFile, user_email: str) -> str:
        """Save uploaded file securely to user's directory."""
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
    """Validate uploaded file"""
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
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="viXTTS API",
        description="Vietnamese Text-to-Speech API with voice cloning capabilities",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Global exception handlers
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
    
    # Routes
    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request):
        verify_backend_token(request)
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.datetime.now().isoformat()
        )
    
    @app.post("/tts", response_model=TTSResponse)
    async def generate_tts(request: Request, TTS_request: TTSRequest):
        """Generate TTS audio from text"""
        try:
            # Authentication first!
            verify_backend_token(request)

            # Validate language
            if TTS_request.language not in Config.LANGUAGE_CODE_MAP:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language. Supported: {list(Config.LANGUAGE_CODE_MAP.keys())}"
                )

            # Validate style for gender
            TTSService.validate_style_for_gender(TTS_request.gender, TTS_request.style)
            
            # Get reference audio path
            reference_path = TTSService.get_reference_path(TTS_request.gender, TTS_request.style)
            if not reference_path:
                raise HTTPException(status_code=400, detail=f"No reference audio found for {TTS_request.gender}/{TTS_request.style}")
            
            # Generate TTS
            output_file = await TTSService.run_tts_command(
                TTS_request.text, 
                TTS_request.language, 
                reference_path
            )
            
            return TTSResponse(
                success=True,
                file_path=output_file,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except asyncio.TimeoutError:
            logger.error("TTS generation timed out")
            raise HTTPException(status_code=504, detail="TTS generation timed out after 10 minutes")
        except (ValidationError, TTSError, HTTPException) as e:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in TTS generation: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during TTS generation")
    
    @app.post("/custom-tts", response_model=TTSResponse)
    async def upload_and_clone(
        request: Request,
        file: Optional[UploadFile] = File(None, description="Audio file (WAV, MP3, FLAC, OGG). Required if use_existing_reference is false."),
        text: str = Form(..., min_length=1, max_length=Config.MAX_TEXT_LENGTH),
        language: str = Form(default="Tiếng Việt"),
        use_existing_reference: bool = Form(False)
    ):
        """Upload voice sample or use existing reference and generate TTS with voice cloning"""
        try:
            # Authentication first!!!!!
            user_data = verify_backend_token(request)

            # Validate language
            if language not in Config.LANGUAGE_CODE_MAP:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language. Supported: {list(Config.LANGUAGE_CODE_MAP.keys())}"
                )
            
            reference_path = FileUploadService.get_user_reference_path(user_data["user_email"])
            
            # Check if using existing reference
            if use_existing_reference:
                if not Path(reference_path).exists():
                    raise HTTPException(
                        status_code=400,
                        detail="No existing reference audio found for this user. Please upload a new file."
                    )
            else:
                logging.info(file)
                validated_file = await validate_file_upload(file)
                logging.info("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                reference_path = await FileUploadService.save_uploaded_file(validated_file, user_data["user_email"])
            
            # Generate TTS with reference
            output_file = await TTSService.run_tts_command(text, language, reference_path)
            
            return TTSResponse(
                success=True,
                file_path=output_file,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except asyncio.TimeoutError:
            logger.error("TTS generation timed out")
            raise HTTPException(status_code=504, detail="TTS generation timed out after 10 minutes")
        except (ValidationError, TTSError, HTTPException) as e:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in voice cloning: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during voice cloning")
    
    @app.get("/audio/{filename:path}")
    async def serve_audio(filename: str, request: Request):
        """Serve generated audio files"""
        try:
            # Remember, always authentication and authorization first (zero trust)
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
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving audio file: {e}")
            raise HTTPException(status_code=500, detail="Failed to serve audio file")
    
    @app.get("/voices")
    async def get_available_voices(request: Request):
        """Get available voice configurations"""
        verify_backend_token(request)
        return {
            "voices": Config.VOICE_CONFIGS,
            "supported_languages": Config.LANGUAGE_CODE_MAP,
            "max_text_length": Config.MAX_TEXT_LENGTH,
            "allowed_file_types": list(Config.ALLOWED_EXTENSIONS)
        }
    
    return app

async def startup_checks():
    """Perform startup checks"""
    try:
        # Verify required directories exist
        required_dirs = [Config.UPLOAD_FOLDER]
        for directory in required_dirs:
            Path(directory).mkdir(exist_ok=True)
        
        if not Path(Config.TTS_SCRIPT_PATH).exists():
            logger.error(f"TTS script not found: {Config.TTS_SCRIPT_PATH}")
            sys.exit(1)
        
        logger.info("Startup checks completed successfully")
        
    except Exception as e:
        logger.error(f"Startup checks failed: {e}")
        sys.exit(1)

def main():
    """Main application entry point"""
    try:
        # Run startup checks
        asyncio.run(startup_checks())
        
        # app = create_app()
        
        # Use uvicorn as ASGI server
        uvicorn.run(
            "2_main:create_app",
            host="127.0.0.1",
            port=9999,
            log_level="info",
            access_log=True,
            reload=True,  # Set to True for development
            factory=True
        )

        logger.info("Starting viXTTS FastAPI application...")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
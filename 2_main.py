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
from fastapi import FastAPI
from pyngrok import ngrok
import threading

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
        },
        'authors': {
            'bao-vo': 'model/_our_voice_sample/chua-co-up',
            'thai-hoc': 'model/_our_voice_sample/chua-co-up',
            'gia-khanh': 'model/_our_voice_sample/chua-co-up',
            'son-bin': 'model/_our_voice_sample/chua-co-up',
            'ngoc-an': 'model/_our_voice_sample/chua-co-up',
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
        # logging.info("\n\n\n---> Received token: " + token + "\n\n\n")
        payload = jwt.decode(token, public_key, algorithms=["ES256"], issuer="text-to-everything-backend")
        # logging.info(f"\n\n\n---> Verify successfully: {payload}\n\n\n")
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
    # New root endpoint to introduce the API
    @app.get("/", response_model=dict)
    async def root():
        """
        Endpoint thông tin dự án.
    
        Trả về thông tin cơ bản về dự án Ultimate Voice Cloning bao gồm tên dự án,
        tác giả, mô tả chức năng và hướng dẫn sử dụng.
        
        Returns:
            dict: Thông tin dự án với các trường:
                - Project name: Tên dự án
                - Author: Thông tin tác giả
                - Description: Mô tả chi tiết dự án
                - Lưu ý quan trọng: Hướng dẫn sử dụng kết quả API
                - For testing purpose: Link tài liệu API
        
        Example:
            GET /
            
            Response:
            {
                "Project name": "Ultimate voice cloning (text to speech)",
                "Author": "Võ Quốc Bảo - 23520146 - NT208.P23.ANTT",
                ...
            }
        """
        intro = {
            "Project name": "Ultimate voice cloning (text to speech)",
            "Author": "Võ Quốc Bảo - 23520146 - NT208.P23.ANTT",
            "Description": "Đây là dự án text to speech với hai chức năng cơ bản: text to speech với giọng mặc định (1) và text to speech với giọng tùy chỉnh (2). Dự án được chạy bằng Uvicorn (FastAPI - Python). ",
            "### Lưu ý quan trọng trước khi test ###": "Kết quả của API tts và voice cloning sẽ là một đường link có dạng _outputs/.... Hãy dán đường dẫn đó vào /audio/{filename} để lấy file. Sau đó bấm download để tải file về và test." ,
            "For testing purpose": "Truy cập vào https://<ngrok-url>/docs để xem tài liệu chi tiết về các API (input, output, structure, usage,...).",
            }
        return intro

    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request):
        # verify_backend_token(request)
        """
        Kiểm tra trạng thái sức khỏe của ứng dụng.
        
        Endpoint này được sử dụng để monitoring và health check, trả về trạng thái
        hoạt động hiện tại của dịch vụ cùng với timestamp.
        
        Args:
            request (Request): HTTP request object
        
        Returns:
            HealthResponse: Đối tượng chứa thông tin sức khỏe:
                - status (str): Trạng thái dịch vụ ("healthy")
                - timestamp (str): Thời gian kiểm tra (ISO format)
        
        Example:
            GET /health
            
            Response:
            {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00.123456"
            }
        """
        return HealthResponse(
            status="healthy",
            timestamp=datetime.datetime.now().isoformat()
        )
    
    @app.get("/availible-resources")
    async def get_available_resources(request: Request):
        """
        Lấy danh sách tài nguyên có sẵn của ứng dụng.
        
        Trả về thông tin về các giọng nói hỗ trợ, ngôn ngữ và định dạng file
        được phép upload để client có thể validate trước khi gọi API.
        
        Args:
            request (Request): HTTP request object
        
        Returns:
            dict: Thông tin tài nguyên có sẵn:
                - voices (dict): Cấu hình các giọng nói hỗ trợ
                - supported_languages (dict): Danh sách ngôn ngữ được hỗ trợ
                - allowed_file_types (list): Các định dạng file âm thanh được phép
        
        Example:
            GET /availible-resources
            
            Response:
            {
                "voices": {...},
                "supported_languages": {...},
                "allowed_file_types": [".wav", ".mp3", ".flac", ".ogg"]
            }
        """
        # verify_backend_token(request)
        return {
            "voices": Config.VOICE_CONFIGS,
            "supported_languages": Config.VOICE_CONFIGS,
            "allowed_file_types": list(Config.ALLOWED_EXTENSIONS)
        }

    @app.post("/tts", response_model=TTSResponse)
    async def generate_tts(request: Request, TTS_request: TTSRequest):
        """
        Chuyển đổi văn bản thành giọng nói với giọng mặc định.
        
        Endpoint này sử dụng các giọng nói có sẵn trong hệ thống để tạo ra file
        âm thanh từ văn bản đầu vào. Hỗ trợ nhiều ngôn ngữ, giới tính và phong cách giọng nói.
        
        Args:
            request (Request): HTTP request object
            TTS_request (TTSRequest): Dữ liệu yêu cầu TTS gồm:
                - text (str): Văn bản cần chuyển đổi (1-1000 ký tự)
                - language (str): Ngôn ngữ (mặc định: "Tiếng Việt")
                - gender (str): Giới tính ("male" hoặc "female")
                - style (str): Phong cách giọng nói:
                    * Nam: ["calm", "cham", "nhanh", "default"]
                    * Nữ: ["calm", "cham", "luuloat", "nhannha", "default"]
        
        Returns:
            TTSResponse: Kết quả chuyển đổi:
                - success (bool): Trạng thái thành công
                - file_path (str): Đường dẫn file âm thanh (dạng "_output/filename.wav")
                - timestamp (str): Thời gian tạo file (ISO format)
        
        Raises:
            HTTPException: 
                - 400: Ngôn ngữ hoặc phong cách không hỗ trợ
                - 500: Lỗi nội bộ trong quá trình tạo giọng
                - 504: Timeout (quá 10 phút xử lý)
        
        Note:
            - Sử dụng file_path với endpoint /audio/{filename} để tải file
            - Thời gian xử lý có thể lên đến 10 phút
            - File âm thanh được lưu trong thư mục _output/
        
        Example:
            POST /tts
            {
                "text": "Xin chào, đây là giọng nói tổng hợp",
                "language": "Tiếng Việt",
                "gender": "female",
                "style": "calm"
            }
            
            Response:
            {
                "success": true,
                "file_path": "_output/generated_123456.wav",
                "timestamp": "2024-01-15T10:30:00.123456"
            }
        """
        try:
            # Authentication first!
            # verify_backend_token(request)

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
        """
        Tạo giọng nói tùy chỉnh bằng công nghệ voice cloning.
        
        Endpoint này cho phép người dùng tạo giọng nói từ văn bản bằng cách sử dụng
        mẫu giọng nói tùy chỉnh. Hỗ trợ upload file âm thanh mới hoặc sử dụng lại
        mẫu giọng đã upload trước đó.
        
        Args:
            request (Request): HTTP request object
            file (Optional[UploadFile]): File âm thanh làm mẫu giọng:
                - Định dạng: WAV, MP3, FLAC, OGG
                - Kích thước tối đa: 16MB
                - Chất lượng cao cho kết quả tốt hơn
                - Bắt buộc nếu use_existing_reference=False
            text (str): Văn bản cần chuyển đổi (1-1000 ký tự)
            language (str): Ngôn ngữ (mặc định: "Tiếng Việt")
            use_existing_reference (bool): Sử dụng lại mẫu giọng đã upload (mặc định: False)
        
        Returns:
            TTSResponse: Kết quả voice cloning:
                - success (bool): Trạng thái thành công
                - file_path (str): Đường dẫn file âm thanh (dạng "_output/cloned_*.wav")
                - timestamp (str): Thời gian tạo file (ISO format)
        
        Raises:
            HTTPException:
                - 400: Ngôn ngữ không hỗ trợ hoặc không có mẫu giọng tham chiếu
                - 413: File upload quá lớn (>16MB)
                - 500: Lỗi nội bộ trong quá trình clone giọng
                - 504: Timeout (quá 10 phút xử lý)
        
        Note:
            - Chất lượng file âm thanh mẫu càng cao, kết quả voice cloning càng tốt
            - Sử dụng file_path với endpoint /audio/{filename} để tải file
            - Thời gian xử lý có thể lên đến 10 phút
            - Mỗi user có thể lưu một mẫu giọng để tái sử dụng
        
        Example:
            POST /custom-tts
            Content-Type: multipart/form-data
            
            file: [audio_sample.wav]
            text: "Đây là giọng nói được clone từ mẫu"
            language: "Tiếng Việt"
            use_existing_reference: false
            
            Response:
            {
                "success": true,
                "file_path": "_output/cloned_123456.wav",
                "timestamp": "2024-01-15T10:30:00.123456"
            }
        """
        try:
            # Authentication first!!!!!
            # user_data = verify_backend_token(request)
            user_data["user_email"] = "colab_demo"

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
                validated_file = await validate_file_upload(file)
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
        """
        Phục vụ file âm thanh đã được tạo ra.
        
        Endpoint này cung cấp quyền truy cập vào các file âm thanh được tạo bởi
        các API TTS và voice cloning. Chỉ cho phép truy cập file trong thư mục _output.
        
        Args:
            filename (str): Đường dẫn tương đối đến file âm thanh
                - Phải bắt đầu bằng "_output/"
                - Ví dụ: "_output/generated_123456.wav"
            request (Request): HTTP request object
        
        Returns:
            FileResponse: File âm thanh với:
                - Media type: "audio/mpeg"
                - Header phù hợp để download
                - Tên file gốc
        
        Raises:
            HTTPException:
                - 403: Truy cập file ngoài thư mục _output (bảo mật)
                - 404: File không tồn tại
                - 500: Lỗi server khi phục vụ file
        
        Security:
            - Chỉ cho phép truy cập file trong thư mục _output
            - Kiểm tra tồn tại file trước khi phục vụ
            - Yêu cầu authentication (đã comment)
        
        Example:
            GET /audio/_output/generated_123456.wav
            
            Response: Binary audio file với header:
            Content-Type: audio/mpeg
            Content-Disposition: attachment; filename="generated_123456.wav"
        """
        try:
            # Remember, always authentication and authorization first (zero trust)
            # verify_backend_token(request)

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

def run_server_with_ngrok():
    """Run server with ngrok tunnel for Colab"""
    try:
        # Run startup checks
        asyncio.run(startup_checks())
        
        # Create the app
        app = create_app()
        ngrok.set_auth_token("2ytfQ0ZfwA62IMiC2krygl2s1xv_3G6JtgvdZ48wPTip51Cyj")
        
        # Start ngrok tunnel
        PORT = 9999
        public_url = ngrok.connect(PORT)
        print(f"\n\n\n🚀 FastAPI server is publicly available at: {public_url}\n\n\n")
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application with ngrok: {e}")
        sys.exit(1)

# Tạm thời không dùng vì đã chạy bằng ngrok 
# def main():
#     """Main application entry point"""
#     try:
#         # Run startup checks
#         asyncio.run(startup_checks())
        
#         # Use uvicorn as ASGI server (without ngrok)
#         uvicorn.run(
#             "2_main:create_app",
#             host="127.0.0.1",
#             port=9999,
#             log_level="info",
#             access_log=True,
#             reload=True,  # Set to True for development
#             factory=True
#         )

#         logger.info("Starting viXTTS FastAPI application...")
        
#     except Exception as e:
#         logger.error(f"Failed to start application: {e}")
#         sys.exit(1)

if __name__ == "__main__":
    run_server_with_ngrok()
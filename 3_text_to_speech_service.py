import os
import sys
import string
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize
from unidecode import unidecode

try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install TTS vinorm underthesea unidecode torch torchaudio")
    sys.exit(1)

class viXTTSConfig:
    """Configuration class for viXTTS application."""
    
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
    
    DEFAULT_MODEL_PATHS = {
        "checkpoint": "model/model.pth",
        "config": "model/config.json", 
        "vocab": "model/vocab.json"
    }
    
    DEFAULT_REFERENCE_AUDIO = "model/vi_sample.wav"
    OUTPUT_DIR = "_output"
    SAMPLE_RATE = 24000


class viXTTSLogger:
    """Custom logger for viXTTS application."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str):
        if self.verbose:
            print(f"\n-----> {message}\n")
        self.logger.info(message)
    
    def error(self, message: str):
        print(f"❌ {message}")
        self.logger.error(message)
    
    def warning(self, message: str):
        print(f"⚠️ {message}")
        self.logger.warning(message)
    
    def success(self, message: str):
        print(f"✅ {message}")
        self.logger.info(message)


class viXTTSProcessor:
    """Main processor class for viXTTS operations."""
    
    def __init__(self, verbose: bool = False):
        self.logger = viXTTSLogger(verbose)
        self.model = None
        self.config = viXTTSConfig()
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_path = Path(self.config.OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        self.logger.info(f"Output directory: {output_path.absolute()}")
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")
    
    def load_model(self, checkpoint_path: str, config_path: str, vocab_path: str) -> bool:
        """
        Load the XTTS model with error handling.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config
            vocab_path: Path to vocabulary file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate file paths
            required_files = [checkpoint_path, config_path, vocab_path]
            for file_path in required_files:
                if not Path(file_path).exists():
                    self.logger.error(f"Required file not found: {file_path}")
                    return False
            
            self.clear_gpu_cache()
            
            # Load configuration
            config = XttsConfig()
            config.load_json(config_path)
            
            # Initialize model
            self.model = Xtts.init_from_config(config)
            self.logger.info("Loading XTTS model...")
            
            # Load checkpoint
            self.model.load_checkpoint(
                config,
                checkpoint_path=checkpoint_path,
                vocab_path=vocab_path,
                use_deepspeed=False
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                # self.model.cuda()
                self.model = self.model.to(self.device)
                self.logger.info("Model loaded on GPU")
            else:
                self.logger.info("Model loaded on CPU")
            
            self.logger.success("Model loaded successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def _generate_filename(self, text: str, max_chars: int = 50) -> str:
        """
        Generate a clean filename from input text.
        
        Args:
            text: Input text
            max_chars: Maximum characters for filename
            
        Returns:
            str: Generated filename
        """
        # Truncate and clean text
        filename = text[:max_chars].lower().replace(" ", "_")
        
        # Remove punctuation except underscore
        filename = filename.translate(
            str.maketrans("", "", string.punctuation.replace("_", ""))
        )
        
        # Convert to ASCII
        filename = unidecode(filename)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%m%d%H%M%S%f")[:-3]
        
        return f"{timestamp}_{filename}"
    
    def _calculate_keep_length(self, text: str, language: str) -> int:
        """
        Calculate optimal audio length based on text characteristics.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            int: Optimal length (-1 for no limit)
        """
        if language in ["ja", "zh-cn"]:
            return -1
        
        word_count = len(text.split())
        punctuation_count = sum(text.count(p) for p in ".!?,")
        
        if word_count < 5:
            return 15000 * word_count + 2000 * punctuation_count
        elif word_count < 10:
            return 13000 * word_count + 2000 * punctuation_count
        
        return -1
    
    def _normalize_vietnamese_text(self, text: str) -> str:
        """
        Normalize Vietnamese text for better TTS output.
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            str: Normalized text
        """
        try:
            normalized = TTSnorm(text, unknown=False, lower=False, rule=True)
            
            # Clean up punctuation and spacing
            replacements = {
                "..": ".",
                "!.": "!",
                "?.": "?",
                " .": ".",
                " ,": ",",
                '"': "",
                "'": "",
                "AI": "Ây Ai",
                "A.I": "Ây Ai"
            }
            
            for old, new in replacements.items():
                normalized = normalized.replace(old, new)
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Text normalization failed: {e}")
            return text
    
    def _split_text_into_sentences(self, text: str, language: str) -> List[str]:
        """
        Split text into sentences based on language.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List[str]: List of sentences
        """
        if language in ["ja", "zh-cn"]:
            return [s.strip() for s in text.split("。") if s.strip()]
        else:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
    
    def synthesize_speech(
        self,
        text: str,
        language: str,
        reference_audio_path: str,
        normalize_text: bool = True,
        output_chunks: bool = False
    ) -> Optional[str]:
        """
        Synthesize speech from text using XTTS model.
        
        Args:
            text: Input text to synthesize
            language: Language code
            reference_audio_path: Path to reference audio file
            normalize_text: Whether to normalize text
            output_chunks: Whether to save individual chunks
            
        Returns:
            Optional[str]: Path to output audio file, None if failed
        """
        if not self.model:
            self.logger.error("Model not loaded. Please load model first.")
            return None
        
        if not Path(reference_audio_path).exists():
            self.logger.error(f"Reference audio file not found: {reference_audio_path}")
            return None
        
        try:
            # Get speaker conditioning
            self.logger.info("Extracting speaker characteristics...")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=reference_audio_path,
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs,
            )
            
            # Normalize text if requested
            if normalize_text and language == "vi":
                text = self._normalize_vietnamese_text(text)
                self.logger.info("Text normalized for Vietnamese")
            
            # Split text into sentences
            sentences = self._split_text_into_sentences(text, language)
            self.logger.info(f"Processing {len(sentences)} sentences")
            
            # Generate audio for each sentence
            wav_chunks = []
            for i, sentence in enumerate(tqdm(sentences, desc="Generating audio")):
                if not sentence.strip():
                    continue
                
                try:
                    # Generate audio chunk
                    wav_chunk = self.model.inference(
                        text=sentence,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=0.3,
                        length_penalty=1.0,
                        repetition_penalty=10.0,
                        top_k=30,
                        top_p=0.85,
                    )
                    
                    # Apply length optimization
                    keep_length = self._calculate_keep_length(sentence, language)
                    if keep_length > 0:
                        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_length])
                    else:
                        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"])
                    
                    # Save individual chunk if requested
                    if output_chunks:
                        chunk_filename = self._generate_filename(sentence)
                        chunk_path = Path(self.config.OUTPUT_DIR) / f"{chunk_filename}.wav"
                        torchaudio.save(
                            chunk_path, 
                            wav_chunk["wav"].unsqueeze(0), 
                            self.config.SAMPLE_RATE
                        )
                        self.logger.info(f"Saved chunk {i+1}: {chunk_path.name}")
                    
                    wav_chunks.append(wav_chunk["wav"])
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process sentence {i+1}: {e}")
                    continue

            if not wav_chunks:
                self.logger.error("No audio chunks generated successfully")
                return None
            
            # Concatenate all chunks
            final_audio = torch.cat(wav_chunks, dim=0).unsqueeze(0)
            
            # Save final output
            output_filename = self._generate_filename(text[:100])
            output_path = Path(self.config.OUTPUT_DIR) / f"{output_filename}.wav"
            
            torchaudio.save(output_path, final_audio, self.config.SAMPLE_RATE)
            
            self.logger.success(f"Audio saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            return None


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="viXTTS - Vietnamese Text-to-Speech Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python vixtts.py --language "Tiếng Việt" --input "Xin chào thế giới" --reference model/vi_sample.wav
            python vixtts.py -l "Tiếng Anh" -i "Hello world" -r model/en_sample.wav --verbose
        """
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        required=True,
        choices=list(viXTTSConfig.LANGUAGE_CODE_MAP.keys()),
        help="Language for text-to-speech"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Text to synthesize"
    )
    
    parser.add_argument(
        '--reference', '-r',
        type=str,
        default=viXTTSConfig.DEFAULT_REFERENCE_AUDIO,
        help="Path to reference audio file"
    )
    
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default=viXTTSConfig.DEFAULT_MODEL_PATHS["checkpoint"],
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        '--model-config',
        type=str,
        default=viXTTSConfig.DEFAULT_MODEL_PATHS["config"],
        help="Path to model config"
    )
    
    parser.add_argument(
        '--model-vocab',
        type=str,
        default=viXTTSConfig.DEFAULT_MODEL_PATHS["vocab"],
        help="Path to model vocabulary"
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help="Disable text normalization"
    )
    
    parser.add_argument(
        '--output-chunks',
        action='store_true',
        help="Save individual sentence chunks"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    return parser


def main():
    """Main application entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize processor
    processor = viXTTSProcessor(verbose=args.verbose)
    
    # Load model
    processor.logger.info("Initializing viXTTS...")
    success = processor.load_model(
        args.model_checkpoint,
        args.model_config,
        args.model_vocab
    )
    
    if not success:
        processor.logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Get language code
    language_code = viXTTSConfig.LANGUAGE_CODE_MAP[args.language]
    
    # Synthesize speech
    output_path = processor.synthesize_speech(
        text=args.input,
        language=language_code,
        reference_audio_path=args.reference,
        normalize_text=not args.no_normalize,
        output_chunks=args.output_chunks
    )
    
    if output_path:
        processor.logger.success(f"Speech synthesis completed: {output_path}")
        return 0
    else:
        processor.logger.error("Speech synthesis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
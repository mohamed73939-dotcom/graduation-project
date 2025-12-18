"""
أدوات معالجة الفيديو المحسّنة
CHANGES:
- Added minor validation & logging improvement.
"""
import subprocess
from pathlib import Path
from logger_config import video_logger

class VideoProcessor:
    def __init__(self, output_dir="outputs/audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            video_logger.info("✓ FFmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "FFmpeg not found. Please install it first."
            video_logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def extract_audio(self, video_path, output_format='wav'):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        audio_filename = f"{video_path.stem}.{output_format}"
        audio_path = self.output_dir / audio_filename
        video_logger.info(f"Extracting & Optimizing audio from: {video_path.name}")
        
        # Combined filters: 
        # 1. silenceremove: aggressive silence removal
        # 2. loudnorm: standard loudness normalization
        filters = "silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-50dB,loudnorm"
        
        command = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',                    # No video
            '-af', filters,           # Apply filters
            '-acodec', 'pcm_s16le',
            '-ar', '16000',           # Resample to 16kHz
            '-ac', '1',               # Mono
            '-threads', 'auto',       # Use all cores
            '-preset', 'ultrafast',   # Minimal CPU for encoding
            '-y',
            str(audio_path)
        ]
        try:
            # FORCE garbage collection before heavy subprocess
            import gc
            gc.collect()
            
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            if audio_path.exists():
                size_mb = audio_path.stat().st_size / (1024 * 1024)
                video_logger.info(f"✓ Audio extracted & optimized ({size_mb:.2f} MB)")
                return str(audio_path)
            raise RuntimeError("Audio file was not created")
        except subprocess.CalledProcessError as e:
            if "1455" in str(e) or (e.stderr and "1455" in e.stderr):
                raise RuntimeError("System out of memory (WinError 1455). Try a smaller video or close other apps.")
            raise RuntimeError(f"Failed to extract/optimize audio: {e.stderr}")
        except OSError as e:
            if e.winerror == 1455:
                 raise RuntimeError("System out of memory (WinError 1455). Try a smaller video or close other apps.")
            raise
    
    def get_video_duration(self, video_path):
        try:
            command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return None
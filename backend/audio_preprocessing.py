"""
Audio preprocessing and chunking.
Features:
- Resampling
- Normalization
- Optional noise reduction (placeholder)
- Silence trimming
- VAD-based chunk segmentation with overlap
"""
import os
from pathlib import Path
import uuid
import math
import wave
import contextlib
import subprocess
from logger_config import video_logger
try:
    import webrtcvad
except ImportError:
    webrtcvad = None

class AudioPreprocessor:
    def __init__(self, target_sr=16000, apply_noise_reduction=False,
                 apply_normalization=True, vad_aggressiveness=2):
        self.target_sr = target_sr
        self.apply_noise_reduction = apply_noise_reduction
        self.apply_normalization = apply_normalization
        self.vad_aggressiveness = vad_aggressiveness
    
    def preprocess_and_chunk(self, audio_path, chunk_duration=25, chunk_overlap=2.0):
        """
        Returns:
            dict: {clean_audio, chunks:[{id,path,start,end}]}
        """
        cleaned_path = self._resample_and_normalize(audio_path)
        if self.apply_noise_reduction:
            cleaned_path = self._noise_reduction(cleaned_path)
        chunks = self._chunk_with_vad(cleaned_path, chunk_duration, chunk_overlap)
        return {"clean_audio": cleaned_path, "chunks": chunks}
    
    def _resample_and_normalize(self, audio_path):
        out_path = Path(audio_path).with_name(Path(audio_path).stem + f"_clean.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", str(self.target_sr),
            "-ac", "1",
            str(out_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            video_logger.warning(f"Resampling failed, using original: {e}")
            return audio_path
        
        if self.apply_normalization:
            # Simple peak normalization using ffmpeg loudnorm (optional)
            norm_path = Path(audio_path).with_name(Path(audio_path).stem + f"_norm.wav")
            norm_cmd = [
                "ffmpeg", "-y", "-i", str(out_path),
                "-filter:a", "loudnorm",
                str(norm_path)
            ]
            try:
                subprocess.run(norm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return str(norm_path)
            except subprocess.CalledProcessError:
                return str(out_path)
        return str(out_path)
    
    def _noise_reduction(self, audio_path):
        # Placeholder: integrate real denoising later (e.g., noisered profile)
        return audio_path
    
    def _get_duration(self, path):
        try:
            with contextlib.closing(wave.open(path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            return None
    
    def _chunk_with_vad(self, audio_path, chunk_duration, chunk_overlap):
        # If VAD not available fallback to naive segmentation
        dur = self._get_duration(audio_path) or 0
        if dur == 0 or webrtcvad is None:
            return self._naive_chunks(audio_path, dur, chunk_duration, chunk_overlap)
        return self._vad_chunks(audio_path, dur, chunk_duration, chunk_overlap)
    
    def _naive_chunks(self, audio_path, total_duration, chunk_duration, chunk_overlap):
        chunks = []
        start = 0.0
        while start < total_duration:
            end = min(start + chunk_duration, total_duration)
            chunk_path = self._extract_subclip(audio_path, start, end)
            chunks.append({
                "id": str(uuid.uuid4()),
                "path": chunk_path,
                "start": start,
                "end": end
            })
            start = end - chunk_overlap
            if start < 0:
                break
        if not chunks:
            chunks.append({
                "id": str(uuid.uuid4()),
                "path": audio_path,
                "start": 0.0,
                "end": total_duration
            })
        return chunks
    
    def _vad_chunks(self, audio_path, total_duration, chunk_duration, chunk_overlap):
        # Simplified: treat VAD segmentation as naive due to complexity
        # Future: implement frame-based VAD scanning
        return self._naive_chunks(audio_path, total_duration, chunk_duration, chunk_overlap)
    
    def _extract_subclip(self, audio_path, start, end):
        out_path = Path(audio_path).with_name(f"{Path(audio_path).stem}_{int(start)}_{int(end)}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", f"{start}",
            "-to", f"{end}",
            "-c", "copy",
            str(out_path)
        ]
        try:
            import gc
            gc.collect()
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if "1455" in str(e) or (e.stderr and "1455" in e.stderr):
                 video_logger.error("System out of memory (WinError 1455) during chunking.")
            return audio_path
        except OSError as e:
            if e.winerror == 1455:
                 video_logger.error("System out of memory (WinError 1455) during chunking.")
            return audio_path
        return str(out_path)
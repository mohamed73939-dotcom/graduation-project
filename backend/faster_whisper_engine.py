# -------------------- Optional Faster-Whisper Engine (CTranslate2) --------------------
# Requires: faster-whisper, ctranslate2 (added to requirements.txt)
# BENEFITS: 2–4x faster on CPU with int8 quantization.
# USAGE: Set config.yaml -> whisper.use_faster_whisper: true
#
# Note: Not identical output to openai-whisper but typically similar or better speed.

from pathlib import Path
from logger_config import whisper_logger

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",   # int8 for max speed
        num_threads: int = 8,         # Increased from 4 for better parallelism
        no_speech_threshold: float = 0.65
    ):
        if WhisperModel is None:
            raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper ctranslate2")

        whisper_logger.info(
            f"Loading Faster-Whisper model '{model_name}' (device={device}, compute_type={compute_type}, threads={num_threads})"
        )
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_threads
        )
        self.no_speech_threshold = no_speech_threshold

    def transcribe_file(self, audio_path: str, language_hint: str | None = None) -> dict:
        """
        Whole file transcription. Returns combined text.
        """
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(
            str(p),
            beam_size=1,
            temperature=0.0,
            language=language_hint,
            vad_filter=True,
            initial_prompt="اللغة العربية الفصحى",  # Normalize dialects to MSA
            no_speech_threshold=self.no_speech_threshold
        )
        # Force generator to execute inside try block to catch errors early
        segments = list(segments)
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        text = " ".join(text_parts)
        whisper_logger.info(f"✓ Faster-Whisper file decoded: {len(text)} chars")
        return {
            "text": text,
            "language": info.language or (language_hint or "unknown"),
            "segments": [ {"start": s.start, "end": s.end, "text": s.text} for s in segments ]
        }

    def transcribe_chunk(self, chunk: dict, language_hint: str | None = None, max_retries: int = 0) -> dict:
        p = Path(chunk["path"])
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        whisper_logger.info(
            f"Faster-Whisper transcribing chunk {chunk['id']} ({chunk['start']:.1f} - {chunk['end']:.1f}s)"
        )
        attempt = 0
        while attempt <= max_retries:
            try:
                segments, info = self.model.transcribe(
                    str(p),
                    beam_size=1,
                    temperature=0.0,
                    language=language_hint,
                    vad_filter=True,
                    initial_prompt="اللغة العربية الفصحى",  # Normalize dialects to MSA
                    no_speech_threshold=self.no_speech_threshold
                )
                segments = list(segments) # Execute generator
                text = " ".join(s.text.strip() for s in segments if s.text.strip())
                return {
                    "id": chunk["id"],
                    "text": text,
                    "language": info.language or (language_hint or "unknown"),
                    "segments": [ {"start": s.start, "end": s.end, "text": s.text} for s in segments ],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "retries": attempt
                }
            except Exception as e:
                whisper_logger.error(f"Chunk {chunk['id']} failed attempt={attempt}: {e}")
                if attempt == max_retries:
                    return {
                        "id": chunk["id"],
                        "text": "",
                        "language": language_hint or "unknown",
                        "segments": [],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "retries": attempt,
                        "error": str(e)
                    }
                attempt += 1
        return {
            "id": chunk["id"],
            "text": "",
            "language": language_hint or "unknown",
            "segments": [],
            "start": chunk["start"],
            "end": chunk["end"],
            "retries": attempt
        }
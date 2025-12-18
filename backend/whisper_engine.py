import os
import torch
import whisper
from pathlib import Path
from logger_config import whisper_logger

class WhisperTranscriber:
    def __init__(self, model_name="base", beam_size=1, temperature=0.0, num_threads=4, no_speech_threshold=0.6):
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.beam_size = beam_size
        self.temperature = temperature
        self.no_speech_threshold = no_speech_threshold

        whisper_logger.info(f"Loading Whisper model '{model_name}' on {self.device} (threads={num_threads})...")
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            whisper_logger.info(f"✓ Whisper model '{model_name}' loaded successfully")
        except Exception as e:
            whisper_logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_chunk(self, chunk, language_hint=None, max_retries=0):
        audio_path = Path(chunk['path'])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        attempt = 0
        final_result = None
        while attempt <= max_retries:
            try:
                result = self.model.transcribe(
                    str(audio_path),
                    language=language_hint,
                    task="transcribe",
                    fp16=False,
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    word_timestamps=False
                )
                final_result = {
                    'id': chunk['id'],
                    'text': result.get('text', '').strip(),
                    'language': result.get('language', language_hint or 'unknown'),
                    'segments': result.get('segments', []),
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'retries': attempt
                }
                break
            except Exception as e:
                whisper_logger.error(f"Chunk {chunk['id']} transcription failed (attempt {attempt}): {e}")
                if attempt == max_retries:
                    final_result = {
                        'id': chunk['id'],
                        'text': '',
                        'language': language_hint or 'unknown',
                        'segments': [],
                        'start': chunk['start'],
                        'end': chunk['end'],
                        'retries': attempt,
                        'error': str(e)
                    }
                else:
                    attempt += 1
                    continue
            attempt += 1

        if final_result:
            preview = final_result['text'][:60] + ('...' if len(final_result['text']) > 60 else '')
            whisper_logger.info(f"✓ Chunk {chunk['id']} ({chunk['start']:.1f}-{chunk['end']:.1f}s) -> {preview}")
        return final_result

    def transcribe_file(self, audio_path, language_hint=None):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        whisper_logger.info(f"Transcribing full file: {audio_path.name}")
        try:
            result = self.model.transcribe(
                str(audio_path),
                language=language_hint,
                task="transcribe",
                fp16=False,
                beam_size=self.beam_size,
                temperature=self.temperature,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=False,
                word_timestamps=False
            )
            whisper_logger.info(f"✓ File transcribed: {len(result.get('text', ''))} chars")
            return result
        except Exception as e:
            whisper_logger.error(f"File transcription failed: {e}")
            raise
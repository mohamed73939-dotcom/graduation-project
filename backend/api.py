from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import uuid
import asyncio
import time
import os
import torch
import html

from logger_config import api_logger
from video_utils import VideoProcessor
from slide_extractor import SlideExtractor


from whisper_engine import WhisperTranscriber
from nlp_utils import TextProcessor
from summarizer import LectureSummarizer
from extractive_summarizer import ExtractiveSummarizer
from audio_preprocessing import AudioPreprocessor
from utils import MetricsAggregator
from caching import CacheManager
from config_loader import load_config

app = FastAPI(title="Sidecut API", version="1.2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG = load_config("config.yaml")

default_threads = int(CONFIG.get("performance", {}).get("cpu_threads", 4))
os.environ.setdefault("OMP_NUM_THREADS", str(default_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(default_threads))

try:
    api_logger.info("Initializing API components...")
    video_processor = VideoProcessor()
    whisper_engine = CONFIG.get("whisper", {}).get("engine", "original")
    if whisper_engine == "faster_whisper":
        from faster_whisper_engine import FasterWhisperTranscriber
        api_logger.info("Using Faster-Whisper engine")
        whisper = FasterWhisperTranscriber(
            model_name=CONFIG.get("whisper", {}).get("model", "base"),
            device="cpu",
            compute_type="int8",
            num_threads=default_threads
        )
    else:
        from whisper_engine import WhisperTranscriber
        api_logger.info("Using OpenAI-Whisper engine")
        whisper = WhisperTranscriber(
            model_name=CONFIG.get("whisper", {}).get("model", "base"),
            beam_size=1,
            temperature=0.0,
            num_threads=default_threads
        )
    text_processor = TextProcessor()
    # Updated to check for custom trained model first
    # If running from backend directory, path is models/custom_mt5-small
    custom_model_path = Path("models/custom_mt5-small") 
    if not custom_model_path.exists():
         # Fallback to check relative to root if running from root
         custom_model_path = Path("backend/models/custom_mt5-small")
         
    model_to_use = str(custom_model_path) if custom_model_path.exists() else CONFIG.get("summarization", {}).get("abstractive_model", "csebuetnlp/mT5_multilingual_XLSum")
    
    if CONFIG.get("summarization", {}).get("use_abstractive", True):
       api_logger.info(f"Initializing abstractive summarizer with {model_to_use}...")
       summarizer = LectureSummarizer(
           model_name=model_to_use,
           strategy=CONFIG.get("summarization", {}).get("model_strategy", "single")
       )
    else:
       summarizer = None
       api_logger.info("Abstractive summarizer disabled by config")
       
    # Force use of summarizer if custom model exists, overriding low memory mode for testing?
    # No, respect config, but log availability.
    if custom_model_path.exists():
        api_logger.info("Custom model found and configured.")
    extractive = ExtractiveSummarizer()
    audio_preprocessor = AudioPreprocessor(
        target_sr=CONFIG.get("audio", {}).get("sample_rate", 16000),
        apply_noise_reduction=CONFIG.get("audio", {}).get("noise_reduction", False),
        apply_normalization=True,
        vad_aggressiveness=CONFIG.get("audio", {}).get("vad_aggressiveness", 2)
    )
    slide_extractor = SlideExtractor(output_dir="outputs/slides")
    cache = CacheManager(enabled=CONFIG.get("performance", {}).get("cache_enabled", True))
    metrics = MetricsAggregator()
    api_logger.info("✓ All API components initialized successfully")
except Exception as e:
    api_logger.error(f"Failed to initialize API components: {e}")
    raise

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_FILE_SIZE = CONFIG.get("limits", {}).get("max_file_mb", 500) * 1024 * 1024


def validate_video_file(filename: str, file_size: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE / (1024*1024):.0f} MB")


async def _transcribe_chunks(chunks, language_hint):
    configured = CONFIG.get("performance", {}).get("parallel_chunks", 4)
    parallel_limit = 1 if not torch.cuda.is_available() else max(1, int(configured))
    sem = asyncio.Semaphore(parallel_limit)
    loop = asyncio.get_running_loop()
    results = []

    async def transcribe_one(chunk):
        async with sem:
            api_logger.info(f"Transcribing chunk {chunk['id']}...")
            try:
                # Add overall timeout for a single chunk transcription (5 minutes max)
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None, 
                        lambda: whisper.transcribe_chunk(
                            chunk,
                            language_hint=language_hint,
                            max_retries=0
                        )
                    ), 
                    timeout=300
                )
            except asyncio.TimeoutError:
                api_logger.error(f"Chunk {chunk['id']} timed out!")
                # Return empty result instead of crashing everything
                return {
                    'id': chunk['id'],
                    'text': '',
                    'language': 'unknown',
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'error': 'timeout'
                }

    tasks = [asyncio.create_task(transcribe_one(c)) for c in chunks]
    for t in asyncio.as_completed(tasks):
        results.append(await t)
    results.sort(key=lambda x: x['start'])
    return results


def _merge_transcriptions(chunk_results):
    merged_text_parts = []
    confidences = []
    per_chunk = []
    for r in chunk_results:
        txt = r.get("text", "").strip()
        if txt:
            merged_text_parts.append(txt)
        conf = r.get("avg_logprob", None)
        confidences.append(conf if conf is not None else None)
        per_chunk.append({
            "chunk_id": r["id"],
            "start": r["start"],
            "end": r["end"],
            "conf": conf,
            "language": r.get("language", "unknown"),
            "tokens": r.get("token_count", 0)
        })
    full_text_raw = " ".join(merged_text_parts)
    detected_lang_raw = text_processor.detect_language(full_text_raw)
    full_text = text_processor.clean_text(full_text_raw, language=detected_lang_raw)
    avg_conf = None
    real_conf_values = [c for c in confidences if c is not None]
    if real_conf_values:
        avg_conf = sum(real_conf_values) / len(real_conf_values)
    detected_lang = text_processor.detect_language(full_text)
    return full_text, avg_conf, detected_lang, per_chunk


def _grounded_summarize(clean_text, language, ocr_context=""):
    # Extractive basis (top sentences + keywords)
    extractive_sentences = extractive.extract_top_sentences(
        clean_text,
        top_n=CONFIG.get("summarization", {}).get("extractive_sentence_limit", 25),
        language=language
    )
    keywords = extractive.extract_keywords(clean_text, top_n=20) if hasattr(extractive, "extract_keywords") else []
    basis = "\n".join(extractive_sentences)

    if summarizer:
        # Strongly grounded hierarchical summarization
        summary, meta = summarizer.hierarchical_summarize(
            basis,
            language=language,
            intermediate_max_length=CONFIG.get("summarization", {}).get("abstractive_intermediate_max", 220),
            final_max_length=CONFIG.get("summarization", {}).get("final_max_length", 450),
            min_length=CONFIG.get("summarization", {}).get("abstractive_min_length", 80),
            chunk_token_limit=1800, # Increased for larger chunks
            beam_size=1,            # Greedy search for speed
            constraints={
                "keywords": keywords,
                "grounding_text": clean_text,
                "ocr_context": ocr_context,  # Pass visual text
                "extractive_summary": basis,  # Pass extractive bullets
                "no_external_entities": True,
                "format": "bullets+paragraphs"
            }
        )
    else:
        # Fallback / Low Memory Mode
        # Format as bullet points for better readability
        # Format as bullet points for better readability
        # Smart Highlighting Logic
        import re
        sentences = [s.strip() for s in basis.split('\n') if s.strip()]
        html_sentences = []
        
        # Sort keywords by length (descending) to prevent nested replacements of substrings
        sorted_kws = sorted(keywords, key=len, reverse=True)
        
        for s in sentences:
            s_html = html.escape(s)
            # Apply highlighting for top 10 keywords only to avoid clutter
            for kw in sorted_kws[:10]:
                if len(kw) < 3: continue 
                # Use regex with word boundaries for cleaner matching
                try:
                    # Enforce word boundaries to prevent splitting words
                    pattern_str = r'\b' + re.escape(kw) + r'\b'
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    s_html = pattern.sub(f'<span class="highlight">{kw}</span>', s_html)
                except Exception:
                    pass # Skip if regex fails for some reason
            html_sentences.append(f"<li>{s_html}</li>")

        summary = f"<ul>{''.join(html_sentences)}</ul>"
        
        # Also keep a plain text version for the 'summary' field if needed
        # But we overwrite it here with the HTML version because the frontend expects 'summary' 
        # to be the main display content (which we verified renders HTML).
        
        meta = {"grounded_score": 1.0, "rerolled": False, "fallback": True, "mode": "extractive_highlighted"}

    # If the summary fails grounding checks, fallback to extractive-only is already handled inside summarizer
    return summary, meta


def _summary_completeness_heuristic(summary: str):
    stripped = summary.strip()
    if not stripped:
        return True
    end_char = stripped[-1]
    incomplete = end_char not in ('.', '؟', '!', '؛') and len(stripped.split()) > 40
    tail = stripped[-10:]
    if ' ' not in tail and len(stripped) > 150:
        incomplete = True
    return incomplete


@app.post("/api/summarize")
async def summarize_video(
    video: UploadFile = File(...),
    language: str = Form("auto")
):
    start_time = time.time()
    video_id = str(uuid.uuid4())

    try:
        api_logger.info(f"[{video_id}] Starting summarization for video: {video.filename}")
        validate_video_file(video.filename, video.size or 0)

        video_path = UPLOAD_DIR / f"{video_id}_{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        loop = asyncio.get_event_loop()
        
        # Parallel Execution: Audio Extraction & Slide Extraction
        # We start slide extraction early as it scans the video
        slide_task = loop.run_in_executor(
            None, 
            lambda: slide_extractor.extract_slides(
                video_path, 
                output_mode="text"  # We want both text and images (handles internally)
            )
        )
        
        # Offload audio extraction
        try:
             audio_path = await loop.run_in_executor(None, video_processor.extract_audio, video_path)
        except RuntimeError as e:
             if "1455" in str(e):
                  raise HTTPException(status_code=503, detail="Server out of memory (WinError 1455). Please try a smaller video or try again later.")
             raise HTTPException(status_code=500, detail=f"Audio extraction failed: {str(e)}")

        short_threshold = int(CONFIG.get("transcription", {}).get("short_audio_seconds", 120))
        try:
            duration = await loop.run_in_executor(None, video_processor.get_video_duration, video_path)
            duration = duration or 0
        except Exception:
            duration = 0

        if duration > 0 and duration <= short_threshold:
            # Short video: Offload full file transcription
            # Short video: Offload full file transcription with 10-minute timeout
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, 
                        lambda: whisper.transcribe_file(audio_path, language_hint=None if language == "auto" else language)
                    ),
                    timeout=600
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Transcription timed out (10 minutes limit exceeded).")
            merged_text = result.get("text", "")
            detected_lang = result.get("language", language if language != "auto" else text_processor.detect_language(merged_text))
            avg_conf = None
            per_chunk_meta = []
        else:
            # Long video: Offload preprocessing
            try:
                prep_result = await loop.run_in_executor(
                    None,
                    lambda: audio_preprocessor.preprocess_and_chunk(
                        audio_path,
                        chunk_duration=CONFIG.get("transcription", {}).get("chunk_duration", 25),
                        chunk_overlap=CONFIG.get("transcription", {}).get("chunk_overlap", 2.0),
                    )
                )
            except Exception as e:
                 api_logger.error(f"Preprocessing failed: {e}")
                 raise HTTPException(status_code=500, detail="Audio preprocessing failed.")
                 
            chunks = prep_result["chunks"]
            api_logger.info(f"[{video_id}] Prepared {len(chunks)} audio chunks for transcription")

            language_hint = None if language == "auto" else language
            cache_key = f"transcription:{video_id}"
            cached = cache.get(cache_key)
            if cached:
                api_logger.info(f"[{video_id}] Using cached transcription")
                chunk_results = cached["chunk_results"]
            else:
                chunk_results = await _transcribe_chunks(chunks, language_hint)
                cache.set(cache_key, {"chunk_results": chunk_results})

            merged_text, avg_conf, detected_lang, per_chunk_meta = _merge_transcriptions(chunk_results)

        cleaned_text = text_processor.clean_text(merged_text, language=detected_lang)
        formatted_text = text_processor.format_text(cleaned_text, add_paragraphs=True)

        # Retrieve Slide Results
        try:
            ocr_text, slides_data = await slide_task
            api_logger.info(f"[{video_id}] Slide extraction complete. Found {len(slides_data)} slides.")
        except Exception as e:
            api_logger.error(f"Slide extraction failed: {e}")
            ocr_text = ""
            slides_data = []

        summary, summary_meta = _grounded_summarize(cleaned_text, detected_lang, ocr_context=ocr_text)
        summary_incomplete = _summary_completeness_heuristic(summary)

        summary_html = "<div>" + "</div><div>".join(
            [html.escape(p.strip()) for p in summary.split("\n") if p.strip()]
        ) + "</div>"

        processing_time = time.time() - start_time
        metrics.observe("latency_seconds", processing_time)
        if avg_conf is not None:
            metrics.observe("avg_transcription_confidence", avg_conf)

        api_logger.info(
            f"[{video_id}] Complete in {processing_time:.2f}s | "
            f"Lang={detected_lang} | Words={len(summary.split())} | "
            f"GroundedScore={summary_meta.get('grounded_score')} "
            f"Rerolled={summary_meta.get('rerolled')} Fallback={summary_meta.get('fallback')}"
        )

        transcription_preview_limit = 1500
        transcription_truncated = len(formatted_text) > transcription_preview_limit
        transcription_preview = formatted_text[:transcription_preview_limit]

        return JSONResponse({
            "status": "success",
            "data": {
                "video_id": video_id,
                "detected_language": detected_lang,
                "transcription_preview": transcription_preview,
                "transcription_full": formatted_text,
                "transcription_truncated": transcription_truncated,
                "summary": summary,
                "summary_html": summary_html,
                "summary_metadata": {
                    "word_count": len(summary.split()),
                    "incomplete": summary_incomplete,
                    "hierarchical": summary_meta,
                },
                "confidence": {
                    "avg_logprob": avg_conf,
                    "summary_score": summary_meta.get('grounded_score', 0.0)
                },
                "chunk_metadata": per_chunk_meta,
                "metrics": {
                    "latency_seconds": processing_time
                },
                "slides": slides_data
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"[{video_id}] Unexpected error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Internal server error",
            "video_id": video_id
        })
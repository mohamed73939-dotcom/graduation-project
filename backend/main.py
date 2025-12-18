"""
ملف رئيسي لمعالجة الفيديو وتلخيصه
CHANGES:
- Integrated new preprocessing, chunk transcription, extractive + abstractive summarization.
- Added caching and metrics.
"""
from pathlib import Path
import time
from logger_config import main_logger
from video_utils import VideoProcessor
from whisper_engine import WhisperTranscriber
from nlp_utils import TextProcessor
from summarizer import LectureSummarizer
from extractive_summarizer import ExtractiveSummarizer
from audio_preprocessing import AudioPreprocessor
from caching import CacheManager
from utils import MetricsAggregator
from config_loader import load_config

class LectureProcessor:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.output_base = Path(self.config.get("paths", {}).get("output_dir", "outputs"))
        self.output_base.mkdir(exist_ok=True)
        
        main_logger.info("Initializing components...")
        try:
            self.video_processor = VideoProcessor(output_dir=self.output_base / "audio")
            self.transcriber = WhisperTranscriber(
                model_name=self.config.get("whisper", {}).get("model", "base"),
                beam_size=self.config.get("transcription", {}).get("initial_beam_size", 1),
                temperature=self.config.get("transcription", {}).get("temperature", 0.0)
            )
            self.text_processor = TextProcessor()
            if self.config.get("summarization", {}).get("use_abstractive", True):
                self.summarizer = LectureSummarizer(
                    model_name=self.config.get("summarization", {}).get("abstractive_model", "csebuetnlp/mT5_multilingual_XLSum"),
                    output_dir=self.output_base / "summaries"
                )
            else:
                self.summarizer = None
                main_logger.info("Skipping abstractive summarizer load (Low Memory Mode)")
            
            self.extractive = ExtractiveSummarizer()
            self.audio_preprocessor = AudioPreprocessor(
                target_sr=self.config.get("audio", {}).get("sample_rate", 16000),
                apply_noise_reduction=self.config.get("audio", {}).get("noise_reduction", False),
                apply_normalization=True,
                vad_aggressiveness=self.config.get("audio", {}).get("vad_aggressiveness", 2)
            )
            self.cache = CacheManager(enabled=self.config.get("performance", {}).get("cache_enabled", True))
            self.metrics = MetricsAggregator()
            main_logger.info("✓ All components initialized successfully!")
        except Exception as e:
            main_logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_video(self, video_path, language='auto', save_transcript=True):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_name = video_path.stem
        
        main_logger.info(f"\n{'='*60}\nProcessing video: {video_name}\n{'='*60}\n")
        start_time = time.time()
        results = {'video_name': video_name, 'video_path': str(video_path), 'success': False}
        
        try:
            audio_path = self.video_processor.extract_audio(video_path)
            prep = self.audio_preprocessor.preprocess_and_chunk(
                audio_path,
                chunk_duration=self.config.get("transcription", {}).get("chunk_duration", 25),
                chunk_overlap=self.config.get("transcription", {}).get("chunk_overlap", 2.0)
            )
            chunks = prep["chunks"]
            main_logger.info(f"Prepared {len(chunks)} chunks")
            
            chunk_results = []
            for c in chunks:
                r = self.transcriber.transcribe_chunk(
                    c,
                    language_hint=None if language == 'auto' else language,
                    min_confidence=self.config.get("transcription", {}).get("min_confidence", -0.35),
                    max_retries=self.config.get("transcription", {}).get("max_retries", 2)
                )
                chunk_results.append(r)
            
            # Merge
            merged_text = " ".join([r.get("text","") for r in chunk_results if r.get("text")])
            cleaned = self.text_processor.clean_text(merged_text, language='ar')
            detected_lang = self.text_processor.detect_language(cleaned)
            formatted = self.text_processor.format_text(cleaned, add_paragraphs=True)
            
            # Two-stage summarization
            extractive_sentences = self.extractive.extract_top_sentences(cleaned, top_n=25, language=detected_lang)
            
            if self.summarizer:
                summary = self.summarizer.summarize(
                    "\n".join(extractive_sentences),
                    max_length=self.config.get("summarization", {}).get("abstractive_max_length", 320),
                    min_length=self.config.get("summarization", {}).get("abstractive_min_length", 80),
                    language=detected_lang
                )
                if not summary or len(summary.strip()) < 20:
                    summary = "\n".join(extractive_sentences[:10])
            else:
                # Fallback / Low Memory Mode
                summary = "\n".join(extractive_sentences)
            
            results.update({
                'raw_transcript': merged_text,
                'cleaned_transcript': formatted,
                'detected_language': detected_lang,
                'summary': summary
            })
            
            if save_transcript:
                transcript_path = self.output_base / "transcripts" / f"{video_name}.txt"
                transcript_path.parent.mkdir(exist_ok=True)
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                results['transcript_path'] = str(transcript_path)
            
            summary_path = self.output_base / "summaries" / f"{video_name}_summary.txt"
            summary_path.parent.mkdir(exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            results['summary_path'] = str(summary_path)
            
            results['success'] = True
            latency = time.time() - start_time
            self.metrics.observe("latency_seconds", latency)
            main_logger.info(f"✓ Processing complete in {latency:.2f}s")
            self._print_results_summary(results)
            return results
        
        except Exception as e:
            main_logger.error(f"Error processing video: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def _print_results_summary(self, results):
        print(f"\n{'='*60}\nPROCESSING COMPLETE!\n{'='*60}")
        print(f"Video: {results['video_name']}")
        print(f"Language: {results.get('detected_language', 'N/A')}")
        print(f"Summary length: {len(results.get('summary','').split())} words")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    processor = LectureProcessor()
    video_file = "path/to/your/lecture.mp4"
    results = processor.process_video(video_file, language='auto')
    if results['success']:
        print("\n✓ Processing successful!")
        print(f"Summary: {results['summary']}")
    else:
        print(f"\n✗ Processing failed: {results.get('error')}")
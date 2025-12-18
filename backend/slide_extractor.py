import cv2
import easyocr
import numpy as np
from pathlib import Path
from logger_config import api_logger

class SlideExtractor:
    def __init__(self, output_dir="frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Initialize EasyOCR reader (lazy load)
        self.reader = None 
        
    def _get_reader(self, language='en'):
        if self.reader is None:
            api_logger.info(f"Initializing EasyOCR for {language}...")
            # Support Arabic and English
            langs = ['en']
            if language == 'ar':
                langs.append('ar')
            self.reader = easyocr.Reader(langs, gpu=False) # GPU False for compatibility/stability
        return self.reader

    def extract_slides(self, video_path, threshold=30.0, output_mode="text"):
        """
        Extracts unique slides from video and optionally performs OCR.
        output_mode: 'text' (returns fused text), 'images' (returns paths), 'both'
        """
        api_logger.info(f"Extracting slides from {video_path}...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 2) # Check every 2 seconds to save time
        
        prev_hist = None
        unique_frames = []
        timestamps = []
        
        frame_count = 0
        saved_count = 0
        
        while True:
            # Skip frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to gray for histogram comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            should_save = False
            if prev_hist is None:
                should_save = True
            else:
                # Compare histograms (Correlation)
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                # If correlation is low, scene changed significantly
                if score < 0.95: # 95% similarity threshold
                    should_save = True
                    
            if should_save:
                # Save frame
                ts = frame_count / fps
                filename = f"slide_{saved_count}_{int(ts)}s.jpg"
                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), frame)
                unique_frames.append(str(filepath))
                timestamps.append(ts)
                
                prev_hist = hist
                saved_count += 1
                api_logger.debug(f"Saved slide: {filename} at {ts:.2f}s")
                
            frame_count += frame_interval
            
        cap.release()
        api_logger.info(f"Extracted {len(unique_frames)} unique slides.")
        
        ocr_text = ""
        slides_data = []

        if output_mode in ["text", "both"]:
            # Perform OCR on extracted frames
            reader = self._get_reader(language='en') # Default to en/ar mix
            api_logger.info("Starting OCR on extracted slides...")
            
            seen_text = set()
            
            for i, frame_path in enumerate(unique_frames):
                try:
                    results = reader.readtext(frame_path)
                    # format: ([[x,y],...], 'text', confidence)
                    frame_text_list = []
                    for (bbox, text, prob) in results:
                        if prob > 0.4: # Filter low confidence
                            frame_text_list.append(text)
                            
                    full_frame_text = " ".join(frame_text_list)
                    
                    # Deduplication (Simple)
                    if full_frame_text and full_frame_text not in seen_text:
                        timestamp = timestamps[i]
                        ocr_text += f"\n[Slide at {int(timestamp)}s]: {full_frame_text}"
                        seen_text.add(full_frame_text)
                        
                        slides_data.append({
                            "time": timestamp,
                            "path": frame_path,
                            "text": full_frame_text
                        })
                        
                except Exception as e:
                    api_logger.warning(f"OCR failed for {frame_path}: {e}")
                    
        return ocr_text, slides_data

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        extractor = SlideExtractor(output_dir="outputs/slides_test")
        text, _ = extractor.extract_slides(sys.argv[1])
        print("Extracted Text:\n", text)

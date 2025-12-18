# -------------------- 4. test_summarizer.py --------------------
"""
اختبارات التلخيص
"""
import pytest
from backend.summarizer import LectureSummarizer

class TestLectureSummarizer:
    
    @pytest.fixture
    def summarizer(self):
        """إنشاء ملخص للاختبار"""
        return LectureSummarizer(model_name="t5-small")  # نموذج صغير
    
    def test_summarize_short_text(self, summarizer):
        """اختبار تلخيص نص قصير"""
        text = """
        Artificial intelligence is the simulation of human intelligence 
        by machines. It includes learning, reasoning, and self-correction. 
        AI is used in various applications like speech recognition, 
        computer vision, and natural language processing.
        """
        
        summary = summarizer.summarize(text, max_length=50, min_length=10)
        
        assert len(summary) > 0
        assert len(summary) < len(text)
    
    def test_summarize_long_text(self, summarizer):
        """اختبار تلخيص نص طويل"""
        # نص طويل (محاكاة نص محاضرة)
        long_text = " ".join([
            "This is a long lecture about artificial intelligence." 
            for _ in range(100)
        ])
        
        summary = summarizer.summarize(long_text)
        assert len(summary.split()) < len(long_text.split())
    
    def test_split_into_chunks(self, summarizer):
        """اختبار تقسيم النص إلى أجزاء"""
        text = " ".join([f"word{i}" for i in range(1500)])
        chunks = summarizer._tokenizer_aware_chunks(text, max_tokens=500)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 500

# ============================================
# Sidecut - Test Suite
# ============================================

import pytest
import os
from pathlib import Path
import tempfile

# -------------------- 1. test_video_utils.py --------------------
"""
اختبارات وحدة معالجة الفيديو
"""
from backend.video_utils import VideoProcessor

class TestVideoProcessor:
    
    @pytest.fixture
    def video_processor(self):
        """إنشاء معالج فيديو للاختبار"""
        return VideoProcessor(output_dir="test_outputs/audio")
    
    def test_extract_audio_success(self, video_processor):
        """اختبار استخراج الصوت بنجاح"""
        # يحتاج ملف فيديو حقيقي للاختبار
        video_path = "tests/fixtures/sample_video.mp4"
        
        if os.path.exists(video_path):
            audio_path = video_processor.extract_audio(video_path)
            assert os.path.exists(audio_path)
            assert audio_path.endswith('.wav')
    
    def test_extract_audio_invalid_file(self, video_processor):
        """اختبار الفشل مع ملف غير صالح"""
        with pytest.raises(Exception):
            video_processor.extract_audio("nonexistent.mp4")
# -------------------- 2. test_whisper_engine.py --------------------
"""
اختبارات وحدة Whisper
"""
from backend.whisper_engine import WhisperTranscriber

class TestWhisperTranscriber:
    
    @pytest.fixture
    def whisper(self):
        """إنشاء محول Whisper"""
        return WhisperTranscriber(model_name="tiny")  # نموذج صغير للاختبار
    
    def test_transcribe_arabic(self, whisper):
        """اختبار تحويل صوت عربي"""
        audio_path = "tests/fixtures/arabic_audio.wav"
        
        if os.path.exists(audio_path):
            result = whisper.transcribe(audio_path, language='ar')
            assert 'text' in result
            assert 'language' in result
            assert len(result['text']) > 0
    
    def test_transcribe_english(self, whisper):
        """اختبار تحويل صوت إنجليزي"""
        audio_path = "tests/fixtures/english_audio.wav"
        
        if os.path.exists(audio_path):
            result = whisper.transcribe(audio_path, language='en')
            assert result['language'] == 'en'
    
    def test_auto_detect_language(self, whisper):
        """اختبار الكشف التلقائي عن اللغة"""
        audio_path = "tests/fixtures/sample_audio.wav"
        
        if os.path.exists(audio_path):
            result = whisper.transcribe(audio_path, language=None)
            assert result['language'] in ['ar', 'en']
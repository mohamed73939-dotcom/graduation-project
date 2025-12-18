"""
اختبارات معالجة النصوص
"""
from backend.nlp_utils import TextProcessor

class TestTextProcessor:
    
    @pytest.fixture
    def text_processor(self):
        return TextProcessor()
    
    def test_clean_arabic_text(self, text_processor):
        """اختبار تنظيف نص عربي"""
        dirty_text = "  مرحباً   بكم   في   المحاضرة  \n\n  اليوم  "
        clean = text_processor.clean_text(dirty_text, language='ar')
        
        assert clean == "مرحباً بكم في المحاضرة اليوم"
        assert '\n' not in clean
        assert '  ' not in clean
    
    def test_clean_english_text(self, text_processor):
        """اختبار تنظيف نص إنجليزي"""
        dirty_text = "  Hello   World!!!   \n\n  Test  "
        clean = text_processor.clean_text(dirty_text, language='en')
        
        assert "Hello World" in clean
    
    def test_split_sentences_arabic(self, text_processor):
        """اختبار تقسيم الجمل العربية"""
        text = "الجملة الأولى. الجملة الثانية. الجملة الثالثة."
        sentences = text_processor.split_sentences(text, language='ar')
        
        assert len(sentences) == 3
        assert "الأولى" in sentences[0]
    
    def test_remove_stopwords_arabic(self, text_processor):
        """اختبار إزالة الكلمات الشائعة"""
        text = "هذا هو درس في الذكاء الاصطناعي"
        filtered = text_processor.remove_stopwords(text, language='ar')
        
        assert "درس" in filtered
        assert "الذكاء" in filtered
        assert "هذا" not in filtered  # كلمة شائعة
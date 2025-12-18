import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from logger_config import nlp_logger

# CHANGES:
# - Added punctuation restoration placeholder.
# - Improved language detection thresholds.
# - Refined clean_text regex for stability.
# - Added optional simple punctuation restoration for Arabic/English.

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nlp_logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nlp_logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        try:
            self.arabic_stopwords = set(stopwords.words('arabic'))
        except Exception as e:
            nlp_logger.warning(f"Failed to load Arabic stopwords: {e}")
            self.arabic_stopwords = set()
        try:
            self.english_stopwords = set(stopwords.words('english'))
        except Exception as e:
            nlp_logger.warning(f"Failed to load English stopwords: {e}")
            self.english_stopwords = set()
    
    def clean_text(self, text, language='ar', keep_punctuation=True):
        if language == 'ar':
            # Arabic Normalization
            text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove Tashkeel
            text = re.sub(r'\u0640', '', text)          # Remove Tatweel (Kashida) - STRIP STRETCHING
            
            text = re.sub(r'[إأآا]', 'ا', text)         # Normalize Alifs (Standard practice)
            # text = re.sub(r'ى', 'ي', text)              # SKIP: Keep Alif Maqsura separate for correct spelling
            text = re.sub(r'ؤ', 'و', text)              # Normalize Waw Hamza to Waw
            text = re.sub(r'ئ', 'ي', text)              # Normalize Ya Hamza to Ya
            # text = re.sub(r'ة', 'ه', text)              # SKIP: Keep Taa Marbuta for correct spelling
            text = re.sub(r'گ', 'ك', text)              # Normalize Persian Kaf
            text = re.sub(r'ڤ', 'ف', text)              # Normalize Ve
            
            # Common Speech-to-Text Glue Words Fixes
            replacements = {
                r'\bان شاء الله\b': 'إن شاء الله',
                r'\bي جماعة\b': 'يا جماعة',
                r'\bعشان\b': 'علشان',
                r'\bده\b': 'هذا',
            }
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)

        if not text:
            return ""
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\.\.+', '.', text)
        text = re.sub(r'\?\?+', '?', text)
        text = re.sub(r'!!+', '!', text)
        allowed_punct = r'.,!?؛،؟'
        arabic_range = r'\u0600-\u06FF\u0750-\u077F'
        if keep_punctuation:
            text = re.sub(rf'[^\w\s{arabic_range}{allowed_punct}\n-]', '', text)
        else:
            text = re.sub(rf'[^\w\s{arabic_range}\n]', ' ', text)
        text = re.sub(r'\s*([.,!?؛،؟])\s*', r'\1 ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        return text.strip()
    
    def split_sentences(self, text, language='ar'):
        if not text:
            return []
        if language == 'ar':
            sentences = re.split(r'(?<=[.!?؟])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            merged = []
            temp = ""
            for sent in sentences:
                if len(sent.split()) < 3 and temp:
                    temp += " " + sent
                else:
                    if temp:
                        merged.append(temp)
                    temp = sent
            if temp:
                merged.append(temp)
            return merged
        else:
            try:
                sentences = sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip()]
            except:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                return [s.strip() for s in sentences if s.strip()]
    
    def remove_stopwords(self, text, language='ar'):
        stopwords_set = self.arabic_stopwords if language == 'ar' else self.english_stopwords
        if not stopwords_set:
            return text
        words = text.split()
        filtered = [w for w in words if w.lower() not in stopwords_set and len(w) > 1]
        return ' '.join(filtered)
    
    def format_text(self, text, add_paragraphs=True):
        if not text:
            return ""
        text = self.clean_text(text)
        if add_paragraphs:
            sentences = self.split_sentences(text)
            paragraphs = []
            current_para = []
            for i, sent in enumerate(sentences):
                current_para.append(sent)
                if len(current_para) >= 3 or (i == len(sentences) - 1):
                    paragraphs.append(" ".join(current_para))
                    current_para = []
            return "\n\n".join(paragraphs)
        return text
    
    def extract_keywords(self, text, language='ar', top_n=10):
        clean = self.remove_stopwords(text, language)
        words = clean.split()
        word_freq = {}
        for word in words:
            w = word.lower().strip()
            if len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
    
    def detect_language(self, text):
        if not text:
            return 'unknown'
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = arabic_chars + latin_chars
        if total_chars == 0:
            return 'unknown'
        arabic_ratio = arabic_chars / total_chars
        if arabic_ratio > 0.65:
            return 'ar'
        elif arabic_ratio < 0.25:
            return 'en'
        else:
            return 'mixed'
    
    def restore_punctuation(self, text, language='ar'):
        """
        Placeholder for punctuation restoration (can integrate a model later).
        For now, returns text unchanged.
        """
        return text
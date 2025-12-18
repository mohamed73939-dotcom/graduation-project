"""
إعدادات pytest المشتركة
"""
import pytest
import shutil
import os
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """إعداد بيئة الاختبار"""
    # إنشاء مجلدات الاختبار
    test_dirs = [
        "test_outputs/audio",
        "test_outputs/summaries",
        "tests/fixtures"
    ]
    
    for directory in test_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # تنظيف بعد الاختبارات
    if os.path.exists("test_outputs"):
        shutil.rmtree("test_outputs")


@pytest.fixture
def sample_text():
    """نص عينة للاختبار"""
    return """
    الذكاء الاصطناعي هو محاكاة الذكاء البشري بواسطة الآلات.
    يشمل التعلم والاستنتاج والتصحيح الذاتي.
    يستخدم الذكاء الاصطناعي في تطبيقات متنوعة مثل التعرف على الكلام
    ورؤية الحاسوب ومعالجة اللغات الطبيعية.
    """
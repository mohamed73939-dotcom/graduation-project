"""
اختبارات API
"""
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

class TestAPI:
    
    def test_health_check(self):
        """اختبار نقطة فحص الصحة"""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_summarize_without_file(self):
        """اختبار الطلب بدون ملف"""
        response = client.post("/api/summarize")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_summarize_with_invalid_file(self):
        """اختبار رفع ملف غير صالح"""
        files = {"video": ("test.txt", b"not a video", "text/plain")}
        data = {"language": "ar"}
        
        response = client.post("/api/summarize", files=files, data=data)
        # قد يختلف حسب التحقق المطبق
        assert response.status_code in [400, 422, 500]
    
    @pytest.mark.skipif(
        not os.path.exists("tests/fixtures/sample_video.mp4"),
        reason="Sample video not available"
    )
    def test_summarize_with_valid_video(self):
        """اختبار رفع فيديو صالح"""
        with open("tests/fixtures/sample_video.mp4", "rb") as f:
            files = {"video": ("sample.mp4", f, "video/mp4")}
            data = {"language": "ar"}
            
            response = client.post("/api/summarize", files=files, data=data)
            assert response.status_code == 200
            
            result = response.json()
            assert result["status"] == "success"
            assert "summary" in result["data"]
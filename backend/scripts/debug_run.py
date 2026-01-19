import requests
from pathlib import Path

# Config
video_path = r"backend/uploads/f743de44-21df-446b-b086-33031f500f2f_IMG_4757.MP4" # Use one of the existing uploads
api_url = "http://localhost:8000/api/summarize"

def test_summarize():
    print(f"Testing summarization for: {video_path}")
    if not Path(video_path).exists():
        print("Error: Video file not found!")
        return

    with open(video_path, "rb") as f:
        files = {"video": (Path(video_path).name, f, "video/mp4")}
        data = {"language": "en"} # Force English to skip auto-detect delay
        
        try:
            print("Sending request to backend...")
            response = requests.post(api_url, files=files, data=data, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                print("\nSUCCESS!")
                print("Summary:", result['data']['summary'])
                print("-" * 50)
                print("Full Transcript:", result['data']['transcription_full'][:500] + "...")
            else:
                print(f"FAILED: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_summarize()

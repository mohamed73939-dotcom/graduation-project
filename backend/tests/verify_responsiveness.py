
import asyncio
import urllib.request
import urllib.error
import time
import sys

# Verification Script (No dependencies)
# Usage: python backend/tests/verify_responsiveness.py

def ping():
    try:
        start = time.time()
        with urllib.request.urlopen("http://127.0.0.1:8000/docs", timeout=2) as response:
            if response.status == 200:
                print(f"Ping success ({time.time() - start:.3f}s)")
                return True
    except urllib.error.URLError as e:
        print(f"Ping failed: {e}")
    except Exception as e:
        print(f"Ping error: {e}")
    return False

async def main():
    print("Checking if backend is reachable...")
    
    if not ping():
        print("Backend not reachable. Please start 'run_app.bat' or 'python backend/main.py'.")
        sys.exit(1)

    print("Backend is up. Monitoring responsiveness for 10s...")
    print("Please upload a video via the UI now if you want to test concurrency.")
    
    for i in range(5):
        ping()
        await asyncio.sleep(2)
    
    print("Done. Backend seems responsive.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())


import time
import os

LOG_FILE = r"d:\_Lecture Video Summarizer__\backend\logs\api.log"

def monitor():
    print(f"Monitoring {LOG_FILE} for completion...")
    if not os.path.exists(LOG_FILE):
        print("Log file not found.")
        return

    # Read to end
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        f.seek(0, 2)
        
        start_wait = time.time()
        while time.time() - start_wait < 60:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            
            print(line.strip())
            if "Processing complete" in line or "Complete in" in line:
                print("\nSUCCESS: Found completion message!")
                return
            if "Error" in line:
                print("\nFAILURE: Found error!")
                pass

if __name__ == "__main__":
    monitor()

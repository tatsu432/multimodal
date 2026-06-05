import cv2
import time
from pathlib import Path

RTMP_URL = "rtmp://localhost:1935/live/gopro"
SAVE_DIR = Path("sampled_frames")
SAVE_DIR.mkdir(exist_ok=True)

INTERVAL_SEC = 2.0

cap = cv2.VideoCapture(RTMP_URL)

if not cap.isOpened():
    raise RuntimeError(f"Could not open RTMP stream: {RTMP_URL}")

last_save_time = 0.0
frame_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        time.sleep(1)
        continue

    now = time.time()

    if now - last_save_time >= INTERVAL_SEC:
        last_save_time = now

        path = SAVE_DIR / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")

        # Later: send this frame to a VLM
        frame_id += 1

    cv2.imshow("GoPro RTMP", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import cv2

RTMP_URL = "rtmp://localhost:1935/live/gopro"

cap = cv2.VideoCapture(RTMP_URL)

if not cap.isOpened():
    raise RuntimeError(f"Could not open RTMP stream: {RTMP_URL}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    cv2.imshow("GoPro RTMP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
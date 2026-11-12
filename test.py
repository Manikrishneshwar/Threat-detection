import cv2, sys
backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
for i in range(5):
    cap = cv2.VideoCapture(i, backend)
    if not cap.isOpened():
        continue
    for _ in range(10):
        ok, f = cap.read()          # throw away a few frames while exposure settles
    if ok:
        print(f"Camera {i}: {f.shape[1]}x{f.shape[0]}, brightness {f.mean():.0f}")
        cv2.imwrite(f"index_{i}.jpg", f)
    cap.release()

#!/usr/bin/env python3
"""Quick test for the ELP-USBFHD01M-L180 (OV2710 sensor, 180 degree fisheye, USB 2.0).

It opens the camera in MJPG (needed for decent FPS over USB 2.0), tries
1080p, 720p and 480p, measures the real FPS of each, saves a still from each,
and then opens a live preview.

  pip install opencv-python
  python test_elp_camera.py                # index 1 (assuming the laptop webcam is 0)
  python test_elp_camera.py --index 0
  python test_elp_camera.py --no-preview   # just the FPS test
  python test_elp_camera.py --preview-mode 720p
"""

import argparse
import sys
import time

import cv2

# (name, width, height, fps we ask for), as advertised for this camera
MODES = [
    ("1080p", 1920, 1080, 30),
    ("720p",  1280,  720, 60),
    ("480p",   640,  480, 100),
]


def open_cam(index):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def set_mode(cap, w, h, fps):
    # Set MJPG before the size and fps, or many UVC cameras fall back to slow YUYV.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):  # let auto-exposure settle
        cap.read()


def fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)) or "?"


def measure(cap, seconds=3.0):
    n, start = 0, time.perf_counter()
    frame = None
    while time.perf_counter() - start < seconds:
        ok, f = cap.read()
        if ok:
            n += 1
            frame = f
    return n / (time.perf_counter() - start), frame


def sweep(cap):
    print(f"{'mode':<7}{'asked for':<16}{'got':<14}{'format':<8}{'real FPS':<14}")
    print("-" * 59)
    for label, w, h, fps in MODES:
        set_mode(cap, w, h, fps)
        gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps, frame = measure(cap)
        print(f"{label:<7}{f'{w}x{h}@{fps}':<16}{f'{gw}x{gh}':<14}"
              f"{fourcc_str(cap):<8}{real_fps:<14.1f}")
        if frame is not None:
            name = f"elp_{label}.jpg"
            cv2.imwrite(name, frame)
            if frame.mean() < 5:
                print(f"        ^ {name} is almost completely black. Lens cap still on, or is the room dark?")
    print("\nSaved a still from each mode: elp_1080p.jpg, elp_720p.jpg, elp_480p.jpg")
    print("Heads up: indoors the FPS is often lower, because the camera needs longer exposures.")


def preview(cap, label):
    mode = next(m for m in MODES if m[0] == label)
    set_mode(cap, *mode[1:])
    print(f"\nLive preview at {label}. Press q to quit, s to save a snapshot.")
    prev, fps_smooth = time.perf_counter(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed, stopping.")
            break
        now = time.perf_counter()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(frame, f"{label}  {fps_smooth:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("ELP fisheye test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(name, frame)
            print(f"Saved {name}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--preview-mode", choices=[m[0] for m in MODES], default="720p")
    args = p.parse_args()

    cap = open_cam(args.index)
    if not cap.isOpened():
        print(f"Couldn't open camera {args.index}. Try --index 0 or --index 2.")
        sys.exit(1)

    try:
        sweep(cap)
        if not args.no_preview:
            preview(cap, args.preview_mode)
    finally:
        cap.release()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture checkerboard images to calibrate one camera.

    python 01_capture_calib_images.py --name left            # index from config.CAMERAS["left"]
    python 01_capture_calib_images.py --name right
    python 01_capture_calib_images.py --name left --index 2  # use a different index

Keys:  SPACE save   a auto-capture on/off   c clear overlay   q quit

Aim for 25-40 images. With a fisheye, where the board lands matters more than
how many shots you take. The green overlay shows where boards have been so far.
Try to fill the whole circle, especially the edges, where the distortion is
strongest. Vary the tilt (up to about 45 degrees), the distance and the rotation,
and keep the board flat and still.
"""
import argparse
import time

import cv2
import numpy as np

import config
from stereo.camera import open_camera
from stereo.fisheye_calib import find_corners_fast
from stereo.viz import fit_width, put_lines

AUTO_INTERVAL_S = 1.0
AUTO_MIN_MOVE_PX = 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="which camera, e.g. left or right")
    ap.add_argument("--index", type=int, help="use this index instead of the one in config")
    args = ap.parse_args()

    index = args.index if args.index is not None else config.CAMERAS[args.name]
    out_dir = config.CALIB_IMG_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob("img_*.png")))
    pattern = (config.BOARD_COLS, config.BOARD_ROWS)

    # Paint in the boards from earlier sessions too, so you can see what's still missing.
    coverage = np.zeros((config.HEIGHT, config.WIDTH, 3), np.uint8)
    for p in sorted(out_dir.glob("img_*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (config.HEIGHT, config.WIDTH):
            continue
        c = find_corners_fast(img, pattern)
        if c is not None:
            cv2.fillConvexPoly(coverage, cv2.convexHull(c.astype(np.int32)), (0, 255, 0))

    cap = open_camera(index, config.WIDTH, config.HEIGHT, config.FPS)
    auto, last_t, last_corners = False, 0.0, None
    already = f"{n} image{'s' if n != 1 else ''} already there" if n else "folder is empty"
    print(f"Capturing for the {args.name} camera (index {index}). "
          f"Saving to {out_dir} ({already}).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed, stopping.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners_fast(gray, pattern)

            view = cv2.addWeighted(frame, 1.0, coverage, 0.35, 0)
            if corners is not None:
                cv2.drawChessboardCorners(view, pattern, corners.astype(np.float32), True)

            save = False
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("a"):
                auto = not auto
            elif k == ord("c"):
                coverage[:] = 0
            elif k == ord(" "):
                save = True
                if corners is None:
                    print("  Couldn't see the board in that one, but saved it anyway. "
                          "Calibration will just skip it if it's unusable.")
            # In auto mode, only save once the board has actually moved somewhere new.
            if auto and corners is not None and time.time() - last_t > AUTO_INTERVAL_S:
                moved = (np.inf if last_corners is None else
                         np.linalg.norm(corners - last_corners, axis=2).mean())
                save = moved > AUTO_MIN_MOVE_PX

            if save:
                path = out_dir / f"img_{n:03d}.png"   # PNG so there's no compression loss
                cv2.imwrite(str(path), frame)
                n += 1
                last_t, last_corners = time.time(), corners
                if corners is not None:
                    hull = cv2.convexHull(corners.astype(np.int32))
                    cv2.fillConvexPoly(coverage, hull, (0, 255, 0))
                print(f"  Saved {path.name}")

            put_lines(view, [
                f"{args.name} camera (index {index})   saved: {n}",
                f"board: {'found' if corners is not None else 'not found'}   auto: {'on' if auto else 'off'}",
                "SPACE save | a auto | c clear | q quit",
            ])
            cv2.imshow("calibration capture", fit_width(view))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"All done, {n} images in {out_dir}.")
    print(f"Next up: python 02_calibrate_intrinsics.py --name {args.name}")


if __name__ == "__main__":
    main()

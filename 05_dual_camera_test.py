#!/usr/bin/env python3
"""Run both cameras at once and check they keep up with each other.

    python 05_dual_camera_test.py
    python 05_dual_camera_test.py --left 2 --right 1
    python 05_dual_camera_test.py --undistort        # needs both *_intrinsics.yml files

Shows left and right side by side, with each camera's actual FPS and how far
apart in time the two latest frames are. Keys:
  s save a pair to data/pairs/   w swap left/right   l guide lines
  u undistortion on/off          q quit

Things to check:
  - Both should get close to config.FPS. If one drops or freezes, the USB bus is
    overloaded. Try different ports (ideally on different controllers) or a lower FPS.
  - The time gap should mostly stay under one frame (33 ms at 30 FPS). The cameras
    aren't hardware-synced, so anything moving will be a little off between them.
    Still scenes are fine.
  - Cover the left lens with a finger. The left image should go dark. If the right
    one does instead, press w and fix config.CAMERAS.
"""
import argparse
import time

import cv2

import config
from stereo.calib_io import load_intrinsics
from stereo.camera import ThreadedCamera
from stereo.fisheye_calib import undistort_maps
from stereo.viz import draw_hlines, fit_width, put_lines, side_by_side


def load_maps():
    size = (config.WIDTH, config.HEIGHT)
    maps = {}
    for name in ("left", "right"):
        c = load_intrinsics(config.intrinsics_path(name))
        m1, m2, _ = undistort_maps(c["K"], c["D"], size, balance=0.0)
        maps[name] = (m1, m2)
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=int, default=config.CAMERAS["left"])
    ap.add_argument("--right", type=int, default=config.CAMERAS["right"])
    ap.add_argument("--undistort", action="store_true")
    args = ap.parse_args()

    maps = None
    if args.undistort:
        try:
            maps = load_maps()
        except FileNotFoundError as e:
            print(f"Running without undistortion. {e}")

    size = (config.WIDTH, config.HEIGHT)
    print(f"Opening left (index {args.left}) and right (index {args.right}) "
          f"at {size[0]}x{size[1]}, {config.FPS} FPS...")
    cams = {"left": ThreadedCamera(args.left, *size, config.FPS),
            "right": ThreadedCamera(args.right, *size, config.FPS)}

    config.PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    n_pair = len(list(config.PAIRS_DIR.glob("left_*.png")))
    last = {k: (time.perf_counter(), 0) for k in cams}
    fps = {k: 0.0 for k in cams}
    lines, use_und = False, maps is not None
    try:
        while True:
            fl, tl, cl = cams["left"].read()
            fr, tr, cr = cams["right"].read()
            if fl is None or fr is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            for k, c in (("left", cl), ("right", cr)):
                t0, c0 = last[k]
                if now - t0 >= 1.0:
                    fps[k] = (c - c0) / (now - t0)
                    last[k] = (now, c)
            skew_ms = (tr - tl) * 1000

            raw_l, raw_r = fl, fr
            if use_und:
                fl = cv2.remap(fl, *maps["left"], cv2.INTER_LINEAR)
                fr = cv2.remap(fr, *maps["right"], cv2.INTER_LINEAR)
            view = side_by_side(fl, fr)
            if lines:
                view = draw_hlines(view)
            put_lines(view, [
                f"left (index {cams['left'].index}): {fps['left']:5.1f} fps   "
                f"right (index {cams['right'].index}): {fps['right']:5.1f} fps   gap {skew_ms:+6.1f} ms",
                f"undistort {'on' if use_und else 'off'}   s save | w swap | l lines | u undistort | q quit",
            ])
            cv2.imshow("dual camera test", fit_width(view))

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("w"):
                cams["left"], cams["right"] = cams["right"], cams["left"]
                if maps is not None:
                    # Each calibration belongs to a physical camera, so it has to swap along with it.
                    maps["left"], maps["right"] = maps["right"], maps["left"]
                print(f"Swapped. Left is now index {cams['left'].index} and right is "
                      f"index {cams['right'].index}. To make it stick, update config.CAMERAS "
                      "and swap the files in calib/ too.")
            elif k == ord("l"):
                lines = not lines
            elif k == ord("u") and maps is not None:
                use_und = not use_und
            elif k == ord("s"):
                # Always save the original frames. Undistortion and rectification happen later.
                cv2.imwrite(str(config.PAIRS_DIR / f"left_{n_pair:03d}.png"), raw_l)
                cv2.imwrite(str(config.PAIRS_DIR / f"right_{n_pair:03d}.png"), raw_r)
                print(f"Saved pair {n_pair:03d} (the frames were {abs(skew_ms):.1f} ms apart)")
                n_pair += 1
    finally:
        for c in cams.values():
            c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

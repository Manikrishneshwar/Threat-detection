#!/usr/bin/env python3
"""Figure out which camera index is which.

Tries each index, prints what it finds, and saves a frame from each one to
output/probe/index_<i>.jpg. Cover one lens with your finger while it runs and
that camera's image will come out dark.

    python 00_list_cameras.py
    python 00_list_cameras.py --max 8
"""
import argparse

import cv2

import config
from stereo.camera import open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=6)
    args = ap.parse_args()

    out = config.OUTPUT_DIR / "probe"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.max):
        try:
            cap = open_camera(i, config.WIDTH, config.HEIGHT, config.FPS)
        except RuntimeError as e:
            msg = str(e)
            if "Could not open" not in msg:
                print(f"Camera {i}: it opens, but {msg[0].lower() + msg[1:]}")
            continue
        ok, f = cap.read()
        cap.release()
        if not ok:
            print(f"Camera {i}: it opens, but no frames are coming through.")
            continue
        path = out / f"index_{i}.jpg"
        cv2.imwrite(str(path), f)
        print(f"Camera {i}: {f.shape[1]}x{f.shape[0]}, brightness {f.mean():5.1f}, saved to {path}")
    print(f"\nWhen you know which is which, put the indices in config.CAMERAS "
          f"(right now it's {config.CAMERAS}).")


if __name__ == "__main__":
    main()

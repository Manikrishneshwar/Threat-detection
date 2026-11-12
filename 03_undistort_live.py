#!/usr/bin/env python3
"""Live view of one camera: original on the left, undistorted on the right.

    python 03_undistort_live.py --name left

Sliders:
  balance    0 crops to the clean middle, 100 keeps as much view as possible (stretched edges)
  fov x100   above 100 zooms out, below 100 zooms in
Keys: l guide lines on/off   s save   q quit

What to look for: door frames, table edges and ceiling lines should come out
straight on the right, even near the borders. If they still bend, the
calibration needs more shots near the edges.
"""
import argparse
import time

import cv2

import config
from stereo.calib_io import load_intrinsics
from stereo.camera import open_camera
from stereo.fisheye_calib import undistort_maps
from stereo.viz import draw_hlines, fit_width, put_lines, side_by_side

WIN = "original | undistorted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--index", type=int)
    args = ap.parse_args()

    calib = load_intrinsics(config.intrinsics_path(args.name))
    size = (config.WIDTH, config.HEIGHT)
    if calib["size"] != size:
        raise SystemExit(f"This calibration was made at {calib['size'][0]}x{calib['size'][1]}, "
                         f"but config is set to {size[0]}x{size[1]}. They need to match.")
    index = args.index if args.index is not None else config.CAMERAS[args.name]
    cap = open_camera(index, *size, config.FPS)

    cv2.namedWindow(WIN)
    cv2.createTrackbar("balance", WIN, 0, 100, lambda v: None)
    cv2.createTrackbar("fov x100", WIN, 100, 200, lambda v: None)
    state, maps, lines = None, None, False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bal = cv2.getTrackbarPos("balance", WIN) / 100
            fov = max(cv2.getTrackbarPos("fov x100", WIN), 20) / 100
            if (bal, fov) != state:   # building the maps is slow, so only redo it when a slider moves
                m1, m2, K_new = undistort_maps(calib["K"], calib["D"], size, bal, fov)
                maps, state = (m1, m2), (bal, fov)
            und = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
            if lines:
                und = draw_hlines(und)
            put_lines(und, [f"balance {bal:.2f}  fov {fov:.2f}  focal {K_new[0,0]:.0f}px"])
            view = side_by_side(frame, und)
            cv2.imshow(WIN, fit_width(view))

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("l"):
                lines = not lines
            if k == ord("s"):
                p = config.OUTPUT_DIR / f"{args.name}_undistort_{int(time.time())}.jpg"
                cv2.imwrite(str(p), view)
                print(f"Saved {p}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

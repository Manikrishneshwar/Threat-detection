#!/usr/bin/env python3
"""Measure distances live with both cameras.

    python 07_depth_live.py
    python 07_depth_live.py --pair 022     # use a saved pair from data/pairs/ instead of the cameras
    python 07_depth_live.py --scale 0.5    # about 4x faster, less precise further out

Shows the left camera (rectified) next to a depth map. Red is close, blue is far,
black means no reading. The distance at the crosshair is shown on screen. Click
anywhere on either half to move it.

Keys: s save   q quit

What to expect with our 117 mm baseline:
  - Nothing closer than about 0.3 m. Closer than that it won't say "no reading",
    it'll just show a wrong value around 0.3-0.4 m.
  - Within about 1.5 m it's good to a percent or two. Past 3 m it gets rough.
  - It needs texture. Blank walls, glare and dark areas come out black.
"""
import argparse
import time

import cv2
import numpy as np

import config
from stereo.calib_io import load_stereo
from stereo.camera import ThreadedCamera
from stereo.depth import DepthEstimator, colorize, distance_at
from stereo.viz import fit_width, put_lines, side_by_side

WIN = "depth"


def render(rect_left, depth, point, fps=None):
    """Left image and depth map side by side, with the crosshair and its distance."""
    x, y = point
    d = distance_at(depth, x, y)
    view = side_by_side(rect_left.copy(), colorize(depth))
    w = rect_left.shape[1]
    for off in (0, w):
        cv2.drawMarker(view, (x + off, y), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    text = f"{d:.2f} m" if d is not None else "no reading here"
    lines = [f"distance: {text}"]
    if fps is not None:
        lines.append(f"{fps:4.1f} fps   click to move the crosshair | s save | q quit")
    put_lines(view, lines)
    return view, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="run on data/pairs/left_<id>.png and right_<id>.png instead of the cameras")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink the images before matching (0.5 is faster)")
    ap.add_argument("--left", type=int, default=config.CAMERAS["left"])
    ap.add_argument("--right", type=int, default=config.CAMERAS["right"])
    args = ap.parse_args()

    stereo = load_stereo(config.STEREO_PATH)
    if stereo["size"] != (config.WIDTH, config.HEIGHT):
        raise SystemExit("The stereo calibration was made at a different resolution than config is set to.")
    est = DepthEstimator(stereo, scale=args.scale)
    print(f"Baseline {est.baseline_mm:.1f} mm, focal length {est.f:.0f} px, "
          f"searching up to {est.num_disp} px of disparity (closest about 0.3 m).")

    cams = None
    if args.pair:
        frames = [cv2.imread(str(config.PAIRS_DIR / f"{s}_{args.pair}.png")) for s in ("left", "right")]
        if any(f is None for f in frames):
            raise SystemExit(f"Couldn't read pair {args.pair} from {config.PAIRS_DIR}.")
    else:
        size = (config.WIDTH, config.HEIGHT)
        print(f"Opening left (index {args.left}) and right (index {args.right})...")
        cams = (ThreadedCamera(args.left, *size, config.FPS), ThreadedCamera(args.right, *size, config.FPS))

    w = int(config.WIDTH * args.scale)
    h = int(config.HEIGHT * args.scale)
    state = {"point": (w // 2, h // 2), "shown_scale": 1.0}

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = int(x / state["shown_scale"]), int(y / state["shown_scale"])
            state["point"] = (min(x % w, w - 1), min(y, h - 1))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)
    fps, last = None, time.perf_counter()
    try:
        while True:
            if cams:
                frames = [c.read()[0] for c in cams]
                if any(f is None for f in frames):
                    time.sleep(0.01)
                    continue
            rl, rr, depth = est(*frames)
            now = time.perf_counter()
            if cams:
                fps = 1 / max(now - last, 1e-6) if fps is None else 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            view, d = render(rl, depth, state["point"], fps)
            shown = fit_width(view)
            state["shown_scale"] = shown.shape[1] / view.shape[1]
            cv2.imshow(WIN, shown)

            k = cv2.waitKey(1 if cams else 30) & 0xFF
            if k == ord("q") or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord("s"):
                stamp = int(time.time())
                config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(config.OUTPUT_DIR / f"depth_{stamp}.jpg"), view)
                np.save(config.OUTPUT_DIR / f"depth_{stamp}.npy", depth)
                where = f"{d:.2f} m" if d is not None else "no reading"
                print(f"Saved depth_{stamp}.jpg and depth_{stamp}.npy to {config.OUTPUT_DIR} "
                      f"(crosshair at {where})")
    finally:
        if cams:
            for c in cams:
                c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

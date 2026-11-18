#!/usr/bin/env python3
"""Capture obstacle images (stairs, poles, trees, ...) to label and train YOLO on later.

    python 08_capture_detect_images.py --cls stairs
    python 08_capture_detect_images.py --cls stairs --interval 0.5     # auto-save twice a second
    python 08_capture_detect_images.py --cls background                # scenes with none of the classes

Shows the rectified left image, which is the one that gets labelled. Keys:
  s save one   a auto-save on/off (every --interval seconds)   q quit

Each save writes three files with the same name to data/detect/<cls>/:
  rect_left/  rectified left image. Label these: the depth map from 07_depth_live.py
              is in this same image, so a box drawn on it lines up with the depth pixels.
  left/ right/  the raw frames, kept in case the calibration is redone or the
              detector should run on something else later.

The folder name is just a note about what you were pointing at. The real classes
come from the labels, so a stairs photo with a pole in it is fine.

Tips for a useful dataset:
  - Vary distance (0.5-5 m), angle, lighting and location. Ten spots beat 500 frames of one.
  - Auto mode skips frames that are blurry or barely changed since the last save,
    so you can walk around with it on.
  - Save some "background" scenes too (corridors, open floor). They cut down on
    false alarms.
"""
import argparse
import re
import time

import cv2
import numpy as np

import config
from stereo.calib_io import load_stereo
from stereo.camera import ThreadedCamera
from stereo.stereo_calib import rectify_maps
from stereo.viz import fit_width, put_lines

WIN = "capture detection images"


def sharpness(img):
    """Variance of the Laplacian. Low means blurry (motion blur when walking, mostly)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def thumb(img):
    """Small grey version for checking whether the scene changed."""
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)


def next_index(folder):
    nums = [int(m.group(1)) for p in folder.glob("*.jpg") if (m := re.search(r"_(\d+)\.jpg$", p.name))]
    return max(nums, default=-1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, help="what you're capturing, e.g. stairs, pole, tree, background")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto-saves")
    ap.add_argument("--min-sharp", type=float, default=60.0,
                    help="auto mode skips frames below this sharpness (shown on screen; tune to your lighting)")
    ap.add_argument("--min-change", type=float, default=6.0,
                    help="auto mode skips frames that differ from the last save by less than this (mean grey level)")
    ap.add_argument("--left", type=int, default=config.CAMERAS["left"])
    ap.add_argument("--right", type=int, default=config.CAMERAS["right"])
    args = ap.parse_args()

    cls = args.cls.strip().lower().replace(" ", "_")
    stereo = load_stereo(config.STEREO_PATH)
    size = (config.WIDTH, config.HEIGHT)
    if stereo["size"] != size:
        raise SystemExit("The stereo calibration was made at a different resolution than config is set to.")
    maps_l = rectify_maps((stereo["K1"], stereo["D1"]), stereo["R1"], stereo["P1"], size)

    out = config.DETECT_DIR / cls
    dirs = {k: out / k for k in ("rect_left", "left", "right")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    n = next_index(dirs["rect_left"])
    n_session = 0

    print(f"Opening left (index {args.left}) and right (index {args.right})...")
    cams = (ThreadedCamera(args.left, *size, config.FPS), ThreadedCamera(args.right, *size, config.FPS))
    print(f"Saving to {out}, starting at #{n:04d}.")

    auto, last_save, last_thumb, note, shown = False, 0.0, None, "", False
    try:
        while True:
            (fl, tl, _), (fr, tr, _) = (c.read() for c in cams)
            if fl is None or fr is None:
                time.sleep(0.01)
                continue
            rl = cv2.remap(fl, *maps_l, cv2.INTER_LINEAR)
            sharp = sharpness(rl)
            now = time.perf_counter()

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or (shown and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1):
                break
            if k == ord("a"):
                auto = not auto
                print(f"Auto-save {'on' if auto else 'off'}.")

            want = k == ord("s")
            if auto and not want and now - last_save >= args.interval:
                th = thumb(rl)
                if sharp < args.min_sharp:
                    note = "auto: skipped (blurry)"
                elif last_thumb is not None and np.abs(th.astype(np.int16) - last_thumb).mean() < args.min_change:
                    note = "auto: skipped (no change)"
                else:
                    want = True

            if want:
                name = f"{cls}_{n:04d}.jpg"
                cv2.imwrite(str(dirs["rect_left"] / name), rl, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(dirs["left"] / name), fl, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(dirs["right"] / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                skew = abs(tr - tl) * 1000
                note = f"saved {name}"
                print(f"Saved {name}  (sharpness {sharp:.0f}, frames {skew:.0f} ms apart)")
                n += 1
                n_session += 1
                last_save, last_thumb = now, thumb(rl).astype(np.int16)

            view = rl.copy()
            put_lines(view, [
                f"class: {cls}   saved this run: {n_session}   next #{n:04d}",
                f"sharpness {sharp:5.0f} (auto min {args.min_sharp:.0f})   auto {'ON' if auto else 'off'}",
                note,
                "s save | a auto | q quit",
            ], color=(0, 0, 255) if auto else (0, 255, 0))
            cv2.imshow(WIN, fit_width(view, 1280))
            shown = True
    finally:
        for c in cams:
            c.release()
        cv2.destroyAllWindows()
        print(f"Saved {n_session} images this run to {out}.")


if __name__ == "__main__":
    main()

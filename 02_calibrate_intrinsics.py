#!/usr/bin/env python3
"""Work out one camera's fisheye intrinsics (K and D) from the saved checkerboard images.

    python 02_calibrate_intrinsics.py --name left
    python 02_calibrate_intrinsics.py --name left --show     # look at each detection

Writes calib/<name>_intrinsics.yml and output/<name>_undistort_check.jpg.

How to tell if it worked:
  RMS error         under 0.5 px is great, under 1.0 px is fine, above that recapture
  Coverage          ideally over 60% of the image, including the edges
  Check image       straight things in the room should look straight
"""
import argparse
from pathlib import Path

import cv2

import config
from stereo.calib_io import save_intrinsics
from stereo.fisheye_calib import (board_object_points, calibrate_fisheye, corner_coverage,
                                  find_corners_accurate, undistort_maps)
from stereo.viz import fit_width, side_by_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", help="read images from here instead of data/calib_images/<name>")
    ap.add_argument("--show", action="store_true", help="show each detected board")
    ap.add_argument("--reject-px", type=float, default=1.0,
                    help="throw out images with more error than this (px) and recalibrate")
    args = ap.parse_args()

    folder = config.CALIB_IMG_DIR / args.name if args.images is None else Path(args.images)
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise SystemExit(f"There are no .png images in {folder}. "
                         "Capture some with 01_capture_calib_images.py first.")

    pattern = (config.BOARD_COLS, config.BOARD_ROWS)
    objp = board_object_points(*pattern, config.SQUARE_MM)
    obj_pts, img_pts, names, size = [], [], [], None

    print(f"Looking for the {pattern[0]}x{pattern[1]} board in {len(paths)} images...")
    for p in paths:
        img = cv2.imread(str(p))
        s = (img.shape[1], img.shape[0])
        if size is None:
            size = s
        elif s != size:
            print(f"  {p.name}: skipping, it's {s[0]}x{s[1]} but the others are {size[0]}x{size[1]}")
            continue
        corners = find_corners_accurate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
        print(f"  {p.name}: {'found it' if corners is not None else 'no board found'}")
        if corners is None:
            continue
        obj_pts.append(objp)
        img_pts.append(corners)
        names.append(p.name)
        if args.show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            cv2.imshow("detections (any key for next, q to stop showing)", fit_width(vis))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                args.show = False
    cv2.destroyAllWindows()

    cov, _ = corner_coverage(img_pts, size)
    print(f"\nFound the board in {len(img_pts)} of {len(paths)} images, "
          f"covering {cov:.0%} of the frame.")
    if cov < 0.6:
        print("  That's not much coverage. Try adding shots with the board near the edges.")

    r = calibrate_fisheye(obj_pts, img_pts, size, names)
    if r["dropped"]:
        print(f"OpenCV couldn't use these, so they're left out: {', '.join(r['dropped'])}")

    # Images that fit much worse than the rest are usually blurry or have a corner
    # detected in the wrong place. Leave them out (the files stay) and recalibrate.
    for _ in range(3):
        bad = {n for n, e in zip(r["names"], r["per_view_err"]) if e > args.reject_px}
        if not bad or len(r["names"]) - len(bad) < 15:
            break
        print(f"RMS was {r['rms']:.3f} px. Leaving out {len(bad)} image(s) that were off "
              f"by more than {args.reject_px} px: {', '.join(sorted(bad))}")
        keep = [i for i, n in enumerate(names) if n in r["names"] and n not in bad]
        obj_pts = [obj_pts[i] for i in keep]
        img_pts = [img_pts[i] for i in keep]
        names = [names[i] for i in keep]
        r = calibrate_fisheye(obj_pts, img_pts, size, names)

    K, D = r["K"], r["D"]
    print(f"\nFinal RMS error: {r['rms']:.3f} px, using {len(r['names'])} images")
    print(f"Focal length fx={K[0,0]:.2f}, fy={K[1,1]:.2f}   center cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print("Distortion k1..k4:", " ".join(f"{d:+.5f}" for d in D.ravel()))
    worst = sorted(zip(r["per_view_err"], r["names"]), reverse=True)[:5]
    print("Worst images (if any are way above the rest, delete them and run this again):")
    for e, n in worst:
        print(f"  {n}: {e:.3f} px")

    out = config.intrinsics_path(args.name)
    save_intrinsics(out, K, D, size, r["rms"], len(r["names"]))
    print(f"\nSaved the calibration to {out}")

    # Undistort the first image so you can eyeball the result.
    sample = cv2.imread(str(paths[0]))
    m1, m2, _ = undistort_maps(K, D, size, balance=0.0)
    und = cv2.remap(sample, m1, m2, cv2.INTER_LINEAR)
    check = config.OUTPUT_DIR / f"{args.name}_undistort_check.jpg"
    check.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(check), side_by_side(sample, und))
    print(f"Saved a before/after image to {check} (original on the left, undistorted on the right)")
    print(f"Next up: python 03_undistort_live.py --name {args.name}")


if __name__ == "__main__":
    main()

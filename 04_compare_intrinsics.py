#!/usr/bin/env python3
"""Compare the two cameras' calibrations. Could one calibration work for both?

    python 04_compare_intrinsics.py                    # left vs right
    python 04_compare_intrinsics.py --baseline-mm 120  # with your real lens spacing

First it lists how the parameters differ. Then the part that actually matters:
if you undistorted camera B using camera A's calibration, how many pixels off
would things be?
  - vertical error breaks row alignment, so stereo matching fails
  - horizontal error shows up as disparity error, so the depth comes out wrong
Finally it turns that into a depth error at a few distances.
"""
import argparse

import cv2
import numpy as np

import config
from stereo.calib_io import load_intrinsics
from stereo.fisheye_calib import undistort_maps, valid_theta_max


def in_valid_range(pts, calib):
    """True for raw pixels that sit inside the angle where this calibration's lens model holds."""
    K, k = calib["K"], np.asarray(calib["D"], np.float64).ravel()
    t = valid_theta_max(calib["D"])
    r_max = t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)
    p = pts.reshape(-1, 2)
    r = np.hypot((p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1])
    return r < r_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="left")
    ap.add_argument("--b", default="right")
    ap.add_argument("--baseline-mm", type=float, default=100.0,
                    help="distance between the two lenses in mm (100 is just a placeholder)")
    ap.add_argument("--balance", type=float, default=0.0)
    args = ap.parse_args()

    A = load_intrinsics(config.intrinsics_path(args.a))
    B = load_intrinsics(config.intrinsics_path(args.b))
    if A["size"] != B["size"]:
        raise SystemExit(f"These two were calibrated at different resolutions "
                         f"({A['size']} vs {B['size']}), so they can't be compared.")
    size = A["size"]

    print(f"{'param':<8}{args.a:>12}{args.b:>12}{'diff':>12}{'diff %':>9}")
    rows = [("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))]
    for name, ij in rows:
        a, b = A["K"][ij], B["K"][ij]
        print(f"{name:<8}{a:>12.2f}{b:>12.2f}{b - a:>12.2f}{100 * (b - a) / a:>8.2f}%")
    for i in range(4):
        a, b = A["D"][i, 0], B["D"][i, 0]
        print(f"{'k' + str(i + 1):<8}{a:>12.5f}{b:>12.5f}{b - a:>12.5f}")
    print(f"{'rms':<8}{A['rms']:>12.3f}{B['rms']:>12.3f}")

    # Push the same grid of raw pixels through each calibration into one shared
    # undistorted camera (A's) and see how far apart the results land.
    _, _, K_new = undistort_maps(A["K"], A["D"], size, balance=args.balance)
    w, h = size
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(0, h - 1, 36))
    raw = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
    pa = cv2.fisheye.undistortPoints(raw, A["K"], A["D"], P=K_new).reshape(-1, 2)
    pb = cv2.fisheye.undistortPoints(raw, B["K"], B["D"], P=K_new).reshape(-1, 2)
    # Only compare pixels where both lens models are valid. Out past that (like the black
    # image corners) OpenCV returns garbage that can be millions of pixels off.
    valid = in_valid_range(raw, A) & in_valid_range(raw, B)
    inside = (valid & (pa[:, 0] >= 0) & (pa[:, 0] < w) & (pa[:, 1] >= 0) & (pa[:, 1] < h)
              & np.isfinite(pa).all(1) & np.isfinite(pb).all(1))
    d = (pb - pa)[inside]
    dx, dy = np.abs(d[:, 0]), np.abs(d[:, 1])

    print(f"\nIf you used {args.a}'s calibration on the {args.b} camera "
          f"(undistorted at f = {K_new[0, 0]:.0f} px, balance {args.balance}):")
    print(f"  vertical error   average {dy.mean():6.2f} px, worst {dy.max():6.2f} px   (row alignment)")
    print(f"  horizontal error average {dx.mean():6.2f} px, worst {dx.max():6.2f} px   (disparity)")

    f, B_mm = K_new[0, 0], args.baseline_mm
    print(f"\nWhat that horizontal error does to depth (baseline {B_mm:.0f} mm, dZ = Z^2 * dd / (f*B)):")
    for Z in (500, 1000, 2000, 5000):
        print(f"  at {Z / 1000:.1f} m, each pixel is worth {Z * Z / (f * B_mm):6.0f} mm, "
              f"so the average error costs {Z * Z * dx.mean() / (f * B_mm):6.0f} mm")

    if max(dy.mean(), dx.mean()) < 0.5:
        verdict = "close enough that one calibration could serve both cameras"
    else:
        verdict = "too different, so keep a separate calibration for each camera"
    print(f"\nVerdict: {verdict}. (Rule of thumb: under 0.5 px on average doesn't matter.)")


if __name__ == "__main__":
    main()

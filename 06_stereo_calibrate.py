#!/usr/bin/env python3
"""Calibrate the two cameras as a pair, then rectify them.

    python 06_stereo_calibrate.py
    python 06_stereo_calibrate.py --reject-px 0.8     # be stricter about bad pairs

Needs both calib/*_intrinsics.yml files and checkerboard pairs in data/pairs/
(captured with 05_dual_camera_test.py, board fully visible in both views).

Writes calib/stereo.yml and output/rectified_check.jpg.

How to tell if it worked:
  RMS error            under 0.5 px is great, under 1.0 px is fine
  Baseline             should be close to what you measured with a tape
  Row mismatch         after rectification, the same corner should sit on the same row
                       in both images, ideally well under 1 px apart
  Check image          things should line up along the yellow lines
"""
import argparse

import cv2
import numpy as np

import config
from stereo.calib_io import load_intrinsics, save_stereo
from stereo.fisheye_calib import board_object_points, find_corners_accurate
from stereo.stereo_calib import (match_corner_order, rectify_maps, rectify_points,
                                 stereo_calibrate, stereo_rectify)
from stereo.viz import draw_hlines, side_by_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reject-px", type=float, default=1.0,
                    help="leave out pairs with more error than this (px) and recalibrate")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="0 crops the rectified images to the clean middle, 1 keeps more of the view")
    args = ap.parse_args()

    size = (config.WIDTH, config.HEIGHT)
    cams = {}
    for name in ("left", "right"):
        c = load_intrinsics(config.intrinsics_path(name))
        if c["size"] != size:
            raise SystemExit(f"The {name} calibration was made at {c['size'][0]}x{c['size'][1]}, "
                             f"but config is set to {size[0]}x{size[1]}.")
        cams[name] = (c["K"], c["D"])

    pattern = (config.BOARD_COLS, config.BOARD_ROWS)
    objp = board_object_points(*pattern, config.SQUARE_MM)
    lefts = sorted(config.PAIRS_DIR.glob("left_*.png"))
    if not lefts:
        raise SystemExit(f"There are no pairs in {config.PAIRS_DIR}. Capture some with 05_dual_camera_test.py first.")

    print(f"Looking for the {pattern[0]}x{pattern[1]} board in {len(lefts)} pairs...")
    ids, left_pts, right_pts = [], [], []
    for lp in lefts:
        pid = lp.stem.split("_", 1)[1]
        rp = lp.with_name(f"right_{pid}.png")
        if not rp.exists():
            print(f"  {pid}: no matching right image, skipping")
            continue
        corners = {}
        for side, path in (("left", lp), ("right", rp)):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            corners[side] = None if img.shape[::-1] != size else find_corners_accurate(img, pattern)
        missing = [s for s in corners if corners[s] is None]
        if missing:
            print(f"  {pid}: no board in the {' or '.join(missing)} image, skipping")
            continue
        right, flipped = match_corner_order(corners["left"], corners["right"], objp,
                                            cams["left"], cams["right"])
        print(f"  {pid}: found it in both{' (fixed the corner order on the right)' if flipped else ''}")
        ids.append(pid)
        left_pts.append(corners["left"])
        right_pts.append(right)

    print(f"\nCalibrating with {len(ids)} pairs...")
    r = stereo_calibrate(objp, left_pts, right_pts, cams["left"], cams["right"], size)

    # Same idea as for a single camera: pairs that fit much worse than the rest
    # usually had the board move between the two frames. Leave them out and redo it.
    for _ in range(3):
        bad = {i for i, e in zip(ids, r["errors"]) if e > args.reject_px}
        if not bad or len(ids) - len(bad) < 10:
            break
        print(f"RMS was {r['rms']:.3f} px. Leaving out {len(bad)} pair(s) that were off "
              f"by more than {args.reject_px} px: {', '.join(sorted(bad))}")
        keep = [k for k, i in enumerate(ids) if i not in bad]
        ids = [ids[k] for k in keep]
        left_pts = [left_pts[k] for k in keep]
        right_pts = [right_pts[k] for k in keep]
        r = stereo_calibrate(objp, left_pts, right_pts, cams["left"], cams["right"], size)

    R, T = r["R"], r["T"]
    baseline = float(np.linalg.norm(T))
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    print(f"\nFinal RMS error: {r['rms']:.3f} px, using {len(ids)} pairs")
    print(f"Baseline: {baseline:.1f} mm (you measured {config.BASELINE_TAPE_MM:.0f} mm with the tape)")
    # T maps points from the left camera's frame into the right one's, so the right
    # camera's position as seen from the left camera is -R^T T.
    pos = (-R.T @ T).ravel()
    print(f"Seen from the left camera, the right one sits at x={pos[0]:+.1f}, y={pos[1]:+.1f}, "
          f"z={pos[2]:+.1f} mm (x is to the right, y down, z forward), turned {angle:.2f} degrees.")
    worst = sorted(zip(r["errors"], ids), reverse=True)[:5]
    print("Worst pairs (if any are way above the rest, delete them and run this again):")
    for e, i in worst:
        print(f"  {i}: {e:.3f} px")

    R1, R2, P1, P2, Q = stereo_rectify(cams["left"], cams["right"], size, R, T, balance=args.balance)

    # The real test of rectification: each corner should land on the same row in both images.
    dy, depth_err = [], []
    f, B = P1[0, 0], -P2[0, 3] / P2[0, 0]
    for l, rt, rv, tv in zip(left_pts, right_pts, r["rvecs"], r["tvecs"]):
        ql = rectify_points(l, cams["left"], R1, P1)
        qr = rectify_points(rt, cams["right"], R2, P2)
        dy.append(np.abs(ql[:, 1] - qr[:, 1]))
        # And depth from disparity should agree with where calibration put the board.
        z_true = (R1 @ (cv2.Rodrigues(rv)[0] @ objp.reshape(-1, 3).T + np.asarray(tv).reshape(3, 1)))[2]
        z_disp = f * B / (ql[:, 0] - qr[:, 0])
        depth_err.append(np.abs(z_disp - z_true) / z_true)
    dy, depth_err = np.concatenate(dy), np.concatenate(depth_err)
    print(f"\nRectified images: focal length {f:.0f} px, baseline {B:.1f} mm")
    print(f"Row mismatch between the two images: average {dy.mean():.2f} px, worst {dy.max():.2f} px")
    print(f"Depth from disparity vs. the calibrated board positions: off by {100 * np.median(depth_err):.1f}% "
          f"on average ({100 * np.percentile(depth_err, 95):.1f}% for the worst 5% of corners)")

    save_stereo(config.STEREO_PATH, size, r["rms"], len(ids), K1=cams["left"][0], D1=cams["left"][1],
                K2=cams["right"][0], D2=cams["right"][1], R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    print(f"\nSaved the stereo calibration to {config.STEREO_PATH}")

    # Rectify one of the pairs so you can eyeball it. A mid-range one shows the most.
    pid = ids[len(ids) // 2]
    views = []
    for side, cam, R_rect, P in (("left", cams["left"], R1, P1), ("right", cams["right"], R2, P2)):
        m1, m2 = rectify_maps(cam, R_rect, P, size)
        views.append(cv2.remap(cv2.imread(str(config.PAIRS_DIR / f"{side}_{pid}.png")), m1, m2, cv2.INTER_LINEAR))
    check = config.OUTPUT_DIR / "rectified_check.jpg"
    check.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(check), draw_hlines(side_by_side(*views)))
    print(f"Saved rectified pair {pid} to {check}. The same point should sit on the same yellow line in both halves.")


if __name__ == "__main__":
    main()

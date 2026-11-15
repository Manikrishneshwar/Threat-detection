"""Checkerboard detection, fisheye calibration and undistortion.

We use OpenCV's fisheye model (K plus four distortion terms k1..k4) because
the regular pinhole model can't handle a 180 degree lens.
"""
import re

import cv2
import numpy as np

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
MIN_VIEWS = 8


def board_object_points(cols, rows, square):
    """Corner positions on the flat board (z = 0), in mm."""
    objp = np.zeros((cols * rows, 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def find_corners_fast(gray, pattern, scale=0.5):
    """Quick, rough detection on a downscaled frame. Good enough for the live preview."""
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    return corners / scale if ok else None


def find_corners_accurate(gray, pattern):
    """Slower, sub-pixel detection for the actual calibration. Returns None if no board."""
    # The SB detector handles heavy fisheye distortion a lot better than the classic one,
    # so try it first and only fall back if it fails.
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
    if ok:
        return corners.reshape(-1, 1, 2).astype(np.float32)
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners.astype(np.float32)


def calibrate_fisheye(obj_points, img_points, image_size, names):
    """Run cv2.fisheye.calibrate, throwing out any view it chokes on.

    With CALIB_CHECK_COND, OpenCV errors out and tells us which view is bad
    (usually a board that's tiny or seen almost edge-on). We drop that one and retry.
    """
    obj = [np.asarray(o, np.float64).reshape(-1, 1, 3) for o in obj_points]
    img = [np.asarray(i, np.float64).reshape(-1, 1, 2) for i in img_points]
    names = list(names)
    w, h = image_size
    # A 180 degree fisheye roughly spanning the image width has f around w / pi.
    # Starting there converges much more reliably than OpenCV's default guess.
    K_guess = np.array([[w / np.pi, 0, w / 2], [0, w / np.pi, h / 2], [0, 0, 1]], np.float64)
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             | cv2.fisheye.CALIB_CHECK_COND
             | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    dropped = []
    while True:
        if len(obj) < MIN_VIEWS:
            raise RuntimeError(f"Only {len(obj)} usable images left, and I need at least {MIN_VIEWS}. "
                               "Take some more with the board in different positions.")
        K, D = K_guess.copy(), np.zeros((4, 1))
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj, img, image_size, K, D, flags=flags, criteria=CRITERIA)
            break
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if m:
                i = int(m.group(1))
                dropped.append(names.pop(i))
                obj.pop(i)
                img.pop(i)
            elif flags & cv2.fisheye.CALIB_CHECK_COND:
                print("  Calibration failed without saying which image was the problem, "
                      "so I'm retrying with the condition check turned off.")
                flags &= ~cv2.fisheye.CALIB_CHECK_COND
            else:
                raise

    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, r, t, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))))

    return dict(rms=rms, K=K, D=D, rvecs=rvecs, tvecs=tvecs, names=names,
                per_view_err=per_view, dropped=dropped)


def corner_coverage(img_points, image_size, grid=8):
    """How much of the image (split into a grid x grid) has at least one corner in it."""
    w, h = image_size
    hit = np.zeros((grid, grid), bool)
    for pts in img_points:
        p = pts.reshape(-1, 2)
        gx = np.clip((p[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((p[:, 1] / h * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return hit.mean(), hit


def undistort_maps(K, D, image_size, balance=0.0, fov_scale=1.0, out_size=None):
    """Build the remap tables that turn a fisheye frame into a normal, straight-line image.

    balance: 0 crops to the clean middle, 1 keeps as much of the view as possible
    (the edges get very stretched). fov_scale above 1 zooms out, below 1 zooms in.
    You can't fit the full 180 degrees into a flat image, so something always gets cut.

    Returns map1, map2 and K_new (the camera matrix of the undistorted image).
    """
    out_size = tuple(out_size or image_size)
    K_new = new_camera_matrix(K, D, image_size, balance, fov_scale, out_size)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, out_size, cv2.CV_16SC2)
    return m1, m2, K_new


# A flat image can never reach 90 degrees off-axis (tan blows up), and past ~80
# the stretching is so bad it's useless anyway.
MAX_THETA_DEG = 80.0


def valid_theta_max(D, cap_deg=MAX_THETA_DEG):
    """Largest angle (radians) where the lens model still makes sense.

    The distortion polynomial is only pinned down where we had checkerboard corners.
    Further out it can bend back on itself, and inverting it there gives nonsense.
    """
    k = np.asarray(D, np.float64).ravel()
    th = np.linspace(0.0, np.radians(cap_deg), 2000)
    t2 = th * th
    dtd = 1 + 3 * k[0] * t2 + 5 * k[1] * t2**2 + 7 * k[2] * t2**3 + 9 * k[3] * t2**4
    bad = np.nonzero(dtd <= 0)[0]
    return th[bad[0] - 1] if len(bad) else th[-1]


def _undistort_normalized(pts, K, D, theta_max):
    """Same as cv2.fisheye.undistortPoints, except points past theta_max get pulled back to it.

    That matters for things like the black image corners outside the fisheye circle,
    where OpenCV happily returns garbage.
    """
    k = np.asarray(D, np.float64).ravel()
    th = np.linspace(0.0, theta_max, 2000)
    t2 = th * th
    td = th * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    x = (p[:, 0] - K[0, 2]) / K[0, 0]
    y = (p[:, 1] - K[1, 2]) / K[1, 1]
    r = np.hypot(x, y)
    theta = np.interp(r, td, th)  # np.interp clamps at the end of the table for us
    s = np.divide(np.tan(theta), r, out=np.ones_like(r), where=r > 1e-12)
    return np.stack([x * s, y * s], 1)


def new_camera_matrix(K, D, image_size, balance=0.0, fov_scale=1.0, out_size=None, R=None):
    """Our replacement for cv2.fisheye.estimateNewCameraMatrixForUndistortRectify.

    It's the same math: undistort the middle of each image edge and pick a focal
    length that fits them. OpenCV's version broke on our left camera (it gave f = 0)
    because those edge points sit outside the fisheye circle, where the lens model
    is meaningless. Here we clamp them to the valid angle first.

    R is the extra rotation applied during stereo rectification, if any.
    """
    w, h = image_size
    balance = min(max(balance, 0.0), 1.0)
    edges = [(w / 2, 0), (w, h / 2), (w / 2, h), (0, h / 2)]
    u = _undistort_normalized(edges, K, D, valid_theta_max(D))
    if R is not None:
        v = np.c_[u, np.ones(len(u))] @ np.asarray(R, np.float64).T
        u = v[:, :2] / v[:, 2:3]
    aspect = K[0, 0] / K[1, 1]
    u[:, 1] *= aspect
    cn = u.mean(0)
    f_cands = [w * 0.5 / (cn[0] - u[:, 0].min()), w * 0.5 / (u[:, 0].max() - cn[0]),
               h * 0.5 * aspect / (cn[1] - u[:, 1].min()), h * 0.5 * aspect / (u[:, 1].max() - cn[1])]
    f = balance * min(f_cands) + (1 - balance) * max(f_cands)
    f /= fov_scale if fov_scale > 0 else 1.0
    fx, fy = f, f / aspect
    cx, cy = -cn[0] * f + w * 0.5, (-cn[1] * f + h * aspect * 0.5) / aspect
    if out_size is not None:
        rx, ry = out_size[0] / w, out_size[1] / h
        fx, cx, fy, cy = fx * rx, cx * rx, fy * ry, cy * ry
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

"""Calibrating the two cameras as a pair, and rectifying them.

Each camera's own calibration (K, D) stays fixed. What we solve for here is how
the right camera sits relative to the left one: a rotation R and a translation T.
"""
import cv2
import numpy as np

from stereo.fisheye_calib import new_camera_matrix

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
MIN_PAIRS = 10


def board_pose(corners, objp, K, D):
    """Where the board is relative to one camera, as (R, t)."""
    n = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2).astype(np.float64), K, D)
    _, rvec, tvec = cv2.solvePnP(objp.reshape(-1, 3), n.reshape(-1, 2), np.eye(3), None)
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def match_corner_order(left, right, objp, cam_l, cam_r):
    """Make sure corner i in the right image is the same physical corner as in the left.

    The detector sometimes numbers the corners starting from the opposite end of the
    board, especially when it's tilted. If we used that as-is, the two cameras would
    look like they're rotated 180 degrees from each other. Our cameras point the same
    way, so we keep whichever order makes them come out closest to parallel.
    Returns (right corners, whether they were flipped).
    """
    R_l, _ = board_pose(left, objp, *cam_l)
    best = None
    for flipped in (False, True):
        r = right[::-1].copy() if flipped else right
        R_r, _ = board_pose(r, objp, *cam_r)
        angle = np.linalg.norm(cv2.Rodrigues(R_r @ R_l.T)[0])
        if best is None or angle < best[0]:
            best = (angle, r, flipped)
    return best[1], best[2]


def pair_errors(objp, left, right, cam_l, cam_r, R, T, rvecs, tvecs):
    """RMS reprojection error (px) of each pair, over both images."""
    errs = []
    for l, r, rv, tv in zip(left, right, rvecs, tvecs):
        pl, _ = cv2.fisheye.projectPoints(objp, rv, tv, *cam_l)
        # Move the board from the left camera's frame into the right camera's frame.
        rv_r = cv2.Rodrigues(R @ cv2.Rodrigues(rv)[0])[0]
        tv_r = R @ np.asarray(tv).reshape(3, 1) + T.reshape(3, 1)
        pr, _ = cv2.fisheye.projectPoints(objp, rv_r, tv_r, *cam_r)
        sq = np.r_[np.sum((pl - l.reshape(-1, 1, 2)) ** 2, 2).ravel(),
                   np.sum((pr - r.reshape(-1, 1, 2)) ** 2, 2).ravel()]
        errs.append(float(np.sqrt(sq.mean())))
    return errs


def stereo_calibrate(objp, left, right, cam_l, cam_r, image_size):
    """Find R, T between the cameras, keeping each camera's own K, D fixed.

    cam_l and cam_r are (K, D) tuples. Returns dict(rms, R, T, rvecs, tvecs, errors).
    """
    if len(left) < MIN_PAIRS:
        raise RuntimeError(f"Only {len(left)} usable pairs, and I need at least {MIN_PAIRS}.")
    obj = [np.asarray(objp, np.float64).reshape(-1, 1, 3)] * len(left)
    pl = [np.asarray(c, np.float64).reshape(-1, 1, 2) for c in left]
    pr = [np.asarray(c, np.float64).reshape(-1, 1, 2) for c in right]
    (K1, D1), (K2, D2) = cam_l, cam_r
    rms, _, _, _, _, R, T, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
        obj, pl, pr, K1, D1, K2, D2, image_size,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC, criteria=CRITERIA)
    T = T.reshape(3, 1)
    errors = pair_errors(obj[0], pl, pr, cam_l, cam_r, R, T, rvecs, tvecs)
    return dict(rms=rms, R=R, T=T, rvecs=rvecs, tvecs=tvecs, errors=errors)


def stereo_rectify(cam_l, cam_r, image_size, R, T, balance=0.0, fov_scale=1.0):
    """Rotations and projections that line both images up row by row.

    The rotations are computed exactly the way cv2.fisheye.stereoRectify does it
    (split R evenly between the cameras, then turn both so the baseline runs along x).
    We can't use that function directly, because it picks the focal length with the
    same broken OpenCV helper that gave us f = 0 before, so that part uses our own
    new_camera_matrix instead.

    Returns R1, R2, P1, P2, Q, the same as cv2.stereoRectify with CALIB_ZERO_DISPARITY.
    """
    (K1, D1), (K2, D2) = cam_l, cam_r
    T = np.asarray(T, np.float64).ravel()
    r_half = cv2.Rodrigues(cv2.Rodrigues(np.asarray(R, np.float64))[0] * -0.5)[0]
    t = r_half @ T
    target = np.array([1.0 if t[0] > 0 else -1.0, 0.0, 0.0])
    axis = np.cross(t, target)
    if np.linalg.norm(axis) > 0:
        axis *= np.arccos(abs(t[0]) / np.linalg.norm(t)) / np.linalg.norm(axis)
    w_rot = cv2.Rodrigues(axis)[0]
    R1 = w_rot @ r_half.T
    R2 = w_rot @ r_half
    t_new = R2 @ T

    k1 = new_camera_matrix(K1, D1, image_size, balance, fov_scale, R=R1)
    k2 = new_camera_matrix(K2, D2, image_size, balance, fov_scale, R=R2)
    f = min(k1[1, 1], k2[1, 1])  # both images need the same focal length to share rows
    cx = (k1[0, 2] + k2[0, 2]) / 2
    cy = (k1[1, 2] + k2[1, 2]) / 2
    P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    P2 = np.array([[f, 0, cx, t_new[0] * f], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1 / t_new[0], 0]], np.float64)
    return R1, R2, P1, P2, Q


def rectify_maps(cam, R_rect, P, image_size):
    K, D = cam
    return cv2.fisheye.initUndistortRectifyMap(K, D, R_rect, P[:, :3], image_size, cv2.CV_16SC2)


def rectify_points(pts, cam, R_rect, P):
    K, D = cam
    return cv2.fisheye.undistortPoints(np.asarray(pts, np.float64).reshape(-1, 1, 2), K, D,
                                       R=R_rect, P=P[:, :3]).reshape(-1, 2)

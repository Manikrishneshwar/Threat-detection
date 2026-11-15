"""Turning a raw left/right pair into a depth map, using the saved stereo calibration."""
import cv2
import numpy as np

from stereo.stereo_calib import rectify_maps

MIN_DIST_M = 0.3   # closest distance we search for; closer than this needs more disparity range


class DepthEstimator:
    """Rectifies a raw pair, matches it with StereoSGBM, and converts disparity to metres.

    scale shrinks the rectified images before matching. 0.5 is about four times faster
    than full size, but it's less precise, and on small or repetitive things far away
    (like a checkerboard at 1 m+) it can lock onto the wrong match.
    """

    def __init__(self, stereo, scale=1.0, block=5):
        self.size = stereo["size"]
        self.scale = scale
        self.maps = {
            "left": rectify_maps((stereo["K1"], stereo["D1"]), stereo["R1"], stereo["P1"], self.size),
            "right": rectify_maps((stereo["K2"], stereo["D2"]), stereo["R2"], stereo["P2"], self.size),
        }
        self.baseline_mm = -stereo["P2"][0, 3] / stereo["P2"][0, 0]
        self.f = stereo["P1"][0, 0] * scale            # focal length in the (scaled) rectified image
        # Disparity grows as things get closer, so the search range is set by the closest distance we care about.
        max_disp = self.f * self.baseline_mm / (MIN_DIST_M * 1000)
        self.num_disp = int(np.ceil(max_disp / 16)) * 16
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.num_disp, blockSize=block,
            P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,   # smoothness: small vs. big jumps in depth
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify(self, left, right):
        out = []
        for img, side in ((left, "left"), (right, "right")):
            r = cv2.remap(img, *self.maps[side], cv2.INTER_LINEAR)
            if self.scale != 1:
                r = cv2.resize(r, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            out.append(r)
        return out

    def __call__(self, left, right):
        """Returns (rectified left, rectified right, depth in metres with NaN where there's no reading)."""
        rl, rr = self.rectify(left, right)
        gl, gr = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (rl, rr))
        disp = self.matcher.compute(gl, gr).astype(np.float32) / 16   # SGBM gives disparity x16
        depth = np.full(disp.shape, np.nan, np.float32)
        ok = disp > 0.5
        depth[ok] = self.f * self.baseline_mm / disp[ok] / 1000
        return rl, rr, depth


def colorize(depth, near=MIN_DIST_M, far=3.0):
    """Red is close, blue is far, black means no reading."""
    v = np.clip((far - np.nan_to_num(depth, nan=far)) / (far - near), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~np.isfinite(depth)] = 0
    return img


def distance_at(depth, x, y, radius=6):
    """Median distance in a small square around (x, y), or None if too little of it has a reading."""
    patch = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    vals = patch[np.isfinite(patch)]
    return float(np.median(vals)) if vals.size >= 0.3 * patch.size else None

"""Little drawing helpers the scripts share."""
import cv2
import numpy as np


def fit_width(img, max_w=1600):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)


def side_by_side(a, b):
    if a.shape[0] != b.shape[0]:
        b = cv2.resize(b, (int(b.shape[1] * a.shape[0] / b.shape[0]), a.shape[0]))
    return np.hstack([a, b])


def draw_hlines(img, step=40, color=(0, 255, 255)):
    """Horizontal guide lines. Once the pair is rectified, matching points should sit on the same line."""
    out = img.copy()
    for y in range(step, out.shape[0], step):
        cv2.line(out, (0, y), (out.shape[1] - 1, y), color, 1)
    return out


def put_lines(img, lines, org=(10, 30), color=(0, 255, 0), scale=0.7):
    x, y = org
    for i, text in enumerate(lines):
        p = (x, y + i * int(32 * scale + 6))
        cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return img

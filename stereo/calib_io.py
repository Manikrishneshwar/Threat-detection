"""Saving and loading calibration results (OpenCV YAML, readable in any text editor)."""
from pathlib import Path

import cv2
import numpy as np


def save_intrinsics(path, K, D, image_size, rms, num_images, model="fisheye"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("model", model)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("K", np.asarray(K, np.float64))
    fs.write("D", np.asarray(D, np.float64).reshape(-1, 1))
    fs.write("rms", float(rms))
    fs.write("num_images", int(num_images))
    fs.release()


def load_intrinsics(path):
    """Returns a dict with model, K, D, size (w, h), rms and num_images."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Can't find {path}. Run 02_calibrate_intrinsics.py for this camera first.")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {
        "model": fs.getNode("model").string(),
        "K": fs.getNode("K").mat(),
        "D": fs.getNode("D").mat(),
        "size": (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())),
        "rms": fs.getNode("rms").real(),
        "num_images": int(fs.getNode("num_images").real()),
    }
    fs.release()
    return out


STEREO_MATS = ("K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2", "Q")


def save_stereo(path, image_size, rms, num_pairs, **mats):
    """Saves the stereo calibration. mats needs every name in STEREO_MATS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("rms", float(rms))
    fs.write("num_pairs", int(num_pairs))
    fs.write("baseline_mm", float(np.linalg.norm(mats["T"])))
    for name in STEREO_MATS:
        fs.write(name, np.asarray(mats[name], np.float64))
    fs.release()


def load_stereo(path):
    """Returns a dict with every matrix in STEREO_MATS, plus size, rms, num_pairs and baseline_mm."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Can't find {path}. Run 06_stereo_calibrate.py first.")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {name: fs.getNode(name).mat() for name in STEREO_MATS}
    out.update(size=(int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())),
               rms=fs.getNode("rms").real(), num_pairs=int(fs.getNode("num_pairs").real()),
               baseline_mm=fs.getNode("baseline_mm").real())
    fs.release()
    return out

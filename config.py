"""All the project settings live here, so the scripts don't need editing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- cameras
# "left" is the camera on your left when you stand behind the rig, looking the
# same way the cameras do. Get the indices from 00_list_cameras.py.
# Windows numbers cameras in the order it finds them, so keep each one plugged
# into the same USB port or the indices can swap after a replug or reboot.
# Probe on 2026-11-15: 0 = laptop webcam, 1 & 4 = black (probably IR / virtual cams),
# 2 & 3 = the two ELP fisheyes. Index 3 is the one mounted on the left of the rig
# (checked with a test pair on 2026-11-15, since near objects shifted the wrong way with 2 as left).
# The names were swapped then, calibration files and image folders included, so
# calib/left_* still belongs to the camera that's physically on the left.
CAMERAS = {
    "left": 3,
    "right": 2,
}

# A calibration only holds at the resolution it was made at.
# 720p shows the full fisheye circle; 480p on this camera is a crop of the middle.
WIDTH, HEIGHT = 1280, 720
FPS = 30          # 30 leaves enough USB 2.0 bandwidth for two cameras; maybe try 60 later

# ---------------------------------------------------------------- checkerboard
# Count INNER corners (squares minus one). One odd and one even count means
# the board can't be mistaken for itself rotated 180 degrees.
BOARD_COLS = 9
BOARD_ROWS = 6
# Measured 2026-11-11 with a tape measure between inner-corner lines:
#   horizontal 223 mm / 8 squares = 27.88 mm, vertical 139 mm / 5 squares = 27.80 mm.
#   That's 0.3% apart, which is just reading error, so the print isn't stretched. Combined 362/13:
SQUARE_MM = 27.85

# ---------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
CALIB_IMG_DIR = DATA_DIR / "calib_images"   # one folder per camera: <name>/img_000.png ...
PAIRS_DIR = DATA_DIR / "pairs"              # left/right pairs captured together
CALIB_DIR = ROOT / "calib"                  # calibration results (.yml)
OUTPUT_DIR = ROOT / "output"                # debug images
DETECT_DIR = DATA_DIR / "detect"            # obstacle images for YOLO: <class>/{rect_left,left,right}/


STEREO_PATH = CALIB_DIR / "stereo.yml"

# Measured 2026-11-15 with a tape between the same point on each lens. Only used as a
# sanity check; the stereo calibration measures the real value.
BASELINE_TAPE_MM = 117.0


def intrinsics_path(name: str) -> Path:
    return CALIB_DIR / f"{name}_intrinsics.yml"

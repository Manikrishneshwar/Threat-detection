# Fisheye stereo depth: 2x ELP-USBFHD01M-L180

Goal: measure the distance to objects with two 180° fisheye USB cameras.

## Layout

```
config.py                  all settings: camera indices, resolution, board size, paths
stereo/                    reusable library code
  camera.py                open_camera(), ThreadedCamera (for running 2 cameras at once)
  fisheye_calib.py         corner detection, fisheye calibration, undistortion maps
  stereo_calib.py          stereo calibration (R, T) and rectification
  depth.py                 rectify a pair, StereoSGBM, disparity -> metres
  calib_io.py              save/load calibration YAML
  viz.py                   drawing helpers
00_list_cameras.py         find which index is which camera
01_capture_calib_images.py step 1: capture checkerboard images (one camera)
02_calibrate_intrinsics.py step 1: compute K, D for one camera
03_undistort_live.py       step 1: live raw vs undistorted check
04_compare_intrinsics.py   step 2: are the two cameras' calibrations interchangeable?
05_dual_camera_test.py     step 3: both cameras live together, FPS + sync skew; s saves pairs
06_stereo_calibrate.py     steps 4-5: stereo calibration + rectification -> calib/stereo.yml
07_depth_live.py           step 6: live depth map and distance at a crosshair
make_checkerboard.py       printable calibration target
calib/                     calibration results (*.yml)
data/calib_images/<name>/  checkerboard captures per camera
data/pairs/                left/right image pairs
output/                    debug images
```

## Pipeline

| # | Step | Status |
|---|------|--------|
| 1 | Per-camera fisheye intrinsics + undistortion | scripts 01–03 |
| 2 | Compare the two cameras' intrinsics | script 04 |
| 3 | Run both cameras at the same time | script 05 |
| 4 | Stereo calibration (R, T between cameras) via `cv2.fisheye.stereoCalibrate` | script 06 |
| 5 | Rectification, so matches share an image row | script 06 |
| 6 | Disparity (StereoSGBM), then depth `Z = f·B / d`, then distance to objects | script 07 |

Current calibration (2026-11-15): stereo RMS 0.45 px from 45 pairs, baseline
117.1 mm (tape: 117 mm), rows line up to 0.3 px after rectification. Depth on the
saved pairs is within about 1% of the true board distance from 0.3 m to 1.2 m.

Undistortion alone does **not** make the rows line up. It only makes lines
straight. Row alignment needs rectification (step 5), which also needs the
rotation and translation between the cameras from step 4. Steps 1–3 are the
foundation for that.

## Run order

```bash
pip install -r requirements.txt
python make_checkerboard.py          # print at 100%, measure, set SQUARE_MM in config.py
python 00_list_cameras.py            # set config.CAMERAS

python 01_capture_calib_images.py --name left
python 02_calibrate_intrinsics.py --name left
python 03_undistort_live.py --name left
# repeat the three commands with --name right

python 04_compare_intrinsics.py --baseline-mm <your lens spacing>
python 05_dual_camera_test.py --undistort   # press s to save checkerboard pairs (board fully in both views, held still)
python 06_stereo_calibrate.py
python 07_depth_live.py
```

## Notes

- **Resolution is fixed at 1280x720.** On this camera, 480p is a cropped view,
  so a calibration made at one resolution is wrong at another.
- **Calibration tips:** 25–40 images. Fill the whole fisheye circle, especially
  the edges. Vary the tilt. Keep the board flat and still. Aim for RMS < 0.5 px.
- **Camera indices** on Windows depend on USB enumeration order. Keep each
  camera in the same port.
- **Don't move anything on the rig** after `06`. If a camera shifts or a lens
  turns, recapture the pairs and rerun `06` (and `01`-`02` for a turned lens).
- **OpenCV's fisheye zoom helper is broken for these lenses** (it returns f = 0
  because the image corners lie outside the fisheye circle).
  `stereo/fisheye_calib.py: new_camera_matrix` replaces it everywhere, including
  inside the stereo rectification.
- **Sync:** the cameras are not hardware-synced. `05` reports the skew, which is
  fine for static scenes. For moving objects, keep the skew low or capture with
  `grab()` on both cameras before calling `retrieve()`.

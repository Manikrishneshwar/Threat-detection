# Fisheye stereo depth: 2x ELP-USBFHD01M-L180

Goal: measure the distance to objects with two 180° fisheye USB cameras.

## Layout

```
config.py                  all settings: camera indices, resolution, board size, paths
stereo/                    reusable library code
  camera.py                open_camera(), ThreadedCamera (for running 2 cameras at once)
  fisheye_calib.py         corner detection, fisheye calibration, undistortion maps
  calib_io.py              save/load calibration YAML
  viz.py                   drawing helpers
00_list_cameras.py         find which index is which camera
01_capture_calib_images.py step 1: capture checkerboard images (one camera)
02_calibrate_intrinsics.py step 1: compute K, D for one camera
03_undistort_live.py       step 1: live raw vs undistorted check
04_compare_intrinsics.py   step 2: are the two cameras' calibrations interchangeable?
05_dual_camera_test.py     step 3: both cameras live together, FPS + sync skew
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
| 4 | Stereo calibration (R, T between cameras) via `cv2.fisheye.stereoCalibrate` | next |
| 5 | Rectification (`cv2.fisheye.stereoRectify`), so matches share an image row | next |
| 6 | Disparity (StereoSGBM), then depth `Z = f·B / d`, then distance to objects | next |

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
python 05_dual_camera_test.py --undistort
```

## Notes

- **Resolution is fixed at 1280x720.** On this camera, 480p is a cropped view,
  so a calibration made at one resolution is wrong at another.
- **Calibration tips:** 25–40 images. Fill the whole fisheye circle, especially
  the edges. Vary the tilt. Keep the board flat and still. Aim for RMS < 0.5 px.
- **Camera indices** on Windows depend on USB enumeration order. Keep each
  camera in the same port.
- **Sync:** the cameras are not hardware-synced. `05` reports the skew, which is
  fine for static scenes. For moving objects, keep the skew low or capture with
  `grab()` on both cameras before calling `retrieve()`.

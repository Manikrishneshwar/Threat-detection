#!/usr/bin/env python3
"""Make a printable checkerboard that matches the settings in config.py.

    python make_checkerboard.py      # saves data/checkerboard_<cols>x<rows>_<size>mm.png at 300 DPI

Print it at 100% / "actual size" (not "fit to page"), then measure a square with a
ruler and put the real size in config.SQUARE_MM. Stick it flat onto something stiff.
"""
import cv2
import numpy as np

import config

DPI = 300
MARGIN_MM = 5   # small enough that a 9x6 board with 25 mm squares still fits on Letter or A4


def main():
    cols, rows = config.BOARD_COLS + 1, config.BOARD_ROWS + 1   # squares, not inner corners
    sq = int(round(config.SQUARE_MM / 25.4 * DPI))
    m = int(round(MARGIN_MM / 25.4 * DPI))
    img = np.full((rows * sq + 2 * m, cols * sq + 2 * m), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[m + r * sq:m + (r + 1) * sq, m + c * sq:m + (c + 1) * sq] = 0
    out = config.DATA_DIR / f"checkerboard_{config.BOARD_COLS}x{config.BOARD_ROWS}_{config.SQUARE_MM:g}mm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    w_mm, h_mm = img.shape[1] / DPI * 25.4, img.shape[0] / DPI * 25.4
    print(f"Saved {out}. It's {w_mm:.0f} x {h_mm:.0f} mm at {DPI} DPI, so print it in landscape.")


if __name__ == "__main__":
    main()

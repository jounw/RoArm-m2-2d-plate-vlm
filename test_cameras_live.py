"""Open camera indices 0-5 at once and show every live feed side by side in
one window, each tile labeled with its index and resolution -- wave a hand in
front of a camera to see which index it is.

Keys: q or ESC = quit, s = save a snapshot of each live feed (cam_live_N.jpg)

Run: python3 test_cameras_live.py            # indices 0-5
     python3 test_cameras_live.py --max 8    # indices 0-8
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

TILE_W, TILE_H = 480, 360
WINDOW_NAME = "Camera index test (q: quit, s: snapshot)"


def label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="show all webcam indices live, labeled")
    ap.add_argument("--max", type=int, default=5, help="highest camera index to try (inclusive)")
    args = ap.parse_args()

    caps: dict[int, cv2.VideoCapture] = {}
    for i in range(args.max + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps[i] = cap
            print(f"index {i}: 열림")
        else:
            print(f"index {i}: 사용 불가")
            cap.release()

    if not caps:
        print("열리는 카메라가 없습니다")
        return 1

    cols = 2 if len(caps) > 1 else 1
    rows = (len(caps) + cols - 1) // cols
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            tiles = []
            for i, cap in caps.items():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    tile = cv2.resize(frame, (TILE_W, TILE_H))
                    label(tile, f"index {i}  ({w}x{h})")
                else:
                    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                    label(tile, f"index {i}  (no frame)")
                tiles.append((i, tile))

            grid_rows = []
            for r in range(rows):
                row_tiles = [t for _, t in tiles[r * cols:(r + 1) * cols]]
                while len(row_tiles) < cols:
                    row_tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
                grid_rows.append(np.hstack(row_tiles))
            grid = np.vstack(grid_rows)

            cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                for i, tile in tiles:
                    path = f"cam_live_{i}.jpg"
                    cv2.imwrite(path, tile)
                    print(f"저장됨: {path}")
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

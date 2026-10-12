"""Quick connectivity check for the iPad LiDAR (Record3D) camera.

Confirms, in order:
  1. the record3d Python package is installed
  2. at least one Record3D device is found over USB
  3. a connection can be established
  4. an actual RGBD frame can be received, with valid depth data

Does NOT touch the robot arm -- purely a camera-side sanity check before
running calibrate_lidar_extrinsic.py or pick_place_lidar.py.

Run: python3 check_lidar_connection.py
Add --show to also pop up a window with the live RGB + depth for a few seconds.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the iPad LiDAR (Record3D) connection.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="also show a live RGB/depth preview window for a few seconds")
    parser.add_argument("--show-seconds", type=float, default=5.0)
    parser.add_argument(
        "--rotate",
        choices=("none", "left", "right", "180"),
        default="left",
        help="rotate the preview window (display only): 'left' = 90deg counter-clockwise (default, matches iPad held upright), 'right' = 90deg clockwise, '180' = upside down",
    )
    args = parser.parse_args()

    print("1) record3d 파이썬 패키지 확인 중...")
    try:
        from record3d import Record3DStream
    except ImportError:
        print("   실패: record3d 패키지가 설치되어 있지 않습니다.")
        print("   설치: pip install record3d --break-system-packages")
        return 1
    print("   OK")

    print("\n2) 연결된 Record3D 기기 검색 중...")
    devices = Record3DStream.get_connected_devices()
    if not devices:
        print("   실패: 연결된 기기를 찾지 못했습니다.")
        print("   확인할 것:")
        print("   - 아이패드가 USB 케이블로 맥에 연결되어 있는지")
        print("   - 아이패드에서 '이 컴퓨터를 신뢰함'을 눌렀는지")
        print("   - Record3D 앱을 열고 Settings 탭에서 'USB Streaming mode'를 켰는지")
        print("   - Record3D 앱이 화면에 켜진 상태(백그라운드 아님)인지")
        return 1
    print(f"   {len(devices)}개 기기 발견:")
    for i, dev in enumerate(devices):
        print(f"     [{i}] product_id={dev.product_id}, udid={dev.udid}")

    if args.device_index >= len(devices):
        print(f"\n   --device-index {args.device_index}가 범위를 벗어났습니다 (기기 {len(devices)}개).")
        return 1

    print(f"\n3) 기기 #{args.device_index}에 연결 중...")
    try:
        from record3d_camera import Record3DCamera
        cam = Record3DCamera(device_index=args.device_index)
    except Exception as e:
        print(f"   실패: {e}")
        return 1
    print("   연결됨")

    print("\n4) 프레임 수신 대기 중 (5초 타임아웃)...")
    try:
        frame = cam.get_frame(timeout=5.0)
    except TimeoutError as e:
        print(f"   실패: {e}")
        print("   Record3D 앱에서 녹화/스트리밍 버튼을 눌렀는지 확인하세요.")
        cam.close()
        return 1
    except Exception as e:
        print(f"   실패: {e}")
        cam.close()
        return 1

    print("   OK -- 프레임 수신 성공")
    print(f"   RGB 해상도: {frame.rgb.shape[1]}x{frame.rgb.shape[0]}")
    print(f"   depth 해상도: {frame.depth_m.shape[1]}x{frame.depth_m.shape[0]}")

    import numpy as np
    valid_depth = frame.depth_m[(frame.depth_m > 0.05) & (frame.depth_m < 5.0) & np.isfinite(frame.depth_m)]
    if valid_depth.size == 0:
        print("   경고: 유효한 depth 값이 없습니다 (전부 0이거나 범위 밖). 카메라가 뭔가를 비추고 있는지 확인하세요.")
    else:
        print(f"   유효 depth 픽셀 비율: {valid_depth.size / frame.depth_m.size * 100:.1f}%")
        print(f"   depth 범위: {valid_depth.min():.2f}m ~ {valid_depth.max():.2f}m (중앙값 {np.median(valid_depth):.2f}m)")
    print(f"   intrinsics fx={frame.intrinsics[0,0]:.1f} fy={frame.intrinsics[1,1]:.1f} cx={frame.intrinsics[0,2]:.1f} cy={frame.intrinsics[1,2]:.1f}")

    if args.show:
        import cv2
        from record3d_camera import rotate_for_display

        print(f"\n5) {args.show_seconds:.0f}초간 라이브 미리보기를 띄웁니다 (회전: {args.rotate}, 아무 키나 누르면 종료)...")
        end = time.time() + args.show_seconds
        while time.time() < end:
            frame = cam.get_frame()
            depth_vis = cv2.normalize(frame.depth_m, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            rgb_show = rotate_for_display(frame.rgb, args.rotate)
            depth_vis = rotate_for_display(depth_vis, args.rotate)
            cv2.imshow("RGB", rgb_show)
            cv2.imshow("Depth", depth_vis)
            if cv2.waitKey(30) != -1:
                break
        cv2.destroyAllWindows()

    cam.close()
    print("\n모든 확인 통과: 아이패드 라이다 연결이 정상입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

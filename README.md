# RoArm-M2 Pick-and-Place — 연결 & 실행 가이드

## 1. 로봇팔 연결

### 방법 A: 시리얼(USB)
1. RoArm-M2를 USB로 컴퓨터에 연결
2. 포트 확인:
   ```bash
   ls /dev/cu.*
   ```
   보통 `/dev/cu.usbserial-XXXX` 형태
3. 스크립트 실행 시 `--serial-port /dev/cu.usbserial-120` 형식으로 지정

### 방법 B: WiFi(HTTP)
1. 로봇팔의 WiFi AP에 연결 (기본 IP `192.168.4.1`)
2. 스크립트 실행 시 `--ip 192.168.4.1` 형식으로 지정 (`--serial-port`는 생략)

두 방법 다 환경변수로도 지정 가능:
```bash
export ROARM_SERIAL_PORT=/dev/cu.usbserial-120
export ROARM_IP=192.168.4.1
```

## 2. 카메라 연결

### 웹캠 (fixed camera 방식)
- `scan_webcams.py`로 인덱스 확인:
  ```bash
  python3 scan_webcams.py
  ```
- 보통 index 0 = 내장캠, index 1/2 = 외장 USB캠

### iPad LiDAR (Record3D 앱)
1. iPad에 **Record3D** 앱 설치
2. iPad를 USB로 Mac에 연결
3. Record3D 앱 → Settings → **USB Streaming mode** 켜기
4. 처음 연결 시 iPad에 "이 컴퓨터를 신뢰함" 팝업 뜨면 허용
5. 연결 확인:
   ```bash
   python3 check_lidar_connection.py --show --rotate left
   ```

## 3. 사전 준비물 설치

```bash
pip install record3d pyserial requests opencv-python numpy --break-system-packages
```

## 4. 캘리브레이션 (순서대로)

### 4-1. 초기 자세 / 작업 범위
```bash
python3 teach_start_pose.py --serial-port /dev/cu.usbserial-120
python3 calibrate_workspace.py --serial-port /dev/cu.usbserial-120
python3 calibrate_floor.py --serial-port /dev/cu.usbserial-120
```

### 4-2. 놓을 위치(place pose) 지정
```bash
python3 teach_place.py --serial-port /dev/cu.usbserial-120
```

### 4-3-A. 바닥 호모그래피 (일반 웹캠 사용 시)
```bash
python3 calibrate_ground_homography.py --camera-index 1 --serial-port /dev/cu.usbserial-120 --num-points 9
```
→ `ground_homography.json` 저장

### 4-3-B. 바닥 호모그래피 (iPad Record3D RGB 사용 시)
```bash
python3 calibrate_ground_homography_record3d.py --serial-port /dev/cu.usbserial-120 --num-points 9 --out ground_homography_record3d.json
```
→ `ground_homography_record3d.json` 저장 (웹캠용과 파일명 다르게 관리 권장)

### 4-4. 라이다 3D 캘리브레이션 (LiDAR 깊이 사용 시)
```bash
python3 calibrate_lidar_extrinsic.py --serial-port /dev/cu.usbserial-120 --num-points 9
```
→ `lidar_extrinsic.json` 저장

캘리브레이션한 9개 지점으로 로봇팔이 잘 찾아가는지 테스트:
```bash
python3 replay_calibration_points.py --serial-port /dev/cu.usbserial-120
```

## 5. 픽업 실행

### 5-1. 일반 웹캠 + 바닥 호모그래피
```bash
python3 pick_place_fixed_cam.py --detector manual --camera-index 1 --serial-port /dev/cu.usbserial-120 --place-mode click
```

### 5-2. iPad LiDAR (깊이 사용, 정확도 우선)
```bash
python3 pick_place_lidar.py --detector manual --place-mode click --serial-port /dev/cu.usbserial-120
```

### 5-3. iPad Record3D RGB만 사용 (라이다 미사용)
```bash
python3 pick_place_record3d_rgb.py --detector manual --place-mode click --serial-port /dev/cu.usbserial-120 --homography ground_homography_record3d.json
```

### 5-4. iPad Record3D 탑다운 + 손목 카메라(eye-in-hand) 2단계 방식
Record3D RGB로 대략적인 위치 접근(1단계) 후, 손목 카메라로 정밀 하강(2단계).
사전에 `calibrate_ground_homography_record3d.py` 캘리브레이션 필요.
```bash
python3 pick_place_record3d_to_eih.py --detector manual --place-mode click \
    --serial-port /dev/cu.usbserial-120
# eye-in-hand 카메라는 기본 인덱스 0; 잡기 직전 y로 +30mm 보정(--grab-offset-y-mm),
# 바닥 위 15mm에서 잡음(--floor-clearance-mm)

```

### 5-5. 손목 카메라(eye-in-hand) 단독 방식
```bash
python3 pick_place.py --target "the red block" --serial-port /dev/cu.usbserial-120
```

## 6. 공통 옵션

| 옵션 | 설명 |
|---|---|
| `--detector manual\|darkest\|vlm` | manual=드래그로 직접 박스 지정, darkest=가장 어두운 물체 자동 감지, vlm=자연어 설명으로 감지 |
| `--target "설명"` | `--detector vlm`일 때 찾을 물체 설명 |
| `--place-mode taught\|click` | taught=미리 지정한 위치에 놓기, click=매번 화면 클릭으로 놓을 위치 지정 |
| `--rotate none\|left\|right\|180` | iPad 화면 회전 보정 (기본 left) |
| `--max-cycles N` | N번 반복 후 종료 (기본 0 = 무한 반복, Ctrl+C로 중단) |

## 7. 참고

- 모든 스크립트는 실제 로봇팔/카메라와 연결된 사용자 컴퓨터의 터미널에서 직접 실행해야 합니다.
- `--serial-port`와 `--ip` 중 하나만 있으면 됩니다 (serial-port가 있으면 시리얼 우선).
- 캘리브레이션 파일(`*_calibration.json`, `*_homography*.json`, `lidar_extrinsic.json`)은 카메라/로봇팔 위치가 바뀌면 다시 캘리브레이션해야 합니다.

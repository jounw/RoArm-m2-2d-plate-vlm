"""One-time download of the ViT-Track ONNX model used by tracker.py."""

from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
OUT_PATH = Path(__file__).parent / "models" / "vittrack.onnx"


def main() -> int:
    OUT_PATH.parent.mkdir(exist_ok=True)
    print(f"downloading {URL} -> {OUT_PATH}")
    urlretrieve(URL, OUT_PATH)
    print(f"done, {OUT_PATH.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

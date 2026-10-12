"""Standalone mic -> whisper check, no robot/camera needed.

Uses the same AudioRecorder.listen() (adaptive noise threshold) and
transcribe() path as pick_place_record3d_to_eih.py --voice, so what works
here works there. Speak after the prompt; Ctrl+C to quit.

Run: python3 test_voice.py
"""

from __future__ import annotations

import time

from pick_place_record3d_to_eih import AudioRecorder
from speech import transcribe

PROMPT = "로봇 팔 명령. 예: 검은 물체 잡아서 3에 놔. 테이프 잡아. 2에 놔. 테이블에 뭐 보여? 종료."


def main() -> int:
    import sys
    mic = sys.argv[1] if len(sys.argv) > 1 else "MacBook"
    audio = AudioRecorder("recordings/test_voice.wav", device=mic or None)
    if not audio.active:
        print("마이크를 열지 못했습니다")
        return 1
    print(f"samplerate={audio.samplerate}Hz. Ctrl+C로 종료.\n")
    try:
        while True:
            seg = audio.listen()
            if seg is None:
                print("(아무 소리도 못 들었습니다)\n")
                continue
            t0 = time.time()
            text = transcribe(seg, samplerate=audio.samplerate, prompt=PROMPT)
            print(f"[나] {text!r}  (STT {time.time() - t0:.1f}s)\n")
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        audio.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

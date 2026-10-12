"""Voice I/O for the pick-and-place demo: OpenAI STT (whisper) + TTS.

listen()  -- record from the default mic until ~1.5s of silence, then
             transcribe Korean speech to text with whisper-1.
speak()   -- synthesize Korean speech and play it through the speakers
             (wav via the TTS API, played with macOS afplay -- no pydub).

Keys come from the same env-var-first/api.txt-fallback scheme as vision.py.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

from vision import _api_key

STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "nova")

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500  # mean |amplitude| below this counts as silence; tune per room
SILENCE_DURATION = 1.5  # this much continuous silence ends the recording
MIN_SPEECH_DURATION = 0.5  # discard blips shorter than this


def _client() -> OpenAI:
    return OpenAI(api_key=_api_key())


def _is_speech(chunk: np.ndarray) -> bool:
    return np.abs(chunk).mean() > SILENCE_THRESHOLD


def record_until_silence(max_seconds: float = 15.0) -> np.ndarray | None:
    """Wait for speech to start, record until it stops. None if nothing usable."""
    print("(듣는 중... 말씀하세요)")
    frames: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0
    silent_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16, blocksize=CHUNK_SIZE) as stream:
        for _ in range(max_chunks):
            chunk, _overflow = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()
            if not speech_started:
                if _is_speech(chunk):
                    speech_started = True
                    frames.append(chunk)
            else:
                frames.append(chunk)
                if _is_speech(chunk):
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                    if silent_chunks >= silent_limit:
                        break

    if not frames:
        return None
    audio = np.concatenate(frames)
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


def transcribe(audio: np.ndarray, samplerate: int = SAMPLE_RATE, prompt: str | None = None) -> str:
    """`prompt` biases whisper toward the expected vocabulary (e.g. short
    Korean robot commands) -- helps a lot with 1-2 word utterances."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        with open(path, "rb") as f:
            kwargs = {"model": STT_MODEL, "file": f, "language": "ko"}
            if prompt:
                kwargs["prompt"] = prompt
            result = _client().audio.transcriptions.create(**kwargs)
        return result.text.strip()
    finally:
        os.unlink(path)


def listen() -> str | None:
    """One utterance from the mic -> Korean text, or None if nothing was heard."""
    audio = record_until_silence()
    if audio is None:
        return None
    return transcribe(audio)


def speak(text: str) -> None:
    """Synthesize `text` and play it. Prints instead if synthesis/playback fails
    (a dead speaker should never kill the robot loop)."""
    print(f"[로봇] {text}")
    try:
        response = _client().audio.speech.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav"
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(response.content)
            path = tmp.name
        try:
            subprocess.run(["afplay", path], check=False)
        finally:
            os.unlink(path)
    except Exception as e:
        print(f"(TTS 실패: {e})")


if __name__ == "__main__":
    # quick mic->STT->TTS loop for testing: speak, hear it echoed back
    while True:
        heard = listen()
        if heard:
            print(f"[나] {heard}")
            speak(f"{heard} 라고 하셨네요")

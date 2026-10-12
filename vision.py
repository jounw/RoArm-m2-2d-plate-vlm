"""VLM object localization for the eye-in-hand camera.

Given a frame and a natural-language description, ask a vision-language model
for the target's bounding box, normalized to 0-1000 (top-left/bottom-right),
matching the convention used throughout this project's prompts.

Three backends (VLM_BACKEND env var or the `backend` argument):
  gemini  -- Gemini Robotics-ER, grounding-trained: precise boxes (default when
             a Gemini key is available)
  openai  -- GPT-4o family: understands language well but its box coordinates
             are unreliable (kept as fallback)
  molmo   -- fully local, no API key/network: a quantized Molmo2-4B GGUF served
             by Ollama (http://localhost:11434), OpenAI-compatible API.
             CAUTION: the only available 4B GGUF build of Molmo2 is an
             unmaintained community conversion; its author warns the vision
             (mmproj) path degrades and is not benchmarked. Box coordinates
             from this backend are unverified -- test against real objects
             before trusting it for unattended picks. Model pull:
               ollama pull hf.co/reubk/Molmo2-4B-GGUF:Q4_K_M
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from openai import OpenAI

VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_VLM_MODEL", "gemini-3.5-flash")
# Keys come from the environment, or from local key files next to this file.
# Both files are gitignored -- never commit keys.
API_KEY_FILE = Path(__file__).with_name("api.txt")
GEMINI_KEY_FILE = Path(__file__).with_name("gemini_api.txt")

# Local Molmo backend via Ollama's OpenAI-compatible endpoint -- no API key.
MOLMO_BASE_URL = os.getenv("MOLMO_BASE_URL", "http://localhost:11434/v1")
MOLMO_MODEL = os.getenv("MOLMO_MODEL", "hf.co/reubk/Molmo2-4B-GGUF:Q4_K_M")


def _read_key(env_var: str, key_file: Path) -> str:
    key = os.getenv(env_var, "").strip()
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    return key


def _api_key() -> str:
    key = _read_key("OPENAI_API_KEY", API_KEY_FILE)
    if not key:
        raise RuntimeError("no OpenAI API key: set OPENAI_API_KEY or put the key in api.txt")
    return key


def _gemini_key() -> str:
    key = _read_key("GEMINI_API_KEY", GEMINI_KEY_FILE)
    if not key:
        raise RuntimeError("no Gemini API key: set GEMINI_API_KEY or put the key in gemini_api.txt")
    return key


@dataclass
class Detection:
    answer: str
    box: tuple[int, int, int, int] | None  # x1, y1, x2, y2 normalized 0-1000

    @property
    def center(self) -> tuple[int, int] | None:
        if self.box is None:
            return None
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def height(self) -> int:
        if self.box is None:
            return 0
        return self.box[3] - self.box[1]


def _client() -> OpenAI:
    return OpenAI(api_key=_api_key())


def _molmo_client() -> OpenAI:
    # Ollama's OpenAI-compatible server ignores the key but the client requires
    # a non-empty string.
    return OpenAI(api_key="ollama", base_url=MOLMO_BASE_URL)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _strip_markdown(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()


def _sanitize_box(box, answer: str) -> Detection:
    """Validate/clamp a 0-1000 (x1,y1,x2,y2) box into a Detection."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return Detection(answer=answer, box=None)
    try:
        x1, y1, x2, y2 = (int(float(v)) for v in box)
    except (TypeError, ValueError):
        return Detection(answer=answer, box=None)
    x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
    y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return Detection(answer=answer, box=None)
    return Detection(answer=answer, box=(x1, y1, x2, y2))


def _gemini_generate(image_path: str, prompt: str, thinking_budget: int = 0) -> str:
    """One image+text call to Gemini; returns the raw text reply."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": thinking_budget}},
    }
    resp = requests.post(url, json=body, headers={"x-goog-api-key": _gemini_key()}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"unexpected Gemini response: {str(data)[:200]}"


def _locate_gemini(image_path: str, description: str) -> Detection:
    """Gemini Robotics-ER: grounding-trained, returns box_2d as [ymin,xmin,ymax,xmax] 0-1000."""
    prompt = (
        f"Find this object in the image: {description}\n"
        'Return JSON only: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "<what it is>"}]\n'
        "box_2d is normalized to 0-1000. Return [] if the object is not visible. "
        "Return at most the single best match."
    )
    # spatial grounding needs no long deliberation; thinking budget 0 keeps latency low
    raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=0))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Gemini occasionally emits ALMOST-json (e.g. a label without its key:
        # {"box_2d": [...], "the blue paper"}). The box coordinates are still
        # there, so salvage them by regex instead of failing the detection.
        m = re.search(r'"box_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', raw)
        if m:
            ymin, xmin, ymax, xmax = (float(v) for v in m.groups())
            return _sanitize_box([xmin, ymin, xmax, ymax], raw)
        return Detection(answer=raw, box=None)

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return Detection(answer="not visible", box=None)

    first = parsed[0]
    answer = str(first.get("label", "")).strip()
    box_2d = first.get("box_2d")
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return Detection(answer=answer, box=None)
    ymin, xmin, ymax, xmax = box_2d  # Gemini order -> our (x1, y1, x2, y2)
    return _sanitize_box([xmin, ymin, xmax, ymax], answer)


def _locate_openai(image_path: str, description: str) -> Detection:
    prompt = f"""
Find this object in the image: {description}

Return JSON only, no other text:
{{"answer": "short Korean description of what you see", "box": [x1, y1, x2, y2]}}

Rules:
- box coordinates are normalized integers 0-1000, x1,y1 = top-left, x2,y2 = bottom-right.
- If the object is not visible, return "box": null.
""".strip()

    client = _client()
    response = client.responses.create(
        model=VLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _encode_image(image_path)},
                ],
            }
        ],
    )

    raw = _strip_markdown(response.output_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Detection(answer=raw, box=None)

    answer = str(data.get("answer", "")).strip()
    return _sanitize_box(data.get("box"), answer)


def _molmo_chat(image_path: str, prompt: str) -> str:
    """One image+text call to the local Ollama-served Molmo model."""
    client = _molmo_client()
    response = client.chat.completions.create(
        model=MOLMO_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _locate_molmo(image_path: str, description: str) -> Detection:
    """Local Molmo2-4B (via Ollama). Same 0-1000 (x1,y1,x2,y2) box convention
    as the other backends. See the module docstring for the accuracy caveat
    on this backend's vision path."""
    prompt = f"""
Find this object in the image: {description}

Return JSON only, no other text:
{{"answer": "short description of what you see", "box": [x1, y1, x2, y2]}}

Rules:
- box coordinates are normalized integers 0-1000, x1,y1 = top-left, x2,y2 = bottom-right.
- If the object is not visible, return "box": null.
""".strip()
    raw = _strip_markdown(_molmo_chat(image_path, prompt))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Detection(answer=raw, box=None)
    answer = str(data.get("answer", "")).strip()
    return _sanitize_box(data.get("box"), answer)


def _pick_backend(backend: str | None) -> str:
    backend = (backend or os.getenv("VLM_BACKEND", "")).strip().lower()
    if not backend:
        backend = "gemini" if _read_key("GEMINI_API_KEY", GEMINI_KEY_FILE) else "openai"
    if backend not in ("gemini", "openai", "molmo"):
        raise ValueError(f"unknown VLM backend: {backend!r} (use 'gemini', 'openai', or 'molmo')")
    return backend


def locate(image_path: str, description: str, backend: str | None = None) -> Detection:
    """Find `description` in the image. Backend: arg > VLM_BACKEND env > auto
    (gemini if a Gemini key is available, else openai)."""
    picked = _pick_backend(backend)
    if picked == "gemini":
        return _locate_gemini(image_path, description)
    if picked == "molmo":
        return _locate_molmo(image_path, description)
    return _locate_openai(image_path, description)


def locate_many(image_path: str, descriptions: list[str], backend: str | None = None) -> list[Detection]:
    """Find several things in ONE image with ONE call (gemini backend); returns
    one Detection per description, in the same order (box=None if not visible).
    Non-gemini backends fall back to one locate() call per description."""
    picked = _pick_backend(backend)
    if picked != "gemini":
        return [locate(image_path, d, backend) for d in descriptions]

    numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descriptions))
    prompt = (
        "Find each of these in the image:\n"
        f"{numbered}\n"
        'Return JSON only: [{"id": <item number from the list>, "box_2d": [ymin, xmin, ymax, xmax], "label": "<what it is>"}]\n'
        "box_2d is normalized to 0-1000. Include exactly one entry per listed item; "
        "omit an item only if it is not visible in the image."
    )
    raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=0))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # combined answer unusable -- fall back to per-item calls rather than failing all
        return [locate(image_path, d, backend) for d in descriptions]
    if isinstance(parsed, dict):
        parsed = [parsed]

    results = [Detection(answer="not visible", box=None) for _ in descriptions]
    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            try:
                idx = int(raw_id) - 1
            except (TypeError, ValueError):
                idx = i
            if not 0 <= idx < len(descriptions):
                continue
            b = item.get("box_2d")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                ymin, xmin, ymax, xmax = b
                results[idx] = _sanitize_box([xmin, ymin, xmax, ymax], str(item.get("label", "")).strip())
    return results


def route(image_path: str, utterance: str, backend: str | None = None) -> dict:
    """Route a spoken utterance: a pick command or a question about the scene.

    Returns {"action": "pick", "target": ...} or {"action": "say", "text": ...}.
    """
    prompt = (
        f'The user said to the robot arm: "{utterance}"\n'
        "Using the attached camera view, respond with JSON only:\n"
        '- if it is a command to pick up/move an object: {"action": "pick", "target": "<description of the object to pick>"}\n'
        '- otherwise (a question or general conversation): {"action": "say", "text": "<concise answer in English, at most 2 sentences>"}\n'
    )
    picked = _pick_backend(backend)
    if picked == "gemini":
        raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=128))
    elif picked == "molmo":
        raw = _strip_markdown(_molmo_chat(image_path, prompt))
    else:
        client = _client()
        response = client.responses.create(
            model=VLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _encode_image(image_path)},
                    ],
                }
            ],
        )
        raw = _strip_markdown(response.output_text)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("action") in ("pick", "say"):
            return data
    except json.JSONDecodeError:
        pass
    return {"action": "say", "text": raw}


def parse_pick_place(
    image_path: str,
    utterance: str,
    backend: str | None = None,
    history: list[str] | None = None,
    holding: str | None = None,
) -> dict:
    """Parse a natural-language pick/place command (likely Korean) against the
    attached top-down camera view.

    `history` is the recent dialogue/action transcript (most recent last) and
    `holding` is a description of the object currently in the gripper (None if
    empty) -- both give the model the context needed to resolve references
    like "그거", "아까 거기", "하나 더".

    Returns {"pick": <str|None>, "place": <str|None>, "answer": <str|None>}:
      - "X 잡아서 3에 놔"  -> {"pick": "<desc of X>", "place": "<desc of the number 3 spot>"}
      - "X 잡아"           -> {"pick": "<desc of X>", "place": None}   (grab and hold)
      - "3에 놔"           -> {"pick": None, "place": "<desc>"}        (place what's held)
      - "테이블에 뭐 있어?" -> {"pick": None, "place": None, "answer": "<Korean answer>"}
    Descriptions are in English, phrased so locate() can find them in the image;
    "answer" handles questions / small talk about the scene without any robot motion.
    """
    context = ""
    if history:
        transcript = "\n".join(history[-12:])
        context += f"Conversation and actions so far (most recent last):\n{transcript}\n"
    context += (
        f"The robot's gripper is currently holding: {holding}.\n" if holding
        else "The robot's gripper is currently empty.\n"
    )
    prompt = (
        f"{context}"
        f'The user now says to the robot arm (likely in Korean): "{utterance}"\n'
        "The attached image is a top-down camera view of the table. Large handwritten "
        "numbers (1 to 5) are written on the table and serve as named place locations.\n"
        "Use the conversation context to resolve references such as '그거', '그것', "
        "'아까 거기', '같은 곳', '하나 더', '다시' -- e.g. if the user previously asked "
        "about an object and now says '그거 잡아', pick refers to that object.\n"
        "Respond with JSON only, no other text:\n"
        '{"pick": <string or null>, "place": <string or null>, "answer": <string or null>}\n'
        '- "pick": an English visual description of the object to pick up, phrased for a '
        "visual detector (e.g. \"the black rectangular object\"). null if the command does "
        "not ask to pick up anything new. IMPORTANT: the robot may already be holding an "
        "object -- if the command only says where to put/place (e.g. '4에 놔', '2에 둬'), "
        '"pick" MUST be null.\n'
        '- "place": an English visual description of where to put it, e.g. "the large '
        'handwritten number 3 on the table". null if the command only says to grab/hold '
        "without placing.\n"
        '- "answer": if the utterance is NOT a manipulation command but a question or '
        "conversation (e.g. asking what is on the table, where something is, or general "
        "chat), answer it concisely in Korean (1-2 sentences) using the image, and set "
        'both "pick" and "place" to null. Otherwise null.\n'
    )
    picked = _pick_backend(backend)
    if picked == "gemini":
        raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=128))
    elif picked == "molmo":
        raw = _strip_markdown(_molmo_chat(image_path, prompt))
    else:
        client = _client()
        response = client.responses.create(
            model=VLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _encode_image(image_path)},
                    ],
                }
            ],
        )
        raw = _strip_markdown(response.output_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"pick": None, "place": None, "error": raw}
    if not isinstance(data, dict):
        return {"pick": None, "place": None, "error": raw}
    pick = data.get("pick")
    place = data.get("place")
    answer = data.get("answer")
    return {
        "pick": str(pick).strip() if isinstance(pick, str) and pick.strip() else None,
        "place": str(place).strip() if isinstance(place, str) and place.strip() else None,
        "answer": str(answer).strip() if isinstance(answer, str) and answer.strip() else None,
    }


def ask(image_path: str, question: str, backend: str | None = None) -> str:
    """Free-form question about the image ("what do you see?"), answered in English."""
    prompt = f"{question}\n\nAnswer concisely in English (at most 2-3 sentences)."
    picked = _pick_backend(backend)
    if picked == "gemini":
        # scene Q&A benefits from a little thinking, unlike box grounding
        return _gemini_generate(image_path, prompt, thinking_budget=256).strip()
    if picked == "molmo":
        return _molmo_chat(image_path, prompt)
    client = _client()
    response = client.responses.create(
        model=VLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _encode_image(image_path)},
                ],
            }
        ],
    )
    return response.output_text.strip()

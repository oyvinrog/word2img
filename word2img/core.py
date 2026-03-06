from __future__ import annotations

import base64
from typing import Any, TypedDict

from .prompts import PromptType, build_normal_prompt, build_prompt


class Word2ImgResult(TypedDict):
    image_bytes: bytes
    mime_type: str
    prompt: str
    model: str


MODEL_NAME = "gpt-image-1"


def _build_prompt(words: list[str]) -> str:
    return build_normal_prompt(words)


def _extract_b64_image(response: Any) -> str:
    # Supports the common OpenAI Images API response shape.
    data = getattr(response, "data", None)
    if not data:
        raise ValueError("image generation response did not include data")
    first = data[0]
    b64_data = getattr(first, "b64_json", None)
    if not b64_data:
        raise ValueError("image generation response did not include b64_json")
    return b64_data


def _generate_from_prompt(prompt: str, api_key: str) -> Word2ImgResult:
    if not api_key or not api_key.strip():
        raise RuntimeError("api_key is required")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install dependencies first.") from exc

    client = OpenAI(api_key=api_key.strip())
    response = client.images.generate(
        model=MODEL_NAME,
        prompt=prompt,
        size="1024x1024",
    )

    image_bytes = base64.b64decode(_extract_b64_image(response))
    return {
        "image_bytes": image_bytes,
        "mime_type": "image/png",
        "prompt": prompt,
        "model": MODEL_NAME,
    }


def text_to_img(prompt: str, api_key: str) -> Word2ImgResult:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    return _generate_from_prompt(prompt.strip(), api_key=api_key)


def words_to_img(words: list[str], api_key: str, prompt_type: PromptType = "loci") -> Word2ImgResult:
    return _generate_from_prompt(build_prompt(words, prompt_type=prompt_type), api_key=api_key)

from __future__ import annotations

import re
import sys
from pathlib import Path

from .auth import resolve_api_key
from .core import words_to_img


def _safe_filename_from_prompt(prompt: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", prompt).strip("-._")
    return safe or "image"


def main() -> int:
    raw = input("Enter comma-separated words: ").strip()
    words = [part.strip() for part in raw.split(",")]

    try:
        api_key = resolve_api_key()
        result = words_to_img(words, api_key=api_key)
    except Exception as exc:  # pragma: no cover - simple CLI failure path
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_name = f"{_safe_filename_from_prompt(result['prompt'])}.png"
    output_path = Path.cwd() / output_name
    output_path.write_bytes(result["image_bytes"])

    print(f"Saved image: {output_path}")
    print(f"Prompt: {result['prompt']}")
    print(f"Model: {result['model']}")
    print(f"MIME type: {result['mime_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

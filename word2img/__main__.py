from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .auth import resolve_api_key
from .core import words_to_img
from .prompts import normalize_words


def _safe_filename_stem(text: str, max_length: int = 120) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("-._")
    return safe or "image"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an image from a list of words.")
    parser.add_argument(
        "--prompt-type",
        choices=("loci", "normal", "scene"),
        default="loci",
        help="Prompt style to use (default: loci)",
    )
    args = parser.parse_args(argv if argv is not None else [])

    raw = input("Enter comma-separated words: ").strip()
    words = [part.strip() for part in raw.split(",")]
    cleaned_words = normalize_words(words)

    try:
        api_key = resolve_api_key()
        result = words_to_img(words, api_key=api_key, prompt_type=args.prompt_type)
    except Exception as exc:  # pragma: no cover - simple CLI failure path
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_name = f"{_safe_filename_stem('-'.join(cleaned_words))}.png"
    output_path = Path.cwd() / output_name
    output_path.write_bytes(result["image_bytes"])

    print(f"Saved image: {output_path}")
    print(f"Prompt: {result['prompt']}")
    print(f"Model: {result['model']}")
    print(f"MIME type: {result['mime_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

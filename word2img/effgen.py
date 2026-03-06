from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .auth import resolve_api_key
from .core import text_to_img

EFF_LARGE_WORDLIST_URL = "https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt"


def _default_cache_path() -> Path:
    return Path.home() / ".cache" / "word2img" / "eff_large_wordlist.txt"


def _safe_filename_from_prompt(prompt: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in prompt).strip("-._") or "image"


def _parse_eff_wordlist(text: str) -> list[str]:
    words: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            words.append(parts[1])
    if not words:
        raise RuntimeError("failed to parse EFF wordlist")
    return words


def _load_eff_words(cache_path: Path | None = None) -> list[str]:
    path = cache_path or _default_cache_path()
    if path.exists():
        return _parse_eff_wordlist(path.read_text(encoding="utf-8"))

    try:
        with urlopen(EFF_LARGE_WORDLIST_URL, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(f"could not download EFF wordlist: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _parse_eff_wordlist(text)


def generate_passphrase(num_words: int = 6, word_pool: list[str] | None = None) -> list[str]:
    if num_words < 1:
        raise ValueError("num_words must be >= 1")
    pool = word_pool if word_pool is not None else _load_eff_words()
    if not pool:
        raise RuntimeError("word pool is empty")
    return [secrets.choice(pool) for _ in range(num_words)]


def build_mnemonic_prompt(words: list[str]) -> str:
    if not words:
        raise ValueError("words must be non-empty")
    word_csv = ", ".join(words)
    return (
        "Create a vivid, memorable scene that clearly represents these concepts as objects or actions: "
        f"{word_csv}. Make it playful, surreal, and easy to recall. "
        "Important: do not include any written text, letters, numbers, captions, signs, or typography in the image."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an EFF passphrase and image mnemonic.")
    parser.add_argument("-n", "--num-words", type=int, default=6, help="Number of passphrase words (default: 6)")
    args = parser.parse_args(argv)

    try:
        words = generate_passphrase(num_words=args.num_words)
        api_key = resolve_api_key()
        mnemonic_prompt = build_mnemonic_prompt(words)
        result = text_to_img(mnemonic_prompt, api_key=api_key)
    except Exception as exc:  # pragma: no cover - simple CLI failure path
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    passphrase = " ".join(words)
    output_name = f"{_safe_filename_from_prompt('-'.join(words))}.png"
    output_path = Path.cwd() / output_name
    output_path.write_bytes(result["image_bytes"])

    print(f"Passphrase: {passphrase}")
    print(f"Saved image: {output_path}")
    print("Image type: mnemonic scene (no text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

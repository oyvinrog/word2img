from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .auth import resolve_api_key
from .core import text_to_img

EFF_LARGE_WORDLIST_URL = "https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt"
TRANSLATION_MODEL = "gpt-4.1-mini"
LANGUAGE_ALIASES = {
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "nb": "Norwegian Bokmal",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "pt": "Portuguese",
    "sv": "Swedish",
}


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


def build_loci_prompt(words: list[str]) -> str:
    if not words:
        raise ValueError("words must be non-empty")
    placements = "; ".join(
        f"at locus {idx + 1}, depict {word} as a concrete visual object or action"
        for idx, word in enumerate(words)
    )
    return (
        "Create a single memory-palace style geographic scene with a clearly visible walkable path that a person could follow on foot. "
        "Show a sequence of distinct, easy-to-recognize locations along the path, like separate landmarks or stops, so the viewer can mentally walk through them in order. "
        f"Place concepts in strict order along the route: {placements}. "
        "Use strong visual separation between loci, with obvious transitions from one stop to the next, so order is unmistakable from start to finish. "
        "Compose the scene from a human walking viewpoint, not a flat map, and make the route progression obvious. "
        "Important: no written text, letters, numbers, captions, signs, or typography in the image."
    )


def _resolve_language_name(lang: str) -> str:
    normalized = lang.strip()
    if not normalized:
        raise ValueError("lang must be non-empty")
    return LANGUAGE_ALIASES.get(normalized.lower(), normalized)


def _parse_translation_output(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        raise RuntimeError("translation response was empty")

    candidates = [raw]
    if "```" in raw:
        fenced_parts = raw.split("```")
        for idx in range(1, len(fenced_parts), 2):
            fenced = fenced_parts[idx].strip()
            if "\n" in fenced:
                _, remainder = fenced.split("\n", 1)
                candidates.append(remainder.strip())
            candidates.append(fenced)
    array_start = raw.find("[")
    array_end = raw.rfind("]")
    if array_start != -1 and array_end != -1 and array_start < array_end:
        candidates.append(raw[array_start : array_end + 1])

    translated: object | None = None
    for candidate in candidates:
        try:
            translated = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(translated, list):
        raise RuntimeError("translation response was not valid JSON")
    cleaned = [str(word).strip() for word in translated]
    if any(not word for word in cleaned):
        raise RuntimeError("translation response included an empty word")
    return cleaned


def translate_words(words: list[str], lang: str, api_key: str) -> list[str]:
    if not words:
        raise ValueError("words must be non-empty")
    language_name = _resolve_language_name(lang)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install dependencies first.") from exc

    client = OpenAI(api_key=api_key.strip())
    response = client.responses.create(
        model=TRANSLATION_MODEL,
        input=(
            f"Translate this ordered list of English passphrase words into {language_name}. "
            "Preserve order. Return exactly one translated word per input word. "
            "If a direct translation is awkward or would need multiple words, choose the closest single-word equivalent. "
            "Return only a JSON array of strings.\n\n"
            f"{json.dumps(words)}"
        ),
    )

    cleaned = _parse_translation_output(response.output_text)
    if len(cleaned) != len(words):
        raise RuntimeError("translation response did not return the expected number of words")
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an EFF passphrase and image mnemonic.")
    parser.add_argument("-n", "--num-words", type=int, default=6, help="Number of passphrase words (default: 6)")
    parser.add_argument(
        "--mnemonic-mode",
        choices=("loci", "scene"),
        default="loci",
        help="Mnemonic prompt style (default: loci)",
    )
    parser.add_argument("--lang", help="Translate the passphrase into this language before building the mnemonic")
    args = parser.parse_args(argv)

    try:
        words = generate_passphrase(num_words=args.num_words)
        passphrase = " ".join(words)
        api_key = resolve_api_key()
        prompt_words = words
        print(f"Passphrase: {passphrase}", flush=True)
        if args.lang:
            prompt_words = translate_words(words, args.lang, api_key=api_key)
            language_name = _resolve_language_name(args.lang)
            print(f"Translated passphrase ({language_name}): {' '.join(prompt_words)}", flush=True)
        if args.mnemonic_mode == "scene":
            mnemonic_prompt = build_mnemonic_prompt(prompt_words)
        else:
            mnemonic_prompt = build_loci_prompt(prompt_words)
        result = text_to_img(mnemonic_prompt, api_key=api_key)
    except Exception as exc:  # pragma: no cover - simple CLI failure path
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_name = f"{_safe_filename_from_prompt('-'.join(prompt_words))}.png"
    output_path = Path.cwd() / output_name
    output_path.write_bytes(result["image_bytes"])

    print(f"Saved image: {output_path}")
    print(f"Image type: mnemonic {args.mnemonic_mode} (no text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

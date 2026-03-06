from __future__ import annotations

from typing import Literal

PromptType = Literal["loci", "normal", "scene"]


def normalize_words(words: list[str]) -> list[str]:
    cleaned = [word.strip() for word in words if word and word.strip()]
    if not cleaned:
        raise ValueError("words must contain at least one non-empty value")
    return cleaned


def build_normal_prompt(words: list[str]) -> str:
    return "-".join(normalize_words(words))


def build_scene_prompt(words: list[str]) -> str:
    cleaned = normalize_words(words)
    word_csv = ", ".join(cleaned)
    return (
        "Create a vivid, memorable scene that clearly represents these concepts as objects or actions: "
        f"{word_csv}. Make it playful, surreal, and easy to recall. "
        "Important: do not include any written text, letters, numbers, captions, signs, or typography in the image."
    )


def build_loci_prompt(words: list[str]) -> str:
    cleaned = normalize_words(words)
    placements = "; ".join(
        f"at locus {idx + 1}, depict {word} as a concrete visual object or action"
        for idx, word in enumerate(cleaned)
    )
    return (
        "Create a single memory-palace style geographic scene with a clearly visible walkable path that a person could follow on foot. "
        "Show a sequence of distinct, easy-to-recognize locations along the path, like separate landmarks or stops, so the viewer can mentally walk through them in order. "
        f"Place concepts in strict order along the route: {placements}. "
        "Use strong visual separation between loci, with obvious transitions from one stop to the next, so order is unmistakable from start to finish. "
        "Compose the scene from a human walking viewpoint, not a flat map, and make the route progression obvious. "
        "Important: no written text, letters, numbers, captions, signs, or typography in the image."
    )


def build_prompt(words: list[str], prompt_type: PromptType = "loci") -> str:
    if prompt_type == "normal":
        return build_normal_prompt(words)
    if prompt_type == "scene":
        return build_scene_prompt(words)
    if prompt_type == "loci":
        return build_loci_prompt(words)
    raise ValueError(f"unsupported prompt_type: {prompt_type}")

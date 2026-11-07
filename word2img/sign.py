"""Generate an animated PNG of Norwegian dictionary signs from the command line."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

from .__main__ import _safe_filename_stem
from .signs import sign_language


def _write_png(data: bytes, destination: Path) -> None:
    from PIL import Image

    frames = []
    durations = []
    try:
        with Image.open(io.BytesIO(data)) as gif:
            for index in range(gif.n_frames):
                gif.seek(index)
                frames.append(gif.convert("RGBA"))
                durations.append(gif.info["duration"])
        frames[0].save(destination, format="PNG", save_all=True,
                       append_images=frames[1:], duration=durations, loop=0)
    finally:
        for frame in frames:
            frame.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Save Norwegian dictionary signs as an animated PNG (APNG)."
    )
    parser.add_argument("sentence", help="Norwegian sentence to sign")
    parser.add_argument("-o", "--output", type=Path,
                        help="output filename (.png or .gif); default: <sentence>.png")
    args = parser.parse_args(argv)
    destination = args.output or Path(f"{_safe_filename_stem(args.sentence)}.png")
    if destination.suffix.lower() not in {".png", ".gif"}:
        parser.error("output must have a .png or .gif extension")
    try:
        result = sign_language(args.sentence)
        if destination.suffix.lower() == ".gif":
            result.write(destination)
        else:
            _write_png(result["image_bytes"], destination)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved animation: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

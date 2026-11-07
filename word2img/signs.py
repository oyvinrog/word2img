"""Norwegian dictionary signs assembled into portable GIFs (no API key needed)."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from typing import Any, TypedDict

DICTIONARY_URL = "https://www.minetegn.no/tegnordbok/xml/tegnordbok.php"
VIDEO_BASE_URL = "https://www.minetegn.no/Tegnordbok-HTML/video_/"


class SignSelection(TypedDict):
    requested: str
    sign: str
    filename: str
    fingerspelled: bool


class SignLanguageResult(dict[str, Any]):
    """GIF bytes and metadata with a convenience method for saving to disk."""

    def write(self, filename: str | os.PathLike[str]) -> Path:
        """Write the GIF, replacing an existing file, and return its Path.

        The destination's parent directory must already exist.
        """
        path = Path(filename)
        path.write_bytes(self["image_bytes"])
        return path


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _cache_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "word2img" / "signs"


def _fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "word2img/0.1"})
        with urlopen(request, timeout=45) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output)
        if not temporary.stat().st_size:
            raise OSError("empty response")
        temporary.replace(path)
    except OSError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_index(path: Path) -> dict[str, tuple[str, str]]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot read sign dictionary {path}: {exc}") from exc
    index: dict[str, tuple[str, str]] = {}
    for entry in root.iter("leksem"):
        label, filename = (entry.get("visningsord") or "").strip(), (entry.get("filnavn") or "").strip()
        if not label or not filename:
            continue
        full = _normalize(label)
        plain = re.sub(r"\s+\d+$", "", full)
        plain = re.sub(r"\s*\([^)]*\)$", "", plain).strip()
        aliases = {full, plain}
        if "/" in plain and " " not in plain:
            aliases.update(plain.split("/"))
        for key in aliases - {""}:
            previous = index.get(key)
            rank = lambda value: (_normalize(value) != key, "(" in value)
            if previous is None or rank(label) < rank(previous[0]):
                index[key] = (label, filename)
    if not index:
        raise ValueError("Sign dictionary contains no usable entries")
    return index


def _base_forms(word: str) -> list[str]:
    irregular = {"er": "være", "var": "være", "vært": "være", "har": "ha", "hadde": "ha",
                 "går": "gå", "gikk": "gå", "gått": "gå", "gjør": "gjøre", "gjorde": "gjøre",
                 "kommer": "komme", "kom": "komme", "barna": "barn", "menn": "mann"}
    forms = [irregular[word]] if word in irregular else []
    for suffix, ending in (("r", ""), ("en", "e"), ("ene", "e"), ("et", "e"),
                           ("te", "e"), ("de", "e"), ("a", "e"), ("ene", ""),
                           ("ane", ""), ("er", ""), ("ar", ""), ("en", ""),
                           ("et", ""), ("a", ""), ("e", ""), ("t", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            forms.append(word[:-len(suffix)] + ending)
    return forms


def _compound(word: str, index: dict, remaining: int = 3) -> list[str] | None:
    if len(word) >= 3 and word in index:
        return [word]
    if remaining <= 1:
        return None
    for cut in range(len(word) - 3, 2, -1):
        if word[:cut] not in index:
            continue
        for start in (cut, cut + 1) if word[cut:cut + 1] in "es" else (cut,):
            tail = _compound(word[start:], index, remaining - 1)
            if tail:
                return [word[:cut], *tail]
    return None


def _resolve(text: str, index: dict[str, tuple[str, str]], language_wash: bool) -> list[SignSelection]:
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", _normalize(text))
    if not words:
        raise ValueError("sentence must contain words or letters")
    result: list[SignSelection] = []
    longest = max(len(key.split()) for key in index)
    position = 0
    while position < len(words):
        spelled = False
        for count in range(min(longest, len(words) - position), 0, -1):
            requested = " ".join(words[position:position + count])
            if requested in index:
                keys = [requested]
                break
        else:
            count = 1
            requested = words[position]
            keys = []
            if language_wash:
                forms = _base_forms(requested)
                keys = next(([form] for form in forms if form in index), [])
                if not keys:
                    for form in [requested, *forms]:
                        keys = _compound(form.replace("-", ""), index) or []
                        if keys:
                            break
            if not keys:
                keys = list(requested.replace("-", "").replace("'", ""))
                spelled = True
                if any(key not in index for key in keys):
                    raise ValueError(f"Cannot sign or fingerspell {requested!r}; no words were skipped")
        for key in keys:
            label, filename = index[key]
            result.append(SignSelection(requested=requested, sign=label, filename=filename,
                                        fingerspelled=spelled))
        position += count
    return result


def _encode_gif(clips: list[Path], size: tuple[int, int], fps: int, max_frames: int) -> bytes:
    from PIL import Image
    from imageio_ffmpeg import read_frames

    width, height = size
    frames = []
    filters = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    try:
        for clip in clips:
            reader = read_frames(str(clip), output_params=["-vf", filters])
            before = len(frames)
            try:
                metadata = next(reader)
                for raw in reader:
                    if len(frames) >= max_frames:
                        raise ValueError("GIF exceeds max_frames; shorten the sentence or lower fps")
                    with Image.frombytes("RGB", metadata["size"], raw) as frame:
                        frames.append(frame.quantize(colors=256))
            finally:
                reader.close()
            if len(frames) == before:
                raise RuntimeError(f"No video frames in {clip}")
        output = io.BytesIO()
        # GIF delays are in hundredths of a second. Distribute rounding error.
        durations = [10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps))
                     for i in range(len(frames))]
        frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, disposal=2, optimize=False)
        return output.getvalue()
    finally:
        for frame in frames:
            frame.close()


def sign_language(
    sentence: str | list[str], *, size: tuple[int, int] = (480, 360), fps: int = 15,
    cache_dir: str | Path | None = None, dictionary: str | Path | None = None,
    refresh: bool = False, language_wash: bool = True, max_frames: int = 1800,
) -> SignLanguageResult:
    """Join Norwegian words/phrases into one looping GIF of Minetegn signs.

    A list is joined in order into a single sentence/GIF. Unknown words are
    fingerspelled; unrepresentable words raise ValueError rather than disappear.
    This is dictionary lookup, not translation of sign-language grammar.
    Install ``word2img[sign-language]`` for the optional rendering dependencies.
    ``dictionary`` accepts a local Minetegn XML index. Downloads are cached until
    ``refresh=True`` (index only); existing cached clips work offline.
    """
    if isinstance(sentence, list) and all(isinstance(item, str) for item in sentence):
        sentence = " ".join(sentence)
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string or list of strings")
    if not re.search(r"[^\W_]", sentence):
        raise ValueError("sentence must contain words or letters")
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 50:
        raise ValueError("fps must be an integer from 1 to 50")
    if (not isinstance(size, tuple) or len(size) != 2 or
            any(type(n) is not int or not 1 <= n <= 1920 for n in size)):
        raise ValueError("size must be a (width, height) tuple with dimensions from 1 to 1920")
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    try:
        import PIL  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install sign language support: pip install "word2img[sign-language]"') from exc
    cache = Path(cache_dir) if cache_dir is not None else _cache_root()
    source = Path(dictionary) if dictionary is not None else cache / "dictionary.xml"
    if dictionary is None and (refresh or not source.is_file() or source.stat().st_size == 0):
        _fetch(DICTIONARY_URL, source)
    selections = _resolve(sentence, _load_index(source), language_wash)
    clips = []
    for selection in selections:
        filename = selection["filename"]
        clip = cache / "clips" / (hashlib.sha256(filename.encode()).hexdigest() + ".mp4")
        if not clip.is_file() or clip.stat().st_size == 0:
            _fetch(VIDEO_BASE_URL + quote(filename, safe="") + ".mp4", clip)
        clips.append(clip)
    return SignLanguageResult(
        image_bytes=_encode_gif(clips, size, fps, max_frames), mime_type="image/gif",
        text=sentence.strip(), signs=selections, source=DICTIONARY_URL,
    )

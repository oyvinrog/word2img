import base64
import types

import pytest

from word2img.core import _build_prompt, words_to_img


def test_build_prompt_hyphen_joins_words() -> None:
    assert _build_prompt(["frog", "does", "dancing"]) == "frog-does-dancing"


def test_build_prompt_trims_and_ignores_empty_values() -> None:
    assert _build_prompt([" frog ", "", "  ", "dance "]) == "frog-dance"


def test_build_prompt_raises_for_empty_cleaned_input() -> None:
    with pytest.raises(ValueError):
        _build_prompt(["", "   "])


def test_words_to_img_calls_openai_and_returns_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    image_bytes = b"fake-png-bytes"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    called = {}

    class FakeImages:
        def generate(self, **kwargs):
            called["kwargs"] = kwargs
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(b64_json=encoded)],
            )

    class FakeClient:
        def __init__(self, api_key: str):
            called["api_key"] = api_key
            self.images = FakeImages()

    fake_openai_module = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)

    result = words_to_img(["frog", "does", "dancing"], api_key="test-key")

    assert called["api_key"] == "test-key"
    assert called["kwargs"]["model"] == "gpt-image-1"
    assert called["kwargs"]["prompt"] == "frog-does-dancing"
    assert result["image_bytes"] == image_bytes
    assert result["mime_type"] == "image/png"
    assert result["prompt"] == "frog-does-dancing"
    assert result["model"] == "gpt-image-1"


def test_words_to_img_raises_when_api_key_missing() -> None:
    with pytest.raises(RuntimeError, match="api_key is required"):
        words_to_img(["frog"], api_key="")

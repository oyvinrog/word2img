from pathlib import Path

import pytest

from word2img.__main__ import main


def test_cli_writes_png_file_with_word_based_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "frog,does,dancing")
    monkeypatch.setattr("word2img.__main__.resolve_api_key", lambda: "test-key")

    fake_bytes = b"\x89PNG\r\n\x1a\n..."

    def fake_words_to_img(words, api_key, prompt_type):
        assert words == ["frog", "does", "dancing"]
        assert api_key == "test-key"
        assert prompt_type == "loci"
        return {
            "image_bytes": fake_bytes,
            "mime_type": "image/png",
            "prompt": "Create a single memory-palace style geographic scene",
            "model": "gpt-image-1",
        }

    monkeypatch.setattr("word2img.__main__.words_to_img", fake_words_to_img)

    rc = main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "Saved image:" in out
    assert (tmp_path / "frog-does-dancing.png").read_bytes() == fake_bytes


def test_cli_passes_prompt_type_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "frog,does,dancing")
    monkeypatch.setattr("word2img.__main__.resolve_api_key", lambda: "test-key")

    fake_bytes = b"\x89PNG\r\n\x1a\n..."

    def fake_words_to_img(words, api_key, prompt_type):
        assert prompt_type == "normal"
        return {
            "image_bytes": fake_bytes,
            "mime_type": "image/png",
            "prompt": "frog-does-dancing",
            "model": "gpt-image-1",
        }

    monkeypatch.setattr("word2img.__main__.words_to_img", fake_words_to_img)

    rc = main(["--prompt-type", "normal"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Saved image:" in out


def test_cli_truncates_long_word_based_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    long_words = ",".join(["extraordinary"] * 20)
    monkeypatch.setattr("builtins.input", lambda _: long_words)
    monkeypatch.setattr("word2img.__main__.resolve_api_key", lambda: "test-key")

    fake_bytes = b"\x89PNG\r\n\x1a\n..."

    def fake_words_to_img(words, api_key, prompt_type):
        return {
            "image_bytes": fake_bytes,
            "mime_type": "image/png",
            "prompt": "very long generated loci prompt",
            "model": "gpt-image-1",
        }

    monkeypatch.setattr("word2img.__main__.words_to_img", fake_words_to_img)

    rc = main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "Saved image:" in out
    saved_files = list(tmp_path.glob("*.png"))
    assert len(saved_files) == 1
    assert len(saved_files[0].stem) <= 120


def test_cli_returns_nonzero_on_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "frog")
    monkeypatch.setattr("word2img.__main__.resolve_api_key", lambda: "test-key")

    def fake_words_to_img(words, api_key, prompt_type):
        raise RuntimeError("boom")

    monkeypatch.setattr("word2img.__main__.words_to_img", fake_words_to_img)

    rc = main()
    err = capsys.readouterr().err

    assert rc == 1
    assert "Error: boom" in err

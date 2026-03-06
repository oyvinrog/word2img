from pathlib import Path

import pytest

from word2img.effgen import _parse_eff_wordlist, generate_passphrase, main


def test_parse_eff_wordlist() -> None:
    text = "11111\tapple\n11112 banana\n"
    assert _parse_eff_wordlist(text) == ["apple", "banana"]


def test_generate_passphrase_uses_random_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_choice(pool):
        calls["count"] += 1
        return pool[0]

    monkeypatch.setattr("word2img.effgen.secrets.choice", fake_choice)
    phrase = generate_passphrase(3, word_pool=["alpha", "beta"])
    assert phrase == ["alpha", "alpha", "alpha"]
    assert calls["count"] == 3


def test_generate_passphrase_validates_count() -> None:
    with pytest.raises(ValueError, match="num_words must be >= 1"):
        generate_passphrase(0, word_pool=["a"])


def test_effgen_cli_generates_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "word2img.effgen.generate_passphrase",
        lambda num_words: ["alpha", "bravo", "charlie"][:num_words],
    )
    monkeypatch.setattr("word2img.effgen.resolve_api_key", lambda: "test-key")

    def fake_words_to_img(words, api_key):
        assert words == ["alpha", "bravo", "charlie"]
        assert api_key == "test-key"
        return {
            "image_bytes": b"img",
            "mime_type": "image/png",
            "prompt": "alpha-bravo-charlie",
            "model": "gpt-image-1",
        }

    monkeypatch.setattr("word2img.effgen.words_to_img", fake_words_to_img)

    rc = main(["--num-words", "3"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Passphrase: alpha bravo charlie" in out
    assert (tmp_path / "alpha-bravo-charlie.png").read_bytes() == b"img"


from pathlib import Path

import pytest

from word2img.effgen import _parse_eff_wordlist, build_loci_prompt, build_mnemonic_prompt, generate_passphrase, main


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


def test_build_mnemonic_prompt_requests_non_text_scene() -> None:
    prompt = build_mnemonic_prompt(["alpha", "bravo", "charlie"])
    assert "alpha, bravo, charlie" in prompt
    assert "do not include any written text" in prompt


def test_build_loci_prompt_requests_ordered_geography() -> None:
    prompt = build_loci_prompt(["alpha", "bravo", "charlie"])
    assert "memory-palace style geographic scene" in prompt
    assert "at locus 1, depict alpha" in prompt
    assert "at locus 2, depict bravo" in prompt
    assert "at locus 3, depict charlie" in prompt
    assert "no written text" in prompt


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

    def fake_text_to_img(prompt, api_key):
        assert "Passphrase: alpha bravo charlie" in capsys.readouterr().out
        assert "at locus 1, depict alpha" in prompt
        assert "at locus 2, depict bravo" in prompt
        assert "at locus 3, depict charlie" in prompt
        assert "no written text" in prompt
        assert api_key == "test-key"
        return {
            "image_bytes": b"img",
            "mime_type": "image/png",
            "prompt": prompt,
            "model": "gpt-image-1",
        }

    monkeypatch.setattr("word2img.effgen.text_to_img", fake_text_to_img)

    rc = main(["--num-words", "3"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Image type: mnemonic loci (no text)" in out
    assert (tmp_path / "alpha-bravo-charlie.png").read_bytes() == b"img"


def test_effgen_cli_scene_mode_uses_scene_prompt(
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

    def fake_text_to_img(prompt, api_key):
        assert "Passphrase: alpha bravo charlie" in capsys.readouterr().out
        assert "alpha, bravo, charlie" in prompt
        assert "do not include any written text" in prompt
        assert api_key == "test-key"
        return {
            "image_bytes": b"img",
            "mime_type": "image/png",
            "prompt": prompt,
            "model": "gpt-image-1",
        }

    monkeypatch.setattr("word2img.effgen.text_to_img", fake_text_to_img)

    rc = main(["--num-words", "3", "--mnemonic-mode", "scene"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Image type: mnemonic scene (no text)" in out

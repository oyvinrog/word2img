import pytest

from word2img.auth import resolve_api_key


def test_resolve_api_key_reads_from_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeKeyring:
        @staticmethod
        def get_password(service, username):
            assert service == "word2img"
            assert username == "openai_api_key"
            return "stored-key"

    monkeypatch.setitem(__import__("sys").modules, "keyring", FakeKeyring)
    assert resolve_api_key() == "stored-key"


def test_resolve_api_key_prompts_and_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, username):
            return None

        @staticmethod
        def set_password(service, username, value):
            calls["saved"] = (service, username, value)

    monkeypatch.setitem(__import__("sys").modules, "keyring", FakeKeyring)
    monkeypatch.setattr("word2img.auth.getpass.getpass", lambda _: "new-key")

    assert resolve_api_key() == "new-key"
    assert calls["saved"] == ("word2img", "openai_api_key", "new-key")


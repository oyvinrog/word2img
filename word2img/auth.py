from __future__ import annotations

import getpass

KEYRING_SERVICE = "word2img"
KEYRING_USERNAME = "openai_api_key"


def resolve_api_key() -> str:
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError("keyring package is required. Install dependencies first.") from exc

    api_key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if api_key:
        return api_key

    entered = getpass.getpass("Enter OpenAI API key: ").strip()
    if not entered:
        raise RuntimeError("OpenAI API key is required")

    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, entered)
    return entered


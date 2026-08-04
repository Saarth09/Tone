from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


def _fernet() -> Fernet:
    # Derive a stable 32-byte key from SECRET_KEY for Fernet
    digest = hashlib.sha256(get_settings().secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(value: str) -> str:
    if not value:
        return ""
    if value.startswith("enc:"):
        return value
    token = _fernet().encrypt(value.encode("utf-8")).decode("utf-8")
    return f"enc:{token}"


def decrypt_secret(value: str) -> str:
    if not value:
        return ""
    if not value.startswith("enc:"):
        return value  # legacy plaintext
    try:
        return _fernet().decrypt(value[4:].encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return ""

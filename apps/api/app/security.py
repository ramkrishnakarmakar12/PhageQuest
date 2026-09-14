"""Tokens and password hashing.

Deliberately small and dependency-light: PBKDF2 from the standard library and
signed tokens via `itsdangerous`. There is no reason to pull in passlib/bcrypt
for a service whose entire user base authenticates with a school-issued
pseudonym, and every dependency here is one more thing a school's IT department
has to accept.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SECRET = os.getenv("PHAGEQUEST_SECRET_KEY", "dev-only-change-me")
TOKEN_MAX_AGE = int(os.getenv("PHAGEQUEST_TOKEN_MAX_AGE", "43200"))  # 12 hours
_ITERATIONS = 240_000

_serializer = URLSafeTimedSerializer(SECRET, salt="phagequest-auth")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def make_token(user_id: str) -> str:
    return _serializer.dumps({"sub": user_id})


def decode_token(token: str) -> Optional[str]:
    try:
        data = _serializer.loads(token, max_age=TOKEN_MAX_AGE)
        return data.get("sub")
    except (BadSignature, SignatureExpired):
        return None

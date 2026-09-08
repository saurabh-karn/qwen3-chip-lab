"""HttpOnly session cookie for the lab PIN gate.

Set LAB_PIN to exactly eight digits to enable the gate. Unset, local serve
stays open. LAB_SESSION_SECRET is optional; it is derived from the PIN if
missing. LAB_COOKIE_SECURE=1 forces the Secure flag (HTTPS).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time

from starlette.requests import Request
from starlette.responses import Response

COOKIE_NAME = "chip_lab_session"
TTL_SECONDS = 7 * 24 * 3600


def pin_configured() -> bool:
    return bool(os.environ.get("LAB_PIN", "").strip())


def _expected_pin() -> str:
    return os.environ.get("LAB_PIN", "").strip()


def _secret() -> bytes:
    explicit = os.environ.get("LAB_SESSION_SECRET", "").strip()
    if explicit:
        return hashlib.sha256(explicit.encode("utf-8")).digest()
    return hashlib.sha256(b"qwen3-chip-lab:" + _expected_pin().encode("utf-8")).digest()


def pin_matches(got: str) -> bool:
    expected = _expected_pin()
    if len(expected) != 8 or not expected.isdigit():
        return False
    if len(got) != 8 or not got.isdigit():
        return False
    return hmac.compare_digest(got.encode("utf-8"), expected.encode("utf-8"))


def issue_token() -> str:
    exp = str(int(time.time()) + TTL_SECONDS)
    sig = hmac.new(_secret(), exp.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def token_ok(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    exp, _, sig = token.partition(".")
    try:
        if int(exp) < int(time.time()):
            return False
    except ValueError:
        return False
    expect = hmac.new(_secret(), exp.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expect)


def request_authed(request: Request) -> bool:
    if not pin_configured():
        return True
    return token_ok(request.cookies.get(COOKIE_NAME))


def _secure_cookie(request: Request) -> bool:
    if os.environ.get("LAB_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded.split(",")[0].strip().lower() == "https":
        return True
    return request.url.scheme == "https"


def attach_session(response: Response, request: Request) -> None:
    response.set_cookie(
        COOKIE_NAME,
        issue_token(),
        httponly=True,
        samesite="lax",
        max_age=TTL_SECONDS,
        path="/",
        secure=_secure_cookie(request),
    )


def clear_session(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")

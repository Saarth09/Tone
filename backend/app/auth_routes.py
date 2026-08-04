from __future__ import annotations

import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token, get_current_user, get_user_by_email, hash_password, verify_password
from app.config import get_settings
from app.db import get_session
from app.db.models import User
from app.schemas import LoginIn, RegisterIn, TokenOut, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Short-lived CSRF state for Google OAuth (in-memory; fine for single-process local/dev)
_oauth_states: set[str] = set()


@router.get("/providers")
async def auth_providers():
    settings = get_settings()
    return {
        "password": True,
        "google": bool(settings.google_client_id and settings.google_client_secret),
    }


@router.post("/register", response_model=TokenOut)
async def register(body: RegisterIn, session: AsyncSession = Depends(get_session)):
    email = body.email.strip().lower()
    if await get_user_by_email(session, email):
        raise HTTPException(400, "Email already registered")
    user = User(
        email=email,
        password_hash=hash_password(body.password),
        name=body.name.strip() or email.split("@")[0],
        auth_provider="password",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return TokenOut(access_token=create_access_token(user_id=user.id, email=user.email))


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, session: AsyncSession = Depends(get_session)):
    email = body.email.strip().lower()
    user = await get_user_by_email(session, email)
    if (
        user is None
        or not user.password_hash
        or not verify_password(body.password, user.password_hash)
    ):
        raise HTTPException(401, "Invalid email or password")
    return TokenOut(access_token=create_access_token(user_id=user.id, email=user.email))


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user


@router.get("/google")
async def google_start():
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            400,
            "Google login is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )
    state = secrets.token_urlsafe(24)
    _oauth_states.add(state)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "select_account",
    }
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    )


@router.get("/google/callback")
async def google_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    settings = get_settings()
    frontend = settings.frontend_url.rstrip("/")

    if error:
        return RedirectResponse(f"{frontend}/?auth_error={error}")
    if not code or not state or state not in _oauth_states:
        return RedirectResponse(f"{frontend}/?auth_error=invalid_state")
    _oauth_states.discard(state)

    async with httpx.AsyncClient(timeout=30.0) as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token_resp.status_code >= 400:
            return RedirectResponse(f"{frontend}/?auth_error=token_exchange_failed")
        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        if not access_token:
            return RedirectResponse(f"{frontend}/?auth_error=missing_access_token")

        profile_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if profile_resp.status_code >= 400:
            return RedirectResponse(f"{frontend}/?auth_error=profile_failed")
        profile = profile_resp.json()

    email = (profile.get("email") or "").strip().lower()
    sub = profile.get("sub")
    if not email or not sub:
        return RedirectResponse(f"{frontend}/?auth_error=email_required")

    user = await session.scalar(select(User).where(User.google_sub == sub))
    if user is None:
        user = await get_user_by_email(session, email)
        if user is None:
            user = User(
                email=email,
                password_hash=None,
                name=profile.get("name") or email.split("@")[0],
                auth_provider="google",
                google_sub=sub,
                avatar_url=profile.get("picture"),
            )
            session.add(user)
        else:
            user.google_sub = sub
            user.auth_provider = user.auth_provider or "google"
            user.avatar_url = profile.get("picture") or user.avatar_url
            if not user.name and profile.get("name"):
                user.name = profile["name"]
    else:
        user.avatar_url = profile.get("picture") or user.avatar_url
        if profile.get("name"):
            user.name = profile["name"]

    await session.commit()
    await session.refresh(user)
    jwt_token = create_access_token(user_id=user.id, email=user.email)
    return RedirectResponse(f"{frontend}/?token={jwt_token}")

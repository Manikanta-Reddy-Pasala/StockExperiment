"""Shared Dragonfly UI read-cache helpers (used by admin_bp + momrot_bp).

Design: FAIL-OPEN. Any cache miss/error/outage falls straight through to the
live view — caching never blocks or breaks an existing flow. Keys live under the
`ui:*` namespace so a single prefix-scan invalidates every cached UI endpoint.
"""
from __future__ import annotations

from functools import wraps

from flask import request, Response

PORTFOLIO_CACHE_KEY = "ui:models_portfolio:v1"
PORTFOLIO_CACHE_TTL = 30  # seconds


def ui_cache():
    """Return the cache service, or None if Dragonfly is unavailable (fail-open)."""
    try:
        from src.services.utils.cache_service import get_cache_service
        cs = get_cache_service()
        return cs if cs and cs.is_available() else None
    except Exception:
        return None


def invalidate_ui_cache():
    """Drop ALL cached UI payloads (ui:* keys). Fail-open."""
    try:
        cs = ui_cache()
        if cs and cs.redis_client:
            keys = list(cs.redis_client.scan_iter(match="ui:*", count=500))
            if keys:
                cs.redis_client.delete(*keys)
    except Exception:
        pass


def ui_cached(name: str, ttl: int = 30):
    """Decorator: cache a GET endpoint's JSON response in Dragonfly for `ttl`s.

    Keyed by name + full request path (query-arg variants cache separately).
    Caches only GET + 2xx. Fail-OPEN: any error runs the view live.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if request.method != "GET":
                return fn(*args, **kwargs)
            key = f"ui:{name}:{request.full_path}"
            cs = ui_cache()
            if cs:
                try:
                    hit = cs.redis_client.get(key)
                    if hit is not None:
                        return Response(hit, mimetype="application/json")
                except Exception:
                    pass
            resp = fn(*args, **kwargs)
            try:
                body, status = (resp if isinstance(resp, tuple)
                                else (resp, getattr(resp, "status_code", 200)))
                if cs and status == 200:
                    data = body.get_data(as_text=True) if hasattr(body, "get_data") else None
                    if data:
                        cs.redis_client.setex(key, ttl, data)
            except Exception:
                pass
            return resp
        return wrapper
    return deco


def invalidate_on_mutation(response):
    """Flask after_request hook: any successful POST/PUT/DELETE/PATCH drops the
    UI cache so the next read is fresh. Fail-open; never blocks the response."""
    try:
        if (request.method in ("POST", "PUT", "DELETE", "PATCH")
                and 200 <= response.status_code < 300):
            invalidate_ui_cache()
    except Exception:
        pass
    return response

"""Telegram notification helper for momentum rotation alerts.

Sends messages to a Telegram chat via Bot API. Reads credentials from env:
  TG_BOT_TOKEN  - bot token from @BotFather
  TG_CHAT_ID    - group/channel chat_id (negative for groups)

Usage:
  python tools/live/telegram_notify.py "Rebalance: SELL HFCL, BUY BSE"
  python tools/live/telegram_notify.py --markdown "*Bold text*"
  python tools/live/telegram_notify.py --test
"""
from __future__ import annotations

import argparse
import os
import sys
from urllib import parse, request


API_BASE = "https://api.telegram.org"


def _post(token: str, chat_id: str, text: str, parse_mode: str = None) -> dict:
    url = f"{API_BASE}/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4000],
        "disable_web_page_preview": "true",
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    data = parse.urlencode(payload).encode()
    try:
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req, timeout=10) as r:
            import json
            return json.loads(r.read().decode())
    except Exception as e:
        # On 400 (Markdown parse error etc), surface body
        import json
        try:
            body = e.read().decode() if hasattr(e, "read") else ""
        except Exception:
            body = ""
        return {"ok": False, "error": str(e), "body": body}


def send(text: str, parse_mode: str = "Markdown",
         token: str = None, chat_id: str = None) -> dict:
    """Send Telegram message. On Markdown parse failure, retry as plain text."""
    token = token or os.environ.get("TG_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TG_CHAT_ID", "")
    if not token or not chat_id:
        return {"ok": False, "error": "TG_BOT_TOKEN or TG_CHAT_ID not set"}

    res = _post(token, chat_id, text, parse_mode=parse_mode)
    if not res.get("ok") and parse_mode:
        # Likely a Markdown parse error — retry as plain text
        res2 = _post(token, chat_id, text, parse_mode=None)
        if res2.get("ok"):
            return res2
        return res
    return res


def alert_data_missing(model: str, detail: str, dedup_seconds: int = 3600) -> dict:
    """Telegram alert that a model has MISSING/STALE data and is NOT trading.

    Fail-open. dedup_seconds>0 suppresses a repeat of the same (model, detail)
    within the window (Dragonfly-backed) so a model that scans many times a
    morning (ORB) doesn't spam the same 'data missing' alert each scan.
    """
    cs = None
    key = None
    if dedup_seconds and dedup_seconds > 0:
        try:
            import hashlib
            from src.services.utils.cache_service import get_cache_service
            cs = get_cache_service()
            key = "alert:datamissing:" + hashlib.sha1(
                f"{model}|{detail}".encode()).hexdigest()
            if cs.get(key):
                return {"ok": True, "deduped": True}
        except Exception:
            cs = None
            key = None
    res = send(f"🛑 *Data missing — not trading* `{model}`\n{detail}\n"
               f"Model skipped this run (fail-safe; will retry when data returns).")
    if cs is not None and key:
        try:
            cs.set(key, "1", dedup_seconds)
        except Exception:
            pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--markdown", action="store_true", default=True)
    ap.add_argument("--plain", dest="markdown", action="store_false")
    ap.add_argument("--test", action="store_true",
                    help="Send a test message")
    args = ap.parse_args()

    text = args.text
    if args.test:
        text = "🤖 momrot test: bot wired, channel reachable."
    if not text:
        print("error: no text supplied", file=sys.stderr)
        sys.exit(1)

    res = send(text, "Markdown" if args.markdown else None)
    if not res.get("ok"):
        print(f"FAIL: {res.get('error') or res}", file=sys.stderr)
        sys.exit(2)
    print(f"sent: message_id={res['result']['message_id']}")


if __name__ == "__main__":
    main()

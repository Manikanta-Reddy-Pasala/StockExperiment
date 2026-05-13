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


def send(text: str, parse_mode: str = "Markdown",
         token: str = None, chat_id: str = None) -> dict:
    token = token or os.environ.get("TG_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TG_CHAT_ID", "")
    if not token or not chat_id:
        return {"ok": False, "error": "TG_BOT_TOKEN or TG_CHAT_ID not set"}

    url = f"{API_BASE}/bot{token}/sendMessage"
    data = parse.urlencode({
        "chat_id": chat_id,
        "text": text[:4000],   # Telegram limit
        "parse_mode": parse_mode,
        "disable_web_page_preview": "true",
    }).encode()
    try:
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req, timeout=10) as r:
            import json
            return json.loads(r.read().decode())
    except Exception as e:
        return {"ok": False, "error": str(e)}


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

"""
Offline backtest harness for the EMA 200/400 1H crossover strategy.

Pulls 1-hour OHLCV from yfinance for a small list of NSE symbols, runs the
strategy state machine directly (no DB), and emits one Markdown report per
symbol summarizing signals + P&L.

Usage:
    venv/bin/python tools/backtests/run_ema_200_400_backtest.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.technical.ema_crossover_strategy import (  # noqa: E402
    EMACrossoverStrategy,
    StrategyConfig,
)
from src.services.data.nifty500_universe import (  # noqa: E402
    load_nifty500_with_meta,
)


# Smoke-test basket — banks, IT, FMCG, energy. Used when --universe=smoke.
SMOKE_SYMBOLS = [
    ("HDFCBANK.NS", "HDFC Bank"),
    ("RELIANCE.NS", "Reliance Industries"),
    ("INFY.NS", "Infosys"),
    ("TCS.NS", "Tata Consultancy"),
    ("ICICIBANK.NS", "ICICI Bank"),
]

# NIFTY 50 constituents (Yahoo .NS format). Source mirrors
# yfinance_data_service.get_nifty50_stocks().
NIFTY50_BASE = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
    'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
    'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
    'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC',
    'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT',
    'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC',
    'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
    'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
    'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'LTIM',
]
# Yahoo .NS format. `&` gets URL-encoded by fetch_1h_yahoo() so M&M.NS works.
# Override map for symbols Yahoo lists under different ticker than NSE.
YAHOO_OVERRIDES = {
    "TATAMOTORS": "TATAMOTORS.NS",  # active again post-2024 demerger
    "LTIM": "LTIM.NS",
}
NIFTY50_SYMBOLS = [
    (YAHOO_OVERRIDES.get(s, f"{s}.NS"), s) for s in NIFTY50_BASE
]

# Indian benchmark indices — 5000-pt target rule applies (image spec).
INDEX_SYMBOLS = [
    ("^NSEI", "NIFTY 50"),
    ("^NSEBANK", "BANK NIFTY"),
    ("^CNXFIN", "NIFTY FIN SERVICE"),
    ("^CNXIT", "NIFTY IT"),
    ("^CNXAUTO", "NIFTY AUTO"),
]


def nifty500_yahoo_symbols(limit: Optional[int] = None) -> List[tuple]:
    """Return ``[(yahoo_symbol, company_name)]`` for the Nifty 500."""
    rows = load_nifty500_with_meta()
    out = []
    for fyers_sym, name, _industry in rows:
        # Fyers ``NSE:RELIANCE-EQ`` -> Yahoo ``RELIANCE.NS``
        plain = fyers_sym.replace("NSE:", "").replace("-EQ", "")
        out.append((f"{plain}.NS", name))
    return out[:limit] if limit else out


YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def fetch_1h_yahoo(symbol: str, days: int = 720) -> pd.DataFrame:
    """Yahoo chart API. Hourly data window cap is ~730 days."""
    from urllib.parse import quote
    params = {"interval": "1h", "range": f"{min(days, 730)}d"}
    try:
        r = requests.get(YAHOO_URL.format(symbol=quote(symbol, safe=".^")),
                         params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  yahoo fetch failed: {e}")
        return pd.DataFrame()

    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()
    res = result[0]
    timestamps = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quote:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if df.empty:
        return df
    df["candle_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True) \
        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", "candle_time", "open", "high", "low", "close", "volume"]]


# ---- Fyers fetcher (production primary) ------------------------------
# Reuses FyersService.history() — same token flow used in production. Needs:
#   1. Postgres reachable (broker_configurations table holds access_token)
#   2. Valid Fyers session for user_id (default 1)
# Falls back to Yahoo when DB or token unavailable.

_FYERS_CACHE = {"service": None, "init_failed": False, "user_id": 1}


def _fyers_service():
    """Lazy-init Fyers; cache result. Returns None if init fails."""
    if _FYERS_CACHE["init_failed"]:
        return None
    if _FYERS_CACHE["service"] is not None:
        return _FYERS_CACHE["service"]
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        cfg = svc.get_broker_config(_FYERS_CACHE["user_id"])
        if not cfg or not cfg.get("access_token"):
            print(f"  fyers: no token for user_id={_FYERS_CACHE['user_id']}")
            _FYERS_CACHE["init_failed"] = True
            return None
        _FYERS_CACHE["service"] = svc
        return svc
    except Exception as e:
        print(f"  fyers init failed: {e}")
        _FYERS_CACHE["init_failed"] = True
        return None


def yahoo_to_fyers_symbol(yahoo_sym: str) -> str:
    """``RELIANCE.NS`` -> ``NSE:RELIANCE-EQ``. Index ``^NSEI`` -> ``NSE:NIFTY50-INDEX``."""
    s = yahoo_sym.upper()
    if s.startswith("^"):
        idx_map = {
            "^NSEI": "NSE:NIFTY50-INDEX",
            "^NSEBANK": "NSE:NIFTYBANK-INDEX",
            "^CNXFIN": "NSE:FINNIFTY-INDEX",
            "^CNXIT": "NSE:NIFTYIT-INDEX",
            "^CNXAUTO": "NSE:NIFTYAUTO-INDEX",
        }
        return idx_map.get(s, s)
    return f"NSE:{s.replace('.NS', '')}-EQ"


def fetch_1h_fyers(symbol: str, days: int = 720,
                   user_id: int = 1) -> pd.DataFrame:
    """Pull Fyers 1H candles, chunking 95-day windows to respect API caps."""
    svc = _fyers_service()
    if svc is None:
        return pd.DataFrame()

    fyers_sym = yahoo_to_fyers_symbol(symbol)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    cursor = start_dt
    all_candles: List[List] = []
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=95), end_dt)
        try:
            res = svc.history(
                user_id=user_id,
                symbol=fyers_sym,
                exchange="NSE",
                interval="1h",
                start_date=cursor.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
            )
            if res and res.get("status") == "success":
                all_candles += res.get("data", {}).get("candles", []) or []
            else:
                msg = (res or {}).get("message", "no response")
                print(f"  fyers chunk fail {fyers_sym} "
                      f"{cursor.date()}..{chunk_end.date()}: {msg}")
        except Exception as e:
            print(f"  fyers chunk error {fyers_sym}: {e}")
        cursor = chunk_end

    if not all_candles:
        return pd.DataFrame()

    # Fyers SDK in this project returns dicts with string-typed values:
    #   [{"timestamp": "1777261500", "open": "787.5", "high": ..., ...}, ...]
    # Older fyers-apiv3 returns flat lists [ts, o, h, l, c, v]. Handle both.
    if isinstance(all_candles[0], dict):
        df = pd.DataFrame(all_candles)
    else:
        df = pd.DataFrame(all_candles,
                          columns=["timestamp", "open", "high",
                                   "low", "close", "volume"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["candle_time"] = pd.to_datetime(df["timestamp"].astype("int64"),
                                       unit="s", utc=True) \
        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", "candle_time", "open", "high", "low",
               "close", "volume"]]


def fetch_1h_data(symbol: str, days: int = 720,
                  source: str = "auto", user_id: int = 1) -> tuple:
    """Return ``(df, source_used)``.

    ``source``:
        - ``fyers``: only Fyers, error if unavailable.
        - ``yahoo``: only Yahoo.
        - ``auto`` (default): Fyers first, Yahoo fallback per-symbol.
    """
    if source == "yahoo":
        return fetch_1h_yahoo(symbol, days), "yahoo"
    if source == "fyers":
        return fetch_1h_fyers(symbol, days, user_id), "fyers"
    # auto: Fyers first, fall back to Yahoo on empty
    df = fetch_1h_fyers(symbol, days, user_id)
    if not df.empty:
        return df, "fyers"
    return fetch_1h_yahoo(symbol, days), "yahoo"


def df_to_candles(df: pd.DataFrame) -> List[SimpleNamespace]:
    """Lightweight stand-ins for HistoricalData1H rows (no DB needed)."""
    return [
        SimpleNamespace(
            timestamp=int(r.timestamp),
            candle_time=r.candle_time,
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=int(r.volume or 0),
        )
        for r in df.itertuples()
    ]


@dataclass
class StrategyState:
    """In-memory mirror of EMACrossoverState. The strategy module mutates it."""
    user_id: int = 0
    symbol: str = ""
    trend: str = "NONE"
    stage: int = 0
    crossover_ts: Optional[int] = None
    crossover_high: Optional[float] = None
    crossover_low: Optional[float] = None
    retest1_ts: Optional[int] = None
    retest1_high: Optional[float] = None
    retest1_low: Optional[float] = None
    retest2_ts: Optional[int] = None
    retest2_high: Optional[float] = None
    retest2_low: Optional[float] = None
    entries_count: int = 0
    entry1_price: Optional[float] = None
    entry1_time: Optional[datetime] = None
    entry2_price: Optional[float] = None
    entry2_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    position_active: bool = False
    last_evaluated_ts: Optional[int] = None
    # v2 BTC rules
    retest1_attempts: int = 0
    retest2_attempts: int = 0
    retest1_invalidated: bool = False
    retest1_pending_cross_ts: Optional[int] = None
    retest2_pending_cross_ts: Optional[int] = None
    alert3_locks_count: int = 0
    positions_json: list = None  # mutable; init in __post_init__

    def __post_init__(self):
        if self.positions_json is None:
            self.positions_json = []


class OfflineStrategy(EMACrossoverStrategy):
    """Subclass that bypasses DB I/O — keeps state in memory."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        # Skip parent __init__ which calls get_database_manager()
        self._states: Dict[str, StrategyState] = {}
        self._signal_log: List[Dict] = []

    def _load_state(self, user_id, symbol):  # type: ignore[override]
        key = f"{user_id}:{symbol}"
        if key not in self._states:
            self._states[key] = StrategyState(user_id=user_id, symbol=symbol)
        return self._states[key]

    def _save_state(self, state):  # type: ignore[override]
        return  # no-op

    def _record_signal(self, user_id, symbol, sig):  # type: ignore[override]
        sig = dict(sig)
        sig["user_id"] = user_id
        sig["symbol"] = symbol
        self._signal_log.append(sig)


INDEX_SUFFIXES = ("-INDEX",)


def _is_index_symbol(symbol: str) -> bool:
    return symbol.startswith("^") or symbol.endswith(INDEX_SUFFIXES) or symbol.upper() in {
        "NIFTY.NS", "BANKNIFTY.NS",
    }


def simulate_pnl(
    signals: List[Dict],
    df: pd.DataFrame,
    symbol: str,
    partial_qty_frac: float = 0.5,
) -> Dict:
    """Walk strategy signals; v2 strategy emits per-position events directly.

    Signal types consumed:
        ENTRY1 / ENTRY2  -> open 1 unit; carries 'sl' and 'target' fields
        PARTIAL          -> book partial_qty_frac of earliest matching open pos
        STOP_HIT         -> close remaining qty of earliest matching open pos
        TARGET_HIT       -> close remaining qty of earliest matching open pos
        EXIT             -> ignored (force-close already emitted as STOP_HIT)
    P&L per unit. FIFO match by trend (PARTIAL/STOP_HIT/TARGET_HIT close
    earliest-open same-trend position).
    """
    if df.empty:
        return _empty_pnl()

    open_positions: List[Dict] = []
    closed: List[Dict] = []
    pnl_total = 0.0

    def _pnl(trend: str, entry: float, exit_price: float, qty: float) -> float:
        if trend == "BUY":
            return qty * (exit_price - entry)
        return qty * (entry - exit_price)

    def _pct(trend: str, entry: float, exit_price: float) -> float:
        """Per-leg % return (independent of position size)."""
        if entry <= 0:
            return 0.0
        if trend == "BUY":
            return (exit_price / entry - 1.0) * 100.0
        return (1.0 - exit_price / entry) * 100.0

    def _match(trend: str, need_qty: bool = True) -> Optional[Dict]:
        for p in open_positions:
            if p["trend"] != trend:
                continue
            if need_qty and p["qty"] <= 0:
                continue
            return p
        return None

    for sig in signals:
        t = sig["signal_type"]
        trend = sig["trend"]
        if t in ("ENTRY1", "ENTRY2"):
            open_positions.append({
                "type": t,
                "trend": trend,
                "entry_alert": "retest1" if t == "ENTRY1" else "retest2",
                "entry_price": float(sig["price"]),
                "entry_time": sig["candle_time"],
                "entry_ts": int(sig["candle_ts"]),
                "target": float(sig.get("target") or 0.0),
                "sl": float(sig.get("sl") or 0.0),
                "qty": 1.0,
                "partial_booked": False,
            })
        elif t == "PARTIAL":
            for p in open_positions:
                if p["trend"] == trend and not p["partial_booked"] and p["qty"] > 0:
                    book = p["qty"] * partial_qty_frac
                    exit_price = float(sig["price"])
                    pnl = _pnl(trend, p["entry_price"], exit_price, book)
                    pnl_total += pnl
                    closed.append({
                        **{k: p[k] for k in ("trend", "entry_alert", "entry_price",
                                              "entry_time", "type")},
                        "exit_price": exit_price,
                        "exit_time": sig["candle_time"],
                        "exit_reason": "PARTIAL",
                        "qty_closed": book,
                        "pnl": pnl,
                        "pct": _pct(trend, p["entry_price"], exit_price),
                    })
                    p["qty"] -= book
                    p["partial_booked"] = True
                    p["sl"] = p["entry_price"]  # trail SL to entry
                    break
        elif t in ("TARGET_HIT", "STOP_HIT"):
            p = _match(trend, need_qty=True)
            if p is not None:
                exit_price = float(sig["price"])
                pnl = _pnl(trend, p["entry_price"], exit_price, p["qty"])
                pnl_total += pnl
                closed.append({
                    **{k: p[k] for k in ("trend", "entry_alert", "entry_price",
                                          "entry_time", "type")},
                    "exit_price": exit_price,
                    "exit_time": sig["candle_time"],
                    "exit_reason": t,
                    "qty_closed": p["qty"],
                    "pnl": pnl,
                    "pct": _pct(trend, p["entry_price"], exit_price),
                })
                p["qty"] = 0.0
            open_positions = [p for p in open_positions if p["qty"] > 0]
        # EXIT, CROSSOVER, ALERT*, ALERT2_SKIP -> no P&L action
    pcts = [c["pct"] for c in closed]
    return {
        "trades_closed": len(closed),
        "trades_open": len(open_positions),
        "winners": sum(1 for c in closed if c["pct"] > 0),
        "losers": sum(1 for c in closed if c["pct"] <= 0),
        "total_pnl": pnl_total,
        "avg_pnl": (pnl_total / len(closed)) if closed else 0.0,
        "sum_pct": sum(pcts),
        "avg_pct": (sum(pcts) / len(pcts)) if pcts else 0.0,
        "median_pct": sorted(pcts)[len(pcts) // 2] if pcts else 0.0,
        "target_hits": sum(1 for c in closed if c["exit_reason"] == "TARGET_HIT"),
        "stop_hits": sum(1 for c in closed if c["exit_reason"] == "STOP_HIT"),
        "partials": sum(1 for c in closed if c["exit_reason"] == "PARTIAL"),
        "closed": closed,
        "open": open_positions,
    }


def _empty_pnl() -> Dict:
    return {
        "trades_closed": 0, "trades_open": 0, "winners": 0, "losers": 0,
        "total_pnl": 0.0, "avg_pnl": 0.0,
        "sum_pct": 0.0, "avg_pct": 0.0, "median_pct": 0.0,
        "target_hits": 0, "stop_hits": 0, "partials": 0,
        "closed": [], "open": [],
    }


def summarize_subset(legs: List[Dict]) -> Dict:
    """Aggregate stats for any subset of closed legs (BUY-only, SELL-only,
    retest1-only, retest2-only, etc.)."""
    if not legs:
        return {
            "legs": 0, "winners": 0, "win_rate": 0.0,
            "avg_pct": 0.0, "sum_pct": 0.0, "median_pct": 0.0,
            "target_hits": 0, "stop_hits": 0, "partials": 0,
        }
    pcts = [c["pct"] for c in legs]
    return {
        "legs": len(legs),
        "winners": sum(1 for c in legs if c["pct"] > 0),
        "win_rate": sum(1 for c in legs if c["pct"] > 0) / len(legs) * 100,
        "avg_pct": sum(pcts) / len(pcts),
        "sum_pct": sum(pcts),
        "median_pct": sorted(pcts)[len(pcts) // 2],
        "target_hits": sum(1 for c in legs if c["exit_reason"] == "TARGET_HIT"),
        "stop_hits": sum(1 for c in legs if c["exit_reason"] == "STOP_HIT"),
        "partials": sum(1 for c in legs if c["exit_reason"] == "PARTIAL"),
    }


# ---- Cycle grouping -------------------------------------------------
# A "cycle" begins at every CROSSOVER. Subsequent signals belong to that cycle
# until the next CROSSOVER (or end of data). Lets us render the user-requested
# stage breakdown:
#   Trend Identification → First Alert → Second Alert (Retest 1)
#   → First Entry → Third Alert (Retest 2) → Second Entry → Exit
STAGE_LABELS_BUY = {
    "CROSSOVER":   "Trend Identification (BUY)",
    "ALERT1":      "First Alert — break + close above crossover candle high",
    "ALERT2":      "Second Alert (Retest 1) — EMA200 retest from above",
    "ALERT2_SKIP": "Retest1 invalidated — EMA400 touched before ENTRY1",
    "ENTRY1":      "First Entry (BUY) — retest1 break (cap 3 attempts)",
    "ALERT3":      "Third Alert (Retest 2) — price touches/crosses EMA400",
    "ENTRY2":      "Second Entry (BUY) — retest2 break (cap 3 attempts)",
    "PARTIAL":     "Partial book — 50% qty @ 15%, trail SL → entry",
    "TARGET_HIT":  "Target hit — 30% from entry",
    "STOP_HIT":    "Stop hit — per-position SL triggered",
    "PENDING":     "Cross detected — sustain check pending",
    "PENDING_CANCEL": "Sustain check cancelled (price retraced)",
    "EXIT":        "Exit — 1H close below EMA400",
}

STAGE_LABELS_SELL = {
    "CROSSOVER":   "Trend Identification (SELL)",
    "ALERT1":      "First Alert — break + close below crossover candle low",
    "ALERT2":      "Second Alert (Retest 1) — EMA200 retest from below",
    "ALERT2_SKIP": "Retest1 invalidated — EMA400 touched before ENTRY1",
    "ENTRY1":      "First Entry (SELL) — retest1 break (cap 3 attempts)",
    "ALERT3":      "Third Alert (Retest 2) — price touches/crosses EMA400",
    "ENTRY2":      "Second Entry (SELL) — retest2 break (cap 3 attempts)",
    "PARTIAL":     "Partial book — 50% qty @ 15%, trail SL → entry",
    "TARGET_HIT":  "Target hit — 30% from entry",
    "STOP_HIT":    "Stop hit — per-position SL triggered",
    "PENDING":     "Cross detected — sustain check pending",
    "PENDING_CANCEL": "Sustain check cancelled (price retraced)",
    "EXIT":        "Exit — 1H close above EMA400",
}


def build_cycles(signals: List[Dict]) -> List[Dict]:
    """Group signals into cycles. New cycle on each CROSSOVER."""
    cycles: List[Dict] = []
    current: Optional[Dict] = None
    for s in signals:
        if s["signal_type"] == "CROSSOVER":
            if current:
                cycles.append(current)
            current = {"trend": s["trend"], "events": [s]}
        elif current:
            current["events"].append(s)
    if current:
        cycles.append(current)
    return cycles


def render_md(symbol: str, name: str, df: pd.DataFrame,
              signals: List[Dict], pnl: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol.replace('.NS', '').lower()}.md"
    path = out_dir / fname

    if df.empty:
        path.write_text(f"# {name} ({symbol})\n\n_No 1H data available._\n")
        return path

    first = df["candle_time"].iloc[0]
    last = df["candle_time"].iloc[-1]
    bar_count = len(df)
    last_close = float(df["close"].iloc[-1])

    # Group signals by type for the summary table.
    type_counts: Dict[str, int] = {}
    for s in signals:
        type_counts[s["signal_type"]] = type_counts.get(s["signal_type"], 0) + 1

    lines = [
        f"# {name} ({symbol})",
        "",
        "## Backtest Summary",
        "",
        f"- **Source:** Yahoo chart API (1H bars)",
        f"- **Window:** {first} → {last} ({bar_count} bars)",
        f"- **Last close:** {last_close:.2f}",
        f"- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)",
        f"- **Target:** entry × (1 ± 30%)",
        f"- **Partial:** 50% qty booked @ 15%, trail SL → entry",
        f"- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high",
        f"- **Re-entry cap:** 3 attempts each at retest1 and retest2",
        "",
        "## Signal Counts",
        "",
        "| Signal | Count |",
        "|--------|-------|",
    ]
    for k in ("CROSSOVER", "ALERT1", "ALERT2", "ALERT2_SKIP", "ALERT3",
              "PENDING", "PENDING_CANCEL",
              "ENTRY1", "ENTRY2", "PARTIAL", "TARGET_HIT", "STOP_HIT", "EXIT"):
        lines.append(f"| {k} | {type_counts.get(k, 0)} |")

    lines += [
        "",
        "## P&L (combined)",
        "",
        f"- **Closed legs:** {pnl['trades_closed']} (incl. partial bookings)",
        f"- **Trades open at end:** {pnl['trades_open']}",
        f"- **Winners / losers:** {pnl['winners']} / {pnl['losers']}",
        f"- **Target hits / Stop hits / Partials:** "
        f"{pnl.get('target_hits', 0)} / {pnl.get('stop_hits', 0)} / {pnl.get('partials', 0)}",
        f"- **Avg / median % per leg:** {pnl.get('avg_pct', 0):.2f}% / {pnl.get('median_pct', 0):.2f}%",
        f"- **Sum % (uncompounded):** {pnl.get('sum_pct', 0):.2f}%",
    ]

    # ---- Direction + alert breakdown ----
    closed = pnl.get("closed") or []
    buy_legs   = [c for c in closed if c["trend"] == "BUY"]
    sell_legs  = [c for c in closed if c["trend"] == "SELL"]
    retest1    = [c for c in closed if c.get("entry_alert") == "retest1"]
    retest2    = [c for c in closed if c.get("entry_alert") == "retest2"]
    buy_r1   = [c for c in buy_legs  if c.get("entry_alert") == "retest1"]
    buy_r2   = [c for c in buy_legs  if c.get("entry_alert") == "retest2"]
    sell_r1  = [c for c in sell_legs if c.get("entry_alert") == "retest1"]
    sell_r2  = [c for c in sell_legs if c.get("entry_alert") == "retest2"]
    buckets = [
        ("BUY (all)",         buy_legs),
        ("BUY @ 2nd Alert (retest1)",  buy_r1),
        ("BUY @ 3rd Alert (retest2)",  buy_r2),
        ("SELL (all)",        sell_legs),
        ("SELL @ 2nd Alert (retest1)", sell_r1),
        ("SELL @ 3rd Alert (retest2)", sell_r2),
        ("retest1 (combined)",         retest1),
        ("retest2 (combined)",         retest2),
    ]
    lines += [
        "",
        "## Direction × Alert breakdown",
        "",
        "| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |",
        "|--------|------|-----|------|-----|----|-----|-------|-------|",
    ]
    for name, subset in buckets:
        s = summarize_subset(subset)
        lines.append(
            f"| {name} | {s['legs']} | {s['winners']} | {s['win_rate']:.1f}% | "
            f"{s['target_hits']} | {s['stop_hits']} | {s['partials']} | "
            f"{s['avg_pct']:.2f}% | {s['sum_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Strategy Cycles",
        "",
        "Each cycle begins at a CROSSOVER (trend flip) and walks through the",
        "configured stages: Trend ID → First Alert → Second Alert (Retest 1)",
        "→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.",
        "",
    ]

    cycles = build_cycles(signals)
    if not cycles:
        lines.append("_No CROSSOVER signals fired in window._")
    for idx, cyc in enumerate(cycles, 1):
        first = cyc["events"][0]
        labels = STAGE_LABELS_BUY if cyc["trend"] == "BUY" else STAGE_LABELS_SELL
        lines += [
            f"### Cycle {idx} — {cyc['trend']} (started {first['candle_time']})",
            "",
            "| Stage | Time | Price | EMA200 | EMA400 | Note |",
            "|-------|------|-------|--------|--------|------|",
        ]
        for ev in cyc["events"]:
            label = labels.get(ev["signal_type"], ev["signal_type"])
            note = (ev.get("note") or "").replace("|", "/")
            lines.append(
                f"| {label} | {ev['candle_time']} | {ev['price']:.2f} | "
                f"{ev['ema_200']:.2f} | {ev['ema_400']:.2f} | {note} |"
            )
        lines.append("")

    if pnl["closed"]:
        lines += [
            "",
            "## Closed Legs",
            "",
            "| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |",
            "|-------|-------|-----------|-------|-----------|------|--------|-----|---|",
        ]
        for t in pnl["closed"]:
            lines.append(
                f"| {t['trend']} | {t.get('entry_alert','')} | {t['entry_time']} | "
                f"{t['entry_price']:.2f} | {t['exit_time']} | "
                f"{t['exit_price']:.2f} | {t.get('exit_reason', '')} | "
                f"{t.get('qty_closed', 1.0):.2f} | {t.get('pct', 0):.2f}% |"
            )

    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=720,
                        help="History window (Yahoo caps 1H at ~730d)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "exports" / "backtests")
    parser.add_argument("--universe", choices=["smoke", "nifty50", "nifty500", "indices"],
                        default="smoke",
                        help="smoke=5-stock sanity; nifty50=NIFTY 50 constituents; "
                             "nifty500=full Nifty 500; indices=NIFTY/BANKNIFTY/sectoral")
    parser.add_argument("--limit", type=int, default=None,
                        help="When --universe=nifty500, cap to first N")
    parser.add_argument("--source", choices=["auto", "fyers", "yahoo"],
                        default="auto",
                        help="Data source. Default 'auto' = Fyers first, Yahoo "
                             "fallback per-symbol (recommended). 'fyers' = Fyers "
                             "only, errors if token missing. 'yahoo' = Yahoo only.")
    parser.add_argument("--user-id", type=int, default=1,
                        help="Fyers user_id (default 1) for token lookup")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single-stock mode. Yahoo (RELIANCE.NS / ^NSEI) or "
                             "Fyers (NSE:RELIANCE-EQ) format. Overrides --universe.")
    parser.add_argument("--from", dest="date_from", type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days; Yahoo only honors range)")
    parser.add_argument("--to", dest="date_to", type=str, default=None,
                        help="End date YYYY-MM-DD (defaults to today when --from given)")
    # ---- Tuning flags (opt-in; default = spec-compliant) ----
    parser.add_argument("--htf-filter", action="store_true",
                        help="Enable HTF (higher-timeframe) trend filter on crossovers")
    parser.add_argument("--htf-buy-period-bars", type=int, default=1400,
                        help="HTF SMA period for BUY (default 1400 ~= 200d on 1H)")
    parser.add_argument("--htf-sell-period-bars", type=int, default=1400,
                        help="HTF SMA period for SELL (default 1400 ~= 200d on 1H)")
    parser.add_argument("--htf-buy-margin-pct", type=float, default=0.0,
                        help="BUY confirm requires close > SMA*(1+margin). 0=disabled")
    parser.add_argument("--htf-sell-margin-pct", type=float, default=0.0,
                        help="SELL confirm requires close < SMA*(1-margin). E.g. 0.03 "
                             "= 3%% below SMA (filters shallow dips). 0=disabled")
    parser.add_argument("--htf-period-bars", type=int, default=0,
                        help="Legacy: single HTF period overrides BUY+SELL when >0")
    parser.add_argument("--max-alert3-locks", type=int, default=0,
                        help="Cap ALERT3 re-locks per cycle (0=unlimited)")
    parser.add_argument("--retest2-sl-cap-pct", type=float, default=0.0,
                        help="Cap ENTRY2 SL distance from entry (e.g. 0.03 = 3%%, 0=disabled)")
    parser.add_argument("--skip-retest2", action="store_true",
                        help="Skip retest2/ENTRY2 phase entirely (retest1 only)")
    args = parser.parse_args()
    _FYERS_CACHE["user_id"] = args.user_id

    # Convert --from/--to to days window (1H window must end at "now" for Yahoo
    # range param; for Fyers we pass exact dates)
    if args.date_from:
        from_dt = datetime.strptime(args.date_from, "%Y-%m-%d")
        to_dt = datetime.strptime(args.date_to, "%Y-%m-%d") if args.date_to else datetime.now()
        delta_days = max(1, (datetime.now() - from_dt).days)
        args.days = min(delta_days, 730)
        print(f"Date range: {from_dt.date()} → {to_dt.date()} ({args.days} days)")

    if args.symbol:
        # Normalize Fyers-style to Yahoo for harness fetcher
        sym = args.symbol.upper()
        if sym.startswith("NSE:") and sym.endswith("-EQ"):
            sym = sym.replace("NSE:", "").replace("-EQ", "") + ".NS"
        elif sym.endswith("-INDEX"):
            sym = "^" + sym.replace("NSE:", "").replace("-INDEX", "")
        symbols_list = [(sym, sym)]
        print(f"Universe: single symbol ({sym})")
    elif args.universe == "nifty50":
        symbols_list = NIFTY50_SYMBOLS
        if args.limit:
            symbols_list = symbols_list[:args.limit]
        print(f"Universe: NIFTY 50 ({len(symbols_list)} symbols)")
    elif args.universe == "nifty500":
        symbols_list = nifty500_yahoo_symbols(limit=args.limit)
        if not symbols_list:
            print("Nifty 500 cache empty. Run tools/refresh_nifty500.py first.")
            return 2
        print(f"Universe: Nifty 500 ({len(symbols_list)} symbols)")
    elif args.universe == "indices":
        symbols_list = INDEX_SYMBOLS
        print(f"Universe: Indian indices ({len(symbols_list)} symbols)")
    else:
        symbols_list = SMOKE_SYMBOLS
        print(f"Universe: smoke ({len(symbols_list)} symbols)")

    config = StrategyConfig(
        htf_filter_enabled=args.htf_filter,
        htf_buy_period_bars=args.htf_buy_period_bars,
        htf_sell_period_bars=args.htf_sell_period_bars,
        htf_buy_margin_pct=args.htf_buy_margin_pct,
        htf_sell_margin_pct=args.htf_sell_margin_pct,
        htf_period_bars=args.htf_period_bars,
        max_alert3_locks_per_cycle=args.max_alert3_locks,
        retest2_sl_cap_pct=args.retest2_sl_cap_pct,
        skip_retest2=args.skip_retest2,
    )
    print(f"Config tuning: htf_filter={config.htf_filter_enabled} "
          f"buy_p={config.htf_buy_period_bars}+m{config.htf_buy_margin_pct} "
          f"sell_p={config.htf_sell_period_bars}+m{config.htf_sell_margin_pct} "
          f"max_alert3_locks={config.max_alert3_locks_per_cycle} "
          f"retest2_sl_cap_pct={config.retest2_sl_cap_pct} "
          f"skip_retest2={config.skip_retest2}")
    strat = OfflineStrategy(config)

    aggregate = []
    source_counts = {"fyers": 0, "yahoo": 0, "none": 0}
    for symbol, name in symbols_list:
        print(f"--- {symbol} ---", flush=True)
        df, src = fetch_1h_data(symbol, days=args.days,
                                source=args.source, user_id=args.user_id)
        source_counts[src if not df.empty else "none"] = \
            source_counts.get(src if not df.empty else "none", 0) + 1
        print(f"  source={src}, bars={len(df)}", flush=True) if df.empty else None
        if df.empty:
            print(f"  no data, skipping")
            render_md(symbol, name, df, [], {"trades_closed": 0, "trades_open": 0,
                                             "winners": 0, "losers": 0,
                                             "total_pnl": 0, "avg_pnl": 0,
                                             "closed": [], "open": []}, args.out)
            continue

        candles = df_to_candles(df)
        # Strategy needs ema_slow_period + 5 bars at minimum.
        if len(candles) < config.ema_slow_period + 5:
            print(f"  only {len(candles)} bars, need {config.ema_slow_period + 5}+")
            continue

        signals = strat.evaluate(user_id=1, symbol=symbol, candles=candles)
        pnl = simulate_pnl(
            signals, df, symbol,
            partial_qty_frac=config.partial_qty_frac,
        )

        path = render_md(symbol, name, df, signals, pnl, args.out)
        # Per-direction aggregates for the global summary
        closed = pnl.get("closed") or []
        buy_stats  = summarize_subset([c for c in closed if c["trend"] == "BUY"])
        sell_stats = summarize_subset([c for c in closed if c["trend"] == "SELL"])
        print(f"  src={src}  bars={len(candles)}  legs={pnl['trades_closed']} "
              f"BUY({buy_stats['legs']} avg{buy_stats['avg_pct']:.1f}%) "
              f"SELL({sell_stats['legs']} avg{sell_stats['avg_pct']:.1f}%) "
              f"-> {path}")
        aggregate.append({
            "symbol": symbol,
            "name": name,
            "bars": len(candles),
            "signals": len(signals),
            "closed_legs": closed,
            "trades_closed": pnl["trades_closed"],
            "winners": pnl["winners"],
            "target_hits": pnl["target_hits"],
            "stop_hits": pnl["stop_hits"],
            "partials": pnl["partials"],
            "sum_pct": pnl["sum_pct"],
            "avg_pct": pnl["avg_pct"],
            "buy": buy_stats,
            "sell": sell_stats,
        })

    # Top-level summary file
    total_closed = sum(a['trades_closed'] for a in aggregate)
    total_winners = sum(a['winners'] for a in aggregate)
    total_target = sum(a['target_hits'] for a in aggregate)
    total_stop = sum(a['stop_hits'] for a in aggregate)
    total_partial = sum(a['partials'] for a in aggregate)
    total_sum_pct = sum(a['sum_pct'] for a in aggregate)
    avg_pct_overall = (total_sum_pct / total_closed) if total_closed else 0.0
    profitable_symbols = sum(1 for a in aggregate if a['sum_pct'] > 0)
    # Universe-wide direction subsets
    all_closed = [c for a in aggregate for c in a.get("closed_legs", [])]
    g_buy  = summarize_subset([c for c in all_closed if c["trend"] == "BUY"])
    g_sell = summarize_subset([c for c in all_closed if c["trend"] == "SELL"])
    g_buy_r1  = summarize_subset([c for c in all_closed if c["trend"] == "BUY"  and c.get("entry_alert") == "retest1"])
    g_buy_r2  = summarize_subset([c for c in all_closed if c["trend"] == "BUY"  and c.get("entry_alert") == "retest2"])
    g_sell_r1 = summarize_subset([c for c in all_closed if c["trend"] == "SELL" and c.get("entry_alert") == "retest1"])
    g_sell_r2 = summarize_subset([c for c in all_closed if c["trend"] == "SELL" and c.get("entry_alert") == "retest2"])
    profitable_buy  = sum(1 for a in aggregate if a['buy']['sum_pct'] > 0)
    profitable_sell = sum(1 for a in aggregate if a['sell']['sum_pct'] > 0)

    def _bucket_row(name: str, s: Dict) -> str:
        return (
            f"| {name} | {s['legs']} | {s['win_rate']:.1f}% | "
            f"{s['target_hits']} | {s['stop_hits']} | {s['partials']} | "
            f"{s['avg_pct']:.2f}% | {s['sum_pct']:.1f}% |"
        )

    summary_lines = [
        "# EMA 200/400 1H Crossover — Backtest Summary",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Headline (combined BUY+SELL)",
        "",
        f"- Symbols processed: {len(aggregate)} (profitable combined: {profitable_symbols})",
        f"- Profitable BUY-only symbols: {profitable_buy} / {len(aggregate)}",
        f"- Profitable SELL-only symbols: {profitable_sell} / {len(aggregate)}",
        f"- Total closed legs: {total_closed}",
        f"- Win rate: {(total_winners / total_closed * 100) if total_closed else 0:.1f}%",
        f"- Target hits / Stop hits / Partials: {total_target} / {total_stop} / {total_partial}",
        f"- **Avg % per leg: {avg_pct_overall:.2f}%**",
        f"- **Sum % across all legs (uncompounded): {total_sum_pct:.1f}%**",
        f"- Data source mix: fyers={source_counts.get('fyers',0)} "
        f"yahoo={source_counts.get('yahoo',0)} none={source_counts.get('none',0)}",
        "",
        "## Direction × Alert breakdown (universe)",
        "",
        "| Bucket | Legs | Win% | Tgt | SL | Prt | Avg % | Sum % |",
        "|--------|------|------|-----|----|-----|-------|-------|",
        _bucket_row("BUY (all)", g_buy),
        _bucket_row("BUY @ 2nd Alert (retest1)", g_buy_r1),
        _bucket_row("BUY @ 3rd Alert (retest2)", g_buy_r2),
        _bucket_row("SELL (all)", g_sell),
        _bucket_row("SELL @ 2nd Alert (retest1)", g_sell_r1),
        _bucket_row("SELL @ 3rd Alert (retest2)", g_sell_r2),
        "",
        "## Per-symbol",
        "",
        "| Symbol | Bars | Sig | Legs | BUY legs | BUY avg% | BUY sum% | SELL legs | SELL avg% | SELL sum% | Tot avg% | Tot sum% |",
        "|--------|------|-----|------|---------|---------|---------|-----------|----------|----------|---------|---------|",
    ]
    for a in aggregate:
        b, s = a["buy"], a["sell"]
        summary_lines.append(
            f"| {a['symbol']} | {a['bars']} | {a['signals']} | {a['trades_closed']} | "
            f"{b['legs']} | {b['avg_pct']:.2f}% | {b['sum_pct']:.1f}% | "
            f"{s['legs']} | {s['avg_pct']:.2f}% | {s['sum_pct']:.1f}% | "
            f"{a['avg_pct']:.2f}% | {a['sum_pct']:.1f}% |"
        )
    summary_path = args.out / "_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary -> {summary_path}")

    # ---- Per-direction summary files ----
    def _write_direction_summary(direction: str, g_all: Dict, g_r1: Dict, g_r2: Dict,
                                  agg_key: str, profitable: int, path: Path) -> None:
        lines_d = [
            f"# EMA 200/400 — {direction} Strategy Summary",
            "",
            f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
            "",
            "## Headline",
            "",
            f"- Symbols processed: {len(aggregate)} (profitable {direction}-only: {profitable})",
            f"- {direction} closed legs: {g_all['legs']}",
            f"- Win rate: {g_all['win_rate']:.1f}%",
            f"- Target hits / Stop hits / Partials: "
            f"{g_all['target_hits']} / {g_all['stop_hits']} / {g_all['partials']}",
            f"- **Avg % per leg: {g_all['avg_pct']:.2f}%**",
            f"- **Sum % across all legs (uncompounded): {g_all['sum_pct']:.1f}%**",
            "",
            "## Alert breakdown",
            "",
            "| Bucket | Legs | Win% | Tgt | SL | Prt | Avg % | Sum % |",
            "|--------|------|------|-----|----|-----|-------|-------|",
            _bucket_row(f"{direction} (all)",            g_all),
            _bucket_row(f"{direction} @ 2nd Alert (retest1)", g_r1),
            _bucket_row(f"{direction} @ 3rd Alert (retest2)", g_r2),
            "",
            "## Per-symbol",
            "",
            "| Symbol | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |",
            "|--------|------|-----|------|-----|----|-----|-------|-------|",
        ]
        for a in aggregate:
            s = a[agg_key]
            lines_d.append(
                f"| {a['symbol']} | {s['legs']} | {s['winners']} | {s['win_rate']:.1f}% | "
                f"{s['target_hits']} | {s['stop_hits']} | {s['partials']} | "
                f"{s['avg_pct']:.2f}% | {s['sum_pct']:.1f}% |"
            )
        path.write_text("\n".join(lines_d) + "\n")
        print(f"{direction} Summary -> {path}")

    _write_direction_summary("BUY", g_buy, g_buy_r1, g_buy_r2, "buy",
                              profitable_buy, args.out / "_summary_buy.md")
    _write_direction_summary("SELL", g_sell, g_sell_r1, g_sell_r2, "sell",
                              profitable_sell, args.out / "_summary_sell.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

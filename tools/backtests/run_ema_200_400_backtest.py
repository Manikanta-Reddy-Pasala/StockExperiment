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
    params = {"interval": "1h", "range": f"{min(days, 730)}d"}
    try:
        r = requests.get(YAHOO_URL.format(symbol=symbol),
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
    target_points: float = 5000.0,
    rr_multiple: float = 3.0,
) -> Dict:
    """Walk bars chronologically with target/SL enforcement.

    Each ENTRY1/ENTRY2 opens 1 unit with target & SL captured at signal time.
        - Target: bar.high >= target (BUY) / bar.low <= target (SELL) -> closed at target.
        - SL:     1H close past EMA400 -> EXIT signal already emitted by strategy,
                  so SL exits arrive as EXIT events (close at signal price).
    Multiple entries close together when EXIT fires (cascade close).
    """
    if df.empty:
        return _empty_pnl()

    bars = df.set_index("timestamp", drop=False).sort_index()
    bar_ts_list: List[int] = list(bars.index)
    is_index = _is_index_symbol(symbol)

    sigs_by_ts: Dict[int, List[Dict]] = {}
    for s in signals:
        sigs_by_ts.setdefault(int(s["candle_ts"]), []).append(s)

    open_entries: List[Dict] = []
    closed: List[Dict] = []
    pnl_total = 0.0

    def _open_entry(sig: Dict) -> Dict:
        ema400 = float(sig["ema_400"])
        entry_price = float(sig["price"])
        if is_index:
            target = entry_price + target_points if sig["trend"] == "BUY" \
                else entry_price - target_points
        else:
            risk = abs(entry_price - ema400)
            target = entry_price + risk * rr_multiple if sig["trend"] == "BUY" \
                else entry_price - risk * rr_multiple
        return {
            "type": sig["signal_type"],
            "trend": sig["trend"],
            "price": entry_price,
            "time": sig["candle_time"],
            "entry_ts": int(sig["candle_ts"]),
            "target": target,
            "sl_ema400": ema400,
        }

    def _close(e: Dict, exit_price: float, exit_time, reason: str) -> float:
        pnl = (exit_price - e["price"]) if e["trend"] == "BUY" else (e["price"] - exit_price)
        closed.append({
            **e,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "exit_reason": reason,
            "pnl": pnl,
        })
        return pnl

    for ts in bar_ts_list:
        bar = bars.loc[ts]

        # 1. Process signals at this timestamp first (entries / EXITs)
        for sig in sigs_by_ts.get(int(ts), []):
            t = sig["signal_type"]
            if t in ("ENTRY1", "ENTRY2"):
                open_entries.append(_open_entry(sig))
            elif t == "EXIT" and open_entries:
                for e in open_entries:
                    pnl_total += _close(e, float(sig["price"]),
                                        sig["candle_time"], "EXIT_EMA400")
                open_entries = []

        # 2. Walk forward bars (skip entry bar itself) checking target hit
        if open_entries:
            still: List[Dict] = []
            for e in open_entries:
                if int(ts) <= e["entry_ts"]:
                    still.append(e)
                    continue
                hit = False
                if e["trend"] == "BUY" and float(bar["high"]) >= e["target"]:
                    pnl_total += _close(e, e["target"], bar["candle_time"], "TARGET")
                    hit = True
                elif e["trend"] == "SELL" and float(bar["low"]) <= e["target"]:
                    pnl_total += _close(e, e["target"], bar["candle_time"], "TARGET")
                    hit = True
                if not hit:
                    still.append(e)
            open_entries = still

    return {
        "trades_closed": len(closed),
        "trades_open": len(open_entries),
        "winners": sum(1 for c in closed if c["pnl"] > 0),
        "losers": sum(1 for c in closed if c["pnl"] <= 0),
        "total_pnl": pnl_total,
        "avg_pnl": (pnl_total / len(closed)) if closed else 0.0,
        "target_hits": sum(1 for c in closed if c["exit_reason"] == "TARGET"),
        "ema_exits": sum(1 for c in closed if c["exit_reason"] == "EXIT_EMA400"),
        "closed": closed,
        "open": open_entries,
    }


def _empty_pnl() -> Dict:
    return {
        "trades_closed": 0, "trades_open": 0, "winners": 0, "losers": 0,
        "total_pnl": 0.0, "avg_pnl": 0.0, "target_hits": 0, "ema_exits": 0,
        "closed": [], "open": [],
    }


# ---- Cycle grouping -------------------------------------------------
# A "cycle" begins at every CROSSOVER. Subsequent signals belong to that cycle
# until the next CROSSOVER (or end of data). Lets us render the user-requested
# stage breakdown:
#   Trend Identification → First Alert → Second Alert (Retest 1)
#   → First Entry → Third Alert (Retest 2) → Second Entry → Exit
STAGE_LABELS_BUY = {
    "CROSSOVER": "Trend Identification (BUY)",
    "ALERT1":    "First Alert — break + close above crossover candle high",
    "ALERT2":    "Second Alert (Retest 1) — EMA200 retest from above",
    "ENTRY1":    "First Entry (BUY) — break of retest1 high, sustain",
    "ALERT3":    "Third Alert (Retest 2) — price touches/crosses EMA400",
    "ENTRY2":    "Second Entry (BUY) — break of retest2 high, sustain",
    "EXIT":      "Exit — 1H close below EMA400",
}

STAGE_LABELS_SELL = {
    "CROSSOVER": "Trend Identification (SELL)",
    "ALERT1":    "First Alert — break + close below crossover candle low",
    "ALERT2":    "Second Alert (Retest 1) — EMA200 retest from below",
    "ENTRY1":    "First Entry (SELL) — break of retest1 low, sustain",
    "ALERT3":    "Third Alert (Retest 2) — price touches/crosses EMA400",
    "ENTRY2":    "Second Entry (SELL) — break of retest2 low, sustain",
    "EXIT":      "Exit — 1H close above EMA400",
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
        f"- **Strategy:** EMA 200/400 1H crossover",
        f"- **Target rule:** {'5000 pts (index)' if _is_index_symbol(symbol) else '1:3 RR (equity)'}",
        f"- **Stop-loss rule:** 1H close on wrong side of EMA400",
        "",
        "## Signal Counts",
        "",
        "| Signal | Count |",
        "|--------|-------|",
    ]
    for k in ("CROSSOVER", "ALERT1", "ALERT2", "ALERT3", "ENTRY1", "ENTRY2", "EXIT"):
        lines.append(f"| {k} | {type_counts.get(k, 0)} |")

    lines += [
        "",
        "## P&L",
        "",
        f"- **Trades closed:** {pnl['trades_closed']}",
        f"- **Trades open at end:** {pnl['trades_open']}",
        f"- **Winners / losers:** {pnl['winners']} / {pnl['losers']}",
        f"- **Target hits / EMA400 exits:** {pnl.get('target_hits', 0)} / {pnl.get('ema_exits', 0)}",
        f"- **Total realized P&L (per unit):** {pnl['total_pnl']:.2f}",
        f"- **Avg P&L per closed trade:** {pnl['avg_pnl']:.2f}",
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
            "## Closed Trades",
            "",
            "| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |",
            "|-------|-----------|-------|-----------|------|--------|-----|",
        ]
        for t in pnl["closed"]:
            lines.append(
                f"| {t['trend']} | {t['time']} | {t['price']:.2f} | "
                f"{t['exit_time']} | {t['exit_price']:.2f} | "
                f"{t.get('exit_reason', '')} | {t['pnl']:.2f} |"
            )

    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=720,
                        help="History window (Yahoo caps 1H at ~730d)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "exports" / "backtests")
    parser.add_argument("--universe", choices=["smoke", "nifty500", "indices"],
                        default="smoke",
                        help="smoke=5-stock sanity; nifty500=full Nifty 500; "
                             "indices=NIFTY/BANKNIFTY/sectoral (5000-pt target)")
    parser.add_argument("--limit", type=int, default=None,
                        help="When --universe=nifty500, cap to first N")
    parser.add_argument("--source", choices=["auto", "fyers", "yahoo"],
                        default="auto",
                        help="Data source (default auto: Fyers first, Yahoo "
                             "fallback). 'fyers' requires Postgres + valid token.")
    parser.add_argument("--user-id", type=int, default=1,
                        help="Fyers user_id (default 1) for token lookup")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single-stock mode. Yahoo (RELIANCE.NS / ^NSEI) or "
                             "Fyers (NSE:RELIANCE-EQ) format. Overrides --universe.")
    parser.add_argument("--from", dest="date_from", type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days; Yahoo only honors range)")
    parser.add_argument("--to", dest="date_to", type=str, default=None,
                        help="End date YYYY-MM-DD (defaults to today when --from given)")
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

    config = StrategyConfig()
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
            target_points=config.target_points,
            rr_multiple=config.rr_multiple,
        )

        path = render_md(symbol, name, df, signals, pnl, args.out)
        print(f"  bars={len(candles)}  signals={len(signals)}  "
              f"closed={pnl['trades_closed']}  tgt={pnl['target_hits']}  "
              f"ema={pnl['ema_exits']}  pnl={pnl['total_pnl']:.2f}  -> {path}")
        aggregate.append({
            "symbol": symbol,
            "name": name,
            "bars": len(candles),
            "signals": len(signals),
            "closed": pnl["trades_closed"],
            "winners": pnl["winners"],
            "target_hits": pnl["target_hits"],
            "ema_exits": pnl["ema_exits"],
            "pnl": pnl["total_pnl"],
        })

    # Top-level summary file
    total_closed = sum(a['closed'] for a in aggregate)
    total_winners = sum(a['winners'] for a in aggregate)
    total_target = sum(a['target_hits'] for a in aggregate)
    total_ema = sum(a['ema_exits'] for a in aggregate)
    total_pnl = sum(a['pnl'] for a in aggregate)

    summary_lines = [
        "# EMA 200/400 1H Crossover — Backtest Summary",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Headline",
        "",
        f"- Symbols processed: {len(aggregate)}",
        f"- Total closed trades: {total_closed}",
        f"- Winners (target hits + EMA-exit > 0): {total_winners}",
        f"- Win rate: {(total_winners / total_closed * 100) if total_closed else 0:.1f}%",
        f"- Target hits / EMA-exit closes: {total_target} / {total_ema}",
        f"- Sum P&L per unit: {total_pnl:.2f}",
        "",
        "## Per-symbol",
        "",
        "| Symbol | Bars | Signals | Closed | Winners | Tgt | EMA | P&L |",
        "|--------|------|---------|--------|---------|-----|-----|-----|",
    ]
    for a in aggregate:
        summary_lines.append(
            f"| {a['symbol']} | {a['bars']} | {a['signals']} | {a['closed']} | "
            f"{a['winners']} | {a['target_hits']} | {a['ema_exits']} | "
            f"{a['pnl']:.2f} |"
        )
    summary_path = args.out / "_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

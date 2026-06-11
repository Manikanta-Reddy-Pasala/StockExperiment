"""Live signal generator for price_meanrev_n500 (limit-order dip-buy, K=3).

Uses the SHARED core `strategy.py` (params / indicators / stop_price /
rank_candidates) — the same logic the backtest uses.

⚠ This model is OBSERVE / PAPER-ONLY by design (signals_only): its edge lives
in LIMIT fills at the dip level (close-fill drops 102.8% -> 36.1% CAGR, see
strategy.py), and the live executor places market orders — so real execution
is deliberately NOT wired. Instead this script keeps its OWN paper ledger with
the backtest's exact limit-fill semantics, validated forward:

Flow (runs daily after the nightly OHLCV pull, stateful):
  1. SETTLE: replay yesterday's emitted limit orders + standing exits against
     the latest completed bar (low-touch fills at min(open, level); stop before
     target on a both-hit day; MAXHOLD time exit) — exactly backtest.py's loop.
  2. EMIT: compute tomorrow's entry levels (SMA50 - 1*ATR from the last bar)
     for PIT-N500 names off cooldown, rank by 60d momentum, top free slots.

Emits an executor-shaped signals file (sells/buys + limit/stop/target extras)
for UI visibility; no order execution job is scheduled for this model.
"""
import sys, json, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union, eligible_at
from tools.models.price_meanrev_n500 import strategy as S

log = logging.getLogger("price_meanrev_n500")
MODEL_NAME = "price_meanrev_n500"
STATE_DIR = Path("/app/logs/price_meanrev_n500")
STATE_FILE = STATE_DIR / "state.json"
PAPER_CAPITAL = 100_000.0          # nominal paper book


def is_model_enabled() -> bool:
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == MODEL_NAME:
                return bool(s.get("enabled"))
    except Exception as e:
        log.warning(f"enabled read failed: {e}")
    return False


def load_panel(days_back=420):
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    eng = _get_engine()
    end = datetime.now().date(); start = end - timedelta(days=days_back)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start, "b": end})
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    piv = lambda v: df.pivot(index="date", columns="symbol", values=v).sort_index()
    cl = piv("close").ffill()
    return (piv("open").reindex_like(cl), piv("high").reindex_like(cl),
            piv("low").reindex_like(cl), cl)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            log.error(f"state file corrupt ({e}) — starting fresh")
    return {"cash": PAPER_CAPITAL, "positions": [], "cooldown": {},
            "pending": None, "trades": []}


def _held_days(dates, entry_date, di) -> int:
    """Trading days held = panel rows between entry date and row di."""
    ei = dates.searchsorted(pd.Timestamp(entry_date))
    return int(di - ei)


def settle(st, dates, O, H, L, C, di):
    """Replay standing exits + yesterday's pending limit buys on bar `di`.

    Mirrors backtest.py's day loop exactly: exits first (stop fills before
    target on a both-hit day), then pending entries (low-touch fills at
    min(open, level)), best-momentum-first order preserved from emit time.
    """
    d_iso = dates[di].date().isoformat()
    # ---- EXITS ----
    keep = []
    for p in st["positions"]:
        s = p["symbol"]
        o = O[s].iloc[di] if s in O.columns else np.nan
        h = H[s].iloc[di] if s in H.columns else np.nan
        l = L[s].iloc[di] if s in L.columns else np.nan
        cp = C[s].iloc[di] if s in C.columns else np.nan
        px = None; why = None
        if pd.isna(cp):
            keep.append(p); continue
        if not pd.isna(l) and l <= p["stop"]:
            px = min(o, p["stop"]) if not pd.isna(o) else p["stop"]; why = "STOP"
        elif not pd.isna(h) and h >= p["target"]:
            px = max(o, p["target"]) if not pd.isna(o) else p["target"]; why = "TARGET"
        elif _held_days(dates, p["entry_date"], di) >= S.MAXHOLD:
            px = float(cp); why = "TIME"
        if px is None:
            keep.append(p); continue
        proc = p["qty"] * float(px) * (1 - S.COST)
        st["cash"] += proc
        st["cooldown"][s] = d_iso
        st["trades"].append({
            "sym": s.replace("NSE:", "").replace("-EQ", ""),
            "entry_date": p["entry_date"], "exit_date": d_iso,
            "qty": p["qty"], "entry_px": round(p["entry"], 2),
            "exit_px": round(float(px), 2),
            "pnl": round(proc - p["qty"] * p["entry"], 0),
            "ret_pct": round((float(px) / p["entry"] - 1) * 100, 2),
            "exit_reason": why})
        log.info(f"PAPER EXIT {s} {why} @ {float(px):.2f} "
                 f"({(float(px)/p['entry']-1)*100:+.1f}%)")
    st["positions"] = keep
    # ---- PENDING LIMIT BUYS (valid for the FIRST new bar after emit) ----
    # Day-order semantics: orders emitted off bar X fill on the next completed
    # bar (> X), whatever calendar date that lands on (holiday-safe). If more
    # than one bar elapsed between runs (missed cron), the orders lapse with
    # the stale pending replaced below — never filled on a later bar.
    pend = st.get("pending")
    prev_iso = dates[di - 1].date().isoformat() if di > 0 else None
    if (pend and pend.get("emitted_from")
            and d_iso > pend["emitted_from"]
            and prev_iso == pend["emitted_from"]):   # first bar after emit only
        held = {p["symbol"] for p in st["positions"]}
        free = S.K - len(held)
        for b in pend.get("buys", []):       # emit order already best-momentum-first
            if free <= 0:
                break
            s = b["symbol"]
            if s in held:
                continue
            l = L[s].iloc[di] if s in L.columns else np.nan
            o = O[s].iloc[di] if s in O.columns else np.nan
            if pd.isna(l) or l > b["limit"]:
                continue                      # limit never touched — order lapses
            fill = min(float(o), b["limit"]) if not pd.isna(o) else b["limit"]
            q = int((st["cash"] / max(1, free)) / fill)
            if q < 1:
                continue
            st["cash"] -= q * fill * (1 + S.COST)
            st["positions"].append({
                "symbol": s, "qty": q, "entry": fill, "entry_date": d_iso,
                "stop": S.stop_price(fill, b["atr"]), "target": b["target"]})
            held.add(s); free -= 1
            log.info(f"PAPER FILL {s} {q} @ {fill:.2f} "
                     f"(limit {b['limit']:.2f}, stop {S.stop_price(fill, b['atr']):.2f}, "
                     f"target {b['target']:.2f})")
    st["pending"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", default=str(STATE_DIR / "signals" / "latest.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = datetime.now()
    sig_path = Path(args.signals_out); sig_path.parent.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    empty = json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})

    if not args.force and not is_model_enabled():
        log.warning(f"{MODEL_NAME} disabled — empty signals.")
        sig_path.write_text(empty); return 0
    if not args.force and today.weekday() >= 5:
        log.info("Weekend — skip."); sig_path.write_text(empty); return 0

    panels = load_panel()
    if panels is None:
        log.error("No data.")
        try:
            from tools.live.telegram_notify import alert_data_missing
            alert_data_missing(MODEL_NAME, "No N500 daily price data returned from the DB.")
        except Exception as _e:
            log.debug(f"tg alert failed: {_e}")
        sig_path.write_text(empty); return 1
    O, H, L, C = panels
    dates = C.index
    di = len(dates) - 1

    # Data-freshness gate (parity with the other models)
    last_day = pd.Timestamp(dates[di]).date()
    if not args.force and (today.date() - last_day).days > 7:
        log.error(f"Panel STALE — last equity day {last_day} > 7d old; refusing to emit.")
        try:
            from tools.live.telegram_notify import alert_data_missing
            alert_data_missing(MODEL_NAME, f"Daily price data STALE — last close {last_day}.")
        except Exception as _e:
            log.debug(f"tg alert failed: {_e}")
        sig_path.write_text(empty); return 1

    atr14, sma50, mom60, entry_lvl = S.indicators(C, H, L)
    st = load_state()

    # 1) SETTLE yesterday's paper orders/exits against the latest completed bar
    settle(st, dates, O, H, L, C, di)

    # 2) EMIT tomorrow's orders from the last bar's levels
    elig500 = eligible_at("n500", last_day)
    held = {p["symbol"] for p in st["positions"]}
    sells = [{"symbol": p["symbol"], "reason": "STANDING_EXIT",
              "target": round(p["target"], 2), "stop": round(p["stop"], 2),
              "time_exit_in_days": max(0, S.MAXHOLD - _held_days(dates, p["entry_date"], di))}
             for p in st["positions"]]
    buys = []
    free = S.K - len(held)
    if free > 0:
        cands = []
        for s in C.columns:
            plain_s = s.replace("NSE:", "").replace("-EQ", "")
            if s in held or plain_s not in elig500:
                continue
            cd = st["cooldown"].get(s)
            if cd is not None:
                cdi = dates.searchsorted(pd.Timestamp(cd))
                if (di - cdi) < S.COOLDOWN:
                    continue
            lv, av, mv = entry_lvl[s].iloc[di], atr14[s].iloc[di], mom60[s].iloc[di]
            if pd.isna(lv) or pd.isna(av) or av <= 0 or pd.isna(mv) or lv <= 0:
                continue
            cands.append((s, float(lv), float(av), float(mv)))
        cands.sort(key=lambda x: -x[3])           # best 60d momentum first
        for s, lv, av, mv in cands[:free * 5]:    # emit a few backups per slot
            buys.append({"symbol": s, "limit": round(lv, 2), "atr": round(av, 4),
                         "target": round(float(sma50[s].iloc[di]), 2),
                         "mom60_pct": round(mv * 100, 1)})

    st["pending"] = {"emitted_from": last_day.isoformat(), "buys": buys}
    STATE_FILE.write_text(json.dumps(st, indent=2))

    payload = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "paper_only": True,
               "sells": sells,
               "buys": [{"symbol": b["symbol"], "limit_price": b["limit"],
                         "target": b["target"], "mom60_pct": b["mom60_pct"]}
                        for b in buys[:S.K]],
               "paper": {"cash": round(st["cash"], 0),
                         "positions": st["positions"],
                         "closed_trades": len(st["trades"]),
                         "realized_pnl": round(sum(t["pnl"] for t in st["trades"]), 0)}}
    sig_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Signals: {len(sells)} standing exits, {len(buys)} limit candidates "
             f"(top-{S.K} emitted) -> {sig_path}")

    # Ranking JSON for Today's Picks UI (same shape as the other models)
    ranking = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "universe_size": len(buys),
               "top_n": [{"rank": i + 1,
                          "symbol": b["symbol"].split(":")[1].replace("-EQ", ""),
                          "name": b["symbol"].split(":")[1].replace("-EQ", ""),
                          "price": b["limit"],
                          "ret_30d_pct": b["mom60_pct"]}
                         for i, b in enumerate(buys[:5])]}
    rp = STATE_DIR / "ranking" / f"{today.strftime('%Y-%m-%d')}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(ranking, indent=2))

    # Audit + notify hooks (parity with the other models; best-effort)
    if not args.force:
        try:
            from src.services.audit_service import write_rankings, write_signal
            write_rankings(MODEL_NAME, today.date(), len(buys), 0,
                           [{"rank": i + 1,
                             "symbol": b["symbol"].split(":")[1].replace("-EQ", ""),
                             "name": b["symbol"].split(":")[1].replace("-EQ", ""),
                             "price": b["limit"]} for i, b in enumerate(buys[:5])])
            for b in buys[:S.K]:
                write_signal(MODEL_NAME, today.date(), "LIMIT_DIP_BUY",
                             b["symbol"], "BUY",
                             reason=f"limit {b['limit']} target {b['target']}")
            if not buys and not sells:
                write_signal(MODEL_NAME, today.date(), "HOLD", "", "NONE",
                             reason="no signal emitted")
        except Exception as _e:
            log.debug(f"audit hook failed: {_e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

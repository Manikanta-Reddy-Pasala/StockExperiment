"""Live signal generator for midcap_breakout_k3 (multi-holding, K=3).

Uses the SHARED core strategy.py (params + PIT universe + breakout scan/exit) —
the same logic the backtest uses — and the multi-holding ledger
(multi_holding_service).

Flow (runs DAILY — this is an event-driven breakout swing, not a monthly model):
  - EXITS: for each holding, recompute peak-since-entry from price history and
    ask the SHARED breakout_exit_reason if TARGET/STOP/TRAIL/MAX_HOLD fires.
  - ENTRIES: if free slots (< K held), scan today's PIT midcap band for fresh
    breakouts (40d high + >200DMA + vol>=2x), rank by volume ratio, emit BUYs to
    fill the free slots.

Emits a multi-holding signals file for tools/live/fyers_executor_multi.py:
  {"model","date","sells":[{symbol,reason}],"buys":[{symbol}]}
Single-holding models / executor untouched.
"""
import sys, json, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.models.midcap_breakout_k3 import strategy as S

log = logging.getLogger("midcap_breakout_k3")
MODEL_NAME = "midcap_breakout_k3"
STATE_DIR = Path("/app/logs/midcap_breakout_k3")


def is_model_enabled() -> bool:
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == MODEL_NAME:
                return bool(s.get("enabled"))
    except Exception as e:
        log.warning(f"enabled read failed: {e}")
    return False


def load_panel(symbols, days_back=500):
    eng = _get_engine()
    end = datetime.now().date(); start = end - timedelta(days=days_back)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": symbols, "a": start, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", default=str(STATE_DIR / "signals" / "latest.json"))
    ap.add_argument("--ranking-out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = datetime.now()
    sig_path = Path(args.signals_out); sig_path.parent.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    empty = json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})

    if not args.force and not is_model_enabled():
        log.warning(f"{MODEL_NAME} disabled — empty signals."); sig_path.write_text(empty); return 0
    if not args.force and today.weekday() >= 5:
        log.info("Weekend — skip."); sig_path.write_text(empty); return 0

    syms = S.n500_union_symbols()
    df = load_panel(syms)
    if df.empty:
        log.error("No data."); sig_path.write_text(empty); return 1
    fresh = S.fresh_symbols(df)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi = df.pivot(index="date", columns="symbol", values="high")
    vol = df.pivot(index="date", columns="symbol", values="volume")
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    # restrict to equity trading days (defensive; union panel is equities-only here)
    dates = cl.index
    di = len(dates) - 1
    sma_long, hh, vol_avg20, adv20 = S.indicators(cl, hi, vol, adv_rs)
    # PIT band as of last ~2 years (year-start pools); pick today's band
    start_for_pools = (today - timedelta(days=800)).date()
    year_starts, pools = S.build_year_pools(adv20, dates, fresh, start_for_pools, today.date())
    band = S.band_for(year_starts, pools, pd.Timestamp(dates[di]))

    from src.services.trading.multi_holding_service import get_holdings
    holdings = get_holdings(MODEL_NAME)
    held = {h["symbol"] for h in holdings}

    sells, buys = [], []
    # ---- EXITS ----
    for h in holdings:
        s = h["symbol"]
        if s not in cl.columns:
            continue
        px = float(cl[s].iloc[di]); entry_px = float(h.get("entry_px") or 0)
        peak = px
        try:
            ed = pd.Timestamp(h.get("entry_date"))
            seg = cl[s].loc[ed:].dropna()
            if len(seg):
                peak = float(seg.max())
            age = (dates[di].date() - ed.date()).days
        except Exception:
            age = 0
        if entry_px <= 0:
            continue
        reason = S.breakout_exit_reason(entry_px, px, peak, age,
                                        target_pct=S.TARGET_PCT, stop_pct=S.STOP_PCT,
                                        trail_pct=S.TRAIL_PCT, profit_trigger=S.PROFIT_TRIG,
                                        max_hold_days=S.MAX_HOLD)
        if reason:
            sells.append({"symbol": s, "reason": reason})

    # ---- ENTRIES (fill free slots) ----
    free = S.K - (len(held) - len(sells))
    cands = S.scan_breakouts(cl, sma_long, hh, vol, vol_avg20, band, di, held)
    log.info(f"breakout candidates today: {len(cands)}; free slots: {max(0, free)}")
    for cand in cands[: max(0, free)]:
        buys.append({"symbol": cand["sym"]})

    payload = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "sells": sells, "buys": buys}
    sig_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Signals: {len(sells)} sells, {len(buys)} buys -> {sig_path}")

    ranking = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "universe_size": len(band),
               "top_n": [{"rank": i + 1, "symbol": c["sym"].split(":")[1].replace("-EQ", ""),
                          "name": c["sym"].split(":")[1].replace("-EQ", ""),
                          "price": round(float(cl[c["sym"]].iloc[di]), 2)}
                         for i, c in enumerate(cands[:5])]}
    rp = Path(args.ranking_out) if args.ranking_out else (STATE_DIR / "ranking" / f"{today.strftime('%Y-%m-%d')}.json")
    rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(ranking, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

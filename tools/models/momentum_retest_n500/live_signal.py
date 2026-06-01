"""Live signal generator for momentum_retest_n500 (multi-holding, K=4).

Uses the SHARED core `strategy.py` (rank_targets / is_retest / params) — the same
logic the backtest uses — and the multi-holding ledger (multi_holding_service).

Flow (runs daily; the strategy is monthly-select + daily-retest-entry, stateful):
  - On the 1st trading day of the month: compute this month's TARGETS (top-K
    momentum leaders) and EXITS (held names out of the top-RETAIN rank). Persist
    the target "watch list" to disk.
  - Every day: for each watched target not yet held, if it is at a 20-EMA retest
    today, emit a BUY. Exits (from the monthly check) are emitted that day.

Emits a multi-holding signals file consumed by tools/live/fyers_executor_multi.py:
  {"model","date","sells":[{symbol,reason}],"buys":[{symbol}]}
Single-holding models / executor are untouched.
"""
import sys, json, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols
from tools.models.momentum_retest_n500 import strategy as S

log = logging.getLogger("momentum_retest_n500")
MODEL_NAME = "momentum_retest_n500"
STATE_DIR = Path("/app/logs/momentum_retest_n500")
WATCH_FILE = STATE_DIR / "watch.json"


def is_model_enabled() -> bool:
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == MODEL_NAME:
                return bool(s.get("enabled"))
    except Exception as e:
        log.warning(f"enabled read failed: {e}")
    return False


def is_first_trading_day_of_month(dates, today_ts) -> bool:
    """True if today is the first session on/after the 1st of its month."""
    m_first = pd.Timestamp(today_ts.year, today_ts.month, 1)
    fut = dates[dates >= m_first]
    return len(fut) > 0 and fut[0].normalize() == pd.Timestamp(today_ts).normalize()


def load_panel(symbols, days_back=420):
    eng = _get_engine()
    end = datetime.now().date(); start = end - timedelta(days=days_back)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": symbols, "a": start, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
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

    if not args.force and not is_model_enabled():
        log.warning(f"{MODEL_NAME} disabled — empty signals."); sig_path.write_text(json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})); return 0
    if not args.force and today.weekday() >= 5:
        log.info("Weekend — skip."); sig_path.write_text(json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})); return 0

    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]
    df = load_panel(n500)
    if df.empty:
        log.error("No data."); sig_path.write_text(json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})); return 1
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    adv20, sma200, ema20 = S.indicators(cl, adv_rs)
    dates = cl.index
    di = len(dates) - 1                                   # today = last row

    from src.services.trading.multi_holding_service import get_holdings
    holds = {h["symbol"] for h in get_holdings(MODEL_NAME)}

    sells, buys = [], []
    rk = S.rank_targets(cl, adv20, sma200, S.load_smallcap(), di)
    log.info(f"PIT ranked {len(rk)} leaders; top-{S.K}: {[s.split(':')[1] for s in rk[:S.K]]}")

    # Monthly check: refresh watch list + flag exits (held out of top-RETAIN)
    if args.force or is_first_trading_day_of_month(dates, today):
        retset = set(rk[:S.RETAIN])
        for s in holds:
            if s not in retset:
                sells.append({"symbol": s, "reason": "RANK_DROP"})
        watch = [s for s in rk[:S.K] if s not in holds]
        WATCH_FILE.write_text(json.dumps({"month": today.strftime("%Y-%m"), "watch": watch}))
        log.info(f"Monthly: {len(sells)} exits, watch={[w.split(':')[1] for w in watch]}")
    else:
        w = json.loads(WATCH_FILE.read_text()) if WATCH_FILE.exists() else {"watch": []}
        watch = [s for s in w.get("watch", []) if s not in holds]

    # Daily retest entry: buy watched targets at a 20-EMA pullback, up to K slots
    slots = S.K - len(holds)
    for s in watch:
        if slots <= 0:
            break
        if s in holds:
            continue
        px = cl[s].iloc[di] if s in cl.columns else None
        ev = ema20[s].iloc[di] if s in ema20.columns else None
        if S.is_retest(px, ev):
            buys.append({"symbol": s})
            slots -= 1

    payload = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "sells": sells, "buys": buys}
    sig_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Signals: {len(sells)} sells, {len(buys)} buys -> {sig_path}")

    # Ranking JSON for Today's Picks UI
    ranking = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "universe_size": len(rk),
               "top_n": [{"rank": i + 1, "symbol": s.split(":")[1].replace("-EQ", ""),
                          "name": s.split(":")[1].replace("-EQ", ""),
                          "price": round(float(cl[s].iloc[di]), 2),
                          "ret_30d_pct": (round((float(cl[s].iloc[di]) / float(cl[s].iloc[di - S.LOOKBACK]) - 1) * 100, 2)
                                          if di >= S.LOOKBACK and pd.notna(cl[s].iloc[di - S.LOOKBACK]) and float(cl[s].iloc[di - S.LOOKBACK]) > 0 else 0.0)}
                         for i, s in enumerate(rk[:5])]}
    rp = Path(args.ranking_out) if args.ranking_out else (STATE_DIR / "ranking" / f"{today.strftime('%Y-%m-%d')}.json")
    rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(ranking, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Live signal generator for emerging_momentum (multi-holding, K=3).

Uses the SHARED core `strategy.py` (build_pools / pool_for_date / rank_targets /
regime_healthy / hit_trail / hit_bear_trail / hit_bear_stop / params) — the same
logic the backtest uses — and the multi-holding ledger (multi_holding_service).

Flow (runs daily; monthly-rebalance + daily-risk, stateful via ledger):
  - DAILY (any session): ALWAYS-ON trailing stop (-25% off the peak since entry,
    recomputed from price history). In the BEAR regime (Nifty50 < 200DMA) also
    arm the hard stop (-10% vs entry) and bear trailing stop (-15% off peak).
    Emit SELL on a hit.
  - MONTHLY (1st trading day): rank the top-K momentum leaders within the PIT
    emerging pool; emit SELL for any holding that dropped out of the top-RETAIN;
    emit BUY for the top-K not held.
  - There is NO retest wait and NO take-profit — buys fill at the rebalance,
    winners ride.

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
from tools.shared.index_membership import universe_union
from tools.models.emerging_momentum import strategy as S

log = logging.getLogger("emerging_momentum")
MODEL_NAME = "emerging_momentum"
STATE_DIR = Path("/app/logs/emerging_momentum")


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
    empty = json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})

    if not args.force and not is_model_enabled():
        log.warning(f"{MODEL_NAME} disabled — empty signals."); sig_path.write_text(empty); return 0
    if not args.force and today.weekday() >= 5:
        log.info("Weekend — skip."); sig_path.write_text(empty); return 0

    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))] + [S.INDEX]
    df = load_panel(syms)
    if df.empty:
        log.error("No data."); sig_path.write_text(empty); return 1
    cl = df.pivot(index="date", columns="symbol", values="close")
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    # The NIFTY50-INDEX trades on dates equities sometimes don't (and vice versa);
    # those index-only rows are all-NaN for equities and would poison the rolling
    # ADV/return windows. Restrict the panel to EQUITY trading days only (the index
    # has data on those days too, so the 200-DMA regime gate is unaffected).
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    adv_rs = adv_rs.loc[equity_dates]
    adv20, sma200, idx_sma200 = S.indicators(cl, adv_rs)
    dates = cl.index
    di = len(dates) - 1                                   # today = last equity day
    healthy = S.regime_healthy(cl, idx_sma200, di)
    log.info(f"regime: {'HEALTHY (trail only)' if healthy else 'BEAR (trail + hard/bear stops)'}")

    # PIT pool in force today (anchored to POOL_ANCHOR_START, rebuilt yearly)
    anchors, pools = S.build_pools(adv20, dates, today.date())
    pool = S.pool_for_date(anchors, pools, dates[di])

    from src.services.trading.multi_holding_service import get_holdings
    holdings = get_holdings(MODEL_NAME)
    held = {h["symbol"] for h in holdings}

    sells, buys = [], []

    # ---- DAILY risk exits: always-on 25% trail, plus bear stops ----
    for h in holdings:
        s = h["symbol"]
        if s not in cl.columns:
            continue
        px = float(cl[s].iloc[di])
        entry_px = float(h.get("entry_px") or 0)
        # peak since entry, from price history (stateless)
        peak = px
        try:
            ed = pd.Timestamp(h.get("entry_date"))
            seg = cl[s].loc[ed:].dropna()
            if len(seg):
                peak = float(seg.max())
        except Exception:
            pass
        if S.hit_trail(px, peak):
            sells.append({"symbol": s, "reason": "TRAIL_STOP"})
        elif S.hit_bear_trail(px, peak, healthy):
            sells.append({"symbol": s, "reason": "BEAR_TRAIL"})
        elif S.hit_bear_stop(px, entry_px, healthy):
            sells.append({"symbol": s, "reason": "HARD_STOP"})

    # ---- MONTHLY rebalance: rank-drop exits + new buys ----
    rk = S.rank_targets(cl, sma200, pool, di)
    log.info(f"ranked {len(rk)} leaders; top-{S.K}: {[s.split(':')[1] for s in rk[:S.K]]}")
    if args.force or is_first_trading_day_of_month(dates, today):
        retset = set(rk[:S.RETAIN])
        sold_syms = {x["symbol"] for x in sells}
        for s in held:
            if s not in retset and s not in sold_syms:
                sells.append({"symbol": s, "reason": "RANK_DROP"})
        # buy top-K not held (and not being sold)
        sold_syms = {x["symbol"] for x in sells}
        slots = S.K - (len(held) - len([x for x in sells if x["symbol"] in held]))
        for s in rk[:S.K]:
            if slots <= 0:
                break
            if s in held or s in sold_syms:
                continue
            buys.append({"symbol": s}); slots -= 1
        log.info(f"Monthly rebalance: {len(sells)} sells, {len(buys)} buys")

    payload = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "sells": sells, "buys": buys}
    sig_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Signals: {len(sells)} sells, {len(buys)} buys -> {sig_path}")

    # Ranking JSON for Today's Picks UI
    ranking = {"model": MODEL_NAME, "date": today.strftime("%Y-%m-%d"),
               "universe_size": len(rk),
               "top_n": [{"rank": i + 1, "symbol": s.split(":")[1].replace("-EQ", ""),
                          "name": s.split(":")[1].replace("-EQ", ""),
                          "price": round(float(cl[s].iloc[di]), 2)}
                         for i, s in enumerate(rk[:5])]}
    rp = Path(args.ranking_out) if args.ranking_out else (STATE_DIR / "ranking" / f"{today.strftime('%Y-%m-%d')}.json")
    rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(ranking, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

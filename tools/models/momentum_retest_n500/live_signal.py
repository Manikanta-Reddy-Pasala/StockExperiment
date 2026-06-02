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
from tools.shared.index_membership import eligible_at
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
        log.error("No data.")
        try:
            from tools.live.telegram_notify import alert_data_missing
            alert_data_missing(MODEL_NAME, "No N500 daily price data returned from the DB.")
        except Exception as _e:
            log.debug(f"tg alert failed: {_e}")
        sig_path.write_text(json.dumps({"model": MODEL_NAME, "sells": [], "buys": []})); return 1
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    adv20, sma200, ema20 = S.indicators(cl, adv_rs)
    dates = cl.index
    di = len(dates) - 1                                   # today = last row

    # Data-freshness gate (parity with n100/pseudo/emerging): if the panel's
    # most recent equity day is stale (failed nightly OHLCV pull / dead feed),
    # abort rather than rank + emit buys off a days-old close. 7-day window
    # spans a long weekend + holiday cluster; fires only on broken data.
    if not args.force and len(dates):
        _last_day = pd.Timestamp(dates[di]).date()
        if (today.date() - _last_day).days > 7:
            log.error(f"Panel STALE — last equity day {_last_day} > 7d before "
                      f"{today.date()}; refusing to emit (fail-safe).")
            try:
                from tools.live.telegram_notify import alert_data_missing
                alert_data_missing(MODEL_NAME, f"Daily price data STALE — last close {_last_day} (>7d old).")
            except Exception as _e:
                log.debug(f"tg alert failed: {_e}")
            sig_path.write_text(json.dumps({"model": MODEL_NAME, "sells": [], "buys": []}))
            return 1

    from src.services.trading.multi_holding_service import get_holdings
    holds = {h["symbol"] for h in get_holdings(MODEL_NAME)}

    sells, buys = [], []
    rk = S.rank_targets(cl, adv20, sma200, S.load_smallcap(), di)
    # POINT-IN-TIME N500 filter (parity with backtest.py:77 — the live path was
    # MISSING this, the same survivorship bug that bought SPARC in ORB on
    # 2026-06-02). rank_targets ranks over the CURRENT nifty500_symbols() CSV;
    # restrict to names actually in the Nifty 500 ON the panel's last bar so a
    # stale-CSV / non-current member can never become a live target. Mirror the
    # backtest exactly: it filters rk through eligible_at("n500", d) per rebalance.
    last_day = pd.Timestamp(dates[di]).date()
    elig500 = eligible_at("n500", last_day)
    rk = [s for s in rk if s.replace("NSE:", "").replace("-EQ", "") in elig500]
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

    # Audit hook — persist ranking snapshot + every signal to the DB
    # (audit_model_signals), parity with the single-position models. retest was
    # file-only (signals/latest.json) before, so its signals never hit the DB.
    if not args.force:
        try:
            from src.services.audit_service import write_rankings, write_signal
            write_rankings(MODEL_NAME, today.date(), len(rk), 0, ranking.get("top_n") or [])
            if sells or buys:
                for sl in sells:
                    write_signal(MODEL_NAME, today.date(), "EXIT",
                                 sl.get("symbol", ""), "SELL",
                                 reason=sl.get("reason", "RANK_DROP"))
                for b in buys:
                    write_signal(MODEL_NAME, today.date(), "RETEST_ENTRY",
                                 b.get("symbol", ""), "BUY")
            else:
                write_signal(MODEL_NAME, today.date(), "HOLD", "", "NONE",
                             reason="no signal emitted")
        except Exception as _e:
            log.debug(f"audit hook failed: {_e}")

    # Telegram/PWA verdict ping (parity with n100/emerging — retest emitted
    # signals but NEVER pinged the decision, so its daily plan was invisible).
    # Weekend-silent + deduped per (model, verdict, day); gated by
    # MOMROT_TG_NOTIFY=1. notify_model_decision wants {signal,symbol,side}.
    if not args.force:
        try:
            from src.services.notification_service import notify_model_decision
            _dec = (
                [{"signal": "EXIT", "symbol": s.get("symbol", ""), "side": "SELL"}
                 for s in sells]
                + [{"signal": "RETEST_ENTRY", "symbol": b.get("symbol", ""), "side": "BUY"}
                   for b in buys]
            )
            # Multi-holding (K=4): pass the full held set so a no-change ping
            # lists ALL names ("holding A, B, C, D"), not "flat".
            notify_model_decision(MODEL_NAME, _dec,
                                  held_symbol=sorted(holds), trigger="CRON")
        except Exception as _ne:
            log.debug(f"notify decision failed: {_ne}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

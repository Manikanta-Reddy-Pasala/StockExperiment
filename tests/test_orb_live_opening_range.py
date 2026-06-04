"""ORB live breakout must be detectable from the first post-range bar (~09:35),
not blocked until 8 bars exist (~09:50).

Bug (2026-06-03): strategy.opening_range() required len >= OR_BARS + 5 (= 8 bars)
before returning a range. The live scanner (live_signal.emit_signals) reuses this
helper, so the 09:30/09:35/09:40/09:45 cron scans could NEVER fire — the earliest
a range was returned was ~09:50 (the 8th 5-min bar). The backtest (orb_trade on a
full ~75-bar day) enters on the first breakout right after the opening range, so
live structurally entered late / missed early breakouts the backtest takes
(proven: the 06-02 SPARC live entry filled 09:50-09:55, exactly the 8-bar gate).

Fix: opening_range gains `min_post_bars` (default 5 -> backtest byte-identical);
the live path passes min_post_bars=1 so a range is returned as soon as there is
one bar beyond the opening range (OR_BARS + 1 = 4 bars, ~09:35).
"""
import pandas as pd

from tools.models.orb_momentum_intraday import strategy as S


def _bars(n: int) -> pd.DataFrame:
    """n synthetic 5-min bars; opening range (first OR_BARS) high=110, low=90."""
    h = [110.0] * S.OR_BARS + [120.0] * (n - S.OR_BARS)
    l = [90.0] * S.OR_BARS + [100.0] * (n - S.OR_BARS)
    c = [105.0] * S.OR_BARS + [119.0] * (n - S.OR_BARS)
    return pd.DataFrame({"h": h[:n], "l": l[:n], "c": c[:n]})


def test_default_gate_unchanged_backtest_safe():
    # Backtest default still demands OR_BARS + 5 bars -> a 4..7 bar day yields None.
    for n in range(S.OR_BARS + 1, S.OR_BARS + 5):
        assert S.opening_range(_bars(n)) is None, f"{n} bars should be None by default"
    rng = S.opening_range(_bars(S.OR_BARS + 5))
    assert rng is not None and rng[0] == 110.0 and rng[1] == 90.0


def test_live_gate_returns_range_from_first_post_range_bar():
    # Live passes min_post_bars=1 -> a range is available at OR_BARS + 1 (~09:35).
    df = _bars(S.OR_BARS + 1)
    rng = S.opening_range(df, min_post_bars=1)
    assert rng is not None
    assert rng == (110.0, 90.0)


def test_live_gate_still_none_with_only_range_bars():
    # With exactly OR_BARS bars there is no post-range bar to test a breakout.
    assert S.opening_range(_bars(S.OR_BARS), min_post_bars=1) is None


def test_live_breakout_high_based_matches_backtest():
    # Post-range bar spikes a HIGH above ORH (110) but CLOSES back under it.
    # Old live logic (last close >= orh) would miss this; live_breakout (high
    # >= orh, like the backtest) must detect it.
    h = [110.0, 110.0, 110.0, 115.0]   # 4th bar high 115 > ORH 110
    l = [90.0, 90.0, 90.0, 100.0]
    c = [105.0, 105.0, 105.0, 104.0]   # 4th bar CLOSES at 104 < ORH
    df = pd.DataFrame({"h": h, "l": l, "c": c})
    assert S.live_breakout(df, 110.0) is True
    assert float(df["c"].iloc[-1]) < 110.0   # the old close-based test would fail


def test_live_breakout_false_when_range_not_pierced():
    h = [110.0, 110.0, 110.0, 109.9]   # never reaches ORH 110
    l = [90.0] * 4
    c = [105.0] * 4
    df = pd.DataFrame({"h": h, "l": l, "c": c})
    assert S.live_breakout(df, 110.0) is False


def test_live_breakout_ignores_opening_range_bars():
    # A high inside the opening range itself must NOT count as a breakout.
    h = [110.0, 110.0, 110.0, 100.0]   # only range bars touch 110; post-range low
    l = [90.0] * 4
    c = [95.0] * 4
    df = pd.DataFrame({"h": h, "l": l, "c": c})
    assert S.live_breakout(df, 110.0) is False


def _day_bars(post_h, post_l, post_c):
    """A full 75-bar trading day (09:15..15:25 IST). Opening range = first OR_BARS
    bars (high 110 / low 90); every post-range bar shares (post_h, post_l, post_c)."""
    n = 75
    dt = pd.date_range("2026-06-03 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    h = [110.0] * S.OR_BARS + [post_h] * (n - S.OR_BARS)
    l = [90.0] * S.OR_BARS + [post_l] * (n - S.OR_BARS)
    c = [105.0] * S.OR_BARS + [post_c] * (n - S.OR_BARS)
    o = c
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "dt": dt})


def test_orb_trade_eod_squares_off_at_eod_flat_min():
    # PARITY: backtest and live both flatten at EOD_FLAT_MIN (15:25). orb_trade's
    # EOD exit must be the 15:25 bar's close, driven by the shared EOD_FLAT_MIN
    # constant (pre-2026-06-04 it was defined but unused — orb_trade held to the
    # last bar regardless; harmless at 15:25 but it broke the moment live used a
    # different time, and locks the live==backtest contract going forward).
    df = _day_bars(post_h=115.0, post_l=100.0, post_c=112.0)  # breaks ORH 110, no stop(90)/target(150)
    t = S.orb_trade(df, "TEST")
    assert t is not None and t.reason == "eod"
    assert t.entry_time == "09:30"          # breakout on the first post-range bar
    assert t.exit_time == "15:15", f"EOD exit must be 15:15 (EOD_FLAT_MIN), got {t.exit_time}"
    assert t.exit_px == 112.0               # close of the 15:15 bar


def test_orb_trade_stop_target_still_take_priority_over_eod():
    # A stop hit before EOD_FLAT must still exit at the stop (EOD gate is last).
    df = _day_bars(post_h=115.0, post_l=80.0, post_c=112.0)   # low 80 <= stop 90
    t = S.orb_trade(df, "TEST")
    assert t is not None and t.reason == "stop"


# ---- live<->backtest EXIT parity (the live cron must exit on the SAME rule) ----

_EXIT_MAP = {"stop": "STOP", "target": "TARGET", "eod": "EOD_FLAT"}


def test_live_exit_reason_matches_orb_trade():
    # For the SAME day bars, the live exit decision (live_exit_reason) must match
    # the backtest's realized exit (orb_trade.reason). Pre-2026-06-04 live had NO
    # intraday stop/target -> it rode every position to EOD, missing the 41% of
    # backtest trades that exit on stop/target. Lock the parity.
    cases = {
        "stop":   _day_bars(post_h=115.0, post_l=80.0,  post_c=112.0),  # low 80 <= ORL 90
        "target": _day_bars(post_h=150.0, post_l=100.0, post_c=140.0),  # high 150 >= target 150
        "eod":    _day_bars(post_h=115.0, post_l=100.0, post_c=112.0),  # neither -> ride to EOD
    }
    for bt_reason, df in cases.items():
        t = S.orb_trade(df, "TEST")
        assert t is not None and t.reason == bt_reason, f"orb_trade should be {bt_reason}"
        # now_mins past EOD so the eod case resolves (stop/target are clock-free)
        live = S.live_exit_reason(df, now_mins=S.EOD_FLAT_MIN)
        assert live == _EXIT_MAP[bt_reason], f"{bt_reason}: live={live}, want {_EXIT_MAP[bt_reason]}"


def test_live_exit_reason_holds_when_no_trigger_pre_eod():
    # No stop/target and before EOD_FLAT -> hold (None), do not square off early.
    df = _day_bars(post_h=115.0, post_l=100.0, post_c=112.0)
    assert S.live_exit_reason(df, now_mins=S.EOD_FLAT_MIN - 5) is None
    # …and the same bars at/after EOD_FLAT -> EOD square-off.
    assert S.live_exit_reason(df, now_mins=S.EOD_FLAT_MIN) == "EOD_FLAT"

"""
EMA 200/400 Crossover Strategy — v2 (BTC trade rules).

Trend rule (1H or 30m bars):
    BUY trend  : EMA200 crosses above EMA400 (crossover candle locked).
    SELL trend : EMA200 crosses below EMA400.

Buy setup state machine (long side):
    Stage 0 : Wait for crossover.
    Stage 1 : Crossover candle locked. Watch for break + close above its high
              -> ALERT1, advance to Stage 2.
    Stage 2 : Wait for retest of EMA200 from above (close < EMA200) -> ALERT2,
              lock retest1 candle, reset retest1_attempts=0,
              retest1_invalidated=False, advance to Stage 3.
    Stage 3 : Retest1 armed. On every new bar:
                - If low <= EMA400 BEFORE next ENTRY1 break -> retest1 path
                  invalidated (omit 1st entry). Advance to Stage 4 (retest2 watch).
                - Edge-detect: prev close <= retest1.high AND curr close >
                  retest1.high -> ENTRY1 fires. retest1_attempts++. Allow up to
                  3 ENTRY1 fires (initial + 2 re-entries) at the same retest1
                  level (price must dip below and re-break).
                - On 3rd attempt OR EMA400 touch -> advance to Stage 4.
              SL for each ENTRY1 position = EMA400 close (close-based exit).
    Stage 4 : Retest2 watch. On low <= EMA400 -> ALERT3, lock retest2 candle,
              reset retest2_attempts=0, advance to Stage 5.
    Stage 5 : Retest2 armed. Edge-detect break of retest2.high -> ENTRY2 fires.
              retest2_attempts++. Allow up to 3 ENTRY2 fires. After 3 OR a new
              ALERT3 lock -> revert to Stage 4 (allow next ALERT3 lock).
              SL for each ENTRY2 position = retest2.low.

Position management (per bar after stage step):
    For each open position:
      - TARGET: bar.high >= entry * (1 + target_pct) -> close all qty at target.
      - PARTIAL: bar.high >= entry * (1 + partial_pct) AND not partial_booked
                 -> book partial_qty_frac of qty at partial price, set
                 trail_sl=entry, mark partial_booked.
      - STOP_HIT: bar.low <= current_sl -> close remaining at SL. SL is
                  per-position (EMA400 for ENTRY1, retest2.low for ENTRY2,
                  entry-price after partial booking).
    The bar-close EMA400 exit (close < EMA400) still emits an EXIT signal that
    closes ALL open positions (overrides per-position SL where applicable).

Sell setup is the mirror image (close > EMA200 retest from below, EMA400 touch
from below for retest2, retest2.high SL for ENTRY2, etc.).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

try:
    from ...models.database import get_database_manager
    from ...models.historical_models import (
        HistoricalData1H,
        HistoricalData15M,
        EMACrossoverState,
        EMACrossoverSignal,
    )
except ImportError:
    from src.models.database import get_database_manager
    from src.models.historical_models import (
        HistoricalData1H,
        HistoricalData15M,
        EMACrossoverState,
        EMACrossoverSignal,
    )

logger = logging.getLogger(__name__)

# Default risk parameters — match Strategy-1 spec.
# Profit target is 10% by default; spec calls it customizable to 10%/15%/20%.
DEFAULT_TARGET_PCT = 0.10
# Partial book — Strategy-1 spec is asymmetric:
#   BUY  ENTRY1=5%,  ENTRY2=15%
#   SELL ENTRY1=5%,  ENTRY2=5%
DEFAULT_PARTIAL_PCT_ENTRY1 = 0.05
DEFAULT_PARTIAL_PCT_ENTRY2_BUY = 0.15
DEFAULT_PARTIAL_PCT_ENTRY2_SELL = 0.05
DEFAULT_PARTIAL_QTY_FRAC = 0.5
# "Allow 3 re-entries only at 2nd Alert" — 1 initial + 3 re-entries = 4 total.
DEFAULT_RE_ENTRY_CAP = 4
DEFAULT_SUSTAIN_MINUTES = 15  # Effective on <=30m intra-bar resolver
# Spec: profit target customizable to 10/15/20%. Code accepts any float; helper
# documents the canonical set.
ALLOWED_TARGET_PCTS = (0.10, 0.15, 0.20)


@dataclass
class StrategyConfig:
    """
    EMA 200/400 1H crossover strategy configuration.

    Every field can be overridden per-user via auto_trading_settings.ema_strategy_config
    (JSONB). Loader: ``EMACrossoverRunner._effective_config(user_id)``. Defaults below
    encode the spec-strict + audit-fix profile that backtests recommend.

    Notes per group:

    * **Profit / partial / SL** — drive position economics.
    * **Spec-strict guards** — eliminate phantom alerts and missed crossovers.
      Toggle off only if you intentionally want pre-audit behavior.
    * **EMA accuracy** — SMA-seeded EMA (Pine convention). Off = raw ewm
      (drifts vs Fyers chart at <365d backfill).
    * **Quality filters** — additive over the spec; set to 0/None to disable.
    * **Opt-in tuning toggles** — HTF, slope, ALERT3 cap, retest2 SL cap, etc.
      All default disabled.
    """
    target_pct: float = DEFAULT_TARGET_PCT
    # Per-entry / per-side partial-book triggers.
    # Strategy-1 spec defaults: BUY entry1=5%/entry2=15%, SELL entry1=5%/entry2=5%.
    partial_pct_entry1: float = DEFAULT_PARTIAL_PCT_ENTRY1
    partial_pct_entry2_buy: float = DEFAULT_PARTIAL_PCT_ENTRY2_BUY
    partial_pct_entry2_sell: float = DEFAULT_PARTIAL_PCT_ENTRY2_SELL
    partial_qty_frac: float = DEFAULT_PARTIAL_QTY_FRAC
    re_entry_cap: int = DEFAULT_RE_ENTRY_CAP        # max attempts per alert
    sustain_minutes: int = DEFAULT_SUSTAIN_MINUTES
    # Optional: override sustain wait for SELL side only. None = same as
    # sustain_minutes. Backtest evidence: BUY benefits from fast (15m), SELL
    # benefits from slow (~75m) sustain confirmation on 1H bars.
    sell_sustain_minutes: Optional[int] = None
    ema_fast_period: int = 200
    ema_slow_period: int = 400

    # ---- Spec-strict guards (default ON; turn OFF for legacy behavior) ----
    # Strategy-1 retest-candle definition: "price moves towards EMA from upside
    # (above EMA)". Lock retest only on a true transition — prev bar must have
    # been above the EMA. Eliminates phantom alerts when price was already below.
    require_retest_from_upside: bool = True
    # Sanity guard: if state.trend disagrees with current EMA200/EMA400 ordering
    # for more than `trend_inversion_grace_bars` consecutive bars (without a
    # true crossover edge), force end-cycle. Catches cases where the cross was
    # missed (gaps, very small EMAs near each other).
    sanity_flip_trend: bool = True
    trend_inversion_grace_bars: int = 1
    # SMA-seeded EMA (Fyers / TradingView Pine convention). First `span` bars
    # use SMA(close, span); after that the EMA recurses from that seed. Pure
    # ewm(adjust=False) drifts vs Fyers chart for hundreds of bars.
    sma_seed_ema: bool = True

    # ---- Sustain check ----
    # During the sustain wait after a retest break, price must hold near the
    # break level — no wick below (BUY) or above (SELL). Tolerance allows
    # small intra-bar noise. 0 = strict (any wick beyond level cancels).
    # 0.005 = 0.5% (default). E.g. BUY break level=100 -> low must stay > 99.5.
    sustain_wick_tolerance_pct: float = 0.005

    # ---- Quality filters ----
    # Minimum EMA200/EMA400 separation at crossover as fraction of price.
    # Touching EMAs (gap below threshold) whipsaw; filter them.
    # Backtest evidence (Nifty50 1y Fyers, threshold sweep):
    #   gap=0      446 legs / 38.3% win / +331% sum / 308 SL
    #   gap=0.0001 209 legs / 37.8% win / +122% sum / 140 SL
    #   gap=0.0002  69 legs / 34.8% win /  +40% sum /  46 SL
    #   gap=0.0003  23 legs / 60.9% win /  +76% sum /   9 SL  <- elbow
    #   gap=0.0004  12 legs / 100%  win /  +85% sum /   0 SL  (small sample)
    # 0.0003 wins on risk-adjusted P&L: meaningful sample, +22pp win-rate
    # uplift, 4.5x avg-per-leg vs unfiltered. Set 0 for spec-strict.
    min_crossover_gap_pct: float = 0.0003
    # Volume confirmation on ENTRY break bar. Skip ENTRY if break-bar volume
    # is below avg(volume, N) × mult. mult=0 disables (default — was too
    # aggressive in 6m backtest, dropped legs from 9 to 3 without improving
    # quality). Toggle on with mult=0.8 for longer windows / liquid stocks.
    volume_confirm_bars: int = 20
    volume_confirm_mult: float = 0.0

    # ---- Opt-in tuning toggles (all default = spec-compliant / disabled) ----
    # 1) Higher-timeframe trend filter: only allow CROSSOVER in matching regime.
    #    BUY  fires only if close > htf_sma_buy at crossover bar.
    #    SELL fires only if close < htf_sma_sell at crossover bar.
    #    Asymmetric periods: BUY uses long-term SMA (200d), SELL uses
    #    medium-term SMA (50d) so SELL can catch stock-specific downtrends
    #    even when broad market is in long-term uptrend.
    htf_filter_enabled: bool = False
    htf_buy_period_bars: int = 1400   # ~200-day SMA on 1H bars (BUY confirm)
    htf_sell_period_bars: int = 1400  # default same as BUY; configurable
    # Optional margin (fraction). Require close < htf_sma_sell * (1 - margin)
    # for SELL. Filters out shallow dips. E.g. 0.02 = require close 2% below
    # SELL SMA. 0 = simple "close below SMA" check.
    htf_sell_margin_pct: float = 0.0
    htf_buy_margin_pct: float = 0.0
    # Legacy single-period field (kept for backwards compat). When non-zero,
    # overrides both htf_buy_period_bars and htf_sell_period_bars.
    htf_period_bars: int = 0

    # 2) Cap number of ALERT3 (retest2 candle) re-locks per cycle.
    #    0 = unlimited (spec). E.g. 2 stops chop near EMA400 from generating
    #    infinite re-entries.
    max_alert3_locks_per_cycle: int = 0

    # 3) Tighten ENTRY2 SL — cap distance from entry. 0 = disabled (use spec
    #    retest2.low/high). E.g. 0.03 caps SL at 3% from entry.
    retest2_sl_cap_pct: float = 0.0

    # 4) Skip retest2 (ENTRY2) phase entirely. retest1 invalidate or cap-reached
    #    ends cycle (await next crossover) instead of advancing to retest2 watch.
    skip_retest2: bool = False

    # 5) Skip SELL trend entirely (BUY-only mode). Bull regimes make SELL
    #    counter-trend; this avoids those trades.
    skip_sell: bool = False
    skip_buy: bool = False  # symmetric option

    # 6) EMA200 slope confirmation for SELL: require EMA200 to be declining
    #    over the last N bars at crossover bar (genuine downtrend momentum).
    #    Threshold = (ema200[t] - ema200[t-N]) / ema200[t-N] <= -slope_min.
    #    Defaults below are tuning suggestions only — slope filter is OFF
    #    until sell_slope_bars > 0. Empirical Nifty50 720d: 350 bars (~50d)
    #    + 0.005 (0.5% drop) flips SELL from −134% to +145% sum.
    sell_slope_bars: int = 0           # set to 350 (50d on 1H) to enable
    sell_slope_min_pct: float = 0.005  # 0.5% drop required (used when bars>0)
    buy_slope_bars: int = 0            # BUY HTF filter alone is sufficient
    buy_slope_min_pct: float = 0.005


class EMACrossoverStrategy:
    """v2 BTC-rules state machine + per-position TP/SL/partial."""

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.db = get_database_manager()

    def _sustain_minutes_for(self, trend: str) -> int:
        """Per-side sustain: BUY uses sustain_minutes; SELL uses
        sell_sustain_minutes if set, else sustain_minutes."""
        if trend == "SELL" and self.config.sell_sustain_minutes is not None:
            return int(self.config.sell_sustain_minutes)
        return int(self.config.sustain_minutes)

    def _volume_ok(self, row) -> bool:
        """Volume confirmation gate. True if break-bar volume >=
        avg(volume, N) × mult, or if filter disabled / volume_sma missing."""
        if not self.config.volume_confirm_bars or self.config.volume_confirm_mult <= 0:
            return True
        try:
            vol = float(row["volume"])
            avg = float(row.get("volume_sma", 0)) if hasattr(row, "get") else 0.0
            if avg <= 0:
                avg = float(row["volume_sma"]) if "volume_sma" in row.index else 0.0
        except Exception:
            return True
        if avg <= 0:
            return True  # warmup or missing — skip filter
        return vol >= avg * self.config.volume_confirm_mult

    # ------------------------------------------------------------------
    @staticmethod
    def _sma_seeded_ema(close: pd.Series, span: int) -> pd.Series:
        """SMA-seeded EMA — Fyers/TradingView Pine convention.

        First `span` bars are NaN. Bar `span-1` is seeded as SMA(close, span).
        Subsequent bars recurse with alpha = 2/(span+1).
        """
        if len(close) < span:
            return pd.Series([float("nan")] * len(close), index=close.index)
        alpha = 2.0 / (span + 1.0)
        out = [float("nan")] * len(close)
        # Seed at index span-1 with SMA over the first `span` closes.
        seed = float(close.iloc[:span].mean())
        out[span - 1] = seed
        prev = seed
        for i in range(span, len(close)):
            prev = (float(close.iloc[i]) - prev) * alpha + prev
            out[i] = prev
        return pd.Series(out, index=close.index)

    @staticmethod
    def compute_emas(df: pd.DataFrame, fast: int = 200, slow: int = 400,
                      htf_buy_period_bars: Optional[int] = None,
                      htf_sell_period_bars: Optional[int] = None,
                      sma_seed: bool = True,
                      volume_sma_bars: int = 0) -> pd.DataFrame:
        df = df.copy()
        if sma_seed:
            df["ema_200"] = EMACrossoverStrategy._sma_seeded_ema(df["close"], fast)
            df["ema_400"] = EMACrossoverStrategy._sma_seeded_ema(df["close"], slow)
        else:
            df["ema_200"] = df["close"].ewm(span=fast, adjust=False).mean()
            df["ema_400"] = df["close"].ewm(span=slow, adjust=False).mean()
        if htf_buy_period_bars and htf_buy_period_bars > 0:
            df["htf_sma_buy"] = df["close"].rolling(htf_buy_period_bars, min_periods=1).mean()
        if htf_sell_period_bars and htf_sell_period_bars > 0:
            df["htf_sma_sell"] = df["close"].rolling(htf_sell_period_bars, min_periods=1).mean()
        if volume_sma_bars and volume_sma_bars > 0 and "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(volume_sma_bars, min_periods=1).mean()
        return df

    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_id: int,
        symbol: str,
        candles: List[HistoricalData1H],
        latest_15m_bar: Optional[HistoricalData15M] = None,
        eval_from_ts: Optional[int] = None,
    ) -> List[dict]:
        if len(candles) < self.config.ema_slow_period + 5:
            logger.debug(
                f"{symbol}: only {len(candles)} candles, need >={self.config.ema_slow_period + 5}"
            )
            return []

        df = self._candles_to_df(candles)
        # Resolve HTF periods: legacy htf_period_bars overrides both if set.
        if self.config.htf_filter_enabled:
            legacy = self.config.htf_period_bars
            buy_p  = legacy if legacy > 0 else self.config.htf_buy_period_bars
            sell_p = legacy if legacy > 0 else self.config.htf_sell_period_bars
        else:
            buy_p = sell_p = None
        df = self.compute_emas(
            df,
            self.config.ema_fast_period,
            self.config.ema_slow_period,
            htf_buy_period_bars=buy_p,
            htf_sell_period_bars=sell_p,
            sma_seed=self.config.sma_seed_ema,
            volume_sma_bars=self.config.volume_confirm_bars or 0,
        )

        state = self._load_state(user_id, symbol)
        signals: List[dict] = []

        start_idx = 0
        if state.last_evaluated_ts:
            mask = df["timestamp"] > state.last_evaluated_ts
            if mask.any():
                start_idx = int(df[mask].index.min())
            else:
                start_idx = len(df)

        # EMA convergence buffer: pure SMA-seed EMA needs ~5 × span bars after
        # seed to drop seed-weight below 1%. For EMA400 that's 2000 bars; if
        # caller hasn't fetched that much history, signals fired during
        # warmup will drift vs Fyers chart values. eval_from_ts (UNIX seconds)
        # lets the caller suppress signal emission until EMAs have converged
        # — strategy still walks every bar to keep state consistent.
        for i in range(max(start_idx, self.config.ema_slow_period), len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            new_signals = self._step_machine(state, row, prev, df, i)
            if eval_from_ts is None or int(row["timestamp"]) >= eval_from_ts:
                for sig in new_signals:
                    signals.append(sig)
            state.last_evaluated_ts = int(row["timestamp"])

        # Intra-bar 15m sustain pass: if a retest break is pending and the
        # latest 15m close still holds beyond the level for >= sustain_minutes,
        # fire ENTRY now instead of waiting for the next 1H close.
        if latest_15m_bar is not None and len(df) > 0:
            last_1h_row = df.iloc[-1]
            signals.extend(self._intra_bar_sustain_15m(state, last_1h_row, latest_15m_bar))

        self._save_state(state)
        for sig in signals:
            self._record_signal(user_id, symbol, sig)
        return signals

    # ------------------------------------------------------------------
    # 15m intra-bar sustain check (entry confirmation only)
    # ------------------------------------------------------------------
    def _intra_bar_sustain_15m(self, state, last_1h_row, bar15m) -> List[dict]:
        """Resolve any pending retest break against the latest 15m bar.

        Only the post-cross sustain confirmation runs on 15m data; trend
        detection, EMA200/400 and retest locking still come from 1H.
        """
        if state.trend not in ("BUY", "SELL"):
            return []
        try:
            bar_ts = int(bar15m.timestamp)
        except Exception:
            return []
        # Build a synthetic row carrying the 15m price/timestamp but the
        # latest known 1H EMAs so signal records remain consistent.
        synth = pd.Series({
            "timestamp": bar_ts,
            "candle_time": bar15m.candle_time,
            "open": float(bar15m.open),
            "high": float(bar15m.high),
            "low": float(bar15m.low),
            "close": float(bar15m.close),
            "ema_200": float(last_1h_row["ema_200"]),
            "ema_400": float(last_1h_row["ema_400"]),
        })

        signals: List[dict] = []
        if state.trend == "BUY":
            if state.stage == 3 and state.retest1_pending_cross_ts and state.retest1_high is not None:
                def _r1_buy_cap():
                    if self.config.skip_retest2:
                        self._reset_state(state, "NONE", synth)
                    else:
                        state.stage = 4
                signals.extend(self._resolve_pending_15m_buy(
                    state, synth,
                    level=float(state.retest1_high),
                    pending_attr="retest1_pending_cross_ts",
                    attempts_attr="retest1_attempts",
                    entry_signal="ENTRY1",
                    entry_alert="retest1",
                    sl=float(last_1h_row["ema_400"]),
                    sl_type="ema400",
                    on_cap_reached=_r1_buy_cap,
                ))
            if state.stage == 5 and state.retest2_pending_cross_ts and state.retest2_high is not None:
                signals.extend(self._resolve_pending_15m_buy(
                    state, synth,
                    level=float(state.retest2_high),
                    pending_attr="retest2_pending_cross_ts",
                    attempts_attr="retest2_attempts",
                    entry_signal="ENTRY2",
                    entry_alert="retest2",
                    sl=float(state.retest2_low),
                    sl_type="static",
                    on_cap_reached=lambda: setattr(state, "stage", 4),
                ))
        else:  # SELL
            if state.stage == 3 and state.retest1_pending_cross_ts and state.retest1_low is not None:
                def _r1_sell_cap():
                    if self.config.skip_retest2:
                        self._reset_state(state, "NONE", synth)
                    else:
                        state.stage = 4
                signals.extend(self._resolve_pending_15m_sell(
                    state, synth,
                    level=float(state.retest1_low),
                    pending_attr="retest1_pending_cross_ts",
                    attempts_attr="retest1_attempts",
                    entry_signal="ENTRY1",
                    entry_alert="retest1",
                    sl=float(last_1h_row["ema_400"]),
                    sl_type="ema400",
                    on_cap_reached=_r1_sell_cap,
                ))
            if state.stage == 5 and state.retest2_pending_cross_ts and state.retest2_low is not None:
                signals.extend(self._resolve_pending_15m_sell(
                    state, synth,
                    level=float(state.retest2_low),
                    pending_attr="retest2_pending_cross_ts",
                    attempts_attr="retest2_attempts",
                    entry_signal="ENTRY2",
                    entry_alert="retest2",
                    sl=float(state.retest2_high),
                    sl_type="static",
                    on_cap_reached=lambda: setattr(state, "stage", 4),
                ))
        return signals

    def _resolve_pending_15m_buy(self, state, row, *, level, pending_attr,
                                  attempts_attr, entry_signal, entry_alert,
                                  sl, sl_type, on_cap_reached) -> List[dict]:
        """15m sustain resolver — mirrors block 1 of _handle_retest_break_buy."""
        signals: List[dict] = []
        pending_ts = getattr(state, pending_attr, None)
        if pending_ts is None:
            return signals
        if int(row["timestamp"]) <= int(pending_ts):
            return signals  # 15m bar predates pending arm
        attempts = getattr(state, attempts_attr, 0) or 0
        cap = attempts < self.config.re_entry_cap

        elapsed_min = (int(row["timestamp"]) - int(pending_ts)) / 60.0
        wick_floor = level * (1.0 - self.config.sustain_wick_tolerance_pct)
        if row["close"] <= level or float(row["low"]) <= wick_floor:
            setattr(state, pending_attr, None)
            reason = ("close back below level" if row["close"] <= level
                      else f"wick below {wick_floor:.2f} (level={level:.2f}, tol={self.config.sustain_wick_tolerance_pct*100:.2f}%)")
            signals.append(self._make_signal(
                row, "PENDING_CANCEL", "BUY",
                note=f"{entry_signal} sustain failed after {elapsed_min:.0f}m (15m): {reason}"
            ))
            return signals
        if elapsed_min < self._sustain_minutes_for("BUY"):
            return signals
        if not cap:
            setattr(state, pending_attr, None)
            return signals
        if not self._volume_ok(row):
            setattr(state, pending_attr, None)
            signals.append(self._make_signal(
                row, "ENTRY_SKIP", "BUY",
                note=f"{entry_signal} volume filter: vol below {self.config.volume_confirm_mult}x avg(volume,{self.config.volume_confirm_bars}) (15m)"
            ))
            return signals

        setattr(state, attempts_attr, attempts + 1)
        new_attempts = attempts + 1
        pos = self._open_position(
            row, trend="BUY", entry_alert=entry_alert, sl=sl, sl_type=sl_type,
        )
        self._append_position(state, pos)
        state.entries_count = (state.entries_count or 0) + 1
        if entry_signal == "ENTRY1" and state.entry1_price is None:
            state.entry1_price = pos["entry_price"]
            state.entry1_time = pos["entry_time"]
        if entry_signal == "ENTRY2" and state.entry2_price is None:
            state.entry2_price = pos["entry_price"]
            state.entry2_time = pos["entry_time"]
        state.stop_loss = pos["sl"]
        state.target_price = pos["target"]
        state.position_active = True
        sig = self._make_signal(
            row, entry_signal, "BUY", price=pos["entry_price"],
            note=f"BUY {entry_signal} attempt {new_attempts}/{self.config.re_entry_cap} "
                 f"({entry_alert} break sustained {elapsed_min:.0f}m on 15m)"
        )
        sig["sl"] = pos["sl"]
        sig["target"] = pos["target"]
        signals.append(sig)
        setattr(state, pending_attr, None)
        if new_attempts >= self.config.re_entry_cap:
            on_cap_reached()
        return signals

    def _resolve_pending_15m_sell(self, state, row, *, level, pending_attr,
                                   attempts_attr, entry_signal, entry_alert,
                                   sl, sl_type, on_cap_reached) -> List[dict]:
        """15m sustain resolver — mirrors block 1 of _handle_retest_break_sell."""
        signals: List[dict] = []
        pending_ts = getattr(state, pending_attr, None)
        if pending_ts is None:
            return signals
        if int(row["timestamp"]) <= int(pending_ts):
            return signals
        attempts = getattr(state, attempts_attr, 0) or 0
        cap = attempts < self.config.re_entry_cap

        elapsed_min = (int(row["timestamp"]) - int(pending_ts)) / 60.0
        wick_ceiling = level * (1.0 + self.config.sustain_wick_tolerance_pct)
        if row["close"] >= level or float(row["high"]) >= wick_ceiling:
            setattr(state, pending_attr, None)
            reason = ("close back above level" if row["close"] >= level
                      else f"wick above {wick_ceiling:.2f} (level={level:.2f}, tol={self.config.sustain_wick_tolerance_pct*100:.2f}%)")
            signals.append(self._make_signal(
                row, "PENDING_CANCEL", "SELL",
                note=f"{entry_signal} sustain failed after {elapsed_min:.0f}m (15m): {reason}"
            ))
            return signals
        if elapsed_min < self._sustain_minutes_for("SELL"):
            return signals
        if not cap:
            setattr(state, pending_attr, None)
            return signals
        if not self._volume_ok(row):
            setattr(state, pending_attr, None)
            signals.append(self._make_signal(
                row, "ENTRY_SKIP", "SELL",
                note=f"{entry_signal} volume filter: vol below {self.config.volume_confirm_mult}x avg(volume,{self.config.volume_confirm_bars}) (15m)"
            ))
            return signals

        setattr(state, attempts_attr, attempts + 1)
        new_attempts = attempts + 1
        pos = self._open_position(
            row, trend="SELL", entry_alert=entry_alert, sl=sl, sl_type=sl_type,
        )
        self._append_position(state, pos)
        state.entries_count = (state.entries_count or 0) + 1
        if entry_signal == "ENTRY1" and state.entry1_price is None:
            state.entry1_price = pos["entry_price"]
            state.entry1_time = pos["entry_time"]
        if entry_signal == "ENTRY2" and state.entry2_price is None:
            state.entry2_price = pos["entry_price"]
            state.entry2_time = pos["entry_time"]
        state.stop_loss = pos["sl"]
        state.target_price = pos["target"]
        state.position_active = True
        sig = self._make_signal(
            row, entry_signal, "SELL", price=pos["entry_price"],
            note=f"SELL {entry_signal} attempt {new_attempts}/{self.config.re_entry_cap} "
                 f"({entry_alert} break sustained {elapsed_min:.0f}m on 15m)"
        )
        sig["sl"] = pos["sl"]
        sig["target"] = pos["target"]
        signals.append(sig)
        setattr(state, pending_attr, None)
        if new_attempts >= self.config.re_entry_cap:
            on_cap_reached()
        return signals

    # ------------------------------------------------------------------
    # Per-bar step
    # ------------------------------------------------------------------
    def _step_machine(self, state, row, prev, df=None, idx=None) -> List[dict]:
        signals: List[dict] = []

        ema200 = row["ema_200"]
        ema400 = row["ema_400"]
        prev_ema200 = prev["ema_200"]
        prev_ema400 = prev["ema_400"]

        cross_up = prev_ema200 <= prev_ema400 and ema200 > ema400
        cross_dn = prev_ema200 >= prev_ema400 and ema200 < ema400

        # Quality filter: require minimum EMA gap as fraction of price.
        # Filters out touching crossings that immediately whipsaw.
        if (cross_up or cross_dn) and self.config.min_crossover_gap_pct > 0:
            gap_pct = abs(ema200 - ema400) / float(row["close"]) if row["close"] else 0.0
            if gap_pct < self.config.min_crossover_gap_pct:
                signals.append(self._make_signal(
                    row, "CROSSOVER_SKIP", "BUY" if cross_up else "SELL",
                    note=f"min_gap filter: gap={gap_pct*100:.3f}% < {self.config.min_crossover_gap_pct*100:.3f}%"
                ))
                cross_up = cross_dn = False

        # Optional HTF filter: only accept crossover if regime matches.
        # Symmetric default (same period for BUY/SELL); configurable margin
        # requires close to be margin% below/above SMA for confirmation.
        htf_ok_buy = htf_ok_sell = True
        if self.config.htf_filter_enabled:
            if "htf_sma_buy" in row.index:
                htf_b = row["htf_sma_buy"]
                if pd.notna(htf_b):
                    threshold = htf_b * (1 + self.config.htf_buy_margin_pct)
                    htf_ok_buy = row["close"] > threshold
            if "htf_sma_sell" in row.index:
                htf_s = row["htf_sma_sell"]
                if pd.notna(htf_s):
                    threshold = htf_s * (1 - self.config.htf_sell_margin_pct)
                    htf_ok_sell = row["close"] < threshold

        # Slope confirm: require EMA200 trending in crossover direction over N bars.
        slope_ok_buy = slope_ok_sell = True
        if df is not None and idx is not None:
            if self.config.buy_slope_bars > 0:
                slope_ok_buy = self._check_slope(df, idx, self.config.buy_slope_bars,
                                                  self.config.buy_slope_min_pct, "up")
            if self.config.sell_slope_bars > 0:
                slope_ok_sell = self._check_slope(df, idx, self.config.sell_slope_bars,
                                                   self.config.sell_slope_min_pct, "down")

        # Trend reset on opposite crossover — closes ALL open positions.
        if cross_up:
            if self.config.skip_buy:
                return signals
            if not htf_ok_buy:
                signals.append(self._make_signal(row, "CROSSOVER_SKIP", "BUY",
                                                 note="HTF filter: close below htf_sma"))
                return signals
            if not slope_ok_buy:
                signals.append(self._make_signal(row, "CROSSOVER_SKIP", "BUY",
                                                 note=f"slope filter: EMA200 not rising {self.config.buy_slope_min_pct*100:.2f}% over {self.config.buy_slope_bars} bars"))
                return signals
            if self._open_positions(state):
                signals.extend(self._close_all_positions(state, row, "CROSSOVER_FLIP"))
            self._reset_state(state, "BUY", row)
            signals.append(self._make_signal(row, "CROSSOVER", "BUY", note="EMA200 above EMA400"))
            return signals
        if cross_dn:
            if self.config.skip_sell:
                return signals
            if not htf_ok_sell:
                signals.append(self._make_signal(row, "CROSSOVER_SKIP", "SELL",
                                                 note="HTF filter: close above htf_sma"))
                return signals
            if not slope_ok_sell:
                signals.append(self._make_signal(row, "CROSSOVER_SKIP", "SELL",
                                                 note=f"slope filter: EMA200 not falling {self.config.sell_slope_min_pct*100:.2f}% over {self.config.sell_slope_bars} bars"))
                return signals
            if self._open_positions(state):
                signals.extend(self._close_all_positions(state, row, "CROSSOVER_FLIP"))
            self._reset_state(state, "SELL", row)
            signals.append(self._make_signal(row, "CROSSOVER", "SELL", note="EMA200 below EMA400"))
            return signals

        if state.trend == "NONE":
            return signals

        # Sanity flip: catch missed crossovers (gaps / EMAs straddling).
        # If state.trend disagrees with current EMA200/EMA400 ordering, end the
        # cycle and close any open positions. Only triggers when no edge cross
        # fired this bar (cross_up/cross_dn handled above already).
        if self.config.sanity_flip_trend:
            inverted = (
                (state.trend == "BUY" and ema200 < ema400)
                or (state.trend == "SELL" and ema200 > ema400)
            )
            if inverted:
                if self._open_positions(state):
                    signals.extend(self._close_all_positions(state, row, "TREND_INVERSION"))
                signals.append(self._make_signal(
                    row, "TREND_RESET", state.trend,
                    note=f"EMA inversion without crossover edge "
                         f"(EMA200={ema200:.2f} EMA400={ema400:.2f}) — end cycle"
                ))
                self._reset_state(state, "NONE", row)
                return signals

        # Per-spec interpretation: SL is per-position (intra-bar).
        # Trend reset happens ONLY on opposite crossover (handled above).
        # No close-below-EMA400 force-close — that conflated SL with trend reset.

        # 1) Stage transitions (alerts + entries)
        if state.trend == "BUY":
            signals.extend(self._step_buy(state, row, prev))
        else:
            signals.extend(self._step_sell(state, row, prev))

        # 2) Per-position TP / partial / SL on the same bar
        signals.extend(self._manage_positions(state, row))

        return signals

    # ------------------------------------------------------------------
    def _step_buy(self, state, row, prev) -> List[dict]:
        signals: List[dict] = []
        ema200 = row["ema_200"]
        ema400 = row["ema_400"]

        # Stage 1 -> ALERT1: break + close above crossover candle high
        if state.stage == 1 and state.crossover_high is not None:
            if row["close"] > state.crossover_high and row["high"] > state.crossover_high:
                state.stage = 2
                signals.append(self._make_signal(row, "ALERT1", "BUY",
                                                 note="Break + close above crossover candle high"))

        # Stage 2 -> ALERT2: retest of EMA200. Spec note: "price moves towards
        # 200 EMA from upside (above 200 EMA)" — require prior bar to have been
        # ABOVE EMA200 (true transition from upside) before locking the retest
        # candle. Without this, alerts fire when price has already been below
        # EMA200 for many bars (Cycle 1 trace bug).
        if state.stage == 2:
            cross_below_200 = row["close"] < ema200 and row["low"] < ema200
            from_upside = True
            if self.config.require_retest_from_upside:
                # prev bar must close above EMA200 (or be NaN-warmup tolerated).
                prev_ema200 = prev["ema_200"]
                if pd.notna(prev_ema200):
                    from_upside = prev["close"] > prev_ema200
            if cross_below_200 and from_upside:
                state.retest1_ts = int(row["timestamp"])
                state.retest1_high = float(row["high"])
                state.retest1_low = float(row["low"])
                state.retest1_attempts = 0
                state.retest1_invalidated = False
                state.retest1_pending_cross_ts = None
                state.stage = 3
                signals.append(self._make_signal(row, "ALERT2", "BUY",
                                                 note="EMA200 retest candle locked (from upside)"))

        # Stage 3: retest1 armed.
        if state.stage == 3 and state.retest1_high is not None:
            # 3a) EMA400 touch behavior depends on whether ENTRY1 already taken:
            #     - 0 entries -> spec invalidates retest1 path ("Omit 1st entry")
            #     - >=1 entry -> SL handled per-position; advance to retest2 watch
            #     Either way we advance to stage 4 so ALERT3 fires same bar.
            if row["low"] <= ema400:
                if (state.retest1_attempts or 0) == 0:
                    state.retest1_invalidated = True
                    signals.append(self._make_signal(row, "ALERT2_SKIP", "BUY",
                                                     note="EMA400 touched before retest1 break — omit ENTRY1"))
                state.retest1_pending_cross_ts = None  # abandon any pending retest1 sustain
                if self.config.skip_retest2:
                    self._reset_state(state, "NONE", row)
                    return signals
                state.stage = 4
            else:
                # 3b) Cross + sustain logic on retest1.high
                def _r1_buy_cap():
                    if self.config.skip_retest2:
                        self._reset_state(state, "NONE", row)
                    else:
                        state.stage = 4
                signals.extend(self._handle_retest_break_buy(
                    state, row, prev,
                    level=state.retest1_high,
                    pending_attr="retest1_pending_cross_ts",
                    attempts_attr="retest1_attempts",
                    entry_signal="ENTRY1",
                    entry_alert="retest1",
                    sl=float(ema400),
                    sl_type="ema400",
                    on_cap_reached=_r1_buy_cap,
                ))
        # Stage 4 -> ALERT3: EMA400 touch / cross below. Spec note: "price moves
        # towards 400 EMA from upside (above 400 EMA)" — require prior bar to
        # have been ABOVE EMA400 before locking retest2 candle.
        if state.stage == 4:
            touch_400 = row["low"] <= ema400
            from_upside = True
            if self.config.require_retest_from_upside:
                prev_ema400 = prev["ema_400"]
                if pd.notna(prev_ema400):
                    from_upside = prev["close"] > prev_ema400
            if touch_400 and from_upside:
                cap_n = self.config.max_alert3_locks_per_cycle
                cur = state.alert3_locks_count or 0
                if cap_n > 0 and cur >= cap_n:
                    # Cap hit: end cycle (await next crossover) instead of re-locking
                    signals.append(self._make_signal(
                        row, "ALERT3_SKIP", "BUY",
                        note=f"max_alert3_locks_per_cycle={cap_n} reached — end cycle"
                    ))
                    self._reset_state(state, "NONE", row)
                    return signals
                state.retest2_ts = int(row["timestamp"])
                state.retest2_high = float(row["high"])
                state.retest2_low = float(row["low"])
                state.retest2_attempts = 0
                state.retest2_pending_cross_ts = None
                state.alert3_locks_count = cur + 1
                state.stage = 5
                signals.append(self._make_signal(row, "ALERT3", "BUY",
                                                 note="EMA400 retest candle locked (from upside)"))

        # Stage 5: retest2 armed
        if state.stage == 5 and state.retest2_high is not None:
            signals.extend(self._handle_retest_break_buy(
                state, row, prev,
                level=state.retest2_high,
                pending_attr="retest2_pending_cross_ts",
                attempts_attr="retest2_attempts",
                entry_signal="ENTRY2",
                entry_alert="retest2",
                sl=float(state.retest2_low),
                sl_type="static",
                on_cap_reached=lambda: setattr(state, "stage", 4),
            ))
        return signals

    # ------------------------------------------------------------------
    def _step_sell(self, state, row, prev) -> List[dict]:
        signals: List[dict] = []
        ema200 = row["ema_200"]
        ema400 = row["ema_400"]

        if state.stage == 1 and state.crossover_low is not None:
            if row["close"] < state.crossover_low and row["low"] < state.crossover_low:
                state.stage = 2
                signals.append(self._make_signal(row, "ALERT1", "SELL",
                                                 note="Break + close below crossover candle low"))

        # SELL mirror: spec note "price moves towards 200 EMA from downside"
        # — require prior bar to have been BELOW EMA200 (true transition).
        if state.stage == 2:
            cross_above_200 = row["close"] > ema200 and row["high"] > ema200
            from_downside = True
            if self.config.require_retest_from_upside:
                prev_ema200 = prev["ema_200"]
                if pd.notna(prev_ema200):
                    from_downside = prev["close"] < prev_ema200
            if cross_above_200 and from_downside:
                state.retest1_ts = int(row["timestamp"])
                state.retest1_high = float(row["high"])
                state.retest1_low = float(row["low"])
                state.retest1_attempts = 0
                state.retest1_invalidated = False
                state.retest1_pending_cross_ts = None
                state.stage = 3
                signals.append(self._make_signal(row, "ALERT2", "SELL",
                                                 note="EMA200 retest candle locked (from downside)"))

        if state.stage == 3 and state.retest1_low is not None:
            if row["high"] >= ema400:
                if (state.retest1_attempts or 0) == 0:
                    state.retest1_invalidated = True
                    signals.append(self._make_signal(row, "ALERT2_SKIP", "SELL",
                                                     note="EMA400 touched before retest1 break — omit ENTRY1"))
                state.retest1_pending_cross_ts = None
                if self.config.skip_retest2:
                    self._reset_state(state, "NONE", row)
                    return signals
                state.stage = 4
            else:
                def _r1_sell_cap():
                    if self.config.skip_retest2:
                        self._reset_state(state, "NONE", row)
                    else:
                        state.stage = 4
                signals.extend(self._handle_retest_break_sell(
                    state, row, prev,
                    level=state.retest1_low,
                    pending_attr="retest1_pending_cross_ts",
                    attempts_attr="retest1_attempts",
                    entry_signal="ENTRY1",
                    entry_alert="retest1",
                    sl=float(ema400),
                    sl_type="ema400",
                    on_cap_reached=_r1_sell_cap,
                ))

        # SELL mirror retest2: require prior bar BELOW EMA400.
        if state.stage == 4:
            touch_400 = row["high"] >= ema400
            from_downside = True
            if self.config.require_retest_from_upside:
                prev_ema400 = prev["ema_400"]
                if pd.notna(prev_ema400):
                    from_downside = prev["close"] < prev_ema400
            if touch_400 and from_downside:
                cap_n = self.config.max_alert3_locks_per_cycle
                cur = state.alert3_locks_count or 0
                if cap_n > 0 and cur >= cap_n:
                    signals.append(self._make_signal(
                        row, "ALERT3_SKIP", "SELL",
                        note=f"max_alert3_locks_per_cycle={cap_n} reached — end cycle"
                    ))
                    self._reset_state(state, "NONE", row)
                    return signals
                state.retest2_ts = int(row["timestamp"])
                state.retest2_high = float(row["high"])
                state.retest2_low = float(row["low"])
                state.retest2_attempts = 0
                state.retest2_pending_cross_ts = None
                state.alert3_locks_count = cur + 1
                state.stage = 5
                signals.append(self._make_signal(row, "ALERT3", "SELL",
                                                 note="EMA400 retest candle locked (from downside)"))

        if state.stage == 5 and state.retest2_low is not None:
            signals.extend(self._handle_retest_break_sell(
                state, row, prev,
                level=state.retest2_low,
                pending_attr="retest2_pending_cross_ts",
                attempts_attr="retest2_attempts",
                entry_signal="ENTRY2",
                entry_alert="retest2",
                sl=float(state.retest2_high),
                sl_type="static",
                on_cap_reached=lambda: setattr(state, "stage", 4),
            ))
        return signals

    # ------------------------------------------------------------------
    @staticmethod
    def _check_slope(df, idx: int, bars: int, min_pct: float, direction: str) -> bool:
        """Check if EMA200 has moved at least min_pct in `direction` over last `bars`.
        direction: 'up' = (now-then)/then >= min_pct
                   'down' = (then-now)/then >= min_pct
        Returns False if window too short or condition not met.
        """
        ref_idx = idx - bars
        if ref_idx < 0:
            return False
        ema_now = df.iloc[idx]["ema_200"]
        ema_then = df.iloc[ref_idx]["ema_200"]
        if ema_then <= 0:
            return False
        change = (ema_now - ema_then) / ema_then
        if direction == "up":
            return change >= min_pct
        return -change >= min_pct  # down: change <= -min_pct

    # ------------------------------------------------------------------
    # Cross-and-sustain trigger (shared BUY/SELL retest1/retest2)
    # ------------------------------------------------------------------
    def _handle_retest_break_buy(self, state, row, prev, *, level, pending_attr,
                                  attempts_attr, entry_signal, entry_alert,
                                  sl, sl_type, on_cap_reached) -> List[dict]:
        """BUY-side cross + sustain-15min logic for retest1.high or retest2.high.

        Detects cross-up edge, marks pending, then triggers ENTRY only when
        elapsed >= sustain_minutes AND price still above level. Cancels pending
        if price closes back below level before sustain elapsed.
        """
        signals: List[dict] = []
        pending_ts = getattr(state, pending_attr, None)
        attempts = getattr(state, attempts_attr, 0) or 0
        cap = attempts < self.config.re_entry_cap

        # 1) Resolve any existing pending first
        if pending_ts is not None:
            elapsed_min = (int(row["timestamp"]) - int(pending_ts)) / 60.0
            wick_floor = level * (1.0 - self.config.sustain_wick_tolerance_pct)
            if row["close"] <= level or float(row["low"]) <= wick_floor:
                # Failed sustain — cancel pending. Allow re-arm via next edge.
                setattr(state, pending_attr, None)
                reason = ("close back below level" if row["close"] <= level
                          else f"wick below {wick_floor:.2f}")
                signals.append(self._make_signal(
                    row, "PENDING_CANCEL", "BUY",
                    note=f"{entry_signal} sustain failed after {elapsed_min:.0f}m: {reason}"
                ))
            elif elapsed_min >= self._sustain_minutes_for("BUY"):
                # Sustained — fire ENTRY (after volume gate)
                if cap and not self._volume_ok(row):
                    setattr(state, pending_attr, None)
                    signals.append(self._make_signal(
                        row, "ENTRY_SKIP", "BUY",
                        note=f"{entry_signal} volume filter: vol below {self.config.volume_confirm_mult}x avg(volume,{self.config.volume_confirm_bars})"
                    ))
                    return signals
                if cap:
                    setattr(state, attempts_attr, attempts + 1)
                    new_attempts = attempts + 1
                    pos = self._open_position(
                        row, trend="BUY", entry_alert=entry_alert,
                        sl=sl, sl_type=sl_type,
                    )
                    self._append_position(state, pos)
                    state.entries_count = (state.entries_count or 0) + 1
                    if entry_signal == "ENTRY1" and state.entry1_price is None:
                        state.entry1_price = pos["entry_price"]
                        state.entry1_time = pos["entry_time"]
                    if entry_signal == "ENTRY2" and state.entry2_price is None:
                        state.entry2_price = pos["entry_price"]
                        state.entry2_time = pos["entry_time"]
                    state.stop_loss = pos["sl"]
                    state.target_price = pos["target"]
                    state.position_active = True
                    sig = self._make_signal(
                        row, entry_signal, "BUY", price=pos["entry_price"],
                        note=f"BUY {entry_signal} attempt {new_attempts}/{self.config.re_entry_cap} "
                             f"({entry_alert} break sustained {elapsed_min:.0f}m)"
                    )
                    sig["sl"] = pos["sl"]
                    sig["target"] = pos["target"]
                    signals.append(sig)
                    setattr(state, pending_attr, None)
                    if new_attempts >= self.config.re_entry_cap:
                        on_cap_reached()
                        return signals
                else:
                    setattr(state, pending_attr, None)
            # else: still pending, not yet elapsed — wait

        # 2) New edge cross — only if no pending and cap not full
        if getattr(state, pending_attr, None) is None and cap:
            edge_break = (prev["close"] <= level and row["close"] > level)
            if edge_break:
                setattr(state, pending_attr, int(row["timestamp"]))
                signals.append(self._make_signal(
                    row, "PENDING", "BUY",
                    note=f"{entry_signal} cross detected — sustain check pending "
                         f"({self._sustain_minutes_for('BUY')}m)"
                ))
        return signals

    def _handle_retest_break_sell(self, state, row, prev, *, level, pending_attr,
                                   attempts_attr, entry_signal, entry_alert,
                                   sl, sl_type, on_cap_reached) -> List[dict]:
        """SELL-side mirror of _handle_retest_break_buy."""
        signals: List[dict] = []
        pending_ts = getattr(state, pending_attr, None)
        attempts = getattr(state, attempts_attr, 0) or 0
        cap = attempts < self.config.re_entry_cap

        if pending_ts is not None:
            elapsed_min = (int(row["timestamp"]) - int(pending_ts)) / 60.0
            wick_ceiling = level * (1.0 + self.config.sustain_wick_tolerance_pct)
            if row["close"] >= level or float(row["high"]) >= wick_ceiling:
                setattr(state, pending_attr, None)
                reason = ("close back above level" if row["close"] >= level
                          else f"wick above {wick_ceiling:.2f}")
                signals.append(self._make_signal(
                    row, "PENDING_CANCEL", "SELL",
                    note=f"{entry_signal} sustain failed after {elapsed_min:.0f}m: {reason}"
                ))
            elif elapsed_min >= self._sustain_minutes_for("SELL"):
                if cap and not self._volume_ok(row):
                    setattr(state, pending_attr, None)
                    signals.append(self._make_signal(
                        row, "ENTRY_SKIP", "SELL",
                        note=f"{entry_signal} volume filter: vol below {self.config.volume_confirm_mult}x avg(volume,{self.config.volume_confirm_bars})"
                    ))
                    return signals
                if cap:
                    setattr(state, attempts_attr, attempts + 1)
                    new_attempts = attempts + 1
                    pos = self._open_position(
                        row, trend="SELL", entry_alert=entry_alert,
                        sl=sl, sl_type=sl_type,
                    )
                    self._append_position(state, pos)
                    state.entries_count = (state.entries_count or 0) + 1
                    if entry_signal == "ENTRY1" and state.entry1_price is None:
                        state.entry1_price = pos["entry_price"]
                        state.entry1_time = pos["entry_time"]
                    if entry_signal == "ENTRY2" and state.entry2_price is None:
                        state.entry2_price = pos["entry_price"]
                        state.entry2_time = pos["entry_time"]
                    state.stop_loss = pos["sl"]
                    state.target_price = pos["target"]
                    state.position_active = True
                    sig = self._make_signal(
                        row, entry_signal, "SELL", price=pos["entry_price"],
                        note=f"SELL {entry_signal} attempt {new_attempts}/{self.config.re_entry_cap} "
                             f"({entry_alert} break sustained {elapsed_min:.0f}m)"
                    )
                    sig["sl"] = pos["sl"]
                    sig["target"] = pos["target"]
                    signals.append(sig)
                    setattr(state, pending_attr, None)
                    if new_attempts >= self.config.re_entry_cap:
                        on_cap_reached()
                        return signals
                else:
                    setattr(state, pending_attr, None)

        if getattr(state, pending_attr, None) is None and cap:
            edge_break = (prev["close"] >= level and row["close"] < level)
            if edge_break:
                setattr(state, pending_attr, int(row["timestamp"]))
                signals.append(self._make_signal(
                    row, "PENDING", "SELL",
                    note=f"{entry_signal} cross detected — sustain check pending "
                         f"({self._sustain_minutes_for('SELL')}m)"
                ))
        return signals

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def _open_position(self, row, trend, entry_alert, sl, sl_type) -> dict:
        """sl_type:
            'ema400'  -> SL = current EMA400 each bar. Used for ENTRY1 pre-partial.
            'ema200'  -> SL = current EMA200 each bar. Used for both ENTRY1 and
                         ENTRY2 post-partial trail (Strategy-1 spec).
            'static'  -> SL = fixed price `sl`. Used for ENTRY2 pre-partial
                         (retest2 candle low/high).
        """
        entry_price = float(row["close"])
        # Per-entry / per-side partial trigger.
        # Strategy-1 spec: BUY entry1=5%/entry2=15%, SELL entry1=5%/entry2=5%.
        if entry_alert == "retest1":
            partial_pct = self.config.partial_pct_entry1
        else:  # retest2
            partial_pct = (
                self.config.partial_pct_entry2_buy if trend == "BUY"
                else self.config.partial_pct_entry2_sell
            )
        if trend == "BUY":
            target = entry_price * (1 + self.config.target_pct)
            partial = entry_price * (1 + partial_pct)
        else:
            target = entry_price * (1 - self.config.target_pct)
            partial = entry_price * (1 - partial_pct)
        # Optional ENTRY2 SL cap — tighten retest2 SL distance from entry.
        if (entry_alert == "retest2" and sl_type == "static"
                and self.config.retest2_sl_cap_pct > 0):
            cap_pct = self.config.retest2_sl_cap_pct
            if trend == "BUY":
                cap_sl = entry_price * (1 - cap_pct)
                sl = max(sl, cap_sl)  # tighter = closer to entry = larger value
            else:
                cap_sl = entry_price * (1 + cap_pct)
                sl = min(sl, cap_sl)
        return {
            "trend": trend,
            "entry_alert": entry_alert,            # 'retest1' | 'retest2'
            "entry_ts": int(row["timestamp"]),
            "entry_time": row["candle_time"].isoformat() if hasattr(row["candle_time"], "isoformat") else str(row["candle_time"]),
            "entry_price": entry_price,
            "sl": sl,
            "sl_type": sl_type,
            "target": target,
            "partial_pct": partial_pct,
            "partial_threshold": partial,
            "partial_booked": False,
            "qty_remaining": 1.0,
        }

    def _open_positions(self, state) -> list:
        return self._load_positions(state)

    def _append_position(self, state, pos: dict) -> None:
        positions = self._load_positions(state)
        positions.append(pos)
        self._save_positions(state, positions)

    def _save_positions(self, state, positions: list) -> None:
        # JSONB column accepts list directly via SQLAlchemy. Keep raw text fallback
        # for non-JSONB backends (sqlite tests).
        try:
            state.positions_json = positions
        except Exception:
            state.positions_json = json.dumps(positions)

    def _load_positions(self, state) -> list:
        raw = getattr(state, "positions_json", None)
        if raw is None:
            return []
        if isinstance(raw, list):
            return list(raw)
        if isinstance(raw, str):
            try:
                return json.loads(raw) or []
            except Exception:
                return []
        return []

    # ------------------------------------------------------------------
    def _manage_positions(self, state, row) -> List[dict]:
        """Per-bar TP / partial / SL on each open position (high/low approx)."""
        signals: List[dict] = []
        positions = self._load_positions(state)
        if not positions:
            return signals

        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])
        cur_ema200 = float(row["ema_200"])
        cur_ema400 = float(row["ema_400"])
        still_open: list = []

        for pos in positions:
            trend = pos["trend"]
            entry = float(pos["entry_price"])
            target = float(pos["target"])
            sl_type = pos.get("sl_type", "static")
            # Dynamic SL types:
            #   'ema400' -> SL = current EMA400 (ENTRY1 pre-partial).
            #   'ema200' -> SL = current EMA200 (post-partial trail per
            #               BTC trade rules v1.2 — both ENTRY1 and ENTRY2).
            #   'static' -> SL = fixed price (ENTRY2 retest2 candle low/high).
            if sl_type == "ema400":
                sl = cur_ema400
            elif sl_type == "ema200":
                sl = cur_ema200
            else:
                sl = float(pos["sl"])
            partial = float(pos["partial_threshold"])
            booked = bool(pos.get("partial_booked"))

            # 1) Target hit (full close)
            if trend == "BUY" and bar_high >= target:
                signals.append(self._make_signal(
                    row, "TARGET_HIT", trend, price=target,
                    note=f"Target hit ({self.config.target_pct*100:.0f}%) qty={pos['qty_remaining']:.2f} alert={pos['entry_alert']}"
                ))
                continue
            if trend == "SELL" and bar_low <= target:
                signals.append(self._make_signal(
                    row, "TARGET_HIT", trend, price=target,
                    note=f"Target hit ({self.config.target_pct*100:.0f}%) qty={pos['qty_remaining']:.2f} alert={pos['entry_alert']}"
                ))
                continue

            # 2) Partial booking @ per-entry partial_pct (5% retest1, 15% retest2).
            # Spec: post-partial SL trails the 200 EMA for the remaining qty.
            pos_partial_pct = float(pos.get("partial_pct", self.config.partial_pct_entry2_buy))
            if not booked:
                if trend == "BUY" and bar_high >= partial:
                    book_qty = pos["qty_remaining"] * self.config.partial_qty_frac
                    pos["qty_remaining"] = pos["qty_remaining"] - book_qty
                    pos["partial_booked"] = True
                    pos["sl_type"] = "ema200"
                    pos["sl"] = cur_ema200
                    sl = cur_ema200
                    booked = True
                    signals.append(self._make_signal(
                        row, "PARTIAL", trend, price=partial,
                        note=f"Partial book {book_qty:.2f} @ {pos_partial_pct*100:.0f}%; trail SL->EMA200 alert={pos['entry_alert']}"
                    ))
                elif trend == "SELL" and bar_low <= partial:
                    book_qty = pos["qty_remaining"] * self.config.partial_qty_frac
                    pos["qty_remaining"] = pos["qty_remaining"] - book_qty
                    pos["partial_booked"] = True
                    pos["sl_type"] = "ema200"
                    pos["sl"] = cur_ema200
                    sl = cur_ema200
                    booked = True
                    signals.append(self._make_signal(
                        row, "PARTIAL", trend, price=partial,
                        note=f"Partial book {book_qty:.2f} @ {pos_partial_pct*100:.0f}%; trail SL->EMA200 alert={pos['entry_alert']}"
                    ))

            # 3) SL hit on remaining qty.
            # Per BTC trade rules v1.2 — all "Price cross below/above" SL
            # triggers are close-based (bar.close beyond level), not wick.
            # Applies uniformly to: ENTRY1 EMA400 SL, ENTRY2 retest2 static SL,
            # and post-partial EMA200 trail.
            if trend == "BUY" and bar_close < sl:
                signals.append(self._make_signal(
                    row, "STOP_HIT", trend, price=bar_close,
                    note=f"SL hit (close<{sl_type}) qty={pos['qty_remaining']:.2f} sl={sl:.2f} alert={pos['entry_alert']}"
                ))
                continue
            if trend == "SELL" and bar_close > sl:
                signals.append(self._make_signal(
                    row, "STOP_HIT", trend, price=bar_close,
                    note=f"SL hit (close>{sl_type}) qty={pos['qty_remaining']:.2f} sl={sl:.2f} alert={pos['entry_alert']}"
                ))
                continue

            still_open.append(pos)

        self._save_positions(state, still_open)
        if not still_open:
            state.position_active = False
        return signals

    def _close_all_positions(self, state, row, reason: str) -> List[dict]:
        """Force-close every open position at row close (CROSSOVER_FLIP / EXIT_EMA400)."""
        signals: List[dict] = []
        positions = self._load_positions(state)
        for pos in positions:
            signals.append(self._make_signal(
                row, "STOP_HIT", pos["trend"], price=float(row["close"]),
                note=f"Force close ({reason}) qty={pos['qty_remaining']:.2f} alert={pos['entry_alert']}"
            ))
        self._save_positions(state, [])
        state.position_active = False
        return signals

    # ------------------------------------------------------------------
    def _reset_state(self, state, trend: str, row) -> None:
        state.trend = trend
        state.stage = 1 if trend in ("BUY", "SELL") else 0
        if trend in ("BUY", "SELL"):
            state.crossover_ts = int(row["timestamp"])
            state.crossover_high = float(row["high"])
            state.crossover_low = float(row["low"])
        else:
            state.crossover_ts = None
            state.crossover_high = None
            state.crossover_low = None
        state.retest1_ts = None
        state.retest1_high = None
        state.retest1_low = None
        state.retest2_ts = None
        state.retest2_high = None
        state.retest2_low = None
        state.entries_count = 0
        state.entry1_price = None
        state.entry1_time = None
        state.entry2_price = None
        state.entry2_time = None
        state.stop_loss = None
        state.target_price = None
        state.position_active = False
        state.retest1_attempts = 0
        state.retest2_attempts = 0
        state.retest1_invalidated = False
        state.retest1_pending_cross_ts = None
        state.retest2_pending_cross_ts = None
        state.alert3_locks_count = 0
        self._save_positions(state, [])

    # ------------------------------------------------------------------
    @staticmethod
    def _candles_to_df(candles: List[HistoricalData1H]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [c.timestamp for c in candles],
                "candle_time": [c.candle_time for c in candles],
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
            }
        )

    @staticmethod
    def _make_signal(row, signal_type: str, trend: str,
                      price: Optional[float] = None, note: str = "") -> dict:
        return {
            "signal_type": signal_type,
            "trend": trend,
            "candle_ts": int(row["timestamp"]),
            "candle_time": row["candle_time"],
            "price": float(price) if price is not None else float(row["close"]),
            "ema_200": float(row["ema_200"]),
            "ema_400": float(row["ema_400"]),
            "note": note,
        }

    # ------------------------------------------------------------------
    def _load_state(self, user_id: int, symbol: str) -> EMACrossoverState:
        with self.db.get_session() as session:
            state = (
                session.query(EMACrossoverState)
                .filter_by(user_id=user_id, symbol=symbol)
                .one_or_none()
            )
            if state is None:
                state = EMACrossoverState(
                    user_id=user_id, symbol=symbol, trend="NONE", stage=0,
                    retest1_attempts=0, retest2_attempts=0,
                    retest1_invalidated=False, positions_json=[],
                    alert3_locks_count=0,
                )
                session.add(state)
                session.commit()
            session.expunge(state)
        return state

    def _save_state(self, state) -> None:
        with self.db.get_session() as session:
            session.merge(state)
            session.commit()

    def _record_signal(self, user_id: int, symbol: str, sig: dict) -> None:
        with self.db.get_session() as session:
            session.add(
                EMACrossoverSignal(
                    user_id=user_id,
                    symbol=symbol,
                    signal_type=sig["signal_type"],
                    trend=sig["trend"],
                    candle_ts=sig["candle_ts"],
                    candle_time=sig["candle_time"],
                    price=sig.get("price"),
                    ema_200=sig.get("ema_200"),
                    ema_400=sig.get("ema_400"),
                    note=sig.get("note", "")[:255],
                )
            )
            session.commit()


_strategy: Optional[EMACrossoverStrategy] = None


def get_ema_crossover_strategy(config: Optional[StrategyConfig] = None) -> EMACrossoverStrategy:
    global _strategy
    if _strategy is None:
        _strategy = EMACrossoverStrategy(config)
    return _strategy

"""EMA Pullback Breakout (Stage 2 / Minervini-style) — daily-bar swing model.

Most-followed Indian retail swing strategy. Identifies stocks in a
confirmed Stage 2 uptrend (close > EMA50 > EMA200), waits for a pullback
to EMA20, enters on breakout of recent high with momentum + volume
confirmation. Position management: half off at 2×ATR, trail remainder on
EMA20.

Designed to mirror the API surface of EMACrossoverStrategy so the same
backtest harness can drive it. Pure Python from OHLCV bars. No DB calls.

Spec sources: ChartInk top-loved swing scanners, Vivek Bajaj/Elearnmarkets
curriculum, Pushkar Raj Thakur, Zerodha Varsity TA module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PullbackConfig:
    # Trend filter (Stage 2)
    ema_fast_period: int = 20      # pullback target
    ema_med_period: int = 50       # trend-strength filter
    ema_slow_period: int = 200     # macro-trend filter

    # Pullback detection
    pullback_lookback_bars: int = 3        # last N bars must wick EMA20
    pullback_proximity_pct: float = 0.005  # within 0.5% of EMA20

    # Breakout trigger
    breakout_lookback_bars: int = 5  # close must exceed prior N-bar high

    # Momentum
    rsi_period: int = 14
    rsi_min: float = 50.0
    rsi_max: float = 70.0

    # Volume confirmation
    volume_sma_bars: int = 20
    volume_mult: float = 1.5

    # Liquidity (NSE retail standard)
    min_price: float = 50.0
    min_adv_inr: float = 5_00_00_000  # ₹5 crore avg-daily-value

    # Exit / risk
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 2.0
    tp1_qty_frac: float = 0.5    # book 50% at T1, trail remainder
    trail_use_ema_fast: bool = True   # trail on EMA20 close-below
    time_stop_bars: int = 10
    time_stop_min_pct: float = 0.03   # exit if not +3% by time-stop bar

    # Direction
    enable_long: bool = True
    enable_short: bool = False  # Indian retail mostly long-only


@dataclass
class PullbackPosition:
    symbol: str
    entry_ts: int
    entry_time: datetime
    entry_price: float
    qty: float
    stop: float
    target: float
    atr_at_entry: float
    bars_held: int = 0
    partial_done: bool = False


@dataclass
class PullbackState:
    user_id: int = 0
    symbol: str = ""
    open_position: Optional[PullbackPosition] = None
    last_evaluated_ts: Optional[int] = None
    closed_count: int = 0


class EMAPullbackBreakoutStrategy:
    """Daily-bar pullback-breakout model. Same evaluate() signature as
    EMACrossoverStrategy so the harness can drive it identically.
    """

    def __init__(self, config: PullbackConfig):
        self.config = config
        self._states: Dict[str, PullbackState] = {}
        self._signal_log: List[Dict] = []

    # ---- helpers ----
    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        dn = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
        roll_dn = dn.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = roll_up / roll_dn.replace(0, 1e-12)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    def _candles_to_df(self, candles) -> pd.DataFrame:
        return pd.DataFrame([{
            "timestamp": c.timestamp,
            "candle_time": c.candle_time,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": int(getattr(c, "volume", 0) or 0),
        } for c in candles])

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast_period, adjust=False).mean()
        df["ema_med"]  = df["close"].ewm(span=cfg.ema_med_period,  adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow_period, adjust=False).mean()
        df["rsi"] = self._rsi(df["close"], cfg.rsi_period)
        df["atr"] = self._atr(df["high"], df["low"], df["close"], cfg.atr_period)
        df["vol_sma"] = df["volume"].rolling(cfg.volume_sma_bars, min_periods=1).mean()
        df["adv_inr"] = (df["close"] * df["volume"]).rolling(cfg.volume_sma_bars, min_periods=1).mean()
        df["prior_high_n"] = df["high"].shift(1).rolling(cfg.breakout_lookback_bars, min_periods=1).max()
        df["pullback_low_n"] = df["low"].rolling(cfg.pullback_lookback_bars, min_periods=1).min()
        return df

    def _load_state(self, user_id, symbol):
        key = f"{user_id}:{symbol}"
        if key not in self._states:
            self._states[key] = PullbackState(user_id=user_id, symbol=symbol)
        return self._states[key]

    def _record(self, sig: Dict):
        self._signal_log.append(sig)

    # ---- main eval ----
    def evaluate(self, user_id: int, symbol: str, candles,
                 candles_15m=None, eval_from_ts: Optional[int] = None) -> List[Dict]:
        cfg = self.config
        if len(candles) < cfg.ema_slow_period + cfg.atr_period + 5:
            return []
        df = self._enrich(self._candles_to_df(candles))
        state = self._load_state(user_id, symbol)
        signals: List[Dict] = []

        for i in range(cfg.ema_slow_period, len(df)):
            row = df.iloc[i]
            ts = int(row["timestamp"])
            if eval_from_ts is not None and ts < eval_from_ts:
                continue

            # Manage open position first
            if state.open_position is not None:
                pos = state.open_position
                pos.bars_held += 1

                # Stop hit
                if row["low"] <= pos.stop:
                    sig = {
                        "signal_type": "STOP_HIT", "trend": "BUY",
                        "candle_ts": ts, "candle_time": row["candle_time"],
                        "price": pos.stop, "ema_200": float(row["ema_slow"]),
                        "ema_400": float(row["ema_fast"]),
                        "note": f"SL hit (bars_held={pos.bars_held})"}
                    signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                    state.closed_count += 1
                    state.open_position = None
                    continue

                # T1 hit (book partial)
                if not pos.partial_done and row["high"] >= pos.target:
                    book_qty = pos.qty * cfg.tp1_qty_frac
                    sig = {
                        "signal_type": "PARTIAL", "trend": "BUY",
                        "candle_ts": ts, "candle_time": row["candle_time"],
                        "price": pos.target, "ema_200": float(row["ema_slow"]),
                        "ema_400": float(row["ema_fast"]),
                        "note": f"T1 booked {cfg.tp1_qty_frac*100:.0f}% @ {pos.target:.2f}"}
                    signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                    pos.qty -= book_qty
                    pos.partial_done = True
                    pos.stop = pos.entry_price  # trail SL to entry after T1

                # Trail exit on EMA-fast close-below (after T1 OR pure trail)
                if cfg.trail_use_ema_fast and pos.partial_done and row["close"] < row["ema_fast"]:
                    sig = {
                        "signal_type": "TARGET_HIT", "trend": "BUY",
                        "candle_ts": ts, "candle_time": row["candle_time"],
                        "price": float(row["close"]),
                        "ema_200": float(row["ema_slow"]), "ema_400": float(row["ema_fast"]),
                        "note": f"Trail-exit close<EMA{cfg.ema_fast_period}"}
                    signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                    state.closed_count += 1
                    state.open_position = None
                    continue

                # Time stop
                if pos.bars_held >= cfg.time_stop_bars:
                    pct = (row["close"] / pos.entry_price - 1.0)
                    if pct < cfg.time_stop_min_pct:
                        sig = {
                            "signal_type": "STOP_HIT", "trend": "BUY",
                            "candle_ts": ts, "candle_time": row["candle_time"],
                            "price": float(row["close"]),
                            "ema_200": float(row["ema_slow"]), "ema_400": float(row["ema_fast"]),
                            "note": f"Time-stop ({cfg.time_stop_bars}d <{cfg.time_stop_min_pct*100:.0f}%)"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        state.closed_count += 1
                        state.open_position = None
                        continue

            # No position → look for entry
            if state.open_position is None and cfg.enable_long:
                # Stage 2 trend filter
                stage2 = (row["close"] > row["ema_med"] > row["ema_slow"])
                # Pullback: low of last N bars wicked EMA20
                pullback = row["pullback_low_n"] <= row["ema_fast"] * (1 + cfg.pullback_proximity_pct)
                # Breakout: today's close > prior 5-bar high AND > prev high
                prior_n = float(row["prior_high_n"]) if pd.notna(row["prior_high_n"]) else float("inf")
                breakout = row["close"] > prior_n
                # Momentum gate
                mom = cfg.rsi_min <= row["rsi"] <= cfg.rsi_max
                # Volume confirmation
                vol_ok = row["volume"] >= cfg.volume_mult * row["vol_sma"]
                # Liquidity
                liq_ok = (row["close"] >= cfg.min_price) and (row["adv_inr"] >= cfg.min_adv_inr)

                if stage2 and pullback and breakout and mom and vol_ok and liq_ok:
                    entry_price = float(row["close"])
                    atr_v = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
                    if atr_v <= 0:
                        continue
                    stop = entry_price - cfg.sl_atr_mult * atr_v
                    target = entry_price + cfg.tp1_atr_mult * atr_v
                    pos = PullbackPosition(
                        symbol=symbol, entry_ts=ts,
                        entry_time=row["candle_time"], entry_price=entry_price,
                        qty=1.0, stop=stop, target=target, atr_at_entry=atr_v,
                    )
                    state.open_position = pos
                    sig = {
                        "signal_type": "ENTRY1", "trend": "BUY",
                        "candle_ts": ts, "candle_time": row["candle_time"],
                        "price": entry_price, "sl": stop, "target": target,
                        "ema_200": float(row["ema_slow"]), "ema_400": float(row["ema_fast"]),
                        "note": (f"Stage2 pullback-breakout "
                                 f"RSI={row['rsi']:.0f} "
                                 f"vol={row['volume']/max(row['vol_sma'],1):.1f}x "
                                 f"ATR={atr_v:.2f}")}
                    signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})

            state.last_evaluated_ts = ts

        return signals

"""15-Minute Opening Range Breakout (ORB) — intraday model.

Most-followed Indian intraday strategy (ChartInk top-loved scanners,
Zerodha "In The Money" deep-dive, Sukhani published system, default
template in Streak/Tradetron/AlgoTest).

Rules:
  - At 09:30 IST, fix ORB_HIGH/ORB_LOW = high/low of bars 09:15-09:29.
  - Long entry: 5-min CLOSE > ORB_HIGH AND > VWAP AND vol > 1.5×SMA(20).
  - Short entry: mirror.
  - Window: 09:30 → 11:15 only. One trade per symbol per day.
  - SL = opposite ORB end OR 1×ATR(14) on 5m, whichever tighter.
  - T1 = 1.5R (50% qty), trail remainder on VWAP cross.
  - EOD square-off 15:20.
  - Skip days where ORB range > 1.5% (gap chase) or < 0.3% (no vol).

Operates on 5-minute bars. ORB derived from same 5m bars (3 bars =
09:15, 09:20, 09:25 forming the 09:15-09:29 window).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ORBConfig:
    # ORB window
    orb_start: dtime = dtime(9, 15)
    orb_end:   dtime = dtime(9, 30)   # inclusive of 09:25 5m bar (close at 09:30)

    # Trading window
    entry_window_start: dtime = dtime(9, 30)
    entry_window_end:   dtime = dtime(11, 15)
    eod_square_off:     dtime = dtime(15, 20)

    # Confirmation
    require_vwap: bool = True
    volume_sma_bars: int = 20
    volume_mult: float = 1.5

    # Range filters
    orb_range_max_pct: float = 0.015   # skip if ORB > 1.5% of price (gap day)
    orb_range_min_pct: float = 0.003   # skip if ORB < 0.3% (no volatility)

    # Risk / exit
    atr_period: int = 14
    sl_use_opposite_orb: bool = True   # SL = opposite ORB end if tighter than ATR
    sl_atr_mult: float = 1.0
    tp1_r_mult: float = 1.5
    tp1_qty_frac: float = 0.5
    trail_on_vwap: bool = True

    # Direction
    enable_long: bool = True
    enable_short: bool = True

    # Penny stock filter — skip entries when close < min_price
    min_price: float = 50.0


@dataclass
class ORBPosition:
    symbol: str
    trend: str         # BUY or SELL
    entry_ts: int
    entry_time: datetime
    entry_price: float
    qty: float
    stop: float
    target: float
    risk_per_share: float
    partial_done: bool = False


@dataclass
class ORBState:
    user_id: int = 0
    symbol: str = ""
    trades_today: int = 0
    last_date: Optional[str] = None
    open_position: Optional[ORBPosition] = None
    closed_count: int = 0


class ORB15MinStrategy:
    """5-min bar evaluator. evaluate(candles=...) where candles are 5m
    SimpleNamespace rows (timestamp, candle_time, open, high, low, close,
    volume) — same shape as the EMA crossover harness."""

    def __init__(self, config: ORBConfig):
        self.config = config
        self._states: Dict[str, ORBState] = {}
        self._signal_log: List[Dict] = []

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
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
        df["date"] = pd.to_datetime(df["candle_time"]).dt.date
        df["time"] = pd.to_datetime(df["candle_time"]).dt.time
        df["vol_sma"] = df["volume"].rolling(cfg.volume_sma_bars, min_periods=1).mean()
        # Per-day cumulative VWAP (resets each new day)
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        df["pv"] = tp * df["volume"]
        df["cum_pv"] = df.groupby("date")["pv"].cumsum()
        df["cum_v"]  = df.groupby("date")["volume"].cumsum().replace(0, 1)
        df["vwap"]   = df["cum_pv"] / df["cum_v"]
        df["atr"] = self._atr(df, cfg.atr_period)
        return df

    def _orb_per_day(self, df: pd.DataFrame) -> Dict:
        """Return {date: (orb_high, orb_low)}. Computed from bars whose
        time falls in [orb_start, orb_end). For a 5m bar the candle_time
        is the open-time; the 09:15, 09:20, 09:25 bars cover 09:15-09:29.
        """
        cfg = self.config
        mask = (df["time"] >= cfg.orb_start) & (df["time"] < cfg.orb_end)
        sub = df[mask].groupby("date").agg(orb_high=("high", "max"),
                                             orb_low=("low", "min")).reset_index()
        return {row.date: (float(row.orb_high), float(row.orb_low))
                for row in sub.itertuples(index=False)}

    def _load_state(self, user_id, symbol):
        key = f"{user_id}:{symbol}"
        if key not in self._states:
            self._states[key] = ORBState(user_id=user_id, symbol=symbol)
        return self._states[key]

    def _record(self, sig: Dict):
        self._signal_log.append(sig)

    def evaluate(self, user_id: int, symbol: str, candles,
                 candles_15m=None, eval_from_ts: Optional[int] = None) -> List[Dict]:
        cfg = self.config
        if len(candles) < cfg.atr_period + 5:
            return []
        df = self._enrich(self._candles_to_df(candles))
        orb_by_day = self._orb_per_day(df)
        state = self._load_state(user_id, symbol)
        signals: List[Dict] = []

        for i in range(cfg.atr_period, len(df)):
            row = df.iloc[i]
            ts = int(row["timestamp"])
            if eval_from_ts is not None and ts < eval_from_ts:
                continue

            d, t = row["date"], row["time"]
            # New day → reset trade counter
            if state.last_date != d:
                state.last_date = d
                state.trades_today = 0
                # Force-close any leftover position from prior day
                if state.open_position is not None:
                    pos = state.open_position
                    sig = {
                        "signal_type": "STOP_HIT", "trend": pos.trend,
                        "candle_ts": ts, "candle_time": row["candle_time"],
                        "price": float(row["open"]), "ema_200": 0.0, "ema_400": 0.0,
                        "note": "EOD overnight gap close"}
                    signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                    state.open_position = None
                    state.closed_count += 1

            # EOD square-off
            if state.open_position is not None and t >= cfg.eod_square_off:
                pos = state.open_position
                exit_price = float(row["close"])
                sig = {
                    "signal_type": "TARGET_HIT" if (
                        (pos.trend == "BUY" and exit_price > pos.entry_price) or
                        (pos.trend == "SELL" and exit_price < pos.entry_price)
                    ) else "STOP_HIT",
                    "trend": pos.trend, "candle_ts": ts,
                    "candle_time": row["candle_time"], "price": exit_price,
                    "ema_200": float(row["vwap"]), "ema_400": 0.0,
                    "note": f"EOD square-off @ {t}"}
                signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                state.open_position = None
                state.closed_count += 1
                continue

            # Manage open position
            if state.open_position is not None:
                pos = state.open_position
                if pos.trend == "BUY":
                    if row["low"] <= pos.stop:
                        sig = {"signal_type": "STOP_HIT", "trend": "BUY",
                               "candle_ts": ts, "candle_time": row["candle_time"],
                               "price": pos.stop, "ema_200": float(row["vwap"]),
                               "ema_400": 0.0, "note": "SL hit"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        state.open_position = None
                        state.closed_count += 1
                        continue
                    if not pos.partial_done and row["high"] >= pos.target:
                        sig = {"signal_type": "PARTIAL", "trend": "BUY",
                               "candle_ts": ts, "candle_time": row["candle_time"],
                               "price": pos.target, "ema_200": float(row["vwap"]),
                               "ema_400": 0.0, "note": f"T1 1.5R @ {pos.target:.2f}"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        pos.qty *= (1.0 - cfg.tp1_qty_frac)
                        pos.partial_done = True
                        pos.stop = pos.entry_price
                    if cfg.trail_on_vwap and pos.partial_done and row["close"] < row["vwap"]:
                        sig = {"signal_type": "TARGET_HIT", "trend": "BUY",
                               "candle_ts": ts, "candle_time": row["candle_time"],
                               "price": float(row["close"]),
                               "ema_200": float(row["vwap"]), "ema_400": 0.0,
                               "note": "Trail-exit close<VWAP"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        state.open_position = None
                        state.closed_count += 1
                        continue
                else:  # SELL
                    if row["high"] >= pos.stop:
                        sig = {"signal_type": "STOP_HIT", "trend": "SELL",
                               "candle_ts": ts, "candle_time": row["candle_time"],
                               "price": pos.stop, "ema_200": float(row["vwap"]),
                               "ema_400": 0.0, "note": "SL hit"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        state.open_position = None
                        state.closed_count += 1
                        continue
                    if not pos.partial_done and row["low"] <= pos.target:
                        sig = {"signal_type": "PARTIAL", "trend": "SELL",
                               "candle_ts": ts, "candle_time": row["candle_time"],
                               "price": pos.target, "ema_200": float(row["vwap"]),
                               "ema_400": 0.0, "note": f"T1 1.5R @ {pos.target:.2f}"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        pos.qty *= (1.0 - cfg.tp1_qty_frac)
                        pos.partial_done = True
                        pos.stop = pos.entry_price
                    if cfg.trail_on_vwap and pos.partial_done and row["close"] > row["vwap"]:
                        sig = {"signal_type": "TARGET_HIT", "trend": "SELL",
                               "candle_ts": ts, "candle_time": row["candle_time"],
                               "price": float(row["close"]),
                               "ema_200": float(row["vwap"]), "ema_400": 0.0,
                               "note": "Trail-exit close>VWAP"}
                        signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})
                        state.open_position = None
                        state.closed_count += 1
                        continue

            # Entry logic — only in window, only if no position, only one per day
            if state.open_position is not None or state.trades_today >= 1:
                continue
            if not (cfg.entry_window_start <= t <= cfg.entry_window_end):
                continue
            if d not in orb_by_day:
                continue
            orb_h, orb_l = orb_by_day[d]
            orb_range_pct = (orb_h - orb_l) / max(row["open"], 1e-9)
            if not (cfg.orb_range_min_pct <= orb_range_pct <= cfg.orb_range_max_pct):
                continue

            close = float(row["close"])
            vwap  = float(row["vwap"])
            vol_ok = row["volume"] >= cfg.volume_mult * row["vol_sma"]
            atr_v = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
            if atr_v <= 0:
                continue

            if close < cfg.min_price:
                continue

            if cfg.enable_long and close > orb_h and (not cfg.require_vwap or close > vwap) and vol_ok:
                # Long entry
                sl_atr = close - cfg.sl_atr_mult * atr_v
                sl = max(orb_l, sl_atr) if cfg.sl_use_opposite_orb else sl_atr
                risk = close - sl
                target = close + cfg.tp1_r_mult * risk
                pos = ORBPosition(symbol=symbol, trend="BUY", entry_ts=ts,
                                   entry_time=row["candle_time"], entry_price=close,
                                   qty=1.0, stop=sl, target=target, risk_per_share=risk)
                state.open_position = pos
                state.trades_today += 1
                sig = {"signal_type": "ENTRY1", "trend": "BUY",
                       "candle_ts": ts, "candle_time": row["candle_time"],
                       "price": close, "sl": sl, "target": target,
                       "ema_200": vwap, "ema_400": 0.0,
                       "note": (f"ORB-long ORB[{orb_l:.2f},{orb_h:.2f}] "
                                f"vol={row['volume']/max(row['vol_sma'],1):.1f}x "
                                f"ATR={atr_v:.2f}")}
                signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})

            elif cfg.enable_short and close < orb_l and (not cfg.require_vwap or close < vwap) and vol_ok:
                sl_atr = close + cfg.sl_atr_mult * atr_v
                sl = min(orb_h, sl_atr) if cfg.sl_use_opposite_orb else sl_atr
                risk = sl - close
                target = close - cfg.tp1_r_mult * risk
                pos = ORBPosition(symbol=symbol, trend="SELL", entry_ts=ts,
                                   entry_time=row["candle_time"], entry_price=close,
                                   qty=1.0, stop=sl, target=target, risk_per_share=risk)
                state.open_position = pos
                state.trades_today += 1
                sig = {"signal_type": "ENTRY1", "trend": "SELL",
                       "candle_ts": ts, "candle_time": row["candle_time"],
                       "price": close, "sl": sl, "target": target,
                       "ema_200": vwap, "ema_400": 0.0,
                       "note": (f"ORB-short ORB[{orb_l:.2f},{orb_h:.2f}] "
                                f"vol={row['volume']/max(row['vol_sma'],1):.1f}x "
                                f"ATR={atr_v:.2f}")}
                signals.append(sig); self._record({**sig, "user_id": user_id, "symbol": symbol})

        return signals

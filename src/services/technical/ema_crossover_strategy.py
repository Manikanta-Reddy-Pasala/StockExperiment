"""
EMA 200/400 Crossover Strategy (1H timeframe).

Trend rule:
    BUY trend  : EMA200 crosses above EMA400.
    SELL trend : EMA200 crosses below EMA400.

Buy setup state machine (long side):
    Stage 0 : Wait for crossover.
    Stage 1 : Crossover candle locked. Watch for break of crossover candle high (Alert 1).
    Stage 2 : Alert 1 fired. Watch for retest of EMA200 (close below + close, locks retest1).
    Stage 3 : Retest1 locked. If price > retest1.high and sustains N minutes -> ENTRY 1.
              Then watch EMA400 touch.
    Stage 4 : EMA400 touched / crossed below. Watch for break of new retest candle high
              that sustains N minutes -> ENTRY 2.
    Exit    : Any 1H close below EMA400 -> exit longs, reset to Stage 0.

Sell setup is the mirror image (price below crossover candle low, retest of EMA200 from
below, EMA400 touch from below, exits on close above EMA400).

Multiple entries allowed (pyramid). Single SL: 1H close on the wrong side of EMA400.
Target: 5000 absolute points for index symbols, or 1:3 RR for equities (configurable).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

try:
    from ...models.database import get_database_manager
    from ...models.historical_models import (
        HistoricalData1H,
        EMACrossoverState,
        EMACrossoverSignal,
    )
except ImportError:
    from src.models.database import get_database_manager
    from src.models.historical_models import (
        HistoricalData1H,
        EMACrossoverState,
        EMACrossoverSignal,
    )

logger = logging.getLogger(__name__)

INDEX_SYMBOLS = {"NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:BANKNIFTY-INDEX"}
DEFAULT_TARGET_POINTS = 5000.0
DEFAULT_RR_MULTIPLE = 3.0
DEFAULT_SUSTAIN_MINUTES = 15  # On 1H timeframe, "sustain" granularity is informational


@dataclass
class StrategyConfig:
    target_points: float = DEFAULT_TARGET_POINTS
    rr_multiple: float = DEFAULT_RR_MULTIPLE
    sustain_minutes: int = DEFAULT_SUSTAIN_MINUTES
    ema_fast_period: int = 200
    ema_slow_period: int = 400


class EMACrossoverStrategy:
    """1H EMA200/400 crossover strategy with multi-stage entry state machine."""

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.db = get_database_manager()

    # ------------------------------------------------------------------
    # Indicator math
    # ------------------------------------------------------------------
    @staticmethod
    def compute_emas(df: pd.DataFrame, fast: int = 200, slow: int = 400) -> pd.DataFrame:
        df = df.copy()
        df["ema_200"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_400"] = df["close"].ewm(span=slow, adjust=False).mean()
        return df

    # ------------------------------------------------------------------
    # Public entry: evaluate one symbol given its 1H candles (ascending order)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_id: int,
        symbol: str,
        candles: List[HistoricalData1H],
    ) -> List[dict]:
        if len(candles) < self.config.ema_slow_period + 5:
            logger.debug(
                f"{symbol}: only {len(candles)} candles, need >={self.config.ema_slow_period + 5}"
            )
            return []

        df = self._candles_to_df(candles)
        df = self.compute_emas(df, self.config.ema_fast_period, self.config.ema_slow_period)

        state = self._load_state(user_id, symbol)
        signals: List[dict] = []

        # Replay only candles after `last_evaluated_ts` to keep the state machine
        # incremental; on a fresh state we replay the full window for backfill.
        start_idx = 0
        if state.last_evaluated_ts:
            mask = df["timestamp"] > state.last_evaluated_ts
            if mask.any():
                start_idx = int(df[mask].index.min())
            else:
                start_idx = len(df)

        # Walk every candle past start_idx
        for i in range(max(start_idx, self.config.ema_slow_period), len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            new_signals = self._step_machine(state, row, prev, df, i)
            for sig in new_signals:
                signals.append(sig)
            state.last_evaluated_ts = int(row["timestamp"])

        self._save_state(state)
        for sig in signals:
            self._record_signal(user_id, symbol, sig)
        return signals

    # ------------------------------------------------------------------
    # State machine core (single-bar step)
    # ------------------------------------------------------------------
    def _step_machine(
        self,
        state: EMACrossoverState,
        row: pd.Series,
        prev: pd.Series,
        df: pd.DataFrame,
        idx: int,
    ) -> List[dict]:
        signals: List[dict] = []

        ema200 = row["ema_200"]
        ema400 = row["ema_400"]
        prev_ema200 = prev["ema_200"]
        prev_ema400 = prev["ema_400"]

        cross_up = prev_ema200 <= prev_ema400 and ema200 > ema400
        cross_dn = prev_ema200 >= prev_ema400 and ema200 < ema400

        # ---- Trend reset on opposite crossover ----
        if cross_up:
            self._reset_state(state, "BUY", row)
            signals.append(self._make_signal(row, "CROSSOVER", "BUY", note="EMA200 above EMA400"))
            return signals
        if cross_dn:
            self._reset_state(state, "SELL", row)
            signals.append(self._make_signal(row, "CROSSOVER", "SELL", note="EMA200 below EMA400"))
            return signals

        # No active trend yet — nothing to do.
        if state.trend == "NONE":
            return signals

        # Stop-loss / exit check (1H close on wrong side of EMA400) -------------------
        if state.position_active:
            if state.trend == "BUY" and row["close"] < ema400:
                signals.append(self._make_signal(row, "EXIT", state.trend, note="Close below EMA400"))
                self._reset_state(state, "NONE", row)
                return signals
            if state.trend == "SELL" and row["close"] > ema400:
                signals.append(self._make_signal(row, "EXIT", state.trend, note="Close above EMA400"))
                self._reset_state(state, "NONE", row)
                return signals

        # ---- Stage transitions per trend direction ----
        if state.trend == "BUY":
            signals.extend(self._step_buy(state, row, prev))
        elif state.trend == "SELL":
            signals.extend(self._step_sell(state, row, prev))

        return signals

    # ------------------------------------------------------------------
    def _step_buy(self, state: EMACrossoverState, row: pd.Series, prev: pd.Series) -> List[dict]:
        signals: List[dict] = []
        ema200 = row["ema_200"]
        ema400 = row["ema_400"]

        # Stage 1: Alert 1 — break of crossover candle high + close above
        if state.stage == 1 and state.crossover_high is not None:
            if row["close"] > state.crossover_high and row["high"] > state.crossover_high:
                state.stage = 2
                signals.append(self._make_signal(row, "ALERT1", "BUY",
                                                 note="Break + close above crossover candle high"))

        # Stage 2: Alert 2 — retest of EMA200 (close below EMA200)
        if state.stage == 2:
            if row["close"] < ema200 and row["low"] < ema200:
                state.retest1_ts = int(row["timestamp"])
                state.retest1_high = float(row["high"])
                state.retest1_low = float(row["low"])
                state.stage = 3
                signals.append(self._make_signal(row, "ALERT2", "BUY",
                                                 note="EMA200 retest candle locked"))

        # Stage 3: Entry 1 — break retest1 high (sustain handled at the 1H close)
        if state.stage == 3 and state.retest1_high is not None:
            if row["close"] > state.retest1_high:
                state.entries_count = (state.entries_count or 0) + 1
                state.entry1_price = float(row["close"])
                state.entry1_time = row["candle_time"]
                state.position_active = True
                state.stop_loss = float(ema400)
                state.target_price = self._calc_target(row["close"], ema400, "BUY", state.symbol)
                state.stage = 4
                signals.append(self._make_signal(row, "ENTRY1", "BUY",
                                                 price=row["close"],
                                                 note="Buy entry 1 (retest1 break)"))

        # Stage 4: Alert 3 — EMA400 touch / cross below
        if state.stage == 4:
            if row["low"] <= ema400:
                state.retest2_ts = int(row["timestamp"])
                state.retest2_high = float(row["high"])
                state.retest2_low = float(row["low"])
                state.stage = 5
                signals.append(self._make_signal(row, "ALERT3", "BUY",
                                                 note="EMA400 retest candle locked"))

        # Stage 5: Entry 2 — break retest2 high
        if state.stage == 5 and state.retest2_high is not None:
            if row["close"] > state.retest2_high:
                state.entries_count = (state.entries_count or 0) + 1
                state.entry2_price = float(row["close"])
                state.entry2_time = row["candle_time"]
                state.stage = 4  # allow further EMA400 retests/entries (pyramid)
                signals.append(self._make_signal(row, "ENTRY2", "BUY",
                                                 price=row["close"],
                                                 note="Buy entry 2 (retest2 break)"))
        return signals

    # ------------------------------------------------------------------
    def _step_sell(self, state: EMACrossoverState, row: pd.Series, prev: pd.Series) -> List[dict]:
        signals: List[dict] = []
        ema200 = row["ema_200"]
        ema400 = row["ema_400"]

        if state.stage == 1 and state.crossover_low is not None:
            if row["close"] < state.crossover_low and row["low"] < state.crossover_low:
                state.stage = 2
                signals.append(self._make_signal(row, "ALERT1", "SELL",
                                                 note="Break + close below crossover candle low"))

        if state.stage == 2:
            if row["close"] > ema200 and row["high"] > ema200:
                state.retest1_ts = int(row["timestamp"])
                state.retest1_high = float(row["high"])
                state.retest1_low = float(row["low"])
                state.stage = 3
                signals.append(self._make_signal(row, "ALERT2", "SELL",
                                                 note="EMA200 retest candle locked"))

        if state.stage == 3 and state.retest1_low is not None:
            if row["close"] < state.retest1_low:
                state.entries_count = (state.entries_count or 0) + 1
                state.entry1_price = float(row["close"])
                state.entry1_time = row["candle_time"]
                state.position_active = True
                state.stop_loss = float(ema400)
                state.target_price = self._calc_target(row["close"], ema400, "SELL", state.symbol)
                state.stage = 4
                signals.append(self._make_signal(row, "ENTRY1", "SELL",
                                                 price=row["close"],
                                                 note="Sell entry 1 (retest1 break)"))

        if state.stage == 4:
            if row["high"] >= ema400:
                state.retest2_ts = int(row["timestamp"])
                state.retest2_high = float(row["high"])
                state.retest2_low = float(row["low"])
                state.stage = 5
                signals.append(self._make_signal(row, "ALERT3", "SELL",
                                                 note="EMA400 retest candle locked"))

        if state.stage == 5 and state.retest2_low is not None:
            if row["close"] < state.retest2_low:
                state.entries_count = (state.entries_count or 0) + 1
                state.entry2_price = float(row["close"])
                state.entry2_time = row["candle_time"]
                state.stage = 4
                signals.append(self._make_signal(row, "ENTRY2", "SELL",
                                                 price=row["close"],
                                                 note="Sell entry 2 (retest2 break)"))
        return signals

    # ------------------------------------------------------------------
    def _calc_target(self, entry_price: float, ema400: float, trend: str, symbol: str) -> float:
        if symbol in INDEX_SYMBOLS:
            return entry_price + self.config.target_points if trend == "BUY" \
                else entry_price - self.config.target_points
        risk = abs(entry_price - ema400)
        if trend == "BUY":
            return entry_price + risk * self.config.rr_multiple
        return entry_price - risk * self.config.rr_multiple

    def _reset_state(self, state: EMACrossoverState, trend: str, row: pd.Series) -> None:
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
    def _make_signal(row: pd.Series, signal_type: str, trend: str,
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
                state = EMACrossoverState(user_id=user_id, symbol=symbol, trend="NONE", stage=0)
                session.add(state)
                session.commit()
            session.expunge(state)
        return state

    def _save_state(self, state: EMACrossoverState) -> None:
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

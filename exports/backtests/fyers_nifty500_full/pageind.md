# Page Industries Ltd. (PAGEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 36780.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -4084.65
- **Avg P&L per closed trade:** -1361.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 42850.00 | 45907.27 | 45912.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 42316.90 | 45702.80 | 45808.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 41600.00 | 41531.62 | 42850.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-27 15:15:00 | 41000.00 | 41860.90 | 42791.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 41000.00 | 41860.90 | 42791.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-28 09:15:00 | 42969.65 | 41871.93 | 42792.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 45970.00 | 43164.18 | 43158.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 47015.00 | 44156.26 | 43721.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 46080.00 | 46086.16 | 45103.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-29 14:15:00 | 46575.00 | 46084.37 | 45136.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 45530.00 | 46122.03 | 45514.46 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 10:15:00 | 45200.00 | 46112.85 | 45512.89 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 43840.00 | 46509.07 | 46516.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 43630.00 | 46480.42 | 46502.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 09:15:00 | 44855.00 | 46010.48 | 46207.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 45485.00 | 45107.44 | 45569.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-17 11:15:00 | 45595.00 | 45115.71 | 45569.17 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 38120.00 | 33873.65 | 33859.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 38230.00 | 33917.00 | 33881.54 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-27 15:15:00 | 41000.00 | 2025-03-28 09:15:00 | 42969.65 | EXIT_EMA400 | -1969.65 |
| BUY | 2025-05-29 14:15:00 | 46575.00 | 2025-06-18 10:15:00 | 45200.00 | EXIT_EMA400 | -1375.00 |
| SELL | 2025-08-28 09:15:00 | 44855.00 | 2025-09-17 11:15:00 | 45595.00 | EXIT_EMA400 | -740.00 |

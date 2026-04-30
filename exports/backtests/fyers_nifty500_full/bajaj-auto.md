# Bajaj Auto Ltd. (BAJAJ-AUTO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 10039.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** 505.67
- **Avg P&L per closed trade:** 84.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 9416.60 | 10783.01 | 10789.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 9200.00 | 10066.48 | 10350.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 8775.00 | 8707.47 | 9054.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 10:15:00 | 8573.00 | 8788.61 | 8986.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 8111.50 | 7883.04 | 8128.72 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 10:15:00 | 8178.00 | 7885.98 | 8128.96 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 8706.00 | 8165.04 | 8163.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 8744.00 | 8170.81 | 8165.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 15:15:00 | 8482.50 | 8494.19 | 8379.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-20 09:15:00 | 8327.00 | 8499.15 | 8398.10 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.98 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 8444.50 | 8364.81 | 8364.42 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8301.00 | 8363.71 | 8363.89 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 8397.00 | 8364.26 | 8364.16 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 8285.50 | 8363.43 | 8363.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 8060.00 | 8359.67 | 8361.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 8245.00 | 8242.30 | 8291.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-08 13:15:00 | 8203.50 | 8241.51 | 8289.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 8274.00 | 8238.95 | 8286.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-11 15:15:00 | 8235.00 | 8238.91 | 8286.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-12 09:15:00 | 8298.00 | 8239.50 | 8286.37 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8826.50 | 8322.29 | 8321.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.25 | 8403.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.41 | 8716.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 12:15:00 | 8948.50 | 8830.95 | 8733.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 8885.00 | 8971.79 | 8857.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-04 09:15:00 | 8798.00 | 8963.07 | 8859.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 9048.50 | 9437.80 | 9439.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9428.24 | 9434.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 9770.00 | 9375.89 | 9374.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 9799.00 | 9398.74 | 9386.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 9490.00 | 9492.89 | 9442.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-29 09:15:00 | 9644.50 | 9494.30 | 9443.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.53 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-14 10:15:00 | 8573.00 | 2025-03-04 13:15:00 | 7333.33 | TARGET | 1239.67 |
| BUY | 2025-06-13 15:15:00 | 8482.50 | 2025-06-20 09:15:00 | 8327.00 | EXIT_EMA400 | -155.50 |
| SELL | 2025-08-08 13:15:00 | 8203.50 | 2025-08-12 09:15:00 | 8298.00 | EXIT_EMA400 | -94.50 |
| SELL | 2025-08-11 15:15:00 | 8235.00 | 2025-08-12 09:15:00 | 8298.00 | EXIT_EMA400 | -63.00 |
| BUY | 2025-10-10 12:15:00 | 8948.50 | 2025-11-04 09:15:00 | 8798.00 | EXIT_EMA400 | -150.50 |
| BUY | 2026-04-29 09:15:00 | 9644.50 | 2026-04-30 09:15:00 | 9374.00 | EXIT_EMA400 | -270.50 |

# NMDC Ltd. (NMDC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 90.37
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -1.85
- **Avg P&L per closed trade:** -0.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 77.28 | 83.04 | 83.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 76.92 | 82.87 | 82.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 82.77 | 81.70 | 82.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-01 11:15:00 | 80.97 | 81.69 | 82.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-25 09:15:00 | 75.80 | 73.03 | 75.30 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 77.30 | 76.05 | 76.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 14:15:00 | 77.82 | 76.07 | 76.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 09:15:00 | 75.52 | 76.08 | 76.06 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 74.72 | 76.03 | 76.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 73.87 | 76.01 | 76.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 75.34 | 75.28 | 75.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 11:15:00 | 74.75 | 75.28 | 75.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 13:15:00 | 75.92 | 75.28 | 75.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 77.91 | 75.87 | 75.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 15:15:00 | 78.10 | 75.93 | 75.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 77.03 | 77.34 | 76.68 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 71.06 | 76.19 | 76.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 10:15:00 | 70.55 | 75.85 | 76.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 68.20 | 68.08 | 70.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 66.68 | 68.07 | 70.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 66.57 | 65.06 | 67.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-21 12:15:00 | 67.55 | 65.12 | 67.40 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 70.02 | 66.51 | 66.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 15:15:00 | 70.32 | 66.70 | 66.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 70.68 | 70.72 | 69.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 71.44 | 69.86 | 69.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 09:15:00 | 67.73 | 69.89 | 69.23 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 75.19 | 79.84 | 79.86 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 13:15:00 | 84.39 | 79.69 | 79.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 84.71 | 79.84 | 79.76 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-01 11:15:00 | 80.97 | 2024-08-05 09:15:00 | 77.06 | TARGET | 3.90 |
| SELL | 2024-11-25 11:15:00 | 74.75 | 2024-11-25 13:15:00 | 75.92 | EXIT_EMA400 | -1.17 |
| SELL | 2025-01-24 09:15:00 | 66.68 | 2025-02-21 12:15:00 | 67.55 | EXIT_EMA400 | -0.87 |
| BUY | 2025-06-27 09:15:00 | 71.44 | 2025-07-01 09:15:00 | 67.73 | EXIT_EMA400 | -3.71 |

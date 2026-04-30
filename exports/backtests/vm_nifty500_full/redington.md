# Redington Ltd. (REDINGTON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 215.87
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 11.27
- **Avg P&L per closed trade:** 1.88

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 12:15:00 | 165.00 | 156.93 | 156.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 12:15:00 | 166.95 | 157.51 | 157.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 175.05 | 175.12 | 170.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-24 09:15:00 | 176.70 | 175.14 | 170.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 193.30 | 200.29 | 191.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 10:15:00 | 191.35 | 200.20 | 191.43 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 202.39 | 210.55 | 210.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 200.48 | 210.31 | 210.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 13:15:00 | 210.01 | 209.96 | 210.28 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 217.02 | 210.56 | 210.55 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 204.19 | 210.51 | 210.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 196.99 | 209.87 | 210.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 203.23 | 203.23 | 206.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-16 14:15:00 | 202.44 | 203.22 | 206.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 205.09 | 203.25 | 206.08 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-20 10:15:00 | 208.21 | 203.30 | 206.09 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 197.63 | 190.27 | 190.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 202.17 | 190.46 | 190.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 202.70 | 203.05 | 198.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 15:15:00 | 206.00 | 201.67 | 198.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 204.83 | 202.26 | 199.53 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-16 13:15:00 | 211.18 | 202.89 | 200.09 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 207.22 | 210.19 | 204.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-28 10:15:00 | 204.73 | 210.13 | 204.81 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 242.85 | 281.46 | 281.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 237.70 | 280.64 | 281.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 266.00 | 248.62 | 258.07 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 287.84 | 265.40 | 265.33 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 253.05 | 267.50 | 267.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 13:15:00 | 252.40 | 266.72 | 267.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 277.45 | 265.52 | 266.52 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 289.60 | 267.61 | 267.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 298.80 | 271.35 | 269.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 281.20 | 282.62 | 277.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 09:15:00 | 285.85 | 282.49 | 277.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 277.80 | 282.40 | 277.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-03 10:15:00 | 277.05 | 282.34 | 277.51 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 272.00 | 275.38 | 275.39 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 280.10 | 275.42 | 275.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 289.60 | 275.66 | 275.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.56 | EMA200 retest candle locked |

### Cycle 12 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 267.45 | 275.69 | 275.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 264.55 | 275.22 | 275.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 269.40 | 268.35 | 271.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 266.00 | 268.37 | 271.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 274.20 | 268.26 | 271.30 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-24 09:15:00 | 176.70 | 2024-02-05 09:15:00 | 195.69 | TARGET | 18.99 |
| SELL | 2024-08-16 14:15:00 | 202.44 | 2024-08-20 10:15:00 | 208.21 | EXIT_EMA400 | -5.77 |
| BUY | 2025-01-07 15:15:00 | 206.00 | 2025-01-21 09:15:00 | 227.50 | TARGET | 21.50 |
| BUY | 2025-01-16 13:15:00 | 211.18 | 2025-01-28 10:15:00 | 204.73 | EXIT_EMA400 | -6.45 |
| BUY | 2025-12-01 09:15:00 | 285.85 | 2025-12-03 10:15:00 | 277.05 | EXIT_EMA400 | -8.80 |
| SELL | 2026-02-02 09:15:00 | 266.00 | 2026-02-03 09:15:00 | 274.20 | EXIT_EMA400 | -8.20 |

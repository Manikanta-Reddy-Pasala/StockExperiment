# Bank of Baroda (BANKBARODA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 263.46
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -24.27
- **Avg P&L per closed trade:** -2.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 15:15:00 | 190.60 | 192.49 | 192.49 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 14:15:00 | 195.40 | 192.52 | 192.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 15:15:00 | 195.90 | 192.55 | 192.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 10:15:00 | 207.55 | 208.20 | 202.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-12 09:15:00 | 208.90 | 208.18 | 203.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-13 09:15:00 | 202.80 | 208.12 | 203.24 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 10:15:00 | 192.00 | 201.15 | 201.17 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 13:15:00 | 209.80 | 200.19 | 200.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 10:15:00 | 211.70 | 200.57 | 200.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 221.80 | 222.14 | 214.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 224.75 | 222.22 | 215.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 251.90 | 264.07 | 251.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-15 11:15:00 | 251.35 | 263.94 | 251.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 256.25 | 267.58 | 267.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 255.35 | 267.11 | 267.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 252.25 | 250.69 | 256.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-04 09:15:00 | 246.15 | 251.24 | 254.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 247.70 | 243.44 | 248.10 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-27 14:15:00 | 249.70 | 243.62 | 248.10 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 11:15:00 | 264.00 | 248.02 | 247.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 265.10 | 248.19 | 248.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 247.30 | 250.14 | 249.14 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 224.55 | 248.09 | 248.19 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 259.52 | 247.83 | 247.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 264.56 | 248.82 | 248.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 252.12 | 252.60 | 250.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 11:15:00 | 254.52 | 252.62 | 250.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 252.18 | 253.31 | 251.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-18 13:15:00 | 250.50 | 253.28 | 251.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 240.05 | 249.46 | 249.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 09:15:00 | 237.13 | 248.85 | 249.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 209.27 | 208.90 | 217.45 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 231.51 | 221.37 | 221.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 233.73 | 221.49 | 221.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 223.66 | 238.20 | 231.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 13:15:00 | 243.65 | 235.98 | 232.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 239.50 | 242.82 | 238.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 12:15:00 | 237.54 | 242.69 | 238.08 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 234.42 | 240.31 | 240.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 233.89 | 239.71 | 240.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 10:15:00 | 236.86 | 238.64 | 239.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 238.13 | 238.55 | 239.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 11:15:00 | 239.41 | 238.56 | 239.27 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 252.50 | 239.91 | 239.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 257.25 | 242.25 | 241.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 284.40 | 284.58 | 274.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 10:15:00 | 287.00 | 284.60 | 275.21 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-02 09:15:00 | 274.95 | 298.99 | 292.09 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 271.90 | 293.73 | 293.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 268.55 | 292.42 | 293.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 09:15:00 | 277.81 | 276.55 | 283.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 269.83 | 276.39 | 283.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 279.80 | 276.29 | 282.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-15 10:15:00 | 278.06 | 276.31 | 282.79 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 279.75 | 276.92 | 282.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 283.40 | 277.06 | 282.48 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-12 09:15:00 | 208.90 | 2023-10-13 09:15:00 | 202.80 | EXIT_EMA400 | -6.10 |
| BUY | 2024-01-11 09:15:00 | 224.75 | 2024-02-01 12:15:00 | 253.77 | TARGET | 29.02 |
| SELL | 2024-09-04 09:15:00 | 246.15 | 2024-09-27 14:15:00 | 249.70 | EXIT_EMA400 | -3.55 |
| BUY | 2024-12-13 11:15:00 | 254.52 | 2024-12-18 13:15:00 | 250.50 | EXIT_EMA400 | -4.02 |
| BUY | 2025-05-23 13:15:00 | 243.65 | 2025-06-13 12:15:00 | 237.54 | EXIT_EMA400 | -6.11 |
| SELL | 2025-09-12 10:15:00 | 236.86 | 2025-09-16 11:15:00 | 239.41 | EXIT_EMA400 | -2.55 |
| BUY | 2025-12-09 10:15:00 | 287.00 | 2026-02-02 09:15:00 | 274.95 | EXIT_EMA400 | -12.05 |
| SELL | 2026-04-13 09:15:00 | 269.83 | 2026-04-21 09:15:00 | 283.40 | EXIT_EMA400 | -13.57 |
| SELL | 2026-04-15 10:15:00 | 278.06 | 2026-04-21 09:15:00 | 283.40 | EXIT_EMA400 | -5.34 |

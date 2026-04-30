# NLC India Ltd. (NLCINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 316.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -43.00
- **Avg P&L per closed trade:** -5.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 11:15:00 | 199.30 | 227.91 | 227.94 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 11:15:00 | 233.13 | 227.68 | 227.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 236.98 | 228.15 | 227.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 269.00 | 271.28 | 257.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-22 09:15:00 | 274.35 | 268.05 | 260.29 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-09 09:15:00 | 261.50 | 272.19 | 265.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 243.55 | 267.84 | 267.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 237.00 | 265.67 | 266.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 260.30 | 259.94 | 263.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 253.75 | 259.84 | 262.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 255.65 | 252.02 | 257.81 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-22 09:15:00 | 257.85 | 252.08 | 257.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 11:15:00 | 248.68 | 229.50 | 229.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 253.24 | 230.45 | 229.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 221.21 | 235.44 | 232.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 10:15:00 | 224.35 | 235.33 | 232.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 224.35 | 235.33 | 232.80 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 11:15:00 | 221.85 | 235.19 | 232.75 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 214.68 | 232.82 | 232.90 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 235.85 | 232.83 | 232.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 248.09 | 233.06 | 232.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 235.76 | 236.81 | 235.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 13:15:00 | 238.03 | 236.67 | 235.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 235.48 | 236.76 | 235.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-06 13:15:00 | 236.30 | 236.75 | 235.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 235.42 | 236.73 | 235.34 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-09 09:15:00 | 238.95 | 236.75 | 235.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-12 13:15:00 | 235.52 | 237.73 | 236.05 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 10:15:00 | 224.04 | 234.70 | 234.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 222.11 | 234.46 | 234.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 229.61 | 229.23 | 231.20 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 246.22 | 232.77 | 232.71 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 232.85 | 234.25 | 234.25 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 236.63 | 234.25 | 234.25 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 232.73 | 234.24 | 234.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 229.67 | 234.16 | 234.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 238.22 | 233.62 | 233.93 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 243.05 | 234.24 | 234.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 246.32 | 234.36 | 234.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 265.45 | 265.50 | 255.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 14:15:00 | 268.45 | 265.53 | 255.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-29 13:15:00 | 254.50 | 264.63 | 257.70 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 246.35 | 255.92 | 255.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 245.65 | 255.49 | 255.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 246.60 | 246.23 | 250.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 12:15:00 | 243.75 | 246.22 | 250.09 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 11:15:00 | 250.30 | 244.28 | 248.30 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 273.65 | 250.95 | 250.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 10:15:00 | 275.70 | 260.52 | 258.11 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-08-22 09:15:00 | 274.35 | 2024-09-09 09:15:00 | 261.50 | EXIT_EMA400 | -12.85 |
| SELL | 2024-11-08 10:15:00 | 253.75 | 2024-11-22 09:15:00 | 257.85 | EXIT_EMA400 | -4.10 |
| BUY | 2025-04-07 10:15:00 | 224.35 | 2025-04-07 11:15:00 | 221.85 | EXIT_EMA400 | -2.50 |
| BUY | 2025-06-06 13:15:00 | 236.30 | 2025-06-09 09:15:00 | 239.19 | TARGET | 2.89 |
| BUY | 2025-06-04 13:15:00 | 238.03 | 2025-06-12 13:15:00 | 235.52 | EXIT_EMA400 | -2.51 |
| BUY | 2025-06-09 09:15:00 | 238.95 | 2025-06-12 13:15:00 | 235.52 | EXIT_EMA400 | -3.43 |
| BUY | 2025-10-15 14:15:00 | 268.45 | 2025-10-29 13:15:00 | 254.50 | EXIT_EMA400 | -13.95 |
| SELL | 2025-12-15 12:15:00 | 243.75 | 2025-12-23 11:15:00 | 250.30 | EXIT_EMA400 | -6.55 |

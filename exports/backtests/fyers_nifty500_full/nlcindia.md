# NLC India Ltd. (NLCINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 316.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -5.96
- **Avg P&L per closed trade:** -0.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 249.50 | 267.18 | 267.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 13:15:00 | 244.40 | 266.39 | 266.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 260.30 | 259.88 | 262.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 253.75 | 259.81 | 262.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 255.65 | 251.99 | 257.60 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-22 09:15:00 | 257.85 | 252.05 | 257.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 09:15:00 | 251.25 | 229.05 | 229.01 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 214.68 | 232.80 | 232.82 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 240.01 | 232.72 | 232.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 248.05 | 233.05 | 232.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 235.76 | 236.81 | 235.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 13:15:00 | 238.03 | 236.67 | 235.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 235.48 | 236.75 | 235.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-06 13:15:00 | 236.30 | 236.75 | 235.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 235.70 | 236.72 | 235.30 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-09 09:15:00 | 238.95 | 236.75 | 235.32 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-12 13:15:00 | 235.52 | 237.73 | 236.01 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 222.81 | 234.59 | 234.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 222.11 | 234.47 | 234.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 229.61 | 229.23 | 231.18 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 245.14 | 232.64 | 232.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 246.22 | 232.77 | 232.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 233.61 | 235.12 | 234.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-04 09:15:00 | 238.65 | 235.15 | 234.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 238.65 | 235.15 | 234.04 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 12:15:00 | 240.40 | 235.26 | 234.11 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 234.72 | 235.95 | 234.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-06 11:15:00 | 234.00 | 235.93 | 234.53 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 232.85 | 234.24 | 234.24 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 236.63 | 234.24 | 234.24 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 232.78 | 234.23 | 234.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 229.67 | 234.15 | 234.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 238.22 | 233.62 | 233.92 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 243.07 | 234.23 | 234.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 246.38 | 234.35 | 234.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 265.35 | 265.54 | 255.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-20 14:15:00 | 266.85 | 265.36 | 256.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-29 13:15:00 | 254.65 | 264.59 | 257.68 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 246.35 | 255.91 | 255.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 245.65 | 255.48 | 255.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 246.60 | 246.24 | 250.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 12:15:00 | 243.75 | 246.23 | 250.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 11:15:00 | 250.30 | 244.28 | 248.29 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 273.65 | 250.93 | 250.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 10:15:00 | 275.70 | 260.52 | 258.12 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 10:15:00 | 253.75 | 2024-11-18 09:15:00 | 226.99 | TARGET | 26.76 |
| BUY | 2025-06-06 13:15:00 | 236.30 | 2025-06-09 09:15:00 | 239.32 | TARGET | 3.02 |
| BUY | 2025-06-04 13:15:00 | 238.03 | 2025-06-12 13:15:00 | 235.52 | EXIT_EMA400 | -2.51 |
| BUY | 2025-06-09 09:15:00 | 238.95 | 2025-06-12 13:15:00 | 235.52 | EXIT_EMA400 | -3.43 |
| BUY | 2025-08-04 09:15:00 | 238.65 | 2025-08-06 11:15:00 | 234.00 | EXIT_EMA400 | -4.65 |
| BUY | 2025-08-04 12:15:00 | 240.40 | 2025-08-06 11:15:00 | 234.00 | EXIT_EMA400 | -6.40 |
| BUY | 2025-10-20 14:15:00 | 266.85 | 2025-10-29 13:15:00 | 254.65 | EXIT_EMA400 | -12.20 |
| SELL | 2025-12-15 12:15:00 | 243.75 | 2025-12-23 11:15:00 | 250.30 | EXIT_EMA400 | -6.55 |

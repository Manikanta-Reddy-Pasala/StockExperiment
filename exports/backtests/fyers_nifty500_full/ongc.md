# Oil & Natural Gas Corporation Ltd. (ONGC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 299.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -17.31
- **Avg P&L per closed trade:** -2.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 293.95 | 305.78 | 305.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 290.90 | 301.13 | 303.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 263.60 | 261.78 | 272.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 11:15:00 | 261.20 | 261.84 | 272.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 12:15:00 | 259.73 | 248.83 | 258.83 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 251.84 | 241.87 | 241.84 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 236.99 | 242.11 | 242.12 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 247.86 | 242.07 | 242.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 251.65 | 242.27 | 242.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 241.10 | 243.84 | 243.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 240.29 | 243.80 | 243.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 239.82 | 239.52 | 241.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-13 13:15:00 | 238.93 | 239.51 | 241.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 240.43 | 239.03 | 240.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-22 11:15:00 | 236.75 | 238.99 | 240.70 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 241.38 | 237.78 | 239.73 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 243.74 | 238.90 | 238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 246.35 | 239.42 | 239.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.20 | 248.70 | 245.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 15:15:00 | 249.90 | 248.69 | 245.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.14 | 245.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-19 13:15:00 | 249.80 | 248.89 | 246.05 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 246.35 | 248.79 | 246.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 10:15:00 | 245.90 | 248.74 | 246.22 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 240.20 | 244.86 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 238.73 | 244.64 | 244.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-01 12:15:00 | 237.93 | 238.73 | 241.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.96 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.57 | 241.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.08 | 241.84 | 241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.70 | 261.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-11 11:15:00 | 272.00 | 269.71 | 261.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.46 | 262.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-04 11:15:00 | 261.20 | 2025-01-03 12:15:00 | 259.73 | EXIT_EMA400 | 1.47 |
| SELL | 2025-08-13 13:15:00 | 238.93 | 2025-08-29 09:15:00 | 231.76 | TARGET | 7.17 |
| SELL | 2025-08-22 11:15:00 | 236.75 | 2025-09-02 09:15:00 | 241.38 | EXIT_EMA400 | -4.63 |
| BUY | 2025-11-11 15:15:00 | 249.90 | 2025-11-24 10:15:00 | 245.90 | EXIT_EMA400 | -4.00 |
| BUY | 2025-11-19 13:15:00 | 249.80 | 2025-11-24 10:15:00 | 245.90 | EXIT_EMA400 | -3.90 |
| SELL | 2026-01-01 12:15:00 | 237.93 | 2026-01-02 10:15:00 | 241.55 | EXIT_EMA400 | -3.62 |
| BUY | 2026-03-11 11:15:00 | 272.00 | 2026-03-16 10:15:00 | 262.20 | EXIT_EMA400 | -9.80 |

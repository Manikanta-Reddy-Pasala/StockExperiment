# Oil & Natural Gas Corporation Ltd. (ONGC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 299.55
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
- **Total realized P&L (per unit):** -15.63
- **Avg P&L per closed trade:** -2.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 292.00 | 305.92 | 305.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 284.35 | 300.97 | 303.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 263.60 | 261.82 | 273.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 11:15:00 | 261.30 | 261.88 | 272.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 12:15:00 | 259.80 | 248.83 | 258.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 251.84 | 241.88 | 241.85 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 236.99 | 242.10 | 242.12 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 247.57 | 242.11 | 242.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 251.70 | 242.26 | 242.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 241.10 | 243.84 | 243.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 240.41 | 243.80 | 243.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 239.81 | 239.52 | 241.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-13 13:15:00 | 238.93 | 239.51 | 241.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 240.43 | 239.03 | 240.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-22 11:15:00 | 236.75 | 238.99 | 240.70 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 241.35 | 237.77 | 239.72 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 243.74 | 238.90 | 238.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 246.43 | 239.42 | 239.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.30 | 248.70 | 245.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 14:15:00 | 249.50 | 248.68 | 245.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.12 | 245.78 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-19 13:15:00 | 249.85 | 248.88 | 246.05 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 246.35 | 248.79 | 246.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 10:15:00 | 245.90 | 248.74 | 246.22 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 240.15 | 244.87 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 238.71 | 244.65 | 244.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-01 12:15:00 | 238.02 | 238.74 | 241.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 10:15:00 | 241.52 | 238.76 | 240.96 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.59 | 241.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.00 | 241.85 | 241.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.54 | 261.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 10:15:00 | 269.70 | 269.54 | 261.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.33 | 262.14 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-16 11:15:00 | 261.05 | 269.18 | 262.13 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-04 11:15:00 | 261.30 | 2025-01-03 12:15:00 | 259.80 | EXIT_EMA400 | 1.50 |
| SELL | 2025-08-13 13:15:00 | 238.93 | 2025-08-29 09:15:00 | 231.76 | TARGET | 7.17 |
| SELL | 2025-08-22 11:15:00 | 236.75 | 2025-09-02 09:15:00 | 241.35 | EXIT_EMA400 | -4.60 |
| BUY | 2025-11-11 14:15:00 | 249.50 | 2025-11-24 10:15:00 | 245.90 | EXIT_EMA400 | -3.60 |
| BUY | 2025-11-19 13:15:00 | 249.85 | 2025-11-24 10:15:00 | 245.90 | EXIT_EMA400 | -3.95 |
| SELL | 2026-01-01 12:15:00 | 238.02 | 2026-01-02 10:15:00 | 241.52 | EXIT_EMA400 | -3.50 |
| BUY | 2026-03-10 10:15:00 | 269.70 | 2026-03-16 11:15:00 | 261.05 | EXIT_EMA400 | -8.65 |

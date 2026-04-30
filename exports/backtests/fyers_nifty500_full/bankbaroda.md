# Bank of Baroda (BANKBARODA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 264.11
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -35.32
- **Avg P&L per closed trade:** -5.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 262.05 | 247.82 | 247.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 265.10 | 248.30 | 248.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 247.30 | 250.23 | 249.11 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 224.55 | 248.15 | 248.16 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 15:15:00 | 260.32 | 247.76 | 247.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 11:15:00 | 261.84 | 248.12 | 247.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 252.12 | 252.63 | 250.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 11:15:00 | 254.61 | 252.64 | 250.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 252.18 | 253.33 | 251.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-18 13:15:00 | 250.50 | 253.30 | 251.07 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 240.05 | 249.47 | 249.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 09:15:00 | 237.04 | 248.86 | 249.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 209.27 | 208.74 | 217.15 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 229.87 | 221.03 | 221.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 233.73 | 221.44 | 221.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 223.66 | 238.17 | 231.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 13:15:00 | 243.65 | 235.96 | 232.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 239.50 | 242.81 | 238.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 12:15:00 | 237.54 | 242.68 | 238.03 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 234.50 | 240.31 | 240.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 233.89 | 239.71 | 240.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 10:15:00 | 236.86 | 238.63 | 239.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 238.10 | 238.55 | 239.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 11:15:00 | 239.43 | 238.55 | 239.26 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 252.50 | 239.90 | 239.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 257.25 | 242.24 | 241.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 284.40 | 284.57 | 274.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 10:15:00 | 287.00 | 284.60 | 275.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 293.25 | 299.16 | 292.17 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-01 11:15:00 | 284.60 | 298.97 | 292.15 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 273.60 | 293.85 | 293.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 272.15 | 293.64 | 293.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 09:15:00 | 277.86 | 276.51 | 283.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 269.91 | 276.34 | 283.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 279.80 | 276.26 | 282.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-15 10:15:00 | 278.06 | 276.27 | 282.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 280.20 | 276.88 | 282.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-20 10:15:00 | 284.05 | 276.95 | 282.45 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-13 11:15:00 | 254.61 | 2024-12-18 13:15:00 | 250.50 | EXIT_EMA400 | -4.11 |
| BUY | 2025-05-23 13:15:00 | 243.65 | 2025-06-13 12:15:00 | 237.54 | EXIT_EMA400 | -6.11 |
| SELL | 2025-09-12 10:15:00 | 236.86 | 2025-09-16 11:15:00 | 239.43 | EXIT_EMA400 | -2.57 |
| BUY | 2025-12-09 10:15:00 | 287.00 | 2026-02-01 11:15:00 | 284.60 | EXIT_EMA400 | -2.40 |
| SELL | 2026-04-13 09:15:00 | 269.91 | 2026-04-20 10:15:00 | 284.05 | EXIT_EMA400 | -14.14 |
| SELL | 2026-04-15 10:15:00 | 278.06 | 2026-04-20 10:15:00 | 284.05 | EXIT_EMA400 | -5.99 |

# The New India Assurance Company Ltd. (NIACL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 160.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 107.97
- **Avg P&L per closed trade:** 15.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 09:15:00 | 221.85 | 237.88 | 237.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 200.25 | 232.65 | 233.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 10:15:00 | 235.04 | 228.52 | 231.24 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 14:15:00 | 237.38 | 233.42 | 233.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 247.62 | 233.60 | 233.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 235.20 | 237.33 | 235.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-01 09:15:00 | 243.07 | 237.21 | 235.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 251.39 | 260.12 | 250.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 09:15:00 | 264.50 | 260.03 | 250.41 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-06 14:15:00 | 253.80 | 267.98 | 257.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 239.55 | 256.38 | 256.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 236.45 | 255.09 | 255.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 196.22 | 192.64 | 207.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 09:15:00 | 190.74 | 201.13 | 204.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 162.50 | 155.63 | 167.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 161.25 | 156.73 | 167.11 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 161.71 | 157.02 | 166.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-26 12:15:00 | 159.63 | 157.11 | 166.84 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 166.90 | 156.97 | 163.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 176.35 | 166.99 | 166.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 178.22 | 167.20 | 167.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 182.85 | 183.33 | 177.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 09:15:00 | 184.24 | 183.34 | 177.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 177.73 | 183.13 | 178.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 181.85 | 189.83 | 189.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 178.82 | 188.49 | 189.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 150.95 | 150.60 | 157.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-26 12:15:00 | 148.46 | 151.27 | 155.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 147.94 | 132.68 | 140.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 168.14 | 146.21 | 146.18 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-07-01 09:15:00 | 243.07 | 2024-07-05 09:15:00 | 265.38 | TARGET | 22.31 |
| BUY | 2024-07-24 09:15:00 | 264.50 | 2024-07-26 13:15:00 | 306.76 | TARGET | 42.26 |
| SELL | 2025-01-10 09:15:00 | 190.74 | 2025-02-18 09:15:00 | 149.10 | TARGET | 41.64 |
| SELL | 2025-03-25 10:15:00 | 161.25 | 2025-04-16 09:15:00 | 166.90 | EXIT_EMA400 | -5.65 |
| SELL | 2025-03-26 12:15:00 | 159.63 | 2025-04-16 09:15:00 | 166.90 | EXIT_EMA400 | -7.27 |
| BUY | 2025-06-18 09:15:00 | 184.24 | 2025-06-19 12:15:00 | 177.73 | EXIT_EMA400 | -6.51 |
| SELL | 2026-02-26 12:15:00 | 148.46 | 2026-03-23 09:15:00 | 127.27 | TARGET | 21.19 |

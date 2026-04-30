# FSN E-Commerce Ventures Ltd. (NYKAA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 264.76
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 8 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 37.50
- **Avg P&L per closed trade:** 3.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 133.70 | 139.39 | 139.40 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 15:15:00 | 146.05 | 139.28 | 139.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 14:15:00 | 146.55 | 139.63 | 139.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 10:15:00 | 142.40 | 142.83 | 141.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-22 09:15:00 | 145.00 | 142.82 | 141.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 145.00 | 142.82 | 141.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-25 11:15:00 | 141.00 | 142.82 | 141.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 146.05 | 165.06 | 165.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 145.15 | 164.87 | 165.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 157.50 | 157.22 | 160.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-06 09:15:00 | 154.80 | 157.50 | 159.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 159.50 | 157.50 | 159.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-11 10:15:00 | 160.25 | 157.53 | 159.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 177.15 | 159.85 | 159.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 178.60 | 160.03 | 159.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 164.40 | 166.07 | 163.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-19 11:15:00 | 166.80 | 166.06 | 163.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 168.35 | 170.48 | 167.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-10 15:15:00 | 169.90 | 170.33 | 167.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-13 09:15:00 | 166.50 | 170.29 | 167.21 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 182.00 | 194.97 | 194.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 179.46 | 194.31 | 194.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 12:15:00 | 187.41 | 186.62 | 189.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 184.26 | 186.74 | 189.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 184.26 | 186.74 | 189.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-11 10:15:00 | 183.81 | 186.62 | 189.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 178.43 | 173.28 | 178.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-18 09:15:00 | 172.35 | 173.43 | 178.65 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-06 09:15:00 | 175.10 | 168.42 | 173.85 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 180.00 | 169.63 | 169.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 183.00 | 172.56 | 171.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 11:15:00 | 195.00 | 196.06 | 189.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 11:15:00 | 197.04 | 195.88 | 189.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 207.05 | 209.95 | 205.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-07 14:15:00 | 209.50 | 209.95 | 205.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 205.30 | 209.90 | 205.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-08 13:15:00 | 204.22 | 209.72 | 205.21 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 234.75 | 253.35 | 253.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 234.00 | 253.16 | 253.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 248.00 | 247.99 | 250.37 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 276.25 | 252.54 | 252.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 280.28 | 253.06 | 252.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 261.67 | 262.26 | 258.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 266.56 | 262.18 | 258.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 259.45 | 262.90 | 259.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 11:15:00 | 258.40 | 262.85 | 259.19 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 236.05 | 256.82 | 256.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 233.75 | 251.90 | 254.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 254.22 | 247.20 | 251.03 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 265.95 | 253.36 | 253.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 267.15 | 255.51 | 254.50 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-22 09:15:00 | 145.00 | 2023-09-25 11:15:00 | 141.00 | EXIT_EMA400 | -4.00 |
| SELL | 2024-03-06 09:15:00 | 154.80 | 2024-03-11 10:15:00 | 160.25 | EXIT_EMA400 | -5.45 |
| BUY | 2024-04-19 11:15:00 | 166.80 | 2024-04-24 13:15:00 | 177.02 | TARGET | 10.22 |
| BUY | 2024-05-10 15:15:00 | 169.90 | 2024-05-13 09:15:00 | 166.50 | EXIT_EMA400 | -3.40 |
| SELL | 2024-11-08 10:15:00 | 184.26 | 2024-11-18 09:15:00 | 167.52 | TARGET | 16.74 |
| SELL | 2024-11-11 10:15:00 | 183.81 | 2024-11-26 10:15:00 | 166.23 | TARGET | 17.58 |
| SELL | 2024-12-18 09:15:00 | 172.35 | 2025-01-06 09:15:00 | 175.10 | EXIT_EMA400 | -2.75 |
| BUY | 2025-06-05 11:15:00 | 197.04 | 2025-07-14 09:15:00 | 219.04 | TARGET | 22.00 |
| BUY | 2025-08-07 14:15:00 | 209.50 | 2025-08-08 13:15:00 | 204.22 | EXIT_EMA400 | -5.28 |
| BUY | 2026-02-25 09:15:00 | 266.56 | 2026-03-02 11:15:00 | 258.40 | EXIT_EMA400 | -8.16 |

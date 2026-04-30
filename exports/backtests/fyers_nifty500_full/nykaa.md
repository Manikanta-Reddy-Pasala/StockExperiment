# FSN E-Commerce Ventures Ltd. (NYKAA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 266.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 41.41
- **Avg P&L per closed trade:** 5.18

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 181.59 | 194.84 | 194.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 179.25 | 193.60 | 194.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 11:15:00 | 186.55 | 186.53 | 189.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 184.26 | 186.66 | 189.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 184.26 | 186.66 | 189.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-11 10:15:00 | 183.85 | 186.54 | 189.59 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 178.48 | 173.27 | 178.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-18 09:15:00 | 172.35 | 173.42 | 178.62 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-06 09:15:00 | 175.10 | 168.42 | 173.83 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 180.00 | 169.64 | 169.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 183.00 | 172.56 | 171.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 11:15:00 | 195.00 | 196.06 | 189.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 11:15:00 | 197.04 | 195.88 | 189.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 193.04 | 196.68 | 191.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 12:15:00 | 193.54 | 196.58 | 191.30 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 207.05 | 209.95 | 205.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-07 14:15:00 | 209.50 | 209.95 | 205.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 205.30 | 209.89 | 205.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-08 13:15:00 | 204.18 | 209.71 | 205.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 234.65 | 253.34 | 253.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 234.00 | 253.15 | 253.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 247.64 | 247.56 | 250.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 12:15:00 | 246.63 | 247.55 | 250.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 13:15:00 | 251.55 | 247.51 | 250.00 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 276.25 | 252.16 | 252.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 280.28 | 252.68 | 252.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 261.67 | 262.06 | 258.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 266.45 | 262.00 | 258.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 261.75 | 262.79 | 259.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 11:15:00 | 258.40 | 262.71 | 259.00 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 235.25 | 256.54 | 256.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 233.75 | 251.85 | 254.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 264.88 | 253.28 | 253.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 267.12 | 255.88 | 254.67 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 10:15:00 | 184.26 | 2024-11-18 09:15:00 | 167.77 | TARGET | 16.49 |
| SELL | 2024-11-11 10:15:00 | 183.85 | 2024-11-18 10:15:00 | 166.63 | TARGET | 17.22 |
| SELL | 2024-12-18 09:15:00 | 172.35 | 2025-01-06 09:15:00 | 175.10 | EXIT_EMA400 | -2.75 |
| BUY | 2025-06-13 12:15:00 | 193.54 | 2025-06-23 10:15:00 | 200.26 | TARGET | 6.72 |
| BUY | 2025-06-05 11:15:00 | 197.04 | 2025-07-14 09:15:00 | 219.06 | TARGET | 22.02 |
| BUY | 2025-08-07 14:15:00 | 209.50 | 2025-08-08 13:15:00 | 204.18 | EXIT_EMA400 | -5.32 |
| SELL | 2026-02-03 12:15:00 | 246.63 | 2026-02-04 13:15:00 | 251.55 | EXIT_EMA400 | -4.92 |
| BUY | 2026-02-25 09:15:00 | 266.45 | 2026-03-02 11:15:00 | 258.40 | EXIT_EMA400 | -8.05 |

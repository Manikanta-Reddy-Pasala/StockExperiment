# NCC Ltd. (NCC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 164.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 66.87
- **Avg P&L per closed trade:** 9.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 301.90 | 315.44 | 315.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 10:15:00 | 299.65 | 315.13 | 315.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 11:15:00 | 308.95 | 307.31 | 310.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 14:15:00 | 302.35 | 307.63 | 310.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-01 17:15:00 | 313.50 | 299.71 | 305.13 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 15:15:00 | 312.20 | 303.59 | 303.57 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 299.40 | 303.54 | 303.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 294.20 | 303.26 | 303.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 257.85 | 255.85 | 270.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 234.00 | 255.64 | 270.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-26 09:15:00 | 216.28 | 198.81 | 216.08 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 238.13 | 216.85 | 216.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 239.94 | 217.28 | 217.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 227.00 | 230.46 | 225.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 229.48 | 230.07 | 225.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 226.10 | 229.91 | 225.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-17 15:15:00 | 225.75 | 229.87 | 225.77 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 222.59 | 225.27 | 225.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 218.82 | 225.21 | 225.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 224.14 | 221.91 | 223.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 218.95 | 222.13 | 223.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 215.60 | 213.65 | 217.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-16 13:15:00 | 215.51 | 213.74 | 217.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 216.31 | 213.82 | 217.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-17 11:15:00 | 216.01 | 213.86 | 217.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 216.77 | 214.14 | 217.20 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-22 09:15:00 | 215.15 | 214.23 | 217.19 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 212.78 | 209.93 | 212.64 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 162.00 | 151.54 | 151.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 164.20 | 151.67 | 151.61 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-21 14:15:00 | 302.35 | 2024-10-25 09:15:00 | 278.15 | TARGET | 24.20 |
| SELL | 2025-02-01 12:15:00 | 234.00 | 2025-03-26 09:15:00 | 216.28 | EXIT_EMA400 | 17.72 |
| BUY | 2025-06-16 13:15:00 | 229.48 | 2025-06-17 15:15:00 | 225.75 | EXIT_EMA400 | -3.73 |
| SELL | 2025-08-14 09:15:00 | 218.95 | 2025-08-29 09:15:00 | 205.77 | TARGET | 13.18 |
| SELL | 2025-09-17 11:15:00 | 216.01 | 2025-09-23 09:15:00 | 212.11 | TARGET | 3.90 |
| SELL | 2025-09-16 13:15:00 | 215.51 | 2025-09-24 10:15:00 | 210.03 | TARGET | 5.48 |
| SELL | 2025-09-22 09:15:00 | 215.15 | 2025-09-24 14:15:00 | 209.03 | TARGET | 6.12 |

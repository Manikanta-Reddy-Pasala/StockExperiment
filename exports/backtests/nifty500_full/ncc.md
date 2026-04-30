# NCC Ltd. (NCC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 163.84
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
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 21.94
- **Avg P&L per closed trade:** 3.13

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 302.35 | 315.58 | 315.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 15:15:00 | 301.55 | 315.44 | 315.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 11:15:00 | 308.95 | 307.30 | 310.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 14:15:00 | 302.35 | 307.64 | 310.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-04 09:15:00 | 309.85 | 299.67 | 305.12 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 15:15:00 | 312.20 | 303.55 | 303.55 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 299.40 | 303.51 | 303.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 294.00 | 303.24 | 303.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 14:15:00 | 196.74 | 196.56 | 218.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 190.43 | 203.44 | 215.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 217.78 | 203.80 | 213.77 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 236.99 | 217.07 | 216.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 239.92 | 217.30 | 217.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 227.00 | 230.47 | 225.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 229.46 | 230.08 | 225.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 226.10 | 229.93 | 225.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-17 15:15:00 | 225.73 | 229.89 | 225.82 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 222.93 | 225.30 | 225.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 218.82 | 225.21 | 225.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 224.00 | 221.91 | 223.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 218.95 | 222.14 | 223.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 215.56 | 213.65 | 217.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-16 13:15:00 | 215.51 | 213.74 | 217.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 216.31 | 213.81 | 217.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-17 11:15:00 | 216.01 | 213.86 | 217.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 216.77 | 214.13 | 217.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-22 09:15:00 | 215.15 | 214.23 | 217.19 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 212.78 | 209.94 | 212.64 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 164.38 | 151.77 | 151.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 165.02 | 152.14 | 151.92 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-21 14:15:00 | 302.35 | 2024-10-25 09:15:00 | 278.09 | TARGET | 24.26 |
| SELL | 2025-04-07 09:15:00 | 190.43 | 2025-04-15 09:15:00 | 217.78 | EXIT_EMA400 | -27.35 |
| BUY | 2025-06-16 13:15:00 | 229.46 | 2025-06-17 15:15:00 | 225.73 | EXIT_EMA400 | -3.73 |
| SELL | 2025-08-14 09:15:00 | 218.95 | 2025-08-29 09:15:00 | 205.73 | TARGET | 13.22 |
| SELL | 2025-09-17 11:15:00 | 216.01 | 2025-09-23 09:15:00 | 212.10 | TARGET | 3.91 |
| SELL | 2025-09-16 13:15:00 | 215.51 | 2025-09-24 10:15:00 | 210.02 | TARGET | 5.49 |
| SELL | 2025-09-22 09:15:00 | 215.15 | 2025-09-24 14:15:00 | 209.02 | TARGET | 6.13 |

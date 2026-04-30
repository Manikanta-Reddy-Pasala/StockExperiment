# Shipping Corporation of India Ltd. (SCI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 304.50
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
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 7.14
- **Avg P&L per closed trade:** 1.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 247.60 | 265.66 | 265.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 11:15:00 | 246.95 | 265.48 | 265.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 259.80 | 259.35 | 262.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-03 13:15:00 | 249.55 | 261.06 | 262.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 238.89 | 228.41 | 239.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-11 11:15:00 | 242.57 | 228.66 | 239.92 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 192.40 | 175.25 | 175.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 198.51 | 175.48 | 175.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 217.73 | 217.77 | 207.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 220.00 | 217.77 | 207.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 214.94 | 218.21 | 211.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 14:15:00 | 211.35 | 217.80 | 211.58 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 221.98 | 235.22 | 235.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 219.95 | 235.06 | 235.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 227.60 | 226.62 | 230.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 12:15:00 | 220.11 | 227.96 | 230.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 218.00 | 217.79 | 223.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-29 10:15:00 | 217.64 | 217.78 | 223.21 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-29 12:15:00 | 223.43 | 217.86 | 223.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 266.96 | 226.29 | 226.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 268.40 | 226.71 | 226.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 249.80 | 249.96 | 241.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-12 12:15:00 | 254.30 | 247.10 | 241.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 242.35 | 247.11 | 241.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-13 12:15:00 | 240.10 | 247.04 | 241.38 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 222.91 | 237.86 | 237.89 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 247.82 | 237.79 | 237.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 249.32 | 237.91 | 237.84 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-03 13:15:00 | 249.55 | 2024-10-23 09:15:00 | 210.45 | TARGET | 39.10 |
| BUY | 2025-07-15 09:15:00 | 220.00 | 2025-08-01 14:15:00 | 211.35 | EXIT_EMA400 | -8.65 |
| SELL | 2026-01-08 12:15:00 | 220.11 | 2026-01-29 12:15:00 | 223.43 | EXIT_EMA400 | -3.32 |
| SELL | 2026-01-29 10:15:00 | 217.64 | 2026-01-29 12:15:00 | 223.43 | EXIT_EMA400 | -5.79 |
| BUY | 2026-03-12 12:15:00 | 254.30 | 2026-03-13 12:15:00 | 240.10 | EXIT_EMA400 | -14.20 |

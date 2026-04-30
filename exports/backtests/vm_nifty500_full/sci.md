# Shipping Corporation of India Ltd. (SCI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 304.48
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -47.56
- **Avg P&L per closed trade:** -9.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 241.85 | 266.73 | 266.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 237.45 | 259.48 | 262.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 238.88 | 228.60 | 240.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 15:15:00 | 222.60 | 228.70 | 239.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 238.20 | 223.77 | 233.58 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 198.51 | 175.48 | 175.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 202.05 | 176.74 | 176.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 217.73 | 217.78 | 207.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 220.00 | 217.78 | 207.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 214.94 | 218.21 | 211.40 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 14:15:00 | 211.35 | 217.80 | 211.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 221.98 | 235.23 | 235.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 219.95 | 235.07 | 235.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 227.60 | 226.62 | 230.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 12:15:00 | 220.11 | 227.96 | 230.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 218.00 | 217.78 | 223.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-29 10:15:00 | 217.64 | 217.78 | 223.20 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-29 12:15:00 | 223.43 | 217.85 | 223.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 12:15:00 | 268.40 | 226.58 | 226.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 274.60 | 228.24 | 227.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 249.80 | 249.91 | 241.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-12 12:15:00 | 254.30 | 247.08 | 241.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 242.35 | 247.09 | 241.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-13 12:15:00 | 240.10 | 247.02 | 241.40 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 222.95 | 237.84 | 237.90 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 249.32 | 237.89 | 237.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 253.45 | 238.40 | 238.11 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-12 15:15:00 | 222.60 | 2024-11-28 09:15:00 | 238.20 | EXIT_EMA400 | -15.60 |
| BUY | 2025-07-15 09:15:00 | 220.00 | 2025-08-01 14:15:00 | 211.35 | EXIT_EMA400 | -8.65 |
| SELL | 2026-01-08 12:15:00 | 220.11 | 2026-01-29 12:15:00 | 223.43 | EXIT_EMA400 | -3.32 |
| SELL | 2026-01-29 10:15:00 | 217.64 | 2026-01-29 12:15:00 | 223.43 | EXIT_EMA400 | -5.79 |
| BUY | 2026-03-12 12:15:00 | 254.30 | 2026-03-13 12:15:00 | 240.10 | EXIT_EMA400 | -14.20 |

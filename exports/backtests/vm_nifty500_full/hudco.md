# Housing & Urban Development Corporation Ltd. (HUDCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 220.87
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -50.91
- **Avg P&L per closed trade:** -7.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 252.40 | 282.89 | 283.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 11:15:00 | 250.45 | 279.97 | 281.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 222.60 | 222.12 | 237.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 219.27 | 222.69 | 236.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 240.04 | 216.30 | 227.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 13:15:00 | 257.23 | 234.48 | 234.42 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 213.72 | 235.19 | 235.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 208.39 | 234.68 | 234.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 230.82 | 230.52 | 232.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 10:15:00 | 220.49 | 230.73 | 232.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 229.07 | 230.39 | 232.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-27 09:15:00 | 207.14 | 229.10 | 231.55 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-31 14:15:00 | 229.75 | 225.31 | 229.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 235.06 | 204.23 | 204.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 239.90 | 206.27 | 205.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 12:15:00 | 208.72 | 216.17 | 211.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 224.55 | 215.78 | 211.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 226.80 | 234.37 | 226.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 11:15:00 | 228.49 | 234.24 | 226.20 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 222.64 | 233.91 | 226.23 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 217.10 | 227.37 | 227.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 212.40 | 226.94 | 227.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 215.05 | 214.66 | 218.89 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 236.21 | 220.37 | 220.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 237.36 | 227.20 | 225.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 228.70 | 229.35 | 226.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 11:15:00 | 234.16 | 229.29 | 226.68 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 09:15:00 | 224.50 | 229.44 | 226.82 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 213.19 | 227.79 | 227.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 211.27 | 225.08 | 226.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 222.25 | 220.48 | 223.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 13:15:00 | 217.70 | 222.93 | 224.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 184.94 | 177.20 | 186.38 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-13 12:15:00 | 187.80 | 177.95 | 186.31 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 218.19 | 191.65 | 191.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 220.87 | 193.98 | 192.75 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 14:15:00 | 219.27 | 2024-11-28 09:15:00 | 240.04 | EXIT_EMA400 | -20.77 |
| SELL | 2025-01-22 10:15:00 | 220.49 | 2025-01-31 14:15:00 | 229.75 | EXIT_EMA400 | -9.26 |
| SELL | 2025-01-27 09:15:00 | 207.14 | 2025-01-31 14:15:00 | 229.75 | EXIT_EMA400 | -22.61 |
| BUY | 2025-05-12 14:15:00 | 224.55 | 2025-06-16 09:15:00 | 222.64 | EXIT_EMA400 | -1.91 |
| BUY | 2025-06-13 11:15:00 | 228.49 | 2025-06-16 09:15:00 | 222.64 | EXIT_EMA400 | -5.85 |
| BUY | 2025-11-10 11:15:00 | 234.16 | 2025-11-11 09:15:00 | 224.50 | EXIT_EMA400 | -9.66 |
| SELL | 2026-01-08 13:15:00 | 217.70 | 2026-01-23 13:15:00 | 198.55 | TARGET | 19.15 |

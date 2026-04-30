# Housing & Urban Development Corporation Ltd. (HUDCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 224.00
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
- **Total realized P&L (per unit):** -50.90
- **Avg P&L per closed trade:** -7.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 252.70 | 284.90 | 285.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 15:15:00 | 245.30 | 278.70 | 281.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 222.70 | 222.06 | 237.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 219.29 | 222.65 | 236.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 240.14 | 216.28 | 228.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 13:15:00 | 257.27 | 234.47 | 234.45 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 213.90 | 235.18 | 235.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 208.41 | 234.67 | 234.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 230.73 | 230.52 | 232.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 10:15:00 | 220.49 | 230.73 | 232.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 229.20 | 230.39 | 232.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-27 09:15:00 | 207.27 | 229.10 | 231.56 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-31 14:15:00 | 229.75 | 225.32 | 229.10 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 234.97 | 204.24 | 204.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 239.90 | 206.28 | 205.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 12:15:00 | 208.72 | 216.18 | 211.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 224.60 | 215.80 | 211.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 226.80 | 234.38 | 226.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 11:15:00 | 228.49 | 234.25 | 226.20 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 222.64 | 233.91 | 226.23 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 217.10 | 227.37 | 227.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 212.40 | 226.94 | 227.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 215.05 | 214.67 | 218.90 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 236.09 | 220.38 | 220.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 237.36 | 227.20 | 225.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 228.70 | 229.35 | 226.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 11:15:00 | 234.16 | 229.30 | 226.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 09:15:00 | 224.50 | 229.45 | 226.83 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 213.19 | 227.80 | 227.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 211.27 | 225.08 | 226.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 222.25 | 220.48 | 223.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 13:15:00 | 217.70 | 222.92 | 224.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 184.94 | 177.17 | 186.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-13 11:15:00 | 186.18 | 177.82 | 186.15 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 214.51 | 191.30 | 191.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 216.87 | 191.55 | 191.38 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 14:15:00 | 219.29 | 2024-11-28 09:15:00 | 240.14 | EXIT_EMA400 | -20.85 |
| SELL | 2025-01-22 10:15:00 | 220.49 | 2025-01-31 14:15:00 | 229.75 | EXIT_EMA400 | -9.26 |
| SELL | 2025-01-27 09:15:00 | 207.27 | 2025-01-31 14:15:00 | 229.75 | EXIT_EMA400 | -22.48 |
| BUY | 2025-05-12 14:15:00 | 224.60 | 2025-06-16 09:15:00 | 222.64 | EXIT_EMA400 | -1.96 |
| BUY | 2025-06-13 11:15:00 | 228.49 | 2025-06-16 09:15:00 | 222.64 | EXIT_EMA400 | -5.85 |
| BUY | 2025-11-10 11:15:00 | 234.16 | 2025-11-11 09:15:00 | 224.50 | EXIT_EMA400 | -9.66 |
| SELL | 2026-01-08 13:15:00 | 217.70 | 2026-01-23 13:15:00 | 198.54 | TARGET | 19.16 |

# Aditya Birla Capital Ltd. (ABCAPITAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 346.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 98.85
- **Avg P&L per closed trade:** 24.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 227.50 | 221.67 | 221.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 230.00 | 221.75 | 221.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 227.79 | 228.07 | 225.35 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 211.63 | 223.95 | 223.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 207.28 | 222.50 | 223.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 199.72 | 199.23 | 206.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 11:15:00 | 198.42 | 199.22 | 206.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 168.41 | 161.83 | 168.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 168.84 | 161.90 | 168.67 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 192.50 | 173.30 | 173.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 193.43 | 173.50 | 173.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 260.60 | 263.69 | 247.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-04 14:15:00 | 278.20 | 260.54 | 249.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 349.00 | 353.57 | 343.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-21 12:15:00 | 351.40 | 353.55 | 343.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 344.00 | 352.91 | 343.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-27 14:15:00 | 349.10 | 352.87 | 343.97 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-29 10:15:00 | 344.00 | 352.34 | 344.14 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 316.00 | 342.52 | 342.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 298.00 | 330.50 | 335.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 342.25 | 332.43 | 332.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 345.40 | 332.66 | 332.50 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-04 11:15:00 | 198.42 | 2025-01-09 14:15:00 | 173.46 | TARGET | 24.96 |
| BUY | 2025-08-04 14:15:00 | 278.20 | 2025-12-09 11:15:00 | 364.58 | TARGET | 86.38 |
| BUY | 2026-01-21 12:15:00 | 351.40 | 2026-01-29 10:15:00 | 344.00 | EXIT_EMA400 | -7.40 |
| BUY | 2026-01-27 14:15:00 | 349.10 | 2026-01-29 10:15:00 | 344.00 | EXIT_EMA400 | -5.10 |

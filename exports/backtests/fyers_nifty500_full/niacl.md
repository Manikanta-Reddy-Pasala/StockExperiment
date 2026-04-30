# The New India Assurance Company Ltd. (NIACL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 160.84
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 12.94
- **Avg P&L per closed trade:** 2.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 236.50 | 255.05 | 255.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 234.25 | 254.85 | 254.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 196.22 | 192.56 | 207.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 09:15:00 | 190.74 | 201.11 | 204.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 162.47 | 155.58 | 167.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 161.25 | 156.68 | 166.94 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 161.71 | 156.96 | 166.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-26 14:15:00 | 158.53 | 157.10 | 166.61 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 166.89 | 156.93 | 163.41 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 15:15:00 | 174.50 | 166.88 | 166.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 176.35 | 166.98 | 166.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 182.82 | 183.34 | 177.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 09:15:00 | 184.24 | 183.34 | 177.86 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 177.73 | 183.14 | 178.02 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 182.04 | 189.83 | 189.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 178.82 | 188.48 | 189.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 152.40 | 151.79 | 159.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 12:15:00 | 149.89 | 151.77 | 159.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-10 10:15:00 | 157.75 | 150.74 | 157.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 163.20 | 146.10 | 146.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 166.89 | 146.31 | 146.18 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-10 09:15:00 | 190.74 | 2025-02-18 09:15:00 | 149.43 | TARGET | 41.31 |
| SELL | 2025-03-25 10:15:00 | 161.25 | 2025-04-16 09:15:00 | 166.89 | EXIT_EMA400 | -5.64 |
| SELL | 2025-03-26 14:15:00 | 158.53 | 2025-04-16 09:15:00 | 166.89 | EXIT_EMA400 | -8.36 |
| BUY | 2025-06-18 09:15:00 | 184.24 | 2025-06-19 12:15:00 | 177.73 | EXIT_EMA400 | -6.51 |
| SELL | 2026-02-01 12:15:00 | 149.89 | 2026-02-10 10:15:00 | 157.75 | EXIT_EMA400 | -7.86 |

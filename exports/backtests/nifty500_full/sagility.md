# Sagility Ltd. (SAGILITY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-12 09:15:00 → 2026-04-30 15:15:00 (2511 bars)
- **Last close:** 41.74
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 4
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 0.32
- **Avg P&L per closed trade:** 0.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 14:15:00 | 41.63 | 44.76 | 44.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 41.00 | 43.72 | 44.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 09:15:00 | 42.63 | 42.36 | 43.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-24 14:15:00 | 42.26 | 42.58 | 43.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-12 09:15:00 | 42.87 | 41.82 | 42.61 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 45.30 | 43.23 | 43.22 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 40.72 | 43.21 | 43.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 39.30 | 43.10 | 43.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 10:15:00 | 41.40 | 40.87 | 41.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-19 12:15:00 | 40.34 | 40.95 | 41.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 41.53 | 40.88 | 41.62 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-24 10:15:00 | 41.66 | 40.89 | 41.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 45.12 | 41.97 | 41.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 45.63 | 42.18 | 42.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 43.04 | 43.14 | 42.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-25 12:15:00 | 43.77 | 43.14 | 42.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 14:15:00 | 42.43 | 43.17 | 42.69 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 47.90 | 50.12 | 50.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 47.32 | 50.04 | 50.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 41.70 | 40.94 | 43.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 14:15:00 | 40.00 | 40.94 | 43.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 43.03 | 41.21 | 43.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-10 11:15:00 | 42.62 | 41.46 | 43.39 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 43.05 | 41.54 | 43.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-20 09:15:00 | 42.60 | 41.81 | 43.28 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 42.56 | 41.86 | 43.19 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-24 11:15:00 | 41.58 | 41.90 | 43.15 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-24 14:15:00 | 42.26 | 2025-05-07 09:15:00 | 39.28 | TARGET | 2.98 |
| SELL | 2025-06-19 12:15:00 | 40.34 | 2025-06-24 10:15:00 | 41.66 | EXIT_EMA400 | -1.32 |
| BUY | 2025-07-25 12:15:00 | 43.77 | 2025-07-28 14:15:00 | 42.43 | EXIT_EMA400 | -1.34 |

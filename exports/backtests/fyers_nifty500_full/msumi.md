# Motherson Sumi Wiring India Ltd. (MSUMI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 40.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 6.12
- **Avg P&L per closed trade:** 1.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 46.87 | 47.66 | 47.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 12:15:00 | 46.66 | 47.57 | 47.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 48.19 | 46.94 | 47.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-01 14:15:00 | 46.70 | 47.14 | 47.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-10 14:15:00 | 43.66 | 42.37 | 43.27 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 37.77 | 35.87 | 35.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 38.00 | 35.89 | 35.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 39.43 | 39.45 | 38.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 39.71 | 39.45 | 38.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 40.48 | 41.54 | 40.37 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-25 15:15:00 | 40.90 | 41.52 | 40.37 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 40.19 | 41.51 | 40.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 37.65 | 39.65 | 39.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 37.20 | 39.63 | 39.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 39.55 | 39.51 | 39.58 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 41.88 | 39.66 | 39.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 42.00 | 39.68 | 39.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 13:15:00 | 45.82 | 45.82 | 43.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 11:15:00 | 46.20 | 45.78 | 44.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 44.54 | 45.68 | 44.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 12:15:00 | 44.90 | 45.65 | 44.27 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 44.33 | 45.61 | 44.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-14 11:15:00 | 44.04 | 45.59 | 44.28 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 43.32 | 46.32 | 46.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 42.94 | 46.29 | 46.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 44.95 | 44.91 | 45.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 10:15:00 | 44.11 | 44.92 | 45.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 40.02 | 39.23 | 40.62 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-22 11:15:00 | 40.67 | 39.27 | 40.61 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-01 14:15:00 | 46.70 | 2024-10-04 09:15:00 | 44.98 | TARGET | 1.72 |
| BUY | 2025-06-20 09:15:00 | 39.71 | 2025-07-09 09:15:00 | 43.68 | TARGET | 3.97 |
| BUY | 2025-07-25 15:15:00 | 40.90 | 2025-07-28 09:15:00 | 40.19 | EXIT_EMA400 | -0.71 |
| BUY | 2025-10-07 11:15:00 | 46.20 | 2025-10-14 11:15:00 | 44.04 | EXIT_EMA400 | -2.16 |
| BUY | 2025-10-13 12:15:00 | 44.90 | 2025-10-14 11:15:00 | 44.04 | EXIT_EMA400 | -0.86 |
| SELL | 2026-02-05 10:15:00 | 44.11 | 2026-03-09 09:15:00 | 39.95 | TARGET | 4.16 |

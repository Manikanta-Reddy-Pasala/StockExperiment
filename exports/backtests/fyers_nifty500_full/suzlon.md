# Suzlon Energy Ltd. (SUZLON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 55.64
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 8.22
- **Avg P&L per closed trade:** 2.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 15:15:00 | 67.30 | 71.84 | 71.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 66.83 | 71.39 | 71.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 66.12 | 65.74 | 67.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 14:15:00 | 64.10 | 66.49 | 67.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-01 09:15:00 | 61.06 | 57.81 | 61.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 56.91 | 56.39 | 56.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 56.98 | 56.40 | 56.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 56.18 | 56.40 | 56.39 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 55.72 | 56.39 | 56.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 55.29 | 56.38 | 56.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 56.18 | 55.78 | 56.07 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 60.85 | 56.32 | 56.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 62.10 | 56.91 | 56.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 64.03 | 64.15 | 61.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 11:15:00 | 64.24 | 64.06 | 61.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 64.12 | 65.52 | 64.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 10:15:00 | 63.86 | 65.50 | 64.09 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 58.30 | 63.44 | 63.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 57.38 | 62.16 | 62.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 59.29 | 59.22 | 60.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-17 13:15:00 | 58.96 | 59.22 | 60.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-19 15:15:00 | 60.60 | 59.26 | 60.52 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 54.60 | 45.87 | 45.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 55.70 | 47.07 | 46.48 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-20 14:15:00 | 64.10 | 2025-01-24 09:15:00 | 53.86 | TARGET | 10.24 |
| BUY | 2025-06-20 11:15:00 | 64.24 | 2025-07-25 10:15:00 | 63.86 | EXIT_EMA400 | -0.38 |
| SELL | 2025-09-17 13:15:00 | 58.96 | 2025-09-19 15:15:00 | 60.60 | EXIT_EMA400 | -1.64 |

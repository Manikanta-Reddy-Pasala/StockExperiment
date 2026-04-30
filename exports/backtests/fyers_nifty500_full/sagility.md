# Sagility Ltd. (SAGILITY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-12 09:15:00 → 2026-04-30 15:15:00 (2529 bars)
- **Last close:** 41.88
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 4
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 0.02
- **Avg P&L per closed trade:** 0.01

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 41.52 | 44.88 | 44.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 41.11 | 44.85 | 44.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 44.83 | 44.29 | 44.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 11:15:00 | 43.66 | 44.30 | 44.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 42.63 | 42.36 | 43.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 43.35 | 42.39 | 43.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 13:15:00 | 44.58 | 43.25 | 43.24 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 40.72 | 43.24 | 43.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 39.30 | 43.10 | 43.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 10:15:00 | 41.41 | 40.87 | 41.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-19 12:15:00 | 40.34 | 40.95 | 41.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 41.53 | 40.88 | 41.62 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-24 10:15:00 | 41.66 | 40.89 | 41.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 45.11 | 41.97 | 41.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 45.63 | 42.18 | 42.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 43.04 | 43.14 | 42.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-25 12:15:00 | 43.77 | 43.15 | 42.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 14:15:00 | 42.44 | 43.17 | 42.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 48.33 | 50.14 | 50.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 47.40 | 50.08 | 50.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 41.70 | 40.92 | 43.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 14:15:00 | 39.98 | 40.92 | 43.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 43.02 | 41.20 | 43.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-10 11:15:00 | 42.62 | 41.45 | 43.37 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 43.04 | 41.53 | 43.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-20 09:15:00 | 42.52 | 41.81 | 43.27 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 42.56 | 41.87 | 43.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-24 13:15:00 | 41.17 | 41.89 | 43.10 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-25 11:15:00 | 43.66 | 2025-04-04 09:15:00 | 40.99 | TARGET | 2.67 |
| SELL | 2025-06-19 12:15:00 | 40.34 | 2025-06-24 10:15:00 | 41.66 | EXIT_EMA400 | -1.32 |
| BUY | 2025-07-25 12:15:00 | 43.77 | 2025-07-28 14:15:00 | 42.44 | EXIT_EMA400 | -1.33 |

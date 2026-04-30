# NMDC Steel Ltd. (NSLNISP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 42.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 3.82
- **Avg P&L per closed trade:** 0.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 15:15:00 | 39.65 | 37.07 | 37.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 39.96 | 37.29 | 37.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 38.20 | 38.55 | 37.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 14:15:00 | 39.49 | 38.39 | 38.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 38.85 | 39.42 | 38.96 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 36.96 | 38.61 | 38.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 36.66 | 38.57 | 38.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 41.80 | 37.76 | 38.14 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 40.15 | 38.48 | 38.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 40.46 | 38.50 | 38.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 38.44 | 38.68 | 38.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-02 10:15:00 | 39.26 | 38.53 | 38.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 39.26 | 38.53 | 38.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-02 11:15:00 | 39.54 | 38.54 | 38.53 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 43.02 | 44.30 | 42.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-17 14:15:00 | 42.82 | 44.28 | 42.91 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 41.99 | 43.06 | 43.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 41.80 | 42.99 | 43.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 41.53 | 41.42 | 42.05 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 45.24 | 42.44 | 42.44 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 40.37 | 42.50 | 42.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 39.86 | 42.47 | 42.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 42.54 | 41.88 | 42.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 11:15:00 | 41.51 | 42.04 | 42.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-02 14:15:00 | 42.30 | 41.97 | 42.18 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 41.16 | 39.63 | 39.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 41.89 | 39.76 | 39.69 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-25 14:15:00 | 39.49 | 2025-07-25 09:15:00 | 38.85 | EXIT_EMA400 | -0.64 |
| BUY | 2025-09-02 10:15:00 | 39.26 | 2025-09-03 11:15:00 | 41.47 | TARGET | 2.21 |
| BUY | 2025-09-02 11:15:00 | 39.54 | 2025-09-03 11:15:00 | 42.58 | TARGET | 3.04 |
| SELL | 2026-02-01 11:15:00 | 41.51 | 2026-02-02 14:15:00 | 42.30 | EXIT_EMA400 | -0.79 |

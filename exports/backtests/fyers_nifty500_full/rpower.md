# Reliance Power Ltd. (RPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 28.77
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 15.30
- **Avg P&L per closed trade:** 2.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 36.70 | 41.13 | 41.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 35.77 | 41.07 | 41.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 40.37 | 40.17 | 40.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 39.32 | 40.47 | 40.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 39.32 | 40.47 | 40.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-11 12:15:00 | 38.88 | 40.43 | 40.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 40.24 | 40.32 | 40.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-13 15:15:00 | 39.14 | 40.29 | 40.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 37.54 | 36.15 | 37.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 15:15:00 | 37.75 | 36.17 | 37.70 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 41.58 | 38.59 | 38.58 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 37.97 | 38.57 | 38.57 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 14:15:00 | 38.98 | 38.58 | 38.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 15:15:00 | 39.20 | 38.58 | 38.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 12:15:00 | 40.60 | 40.70 | 39.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 42.72 | 40.23 | 39.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 59.50 | 63.60 | 59.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 11:15:00 | 62.94 | 63.56 | 59.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 56.78 | 63.24 | 59.30 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 45.08 | 56.64 | 56.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 43.77 | 56.39 | 56.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 47.93 | 47.85 | 50.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 12:15:00 | 47.45 | 47.90 | 50.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 48.56 | 46.50 | 48.46 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 09:15:00 | 39.32 | 2025-02-27 11:15:00 | 35.22 | TARGET | 4.10 |
| SELL | 2025-02-13 15:15:00 | 39.14 | 2025-02-27 12:15:00 | 34.85 | TARGET | 4.29 |
| SELL | 2025-02-11 12:15:00 | 38.88 | 2025-02-28 11:15:00 | 33.52 | TARGET | 5.36 |
| BUY | 2025-05-12 09:15:00 | 42.72 | 2025-05-23 12:15:00 | 51.53 | TARGET | 8.81 |
| BUY | 2025-07-23 11:15:00 | 62.94 | 2025-07-25 09:15:00 | 56.78 | EXIT_EMA400 | -6.16 |
| SELL | 2025-09-18 12:15:00 | 47.45 | 2025-10-10 09:15:00 | 48.56 | EXIT_EMA400 | -1.11 |

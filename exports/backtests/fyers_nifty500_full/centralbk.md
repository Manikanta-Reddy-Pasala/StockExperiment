# Central Bank of India (CENTRALBK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 36.43
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 1.86
- **Avg P&L per closed trade:** 0.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 60.16 | 62.71 | 62.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 59.95 | 62.68 | 62.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 61.38 | 61.22 | 61.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-22 15:15:00 | 60.77 | 61.21 | 61.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 61.47 | 61.13 | 61.67 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-30 11:15:00 | 61.12 | 61.13 | 61.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 60.73 | 60.27 | 61.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-18 12:15:00 | 59.65 | 60.24 | 60.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-20 15:15:00 | 60.88 | 60.00 | 60.75 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 37.67 | 37.15 | 37.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 37.82 | 37.17 | 37.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 37.05 | 37.18 | 37.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 10:15:00 | 37.38 | 37.16 | 37.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 37.38 | 37.16 | 37.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-15 11:15:00 | 38.06 | 37.17 | 37.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-17 14:15:00 | 36.99 | 37.27 | 37.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 36.42 | 37.94 | 37.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 36.25 | 37.89 | 37.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 37.25 | 37.22 | 37.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 15:15:00 | 37.00 | 37.20 | 37.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 37.41 | 37.13 | 37.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-31 11:15:00 | 37.51 | 37.14 | 37.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 38.44 | 37.63 | 37.62 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 36.95 | 37.61 | 37.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 36.43 | 37.60 | 37.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 37.50 | 37.28 | 37.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 09:15:00 | 36.90 | 37.28 | 37.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 36.50 | 37.15 | 37.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 37.70 | 37.11 | 37.30 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 38.35 | 37.44 | 37.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 38.89 | 37.50 | 37.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 37.57 | 38.19 | 37.86 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 35.60 | 37.59 | 37.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 35.44 | 37.55 | 37.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 35.23 | 34.99 | 36.00 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-30 11:15:00 | 61.12 | 2024-09-04 09:15:00 | 59.47 | TARGET | 1.65 |
| SELL | 2024-08-22 15:15:00 | 60.77 | 2024-09-09 09:15:00 | 57.62 | TARGET | 3.15 |
| SELL | 2024-09-18 12:15:00 | 59.65 | 2024-09-20 15:15:00 | 60.88 | EXIT_EMA400 | -1.23 |
| BUY | 2025-10-15 10:15:00 | 37.38 | 2025-10-15 11:15:00 | 38.05 | TARGET | 0.67 |
| BUY | 2025-10-15 11:15:00 | 38.06 | 2025-10-17 14:15:00 | 36.99 | EXIT_EMA400 | -1.07 |
| SELL | 2025-12-24 15:15:00 | 37.00 | 2025-12-31 11:15:00 | 37.51 | EXIT_EMA400 | -0.51 |
| SELL | 2026-02-01 09:15:00 | 36.90 | 2026-02-09 09:15:00 | 37.70 | EXIT_EMA400 | -0.80 |

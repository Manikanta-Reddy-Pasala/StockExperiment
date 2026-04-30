# Central Bank of India (CENTRALBK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 36.42
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -4.28
- **Avg P&L per closed trade:** -0.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 09:15:00 | 59.54 | 63.38 | 63.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 58.95 | 63.22 | 63.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 61.42 | 61.25 | 62.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-22 15:15:00 | 60.64 | 61.24 | 62.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 60.73 | 60.29 | 61.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-18 12:15:00 | 59.66 | 60.26 | 61.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 60.88 | 60.02 | 60.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-23 12:15:00 | 61.02 | 60.04 | 60.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 37.65 | 37.15 | 37.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 37.82 | 37.17 | 37.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 37.15 | 37.18 | 37.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 10:15:00 | 37.38 | 37.16 | 37.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 37.38 | 37.16 | 37.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-15 11:15:00 | 38.06 | 37.17 | 37.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-17 14:15:00 | 36.99 | 37.27 | 37.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 36.59 | 37.96 | 37.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 36.42 | 37.94 | 37.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 37.25 | 37.22 | 37.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 13:15:00 | 37.05 | 37.21 | 37.51 | Sell entry 1 (retest1 break) |
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
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 36.10 | 37.27 | 37.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 36.50 | 37.19 | 37.38 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 37.70 | 37.14 | 37.33 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 38.35 | 37.46 | 37.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 38.89 | 37.52 | 37.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 37.56 | 38.20 | 37.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 15:15:00 | 37.77 | 38.10 | 37.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 37.77 | 38.10 | 37.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-06 09:15:00 | 37.69 | 38.09 | 37.83 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 35.84 | 37.61 | 37.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 35.60 | 37.59 | 37.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 35.23 | 34.99 | 36.00 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-22 15:15:00 | 60.64 | 2024-09-23 12:15:00 | 61.02 | EXIT_EMA400 | -0.38 |
| SELL | 2024-09-18 12:15:00 | 59.66 | 2024-09-23 12:15:00 | 61.02 | EXIT_EMA400 | -1.36 |
| BUY | 2025-10-15 10:15:00 | 37.38 | 2025-10-15 11:15:00 | 38.05 | TARGET | 0.67 |
| BUY | 2025-10-15 11:15:00 | 38.06 | 2025-10-17 14:15:00 | 36.99 | EXIT_EMA400 | -1.07 |
| SELL | 2025-12-24 13:15:00 | 37.05 | 2025-12-31 11:15:00 | 37.51 | EXIT_EMA400 | -0.46 |
| SELL | 2026-02-02 09:15:00 | 36.10 | 2026-02-09 09:15:00 | 37.70 | EXIT_EMA400 | -1.60 |
| BUY | 2026-03-05 15:15:00 | 37.77 | 2026-03-06 09:15:00 | 37.69 | EXIT_EMA400 | -0.08 |

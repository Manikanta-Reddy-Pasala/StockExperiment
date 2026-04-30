# Samvardhana Motherson International Ltd. (MOTHERSON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 121.21
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 32.41
- **Avg P&L per closed trade:** 5.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 61.73 | 63.05 | 63.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 13:15:00 | 61.30 | 63.01 | 63.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 61.67 | 60.71 | 61.58 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 15:15:00 | 65.93 | 62.08 | 62.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 66.23 | 62.12 | 62.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 62.50 | 62.74 | 62.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-26 11:15:00 | 64.03 | 62.76 | 62.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 74.37 | 77.10 | 74.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 12:15:00 | 74.27 | 77.07 | 74.29 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 118.50 | 129.20 | 129.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 117.81 | 129.09 | 129.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 116.10 | 115.71 | 120.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 11:15:00 | 114.92 | 115.70 | 120.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 88.58 | 85.88 | 90.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 88.38 | 85.90 | 90.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 88.83 | 86.38 | 90.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-28 10:15:00 | 88.70 | 86.40 | 90.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 87.43 | 84.19 | 87.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 13:15:00 | 88.29 | 84.26 | 87.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 94.60 | 89.55 | 89.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 95.28 | 89.79 | 89.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 99.73 | 99.84 | 96.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-19 10:15:00 | 100.61 | 99.85 | 96.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-22 10:15:00 | 99.50 | 101.64 | 99.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 89.94 | 98.76 | 98.78 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 107.30 | 97.72 | 97.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 107.91 | 97.82 | 97.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 103.15 | 104.02 | 101.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 10:15:00 | 104.47 | 103.85 | 101.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 104.06 | 105.26 | 103.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-06 11:15:00 | 103.46 | 105.23 | 103.62 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 12:15:00 | 110.78 | 119.62 | 119.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 107.25 | 118.30 | 118.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 117.41 | 115.15 | 117.12 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 130.89 | 118.47 | 118.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 131.47 | 118.60 | 118.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 120.05 | 121.11 | 119.93 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-26 11:15:00 | 64.03 | 2024-01-01 09:15:00 | 68.75 | TARGET | 4.72 |
| SELL | 2024-12-06 11:15:00 | 114.92 | 2025-01-10 09:15:00 | 98.25 | TARGET | 16.67 |
| SELL | 2025-03-28 10:15:00 | 88.70 | 2025-04-04 09:15:00 | 82.95 | TARGET | 5.75 |
| SELL | 2025-03-25 10:15:00 | 88.38 | 2025-04-04 10:15:00 | 80.98 | TARGET | 7.40 |
| BUY | 2025-06-19 10:15:00 | 100.61 | 2025-07-22 10:15:00 | 99.50 | EXIT_EMA400 | -1.11 |
| BUY | 2025-10-10 10:15:00 | 104.47 | 2025-11-06 11:15:00 | 103.46 | EXIT_EMA400 | -1.01 |

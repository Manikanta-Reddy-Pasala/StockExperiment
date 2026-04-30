# Ashok Leyland Ltd. (ASHOKLEY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 162.09
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 11 |
| ENTRY1 | 8 |
| ENTRY2 | 7 |
| EXIT | 8 |

## P&L

- **Trades closed:** 15
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / EMA400 exits:** 5 / 10
- **Total realized P&L (per unit):** 3.77
- **Avg P&L per closed trade:** 0.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 10:15:00 | 85.12 | 88.37 | 88.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 84.62 | 88.33 | 88.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 13:15:00 | 86.62 | 86.16 | 87.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-09 14:15:00 | 85.50 | 86.20 | 87.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 85.50 | 86.20 | 87.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-13 09:15:00 | 87.03 | 86.22 | 87.01 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 90.40 | 87.45 | 87.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 13:15:00 | 90.65 | 87.51 | 87.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 14:15:00 | 87.28 | 87.76 | 87.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-06 11:15:00 | 88.05 | 87.73 | 87.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 88.05 | 87.73 | 87.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-06 13:15:00 | 88.30 | 87.73 | 87.61 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 87.80 | 87.81 | 87.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-12-08 12:15:00 | 87.62 | 87.81 | 87.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-12-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 11:15:00 | 84.95 | 87.54 | 87.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 14:15:00 | 84.68 | 87.46 | 87.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 09:15:00 | 87.35 | 87.34 | 87.44 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 15:15:00 | 90.78 | 87.53 | 87.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 92.80 | 87.58 | 87.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 87.78 | 88.55 | 88.11 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 85.20 | 87.83 | 87.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 84.20 | 87.76 | 87.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 87.97 | 87.33 | 87.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-02 13:15:00 | 86.72 | 87.33 | 87.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 87.20 | 87.33 | 87.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-02-05 10:15:00 | 87.00 | 87.32 | 87.54 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-02-05 11:15:00 | 88.55 | 87.33 | 87.54 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 14:15:00 | 87.12 | 86.17 | 86.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 10:15:00 | 88.00 | 86.21 | 86.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 86.22 | 86.32 | 86.25 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 10:15:00 | 85.80 | 86.17 | 86.17 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 86.28 | 86.17 | 86.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 86.55 | 86.18 | 86.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 100.53 | 102.93 | 97.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 13:15:00 | 109.80 | 103.27 | 97.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 113.72 | 114.20 | 110.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 14:15:00 | 116.25 | 114.24 | 110.20 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 121.47 | 125.28 | 121.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-09 14:15:00 | 121.95 | 125.21 | 121.14 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 121.28 | 124.98 | 121.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-11 13:15:00 | 120.68 | 124.94 | 121.26 | Close below EMA400 |

### Cycle 9 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 111.53 | 119.82 | 119.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 14:15:00 | 111.25 | 119.48 | 119.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 111.04 | 110.13 | 113.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 107.62 | 110.32 | 113.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 112.28 | 110.17 | 112.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-19 14:15:00 | 110.44 | 110.17 | 112.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 114.70 | 110.25 | 112.67 | Close above EMA400 |

### Cycle 10 — BUY (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 14:15:00 | 115.37 | 114.08 | 114.07 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 112.34 | 114.06 | 114.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 111.49 | 114.03 | 114.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 114.46 | 112.22 | 112.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-07 11:15:00 | 112.19 | 112.88 | 113.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 112.19 | 112.88 | 113.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-08 09:15:00 | 111.37 | 112.85 | 113.22 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 108.83 | 106.43 | 108.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-03 09:15:00 | 100.65 | 106.44 | 108.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-12 14:15:00 | 109.50 | 105.56 | 107.84 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 115.31 | 106.41 | 106.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 115.60 | 109.60 | 108.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 116.43 | 117.39 | 114.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 117.01 | 117.38 | 114.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 119.55 | 122.97 | 120.59 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 12:15:00 | 171.50 | 185.30 | 185.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 164.38 | 184.68 | 185.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 175.71 | 173.58 | 178.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 172.56 | 173.77 | 178.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 177.40 | 173.74 | 178.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 178.63 | 173.78 | 178.27 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-09 14:15:00 | 85.50 | 2023-11-13 09:15:00 | 87.03 | EXIT_EMA400 | -1.53 |
| BUY | 2023-12-06 11:15:00 | 88.05 | 2023-12-07 14:15:00 | 89.39 | TARGET | 1.34 |
| BUY | 2023-12-06 13:15:00 | 88.30 | 2023-12-08 12:15:00 | 87.62 | EXIT_EMA400 | -0.68 |
| SELL | 2024-02-02 13:15:00 | 86.72 | 2024-02-05 11:15:00 | 88.55 | EXIT_EMA400 | -1.83 |
| SELL | 2024-02-05 10:15:00 | 87.00 | 2024-02-05 11:15:00 | 88.55 | EXIT_EMA400 | -1.55 |
| BUY | 2024-09-09 14:15:00 | 121.95 | 2024-09-10 12:15:00 | 124.37 | TARGET | 2.42 |
| BUY | 2024-06-05 13:15:00 | 109.80 | 2024-09-11 13:15:00 | 120.68 | EXIT_EMA400 | 10.88 |
| BUY | 2024-07-24 14:15:00 | 116.25 | 2024-09-11 13:15:00 | 120.68 | EXIT_EMA400 | 4.43 |
| SELL | 2024-11-13 09:15:00 | 107.62 | 2024-11-25 09:15:00 | 114.70 | EXIT_EMA400 | -7.07 |
| SELL | 2024-11-19 14:15:00 | 110.44 | 2024-11-25 09:15:00 | 114.70 | EXIT_EMA400 | -4.25 |
| SELL | 2025-01-07 11:15:00 | 112.19 | 2025-01-09 11:15:00 | 109.01 | TARGET | 3.18 |
| SELL | 2025-01-08 09:15:00 | 111.37 | 2025-01-10 09:15:00 | 105.81 | TARGET | 5.56 |
| SELL | 2025-02-03 09:15:00 | 100.65 | 2025-02-12 14:15:00 | 109.50 | EXIT_EMA400 | -8.85 |
| BUY | 2025-06-13 10:15:00 | 117.01 | 2025-06-27 09:15:00 | 124.79 | TARGET | 7.77 |
| SELL | 2026-04-13 09:15:00 | 172.56 | 2026-04-15 10:15:00 | 178.63 | EXIT_EMA400 | -6.07 |

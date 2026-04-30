# Steel Authority of India Ltd. (SAIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 184.62
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 5 |
| EXIT | 6 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / EMA400 exits:** 4 / 7
- **Total realized P&L (per unit):** 48.46
- **Avg P&L per closed trade:** 4.41

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 12:15:00 | 88.05 | 91.01 | 91.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 87.20 | 90.85 | 90.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 13:15:00 | 87.30 | 87.22 | 88.64 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 11:15:00 | 94.25 | 89.30 | 89.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 95.90 | 89.60 | 89.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 109.80 | 111.81 | 105.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 09:15:00 | 114.05 | 111.87 | 105.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 11:15:00 | 122.20 | 128.94 | 122.52 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 140.03 | 150.41 | 150.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 135.14 | 147.82 | 148.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 09:15:00 | 134.61 | 132.37 | 136.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-08 09:15:00 | 131.06 | 134.89 | 137.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 134.69 | 134.15 | 136.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-14 10:15:00 | 134.05 | 134.15 | 136.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 122.67 | 118.54 | 123.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-04 10:15:00 | 121.73 | 118.57 | 123.26 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 122.08 | 118.71 | 123.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-05 09:15:00 | 121.60 | 118.78 | 123.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 122.50 | 118.92 | 123.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-06 10:15:00 | 123.90 | 119.08 | 123.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 116.60 | 111.10 | 111.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 117.75 | 111.21 | 111.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 103.43 | 111.98 | 111.56 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 14:15:00 | 104.77 | 111.13 | 111.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 102.01 | 110.98 | 111.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 111.80 | 110.45 | 110.79 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 114.64 | 111.09 | 111.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 116.20 | 111.15 | 111.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 113.03 | 113.19 | 112.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 09:15:00 | 116.11 | 113.22 | 112.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 112.92 | 113.23 | 112.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-02 13:15:00 | 113.39 | 113.23 | 112.33 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 111.82 | 113.31 | 112.44 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 121.35 | 126.97 | 127.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 121.17 | 126.92 | 126.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 124.41 | 124.04 | 125.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-02 15:15:00 | 122.90 | 124.00 | 125.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 125.00 | 124.01 | 125.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-03 10:15:00 | 126.52 | 124.04 | 125.24 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 131.75 | 126.17 | 126.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 131.80 | 126.23 | 126.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 15:15:00 | 130.69 | 130.77 | 128.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 133.22 | 130.79 | 128.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 131.25 | 132.02 | 130.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 11:15:00 | 131.62 | 132.01 | 130.14 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 11:15:00 | 129.64 | 131.97 | 130.19 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 129.83 | 133.46 | 133.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 127.59 | 132.97 | 133.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 10:15:00 | 132.24 | 132.20 | 132.78 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 147.34 | 133.23 | 133.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 13:15:00 | 149.25 | 136.22 | 134.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 143.94 | 146.34 | 141.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-02 14:15:00 | 147.95 | 146.29 | 141.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 146.93 | 156.74 | 151.56 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-19 09:15:00 | 114.05 | 2024-02-06 13:15:00 | 139.21 | TARGET | 25.16 |
| SELL | 2024-10-14 10:15:00 | 134.05 | 2024-10-18 09:15:00 | 127.19 | TARGET | 6.86 |
| SELL | 2024-10-08 09:15:00 | 131.06 | 2024-10-25 09:15:00 | 113.17 | TARGET | 17.89 |
| SELL | 2024-12-04 10:15:00 | 121.73 | 2024-12-06 10:15:00 | 123.90 | EXIT_EMA400 | -2.17 |
| SELL | 2024-12-05 09:15:00 | 121.60 | 2024-12-06 10:15:00 | 123.90 | EXIT_EMA400 | -2.30 |
| BUY | 2025-05-02 09:15:00 | 116.11 | 2025-05-06 14:15:00 | 111.82 | EXIT_EMA400 | -4.29 |
| BUY | 2025-05-02 13:15:00 | 113.39 | 2025-05-06 14:15:00 | 111.82 | EXIT_EMA400 | -1.57 |
| SELL | 2025-09-02 15:15:00 | 122.90 | 2025-09-03 10:15:00 | 126.52 | EXIT_EMA400 | -3.62 |
| BUY | 2025-09-29 09:15:00 | 133.22 | 2025-10-14 11:15:00 | 129.64 | EXIT_EMA400 | -3.58 |
| BUY | 2025-10-13 11:15:00 | 131.62 | 2025-10-14 11:15:00 | 129.64 | EXIT_EMA400 | -1.98 |
| BUY | 2026-02-02 14:15:00 | 147.95 | 2026-02-25 09:15:00 | 166.02 | TARGET | 18.07 |

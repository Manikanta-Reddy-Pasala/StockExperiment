# Godfrey Phillips India Ltd. (GODFRYPHLP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 2251.00
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
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 1259.67
- **Avg P&L per closed trade:** 157.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 13:15:00 | 690.82 | 700.09 | 700.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 689.23 | 699.98 | 700.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 10:15:00 | 702.08 | 699.90 | 700.02 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 11:15:00 | 708.37 | 700.17 | 700.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 690.00 | 700.10 | 700.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 683.37 | 699.86 | 700.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 701.78 | 699.72 | 699.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-22 12:15:00 | 694.63 | 699.67 | 699.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-22 14:15:00 | 702.27 | 699.64 | 699.88 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 14:15:00 | 717.90 | 700.09 | 700.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 14:15:00 | 737.83 | 704.93 | 702.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 707.72 | 715.19 | 708.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 13:15:00 | 718.72 | 714.07 | 708.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1182.73 | 1197.64 | 1116.09 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-04 13:15:00 | 1190.82 | 1197.57 | 1116.46 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1368.85 | 1404.16 | 1340.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-05 10:15:00 | 1339.22 | 1403.51 | 1340.75 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 1927.83 | 2082.55 | 2082.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 1897.22 | 2075.95 | 2079.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 2013.00 | 2012.68 | 2043.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 13:15:00 | 1961.50 | 2008.36 | 2036.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-03 11:15:00 | 1732.80 | 1583.01 | 1715.14 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 2170.02 | 1772.53 | 1772.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 2228.35 | 1919.15 | 1860.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 2708.67 | 2709.09 | 2490.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 13:15:00 | 2793.17 | 2710.50 | 2495.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2629.67 | 2743.04 | 2612.94 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-24 10:15:00 | 2687.67 | 2732.59 | 2616.91 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3305.00 | 3437.52 | 3302.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 12:15:00 | 3288.00 | 3433.66 | 3302.23 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 3102.00 | 3283.94 | 3284.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 3028.00 | 3276.29 | 3280.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 2302.90 | 2174.28 | 2377.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-02 10:15:00 | 2062.00 | 2196.42 | 2342.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 2130.70 | 2027.27 | 2131.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-15 13:15:00 | 2084.60 | 2028.62 | 2131.39 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 2122.00 | 2031.45 | 2130.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 2230.90 | 2037.80 | 2131.07 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-22 12:15:00 | 694.63 | 2023-12-22 14:15:00 | 702.27 | EXIT_EMA400 | -7.63 |
| BUY | 2024-01-19 13:15:00 | 718.72 | 2024-01-25 09:15:00 | 747.94 | TARGET | 29.23 |
| BUY | 2024-06-04 13:15:00 | 1190.82 | 2024-06-18 10:15:00 | 1413.87 | TARGET | 223.06 |
| SELL | 2024-12-17 13:15:00 | 1961.50 | 2024-12-23 09:15:00 | 1735.64 | TARGET | 225.86 |
| BUY | 2025-06-24 10:15:00 | 2687.67 | 2025-06-26 09:15:00 | 2899.93 | TARGET | 212.26 |
| BUY | 2025-05-28 13:15:00 | 2793.17 | 2025-08-07 09:15:00 | 3685.26 | TARGET | 892.09 |
| SELL | 2026-03-02 10:15:00 | 2062.00 | 2026-04-17 09:15:00 | 2230.90 | EXIT_EMA400 | -168.90 |
| SELL | 2026-04-15 13:15:00 | 2084.60 | 2026-04-17 09:15:00 | 2230.90 | EXIT_EMA400 | -146.30 |

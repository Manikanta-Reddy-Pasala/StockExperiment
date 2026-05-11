# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 425.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 200 |
| ALERT1 | 134 |
| ALERT2 | 134 |
| ALERT2_SKIP | 87 |
| ALERT3 | 284 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 117 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 83
- **Target hits / Stop hits / Partials:** 8 / 111 / 13
- **Avg / median % per leg:** 0.59% / -0.71%
- **Sum % (uncompounded):** 78.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 17 | 34.7% | 7 | 42 | 0 | 0.84% | 41.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 49 | 17 | 34.7% | 7 | 42 | 0 | 0.84% | 41.1% |
| SELL (all) | 83 | 32 | 38.6% | 1 | 69 | 13 | 0.45% | 37.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.23% | -0.5% |
| SELL @ 3rd Alert (retest2) | 81 | 32 | 39.5% | 1 | 67 | 13 | 0.46% | 37.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.23% | -0.5% |
| retest2 (combined) | 130 | 49 | 37.7% | 8 | 109 | 13 | 0.60% | 78.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 12:15:00 | 107.25 | 107.66 | 107.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 13:15:00 | 106.85 | 107.13 | 107.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 108.20 | 105.79 | 106.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 108.20 | 105.79 | 106.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 108.20 | 105.79 | 106.22 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 107.80 | 106.48 | 106.47 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 14:15:00 | 105.90 | 106.59 | 106.66 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 107.60 | 106.75 | 106.72 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 11:15:00 | 106.55 | 106.70 | 106.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 13:15:00 | 105.95 | 106.55 | 106.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 106.10 | 105.71 | 106.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 106.10 | 105.71 | 106.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 106.10 | 105.71 | 106.01 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 11:15:00 | 107.55 | 105.82 | 105.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 108.55 | 106.37 | 105.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 09:15:00 | 110.60 | 111.25 | 109.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 109.60 | 111.23 | 110.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 109.60 | 111.23 | 110.46 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 110.35 | 111.30 | 111.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 14:15:00 | 109.75 | 110.71 | 111.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 110.50 | 110.46 | 110.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 112.40 | 110.84 | 110.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 112.40 | 110.84 | 110.97 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 112.30 | 111.14 | 111.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 114.40 | 112.14 | 111.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 13:15:00 | 113.00 | 113.31 | 112.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 14:15:00 | 112.70 | 113.19 | 112.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 14:15:00 | 112.70 | 113.19 | 112.57 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 124.75 | 126.97 | 127.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 122.75 | 126.13 | 126.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 09:15:00 | 125.85 | 125.03 | 125.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 125.85 | 125.03 | 125.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 125.85 | 125.03 | 125.98 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 09:15:00 | 128.00 | 126.36 | 126.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 10:15:00 | 128.70 | 126.83 | 126.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 144.25 | 147.70 | 144.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 144.25 | 147.70 | 144.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 144.25 | 147.70 | 144.56 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 173.25 | 178.61 | 179.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 12:15:00 | 172.00 | 177.29 | 178.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 12:15:00 | 171.15 | 170.95 | 172.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 13:15:00 | 172.05 | 171.17 | 172.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 13:15:00 | 172.05 | 171.17 | 172.47 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 174.45 | 172.88 | 172.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 10:15:00 | 176.10 | 174.58 | 174.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 13:15:00 | 174.20 | 174.71 | 174.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 13:15:00 | 174.20 | 174.71 | 174.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 174.20 | 174.71 | 174.25 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 09:15:00 | 174.15 | 174.93 | 175.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 12:15:00 | 172.80 | 174.26 | 174.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 169.70 | 168.87 | 170.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 12:15:00 | 171.35 | 169.22 | 170.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 171.35 | 169.22 | 170.65 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 13:15:00 | 171.50 | 171.00 | 170.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 173.75 | 171.66 | 171.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 12:15:00 | 174.50 | 174.50 | 173.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 12:15:00 | 174.50 | 174.50 | 173.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 174.50 | 174.50 | 173.45 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 12:15:00 | 218.00 | 220.63 | 220.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 212.95 | 217.71 | 219.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 216.30 | 213.69 | 215.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 216.30 | 213.69 | 215.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 216.30 | 213.69 | 215.90 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 12:15:00 | 221.85 | 217.70 | 217.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 227.15 | 220.80 | 219.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 14:15:00 | 246.35 | 248.23 | 242.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 247.85 | 250.85 | 248.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 247.85 | 250.85 | 248.62 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 15:15:00 | 244.00 | 247.40 | 247.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 12:15:00 | 240.85 | 244.76 | 246.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 10:15:00 | 241.75 | 241.70 | 243.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 14:15:00 | 241.55 | 241.38 | 242.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 241.55 | 241.38 | 242.92 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 247.00 | 243.87 | 243.77 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 234.40 | 243.34 | 244.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 217.20 | 231.75 | 237.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 227.20 | 226.46 | 231.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 227.20 | 226.46 | 231.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 227.20 | 226.46 | 231.16 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 13:15:00 | 233.30 | 231.26 | 231.19 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 14:15:00 | 227.50 | 230.50 | 230.85 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 232.50 | 231.28 | 231.15 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 13:15:00 | 230.00 | 231.06 | 231.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 229.45 | 230.74 | 230.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 218.05 | 215.08 | 218.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 218.05 | 215.08 | 218.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 218.05 | 215.08 | 218.19 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 15:15:00 | 217.45 | 213.93 | 213.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 09:15:00 | 220.60 | 215.27 | 214.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 222.40 | 226.02 | 224.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 222.40 | 226.02 | 224.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 222.40 | 226.02 | 224.15 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 290.75 | 295.25 | 295.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 290.10 | 294.22 | 295.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 295.60 | 292.55 | 293.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 295.60 | 292.55 | 293.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 295.60 | 292.55 | 293.89 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 15:15:00 | 290.40 | 288.64 | 288.59 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 13:15:00 | 287.75 | 288.58 | 288.62 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 09:15:00 | 299.50 | 290.78 | 289.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 14:15:00 | 303.95 | 299.15 | 297.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 10:15:00 | 337.40 | 343.08 | 334.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 10:15:00 | 337.40 | 343.08 | 334.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 337.40 | 343.08 | 334.38 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 12:15:00 | 332.40 | 336.42 | 336.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-15 13:15:00 | 328.00 | 334.74 | 335.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 319.55 | 318.55 | 324.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 10:15:00 | 319.65 | 316.65 | 318.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 319.65 | 316.65 | 318.44 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 324.60 | 319.39 | 319.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 14:15:00 | 332.65 | 322.04 | 320.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 09:15:00 | 329.45 | 331.11 | 327.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 10:15:00 | 331.45 | 331.17 | 328.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 331.45 | 331.17 | 328.07 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 13:15:00 | 328.15 | 329.27 | 329.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 15:15:00 | 327.00 | 328.57 | 328.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 325.80 | 325.26 | 326.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 325.80 | 325.26 | 326.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 325.80 | 325.26 | 326.78 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 333.80 | 328.18 | 327.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 339.95 | 335.69 | 333.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 14:15:00 | 337.25 | 337.36 | 335.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 329.35 | 335.62 | 334.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 329.35 | 335.62 | 334.74 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 11:15:00 | 328.70 | 333.25 | 333.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 328.15 | 332.23 | 333.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 09:15:00 | 324.95 | 324.43 | 327.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 320.50 | 322.76 | 324.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 320.50 | 322.76 | 324.93 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 322.05 | 319.80 | 319.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 325.75 | 321.50 | 320.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 09:15:00 | 322.65 | 322.94 | 321.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 12:15:00 | 319.00 | 321.98 | 321.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 319.00 | 321.98 | 321.56 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 14:15:00 | 316.75 | 320.46 | 320.91 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 331.40 | 319.99 | 318.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 11:15:00 | 336.90 | 323.37 | 320.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 323.40 | 325.86 | 322.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 15:15:00 | 325.00 | 325.69 | 322.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 325.00 | 325.69 | 322.86 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 15:15:00 | 359.00 | 360.24 | 360.33 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 09:15:00 | 367.60 | 361.71 | 360.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 11:15:00 | 371.00 | 364.86 | 362.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 10:15:00 | 378.00 | 379.87 | 375.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 382.55 | 387.75 | 385.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 382.55 | 387.75 | 385.85 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 12:15:00 | 379.30 | 384.15 | 384.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 377.85 | 381.63 | 383.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 15:15:00 | 365.00 | 364.83 | 370.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 365.15 | 364.01 | 366.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 365.15 | 364.01 | 366.95 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 366.55 | 357.25 | 356.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 376.90 | 365.61 | 361.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 12:15:00 | 355.05 | 366.33 | 363.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 12:15:00 | 355.05 | 366.33 | 363.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 355.05 | 366.33 | 363.20 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 14:15:00 | 349.65 | 361.10 | 361.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 09:15:00 | 343.95 | 356.42 | 359.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 12:15:00 | 343.65 | 340.54 | 346.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 12:15:00 | 343.65 | 340.54 | 346.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 343.65 | 340.54 | 346.14 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 349.50 | 337.82 | 337.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 11:15:00 | 355.70 | 343.55 | 340.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 348.00 | 350.10 | 345.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 348.00 | 350.10 | 345.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 348.00 | 350.10 | 345.27 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 09:15:00 | 371.80 | 375.52 | 375.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 10:15:00 | 367.55 | 373.92 | 374.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 374.50 | 369.76 | 371.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 374.50 | 369.76 | 371.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 374.50 | 369.76 | 371.84 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 11:15:00 | 383.10 | 374.45 | 373.73 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 15:15:00 | 378.00 | 379.99 | 380.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 10:15:00 | 376.85 | 379.03 | 379.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 375.15 | 375.09 | 377.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 375.15 | 375.09 | 377.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 375.15 | 375.09 | 377.23 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 09:15:00 | 386.90 | 378.19 | 377.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 10:15:00 | 389.50 | 380.45 | 378.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 407.15 | 407.31 | 402.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 14:15:00 | 402.15 | 405.75 | 403.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 402.15 | 405.75 | 403.52 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 394.85 | 401.82 | 402.15 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 15:15:00 | 403.25 | 401.88 | 401.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 09:15:00 | 408.75 | 403.25 | 402.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 10:15:00 | 396.55 | 405.54 | 404.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 10:15:00 | 396.55 | 405.54 | 404.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 396.55 | 405.54 | 404.82 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 393.75 | 403.18 | 403.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 393.00 | 397.78 | 400.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 14:15:00 | 400.30 | 378.31 | 385.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 14:15:00 | 400.30 | 378.31 | 385.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 400.30 | 378.31 | 385.08 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 375.10 | 370.42 | 370.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 378.10 | 372.97 | 371.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 15:15:00 | 398.00 | 398.31 | 390.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 15:15:00 | 428.00 | 428.89 | 421.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 428.00 | 428.89 | 421.99 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 425.25 | 430.19 | 430.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 09:15:00 | 419.75 | 427.19 | 429.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 425.10 | 421.57 | 424.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 425.10 | 421.57 | 424.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 425.10 | 421.57 | 424.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 425.10 | 421.57 | 424.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 429.25 | 423.10 | 425.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:45:00 | 429.45 | 423.10 | 425.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 429.55 | 424.39 | 425.44 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 13:15:00 | 429.60 | 426.09 | 426.07 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 424.40 | 425.85 | 426.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 418.35 | 424.16 | 425.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 11:15:00 | 409.90 | 408.92 | 412.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 405.80 | 408.16 | 411.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 405.80 | 408.16 | 411.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 14:45:00 | 402.90 | 405.73 | 408.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 09:45:00 | 404.00 | 405.18 | 407.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 417.60 | 407.67 | 408.79 | SL hit (close>static) qty=1.00 sl=415.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 12:15:00 | 415.00 | 410.43 | 409.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 11:15:00 | 416.55 | 414.34 | 412.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 11:15:00 | 418.05 | 418.65 | 416.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 12:00:00 | 418.05 | 418.65 | 416.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 416.90 | 418.30 | 416.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:45:00 | 416.65 | 418.30 | 416.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 415.10 | 417.66 | 416.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 13:30:00 | 415.10 | 417.66 | 416.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 418.05 | 417.74 | 416.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 421.60 | 417.89 | 416.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 12:15:00 | 414.65 | 417.63 | 417.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 12:15:00 | 414.65 | 417.63 | 417.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 14:15:00 | 411.30 | 415.91 | 416.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 415.85 | 415.40 | 416.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 415.85 | 415.40 | 416.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 415.85 | 415.40 | 416.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 11:45:00 | 413.35 | 415.18 | 416.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 12:30:00 | 413.65 | 414.93 | 415.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 13:00:00 | 413.95 | 414.93 | 415.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:30:00 | 414.05 | 414.81 | 415.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 414.75 | 414.75 | 415.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:30:00 | 413.85 | 414.72 | 415.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 11:45:00 | 413.70 | 414.37 | 415.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 392.68 | 405.75 | 409.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 392.97 | 405.75 | 409.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 393.25 | 405.75 | 409.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 393.35 | 405.75 | 409.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 393.16 | 405.75 | 409.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 393.01 | 405.75 | 409.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-06 15:15:00 | 401.00 | 400.94 | 404.84 | SL hit (close>ema200) qty=0.50 sl=400.94 alert=retest2 |

### Cycle 56 — BUY (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 14:15:00 | 415.00 | 398.23 | 397.31 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 10:15:00 | 395.30 | 398.22 | 398.43 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 407.90 | 399.03 | 398.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 408.75 | 400.98 | 399.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 09:15:00 | 412.30 | 413.72 | 411.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 412.30 | 413.72 | 411.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 412.30 | 413.72 | 411.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:30:00 | 411.70 | 413.72 | 411.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 413.00 | 413.58 | 411.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 412.95 | 413.58 | 411.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 411.35 | 413.07 | 411.57 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 400.15 | 409.60 | 410.19 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 409.00 | 404.77 | 404.27 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 397.75 | 403.35 | 403.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 394.45 | 398.32 | 400.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 387.95 | 383.75 | 388.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 387.95 | 383.75 | 388.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 387.95 | 383.75 | 388.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 387.95 | 383.75 | 388.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 385.00 | 384.00 | 388.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 401.35 | 384.00 | 388.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 399.85 | 387.17 | 389.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:30:00 | 402.70 | 387.17 | 389.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 396.05 | 390.24 | 390.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:00:00 | 396.05 | 390.24 | 390.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 393.60 | 390.92 | 390.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 09:15:00 | 398.20 | 392.51 | 391.47 | Break + close above crossover candle high |

### Cycle 63 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 378.40 | 389.68 | 390.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 364.05 | 384.56 | 387.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 384.65 | 384.16 | 387.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 384.65 | 384.16 | 387.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 384.65 | 384.16 | 387.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 386.55 | 384.16 | 387.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 387.00 | 383.21 | 386.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 387.10 | 383.21 | 386.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 380.45 | 382.65 | 385.56 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 403.40 | 386.18 | 386.06 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 404.20 | 405.50 | 405.64 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 407.60 | 405.92 | 405.82 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 10:15:00 | 403.90 | 405.51 | 405.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 11:15:00 | 402.30 | 404.87 | 405.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 404.45 | 401.74 | 403.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 404.45 | 401.74 | 403.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 404.45 | 401.74 | 403.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 404.50 | 401.74 | 403.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 402.05 | 401.80 | 403.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 400.60 | 401.80 | 403.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 09:45:00 | 401.75 | 400.86 | 402.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 405.45 | 401.78 | 402.33 | SL hit (close>static) qty=1.00 sl=404.80 alert=retest2 |

### Cycle 68 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 406.85 | 403.47 | 403.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 418.05 | 406.39 | 404.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 421.10 | 424.76 | 417.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 421.10 | 424.76 | 417.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 438.45 | 441.28 | 439.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 438.45 | 441.28 | 439.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 439.40 | 440.90 | 439.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 449.55 | 440.90 | 439.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-28 09:15:00 | 494.51 | 473.78 | 461.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 12:15:00 | 487.90 | 489.61 | 489.65 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 498.95 | 491.44 | 490.45 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 12:15:00 | 488.05 | 489.89 | 489.91 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 493.45 | 490.60 | 490.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 495.00 | 491.48 | 490.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 14:15:00 | 502.65 | 505.13 | 502.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 14:15:00 | 502.65 | 505.13 | 502.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 502.65 | 505.13 | 502.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 502.65 | 505.13 | 502.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 502.50 | 504.61 | 502.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 499.25 | 503.82 | 502.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 500.95 | 503.25 | 502.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 500.95 | 503.25 | 502.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 500.70 | 502.74 | 501.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 500.55 | 502.74 | 501.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 500.20 | 502.16 | 501.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 500.20 | 502.16 | 501.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 503.55 | 502.44 | 501.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:30:00 | 512.90 | 505.28 | 503.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 523.85 | 528.26 | 525.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-24 09:15:00 | 564.19 | 548.50 | 536.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 564.60 | 570.37 | 571.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 560.05 | 567.67 | 569.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 14:15:00 | 562.20 | 561.13 | 565.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 15:00:00 | 562.20 | 561.13 | 565.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 546.55 | 537.98 | 548.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 15:00:00 | 546.55 | 537.98 | 548.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 538.80 | 538.14 | 547.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 549.15 | 538.14 | 547.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 545.70 | 539.65 | 547.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:15:00 | 539.05 | 542.25 | 546.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:30:00 | 538.60 | 539.08 | 542.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 537.80 | 539.08 | 542.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 535.85 | 539.56 | 542.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 533.10 | 536.00 | 538.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 540.60 | 538.40 | 538.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 11:15:00 | 540.60 | 538.40 | 538.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 543.45 | 539.41 | 538.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 556.70 | 558.08 | 551.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 556.70 | 558.08 | 551.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 553.45 | 557.01 | 551.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 550.85 | 557.01 | 551.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 553.40 | 556.29 | 551.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 553.40 | 556.29 | 551.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 553.65 | 555.76 | 552.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:15:00 | 554.35 | 555.76 | 552.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:45:00 | 554.95 | 555.46 | 552.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:00:00 | 562.40 | 556.85 | 553.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 12:15:00 | 550.00 | 560.15 | 560.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 12:15:00 | 550.00 | 560.15 | 560.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 533.00 | 550.75 | 555.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 14:15:00 | 543.80 | 543.27 | 549.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 14:45:00 | 543.40 | 543.27 | 549.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 536.25 | 541.31 | 547.35 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 573.85 | 550.48 | 548.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 14:15:00 | 598.60 | 575.77 | 563.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 15:15:00 | 585.95 | 586.82 | 577.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 09:15:00 | 596.00 | 586.82 | 577.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 612.20 | 612.37 | 604.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 610.65 | 612.37 | 604.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 610.00 | 611.20 | 607.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 13:15:00 | 622.30 | 615.40 | 612.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 630.00 | 615.15 | 612.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 13:30:00 | 626.15 | 623.46 | 618.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 636.75 | 643.58 | 644.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 636.75 | 643.58 | 644.27 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 15:15:00 | 649.00 | 644.87 | 644.68 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 643.25 | 644.36 | 644.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 15:15:00 | 641.00 | 643.00 | 643.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 15:15:00 | 641.00 | 640.72 | 641.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:15:00 | 640.60 | 640.72 | 641.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 80 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 652.55 | 643.08 | 642.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 655.00 | 647.58 | 645.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 708.50 | 710.77 | 693.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 708.50 | 710.77 | 693.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 693.00 | 707.22 | 693.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 693.00 | 707.22 | 693.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 694.50 | 704.67 | 693.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 691.00 | 704.67 | 693.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 709.40 | 705.62 | 695.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:30:00 | 704.60 | 705.62 | 695.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 699.00 | 702.92 | 696.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 707.55 | 702.92 | 696.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:30:00 | 704.95 | 703.28 | 697.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:00:00 | 706.25 | 703.28 | 697.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:30:00 | 714.25 | 709.13 | 703.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 703.25 | 707.95 | 703.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 703.25 | 707.95 | 703.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 701.95 | 706.75 | 703.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:45:00 | 701.90 | 706.75 | 703.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 704.60 | 706.32 | 703.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 688.95 | 699.97 | 701.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 688.95 | 699.97 | 701.12 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 707.00 | 701.33 | 701.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 713.65 | 705.10 | 703.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 767.70 | 768.93 | 757.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:45:00 | 767.20 | 768.93 | 757.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 757.60 | 765.25 | 758.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 757.60 | 765.25 | 758.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 760.00 | 764.20 | 758.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 751.70 | 764.20 | 758.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 735.00 | 758.36 | 756.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 735.00 | 758.36 | 756.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 732.95 | 753.28 | 754.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 14:15:00 | 717.75 | 735.20 | 744.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 13:15:00 | 721.95 | 721.02 | 732.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 14:00:00 | 721.95 | 721.02 | 732.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 715.10 | 715.98 | 726.71 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 740.10 | 727.56 | 727.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 750.05 | 733.88 | 730.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 740.20 | 743.05 | 737.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 12:00:00 | 740.20 | 743.05 | 737.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 736.85 | 741.81 | 737.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:30:00 | 736.65 | 741.81 | 737.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 732.00 | 739.85 | 737.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 732.00 | 739.85 | 737.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 730.85 | 738.05 | 736.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:45:00 | 726.40 | 738.05 | 736.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 723.25 | 734.73 | 735.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 711.90 | 724.85 | 729.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 703.00 | 699.89 | 712.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 703.00 | 699.89 | 712.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 709.30 | 700.64 | 709.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 709.30 | 700.64 | 709.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 713.65 | 703.25 | 709.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 712.05 | 703.25 | 709.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 711.65 | 704.93 | 709.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:45:00 | 708.00 | 707.04 | 710.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 707.90 | 709.43 | 710.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 708.35 | 709.18 | 710.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 708.20 | 709.18 | 710.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 718.10 | 710.97 | 711.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 718.10 | 710.97 | 711.00 | SL hit (close>static) qty=1.00 sl=716.50 alert=retest2 |

### Cycle 86 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 721.95 | 713.16 | 712.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 728.80 | 717.54 | 714.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 720.65 | 720.80 | 716.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:45:00 | 721.20 | 720.80 | 716.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 719.00 | 720.16 | 717.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 720.00 | 720.16 | 717.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 718.50 | 719.83 | 717.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 722.85 | 717.91 | 717.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 721.10 | 739.32 | 740.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 721.10 | 739.32 | 740.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 710.35 | 727.49 | 733.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 734.00 | 724.81 | 730.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 734.00 | 724.81 | 730.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 734.00 | 724.81 | 730.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 734.00 | 724.81 | 730.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 724.60 | 724.77 | 729.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 720.40 | 724.77 | 729.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 684.38 | 692.30 | 704.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 698.95 | 687.61 | 696.68 | SL hit (close>ema200) qty=0.50 sl=687.61 alert=retest2 |

### Cycle 88 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 674.50 | 672.82 | 672.82 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 670.25 | 672.31 | 672.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 663.10 | 670.47 | 671.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 674.75 | 665.18 | 667.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 674.75 | 665.18 | 667.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 674.75 | 665.18 | 667.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 674.75 | 665.18 | 667.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 668.70 | 665.88 | 668.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 662.75 | 665.88 | 668.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 690.00 | 660.35 | 658.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 690.00 | 660.35 | 658.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 698.15 | 679.32 | 669.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 698.70 | 701.29 | 691.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 10:45:00 | 698.20 | 701.29 | 691.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 694.50 | 696.85 | 692.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 690.80 | 696.85 | 692.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 689.85 | 695.45 | 692.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 696.00 | 695.45 | 692.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 683.40 | 699.20 | 699.06 | SL hit (close<static) qty=1.00 sl=685.35 alert=retest2 |

### Cycle 91 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 682.95 | 695.95 | 697.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 13:15:00 | 668.25 | 687.39 | 693.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 688.00 | 681.72 | 688.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 688.00 | 681.72 | 688.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 688.00 | 681.72 | 688.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:45:00 | 687.25 | 681.72 | 688.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 660.40 | 677.45 | 685.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 650.30 | 666.19 | 678.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:30:00 | 655.85 | 662.42 | 672.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 693.00 | 673.21 | 672.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 693.00 | 673.21 | 672.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 698.35 | 678.24 | 674.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 689.70 | 703.05 | 695.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 689.70 | 703.05 | 695.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 689.70 | 703.05 | 695.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 693.00 | 703.05 | 695.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 687.15 | 699.87 | 694.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:30:00 | 686.65 | 699.87 | 694.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 696.20 | 709.75 | 704.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 696.20 | 709.75 | 704.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 692.95 | 706.39 | 703.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 702.20 | 706.39 | 703.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:45:00 | 699.80 | 702.82 | 702.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 11:15:00 | 695.55 | 701.37 | 701.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 695.55 | 701.37 | 701.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 693.10 | 698.74 | 700.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 696.80 | 693.98 | 696.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 14:15:00 | 696.80 | 693.98 | 696.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 696.80 | 693.98 | 696.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 696.80 | 693.98 | 696.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 694.50 | 694.09 | 696.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 699.05 | 694.09 | 696.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 700.00 | 695.27 | 696.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 699.60 | 695.27 | 696.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 700.90 | 696.40 | 696.96 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 12:15:00 | 699.50 | 697.75 | 697.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 13:15:00 | 703.70 | 698.94 | 698.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 12:15:00 | 721.80 | 724.26 | 716.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 13:00:00 | 721.80 | 724.26 | 716.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 720.75 | 725.24 | 722.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 720.75 | 725.24 | 722.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 713.40 | 722.87 | 721.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 711.55 | 722.87 | 721.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 706.90 | 719.68 | 720.10 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 729.90 | 720.00 | 719.90 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 718.40 | 719.83 | 719.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 712.85 | 718.43 | 719.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 719.00 | 718.55 | 719.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 11:15:00 | 719.00 | 718.55 | 719.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 719.00 | 718.55 | 719.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:00:00 | 719.00 | 718.55 | 719.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 719.95 | 718.83 | 719.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:30:00 | 720.00 | 718.83 | 719.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 720.10 | 719.08 | 719.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:30:00 | 720.90 | 719.08 | 719.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 720.25 | 719.32 | 719.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:00:00 | 720.25 | 719.32 | 719.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 719.50 | 719.35 | 719.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 721.90 | 719.35 | 719.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 729.95 | 721.47 | 720.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 742.45 | 725.67 | 722.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 729.20 | 730.13 | 726.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 15:00:00 | 729.20 | 730.13 | 726.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 753.90 | 760.41 | 747.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 753.90 | 760.41 | 747.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 750.65 | 755.87 | 748.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 747.40 | 755.87 | 748.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 750.65 | 754.83 | 748.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:45:00 | 756.55 | 755.62 | 749.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 743.10 | 757.78 | 757.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 743.10 | 757.78 | 757.82 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 761.60 | 755.36 | 755.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 764.15 | 757.12 | 755.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 760.85 | 761.14 | 758.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:15:00 | 757.85 | 761.14 | 758.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 755.30 | 759.97 | 758.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 755.30 | 759.97 | 758.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 756.00 | 759.18 | 758.30 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 743.25 | 755.62 | 756.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 738.80 | 748.54 | 752.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 14:15:00 | 753.40 | 749.02 | 752.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 14:15:00 | 753.40 | 749.02 | 752.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 753.40 | 749.02 | 752.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 753.40 | 749.02 | 752.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 752.80 | 749.78 | 752.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 743.85 | 749.78 | 752.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 15:15:00 | 733.00 | 725.10 | 724.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 733.00 | 725.10 | 724.85 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 10:15:00 | 721.70 | 724.33 | 724.54 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 727.40 | 724.67 | 724.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 744.10 | 729.02 | 726.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 726.70 | 730.80 | 727.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 726.70 | 730.80 | 727.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 726.70 | 730.80 | 727.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 745.50 | 733.02 | 729.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 13:15:00 | 752.75 | 765.99 | 766.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 752.75 | 765.99 | 766.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 742.35 | 761.26 | 764.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 710.00 | 703.00 | 722.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 15:00:00 | 710.00 | 703.00 | 722.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 547.10 | 519.81 | 535.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 547.10 | 519.81 | 535.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 543.60 | 524.57 | 536.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:45:00 | 532.35 | 530.77 | 535.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 505.73 | 526.09 | 532.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 09:15:00 | 479.12 | 496.59 | 512.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 106 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 454.10 | 449.22 | 448.80 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 440.25 | 447.47 | 448.11 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 488.90 | 454.45 | 451.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 502.70 | 480.14 | 466.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 15:15:00 | 502.50 | 505.97 | 490.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 09:15:00 | 506.50 | 505.97 | 490.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 504.30 | 505.64 | 491.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 511.55 | 507.25 | 493.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 511.85 | 505.03 | 497.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:45:00 | 513.50 | 505.66 | 498.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:00:00 | 512.00 | 506.93 | 499.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-04 14:15:00 | 562.71 | 527.84 | 512.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 539.80 | 544.03 | 544.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 525.35 | 539.17 | 542.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 522.40 | 516.31 | 523.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 522.40 | 516.31 | 523.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 517.20 | 516.49 | 523.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 505.90 | 516.49 | 523.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 12:00:00 | 514.00 | 513.47 | 519.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:30:00 | 514.20 | 513.70 | 518.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 14:15:00 | 524.95 | 520.57 | 520.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 14:15:00 | 524.95 | 520.57 | 520.02 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 511.50 | 519.15 | 519.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 503.80 | 516.08 | 518.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 498.45 | 498.19 | 506.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 09:15:00 | 498.45 | 498.19 | 506.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 498.45 | 498.19 | 506.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:45:00 | 503.25 | 498.19 | 506.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 499.10 | 498.37 | 505.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 504.35 | 498.37 | 505.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 501.00 | 497.65 | 503.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 500.10 | 497.65 | 503.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 507.15 | 499.55 | 503.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 507.15 | 499.55 | 503.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 504.00 | 500.44 | 503.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 497.10 | 500.44 | 503.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 09:15:00 | 472.25 | 475.57 | 480.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 15:15:00 | 468.55 | 463.42 | 468.43 | SL hit (close>ema200) qty=0.50 sl=463.42 alert=retest2 |

### Cycle 112 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 464.60 | 454.01 | 453.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 467.30 | 460.95 | 457.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 457.10 | 461.81 | 459.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 457.10 | 461.81 | 459.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 457.10 | 461.81 | 459.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:45:00 | 457.80 | 461.81 | 459.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 458.00 | 461.05 | 459.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:15:00 | 454.90 | 461.05 | 459.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 455.60 | 459.96 | 458.89 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 14:15:00 | 449.95 | 457.96 | 458.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 15:15:00 | 448.85 | 456.14 | 457.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 417.20 | 416.77 | 427.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:45:00 | 418.15 | 416.77 | 427.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 422.00 | 416.35 | 423.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 422.00 | 416.35 | 423.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 421.90 | 417.46 | 423.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 427.50 | 417.46 | 423.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 422.35 | 418.44 | 423.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 427.45 | 418.44 | 423.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 424.75 | 419.70 | 423.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 427.20 | 419.70 | 423.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 422.70 | 420.30 | 423.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 418.30 | 423.08 | 423.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 425.30 | 423.68 | 424.07 | SL hit (close>static) qty=1.00 sl=425.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 433.15 | 425.81 | 424.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 442.60 | 435.40 | 432.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 474.30 | 483.63 | 474.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 474.30 | 483.63 | 474.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 474.30 | 483.63 | 474.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 474.30 | 483.63 | 474.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 473.30 | 481.57 | 474.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 473.30 | 481.57 | 474.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 470.60 | 479.37 | 474.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 468.75 | 479.37 | 474.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 469.00 | 474.19 | 473.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 474.05 | 474.19 | 473.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 475.00 | 474.95 | 473.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 469.75 | 473.64 | 473.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 469.75 | 473.64 | 473.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 466.65 | 471.77 | 472.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 471.00 | 470.97 | 472.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:15:00 | 475.95 | 470.97 | 472.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 477.15 | 472.20 | 472.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 479.10 | 472.20 | 472.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 476.65 | 473.09 | 472.96 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 467.75 | 471.95 | 472.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 463.65 | 468.73 | 470.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 465.60 | 462.13 | 465.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 465.60 | 462.13 | 465.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 465.60 | 462.13 | 465.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 463.85 | 462.13 | 465.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 479.30 | 465.57 | 466.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 489.70 | 465.57 | 466.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 490.50 | 470.55 | 468.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 497.50 | 479.33 | 473.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 487.05 | 501.86 | 493.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 487.05 | 501.86 | 493.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 487.05 | 501.86 | 493.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 487.05 | 501.86 | 493.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 491.60 | 499.81 | 493.46 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 466.50 | 486.37 | 488.82 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 489.85 | 485.82 | 485.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 494.05 | 488.50 | 487.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 484.50 | 488.61 | 487.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 484.50 | 488.61 | 487.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 484.50 | 488.61 | 487.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:30:00 | 485.95 | 488.61 | 487.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 485.60 | 488.01 | 487.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:30:00 | 490.45 | 488.61 | 487.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 522.95 | 525.98 | 526.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 522.95 | 525.98 | 526.00 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 530.55 | 526.46 | 526.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 537.90 | 529.22 | 527.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 11:15:00 | 529.90 | 530.58 | 528.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 11:45:00 | 530.25 | 530.58 | 528.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 524.25 | 529.46 | 528.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 524.25 | 529.46 | 528.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 519.85 | 527.54 | 527.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 15:15:00 | 518.95 | 525.82 | 527.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 519.25 | 509.89 | 515.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 519.25 | 509.89 | 515.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 519.25 | 509.89 | 515.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 518.05 | 509.89 | 515.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 516.55 | 511.22 | 515.81 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 518.00 | 517.07 | 517.05 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 516.10 | 517.24 | 517.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 509.55 | 515.04 | 516.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 519.70 | 512.71 | 514.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 519.70 | 512.71 | 514.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 519.70 | 512.71 | 514.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 519.70 | 512.71 | 514.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 521.15 | 514.40 | 514.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 520.80 | 514.40 | 514.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 520.00 | 515.52 | 515.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 524.30 | 517.27 | 516.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 516.80 | 521.44 | 518.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 516.80 | 521.44 | 518.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 516.80 | 521.44 | 518.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 516.80 | 521.44 | 518.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 515.05 | 520.16 | 518.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 515.05 | 520.16 | 518.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 512.40 | 517.41 | 517.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 505.75 | 515.08 | 516.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 511.20 | 508.52 | 512.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:00:00 | 511.20 | 508.52 | 512.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 515.65 | 509.94 | 512.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 513.95 | 509.94 | 512.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 518.95 | 511.75 | 512.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 518.95 | 511.75 | 512.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 522.80 | 513.96 | 513.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 536.00 | 522.48 | 519.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 14:15:00 | 552.15 | 552.52 | 545.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:30:00 | 552.00 | 552.52 | 545.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 558.30 | 558.78 | 554.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 555.95 | 558.78 | 554.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 553.50 | 558.60 | 556.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 552.25 | 558.60 | 556.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 551.95 | 557.27 | 556.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 551.95 | 557.27 | 556.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 549.20 | 554.62 | 555.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 545.20 | 552.74 | 554.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 555.10 | 551.56 | 553.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 555.10 | 551.56 | 553.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 555.10 | 551.56 | 553.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 554.55 | 551.56 | 553.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 555.75 | 552.40 | 553.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 557.10 | 552.40 | 553.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 551.20 | 551.75 | 552.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:45:00 | 549.50 | 550.92 | 552.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 548.90 | 550.67 | 551.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 547.85 | 550.13 | 551.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 554.55 | 551.02 | 551.77 | SL hit (close>static) qty=1.00 sl=553.55 alert=retest2 |

### Cycle 130 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 555.55 | 552.63 | 552.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 563.00 | 554.98 | 553.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 10:15:00 | 565.90 | 566.27 | 561.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:00:00 | 565.90 | 566.27 | 561.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 561.60 | 564.47 | 561.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 561.60 | 564.47 | 561.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 563.25 | 564.22 | 562.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 564.45 | 563.98 | 562.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 564.30 | 564.70 | 562.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 562.65 | 565.04 | 565.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 562.65 | 565.04 | 565.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 553.95 | 560.52 | 562.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 558.40 | 556.93 | 559.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 558.40 | 556.93 | 559.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 555.20 | 556.10 | 558.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 553.25 | 556.75 | 557.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:00:00 | 554.10 | 556.04 | 556.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 552.85 | 555.69 | 556.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:45:00 | 553.80 | 554.15 | 555.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 554.50 | 554.22 | 555.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 556.55 | 554.22 | 555.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 556.50 | 554.67 | 555.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 556.50 | 554.67 | 555.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 566.80 | 557.10 | 556.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 566.80 | 557.10 | 556.57 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 551.50 | 556.10 | 556.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 546.85 | 552.00 | 554.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 15:15:00 | 547.80 | 547.64 | 550.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 546.35 | 547.64 | 550.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 521.60 | 520.30 | 523.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 520.25 | 520.73 | 522.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 520.45 | 517.56 | 518.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 518.00 | 515.42 | 515.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 525.00 | 517.43 | 516.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 525.00 | 517.43 | 516.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 536.30 | 523.63 | 520.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 539.30 | 539.78 | 533.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:45:00 | 537.15 | 539.78 | 533.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 545.40 | 546.43 | 541.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 545.40 | 546.43 | 541.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 546.90 | 546.94 | 543.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:30:00 | 542.65 | 546.94 | 543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 587.65 | 584.55 | 580.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 591.45 | 585.80 | 581.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 589.95 | 586.20 | 581.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 566.60 | 580.83 | 580.75 | SL hit (close<static) qty=1.00 sl=570.75 alert=retest2 |

### Cycle 135 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 568.40 | 578.35 | 579.62 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 582.30 | 578.35 | 578.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 589.50 | 581.34 | 579.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 578.50 | 581.39 | 580.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 578.50 | 581.39 | 580.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 578.50 | 581.39 | 580.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 578.50 | 581.39 | 580.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 580.20 | 581.15 | 580.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 578.45 | 581.15 | 580.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 579.15 | 580.75 | 579.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 579.15 | 580.75 | 579.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 580.80 | 580.76 | 580.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 581.65 | 580.53 | 580.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 589.90 | 593.45 | 593.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 589.90 | 593.45 | 593.70 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 612.60 | 595.53 | 593.61 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 590.45 | 597.36 | 598.21 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 602.50 | 597.48 | 597.40 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 593.00 | 596.59 | 597.00 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 599.70 | 597.74 | 597.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 604.60 | 599.59 | 598.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 593.20 | 601.70 | 600.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 593.20 | 601.70 | 600.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 593.20 | 601.70 | 600.86 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 591.25 | 599.61 | 599.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 587.30 | 595.34 | 597.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 592.30 | 586.53 | 590.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 592.30 | 586.53 | 590.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 592.30 | 586.53 | 590.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 592.30 | 586.53 | 590.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 594.75 | 588.17 | 590.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 594.75 | 588.17 | 590.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 596.20 | 589.78 | 591.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 596.75 | 589.78 | 591.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 591.55 | 591.10 | 591.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 591.80 | 591.10 | 591.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 592.15 | 591.31 | 591.49 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 596.10 | 592.27 | 591.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 598.15 | 593.40 | 592.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 590.85 | 593.47 | 592.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 590.85 | 593.47 | 592.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 590.85 | 593.47 | 592.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 592.90 | 593.47 | 592.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 589.45 | 592.66 | 592.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 588.45 | 592.66 | 592.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 592.25 | 592.58 | 592.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 594.60 | 593.37 | 592.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 590.35 | 592.34 | 592.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 590.35 | 592.34 | 592.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 587.50 | 591.37 | 591.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 591.20 | 584.52 | 587.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 591.20 | 584.52 | 587.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 591.20 | 584.52 | 587.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 591.20 | 584.52 | 587.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 598.10 | 587.23 | 588.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 555.40 | 587.23 | 588.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 543.45 | 578.48 | 584.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 525.35 | 551.55 | 567.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 525.30 | 535.95 | 548.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 538.50 | 529.70 | 529.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 538.50 | 529.70 | 529.69 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 525.80 | 529.78 | 529.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 524.60 | 528.74 | 529.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 512.00 | 510.75 | 516.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:30:00 | 512.20 | 510.75 | 516.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 512.65 | 510.72 | 513.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 514.05 | 510.72 | 513.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 513.50 | 511.28 | 513.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 513.30 | 511.28 | 513.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 512.10 | 511.44 | 513.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 510.20 | 511.57 | 513.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 514.85 | 512.01 | 512.97 | SL hit (close>static) qty=1.00 sl=513.60 alert=retest2 |

### Cycle 148 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 510.20 | 507.62 | 507.62 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 502.60 | 506.96 | 507.34 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 510.70 | 506.98 | 506.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 514.50 | 509.18 | 507.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 508.10 | 509.54 | 508.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 508.10 | 509.54 | 508.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 508.10 | 509.54 | 508.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 506.50 | 509.54 | 508.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 508.80 | 509.39 | 508.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 518.25 | 509.25 | 508.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 503.15 | 510.86 | 510.35 | SL hit (close<static) qty=1.00 sl=507.40 alert=retest2 |

### Cycle 151 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 503.10 | 509.31 | 509.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 501.10 | 505.42 | 507.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 503.80 | 502.93 | 505.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 503.80 | 502.93 | 505.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 503.80 | 502.93 | 505.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 503.80 | 502.93 | 505.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 506.85 | 503.47 | 504.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 506.85 | 503.47 | 504.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 508.85 | 504.55 | 505.18 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 507.80 | 505.71 | 505.63 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 500.00 | 504.92 | 505.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 11:15:00 | 498.90 | 503.07 | 504.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 501.35 | 501.20 | 502.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:15:00 | 501.70 | 501.20 | 502.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 501.85 | 501.33 | 502.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 500.25 | 501.18 | 502.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 500.20 | 501.18 | 502.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:30:00 | 500.20 | 501.22 | 502.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 506.70 | 502.76 | 502.91 | SL hit (close>static) qty=1.00 sl=505.70 alert=retest2 |

### Cycle 154 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 507.00 | 503.61 | 503.28 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 502.65 | 504.32 | 504.41 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 506.50 | 504.77 | 504.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 508.20 | 505.55 | 505.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 517.50 | 519.89 | 515.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 517.50 | 519.89 | 515.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 512.95 | 518.50 | 515.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 512.95 | 518.50 | 515.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 514.05 | 517.61 | 515.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 514.05 | 517.61 | 515.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 514.00 | 516.45 | 514.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 514.20 | 516.45 | 514.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 514.10 | 515.98 | 514.87 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 512.70 | 514.18 | 514.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 511.20 | 513.58 | 513.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 458.00 | 457.77 | 468.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 455.00 | 457.77 | 468.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 458.55 | 453.02 | 459.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 455.00 | 453.50 | 458.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 454.20 | 453.72 | 458.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 462.45 | 458.98 | 458.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 462.45 | 458.98 | 458.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 465.90 | 460.36 | 459.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 481.50 | 481.80 | 475.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:45:00 | 481.70 | 481.80 | 475.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 486.50 | 487.21 | 484.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 486.50 | 487.21 | 484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 480.60 | 485.89 | 484.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 480.60 | 485.89 | 484.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 483.30 | 485.37 | 483.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 483.85 | 485.45 | 484.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:45:00 | 483.90 | 484.39 | 484.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 483.95 | 484.01 | 483.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 481.05 | 484.11 | 484.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 481.05 | 484.11 | 484.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 475.50 | 481.36 | 482.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 471.75 | 471.58 | 475.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 472.40 | 471.58 | 475.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 475.65 | 472.39 | 475.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 475.60 | 472.39 | 475.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 477.35 | 473.38 | 475.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 478.20 | 473.38 | 475.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 477.80 | 474.27 | 475.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 477.80 | 474.27 | 475.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 476.55 | 475.50 | 475.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 478.20 | 475.50 | 475.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 478.20 | 476.04 | 476.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 478.35 | 476.04 | 476.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 484.60 | 477.75 | 476.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 489.10 | 480.02 | 477.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 491.30 | 493.46 | 490.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 491.30 | 493.46 | 490.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 491.30 | 493.46 | 490.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 491.30 | 493.46 | 490.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 494.05 | 497.43 | 494.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 494.05 | 497.43 | 494.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 494.05 | 496.76 | 494.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 495.15 | 496.76 | 494.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 495.40 | 496.17 | 495.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 515.00 | 516.74 | 516.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 515.00 | 516.74 | 516.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 505.50 | 514.49 | 515.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 513.65 | 511.85 | 513.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 513.65 | 511.85 | 513.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 513.65 | 511.85 | 513.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 513.65 | 511.85 | 513.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 513.85 | 512.25 | 513.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 507.80 | 512.25 | 513.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 511.30 | 512.06 | 513.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 505.80 | 511.67 | 512.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 503.90 | 506.46 | 509.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 504.60 | 508.30 | 509.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 505.70 | 508.30 | 509.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 497.35 | 492.97 | 494.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 497.35 | 492.97 | 494.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 498.50 | 494.07 | 494.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 498.25 | 494.07 | 494.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-19 15:15:00 | 501.00 | 496.28 | 495.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 501.00 | 496.28 | 495.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 509.80 | 499.90 | 497.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 502.30 | 504.11 | 501.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 502.30 | 504.11 | 501.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 502.30 | 504.11 | 501.36 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 496.75 | 499.92 | 500.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 494.50 | 498.69 | 499.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 492.75 | 486.05 | 489.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 492.75 | 486.05 | 489.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 492.75 | 486.05 | 489.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 492.75 | 486.05 | 489.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 494.30 | 487.70 | 489.84 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 496.60 | 491.54 | 491.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 497.90 | 492.81 | 491.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 492.80 | 493.96 | 492.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 492.80 | 493.96 | 492.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 492.80 | 493.96 | 492.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 492.80 | 493.96 | 492.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 490.60 | 493.29 | 492.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 490.60 | 493.29 | 492.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 492.00 | 493.03 | 492.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 493.00 | 493.13 | 492.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 493.50 | 493.13 | 492.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 495.55 | 501.27 | 501.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 495.55 | 501.27 | 501.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 494.00 | 499.81 | 501.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 491.20 | 490.93 | 493.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:45:00 | 490.00 | 490.93 | 493.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 492.00 | 490.91 | 492.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 491.00 | 490.91 | 492.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 492.05 | 491.14 | 492.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 492.95 | 491.14 | 492.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 487.70 | 490.45 | 492.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 485.75 | 489.26 | 491.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 461.46 | 477.32 | 483.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 469.20 | 467.21 | 471.83 | SL hit (close>ema200) qty=0.50 sl=467.21 alert=retest2 |

### Cycle 166 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 477.20 | 472.95 | 472.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 479.20 | 474.20 | 473.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 478.60 | 479.79 | 477.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 475.65 | 479.79 | 477.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 473.55 | 478.54 | 477.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 475.30 | 478.54 | 477.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 474.35 | 477.70 | 476.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 477.45 | 477.82 | 477.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 473.35 | 476.65 | 476.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 473.35 | 476.65 | 476.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 472.25 | 474.98 | 476.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 473.50 | 471.86 | 473.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 473.50 | 471.86 | 473.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 473.50 | 471.86 | 473.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 473.70 | 471.86 | 473.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 474.00 | 472.29 | 473.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 476.05 | 472.29 | 473.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 476.15 | 473.06 | 473.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 476.15 | 473.06 | 473.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 474.80 | 473.41 | 473.78 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 476.50 | 474.03 | 474.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 485.20 | 476.26 | 475.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 488.85 | 488.98 | 486.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 488.85 | 488.98 | 486.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 486.85 | 488.49 | 486.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 486.85 | 488.49 | 486.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 485.85 | 487.96 | 486.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 485.85 | 487.96 | 486.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 487.75 | 487.92 | 486.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 492.50 | 487.92 | 486.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 492.35 | 488.80 | 486.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 496.40 | 491.55 | 489.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 486.80 | 488.85 | 488.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 486.80 | 488.85 | 488.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 483.60 | 487.39 | 488.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 484.95 | 484.59 | 486.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 484.95 | 484.59 | 486.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 483.50 | 483.87 | 485.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:00:00 | 482.15 | 484.33 | 485.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 481.85 | 483.72 | 484.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 490.75 | 485.24 | 485.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 490.75 | 485.24 | 485.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 494.75 | 487.14 | 485.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 14:15:00 | 499.70 | 501.46 | 498.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 499.70 | 501.46 | 498.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 499.70 | 501.46 | 498.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 507.35 | 501.06 | 498.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 504.75 | 509.43 | 509.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 504.75 | 509.43 | 509.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 497.40 | 506.09 | 508.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 501.10 | 500.78 | 504.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 502.80 | 500.78 | 504.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 495.20 | 499.78 | 503.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 491.25 | 496.77 | 500.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 492.00 | 495.81 | 500.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 485.35 | 495.19 | 498.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:15:00 | 467.40 | 479.72 | 487.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:15:00 | 466.69 | 477.09 | 485.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 461.08 | 466.97 | 476.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 461.45 | 458.87 | 468.66 | SL hit (close>ema200) qty=0.50 sl=458.87 alert=retest2 |

### Cycle 172 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 374.70 | 366.55 | 366.44 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 362.15 | 365.89 | 366.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 357.50 | 364.21 | 365.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 365.55 | 362.26 | 363.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 365.55 | 362.26 | 363.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 365.55 | 362.26 | 363.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 365.55 | 362.26 | 363.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 367.95 | 363.40 | 364.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 383.95 | 363.40 | 364.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 383.80 | 367.48 | 366.01 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 366.30 | 376.33 | 377.55 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 421.55 | 385.35 | 380.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 425.70 | 393.42 | 384.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 429.25 | 430.77 | 419.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 429.25 | 430.77 | 419.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 426.90 | 427.96 | 423.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:00:00 | 432.20 | 427.59 | 424.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 432.75 | 428.62 | 425.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 420.45 | 427.53 | 425.28 | SL hit (close<static) qty=1.00 sl=423.15 alert=retest2 |

### Cycle 177 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 419.45 | 423.18 | 423.63 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 427.50 | 424.20 | 423.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 428.30 | 425.02 | 424.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 09:15:00 | 426.15 | 426.30 | 425.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 426.15 | 426.30 | 425.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 426.15 | 426.30 | 425.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:30:00 | 423.45 | 426.30 | 425.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 422.30 | 425.50 | 424.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 422.30 | 425.50 | 424.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 422.10 | 424.82 | 424.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:00:00 | 425.75 | 425.01 | 424.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 421.90 | 424.38 | 424.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 421.90 | 424.38 | 424.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 420.75 | 423.66 | 424.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 418.15 | 417.76 | 420.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:45:00 | 417.60 | 417.76 | 420.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 412.15 | 416.65 | 419.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 411.80 | 416.65 | 419.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 407.10 | 405.23 | 405.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 407.10 | 405.23 | 405.11 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 403.60 | 404.95 | 405.01 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 408.25 | 405.64 | 405.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 413.80 | 408.06 | 406.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 408.65 | 411.05 | 408.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 408.65 | 411.05 | 408.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 408.65 | 411.05 | 408.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 408.65 | 411.05 | 408.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 409.75 | 410.79 | 408.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 402.45 | 410.79 | 408.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 405.05 | 409.64 | 408.55 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 395.90 | 405.60 | 406.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 393.70 | 403.22 | 405.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 402.40 | 401.79 | 404.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 402.40 | 401.79 | 404.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 392.25 | 393.52 | 397.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 391.20 | 393.52 | 397.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 399.55 | 394.20 | 396.25 | SL hit (close>static) qty=1.00 sl=399.45 alert=retest2 |

### Cycle 184 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 397.55 | 392.40 | 392.06 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 389.75 | 392.09 | 392.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 388.60 | 391.39 | 391.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 390.80 | 389.72 | 390.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 390.80 | 389.72 | 390.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 390.80 | 389.72 | 390.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 390.80 | 389.72 | 390.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 391.35 | 390.04 | 390.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 392.30 | 390.04 | 390.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 391.95 | 390.43 | 390.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 392.60 | 390.43 | 390.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 389.70 | 390.28 | 390.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 382.30 | 390.12 | 390.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 385.00 | 381.88 | 381.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 385.00 | 381.88 | 381.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 388.70 | 384.24 | 383.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 376.45 | 384.82 | 383.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 376.45 | 384.82 | 383.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 376.45 | 384.82 | 383.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 376.45 | 384.82 | 383.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 376.80 | 383.21 | 383.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 375.55 | 380.09 | 381.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 387.30 | 379.92 | 381.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 387.30 | 379.92 | 381.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 387.30 | 379.92 | 381.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 385.55 | 379.92 | 381.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 387.10 | 381.35 | 381.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 387.65 | 381.35 | 381.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 384.60 | 382.00 | 381.89 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 368.20 | 379.84 | 381.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 363.45 | 376.56 | 379.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 368.60 | 367.38 | 371.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 368.60 | 367.38 | 371.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 373.40 | 368.58 | 371.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 372.85 | 368.58 | 371.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 374.55 | 369.78 | 372.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 374.55 | 369.78 | 372.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 372.15 | 370.49 | 372.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 383.45 | 370.49 | 372.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 388.80 | 374.15 | 373.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 396.65 | 381.43 | 377.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 385.10 | 387.63 | 382.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 385.30 | 387.63 | 382.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 379.95 | 387.32 | 385.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 379.95 | 387.32 | 385.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 378.05 | 385.47 | 384.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 378.05 | 385.47 | 384.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 379.35 | 383.39 | 383.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 376.30 | 381.22 | 382.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 387.80 | 381.75 | 382.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 387.80 | 381.75 | 382.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 387.80 | 381.75 | 382.56 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 390.70 | 384.31 | 383.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 395.50 | 387.61 | 385.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 414.00 | 414.27 | 406.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:00:00 | 414.00 | 414.27 | 406.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 440.10 | 446.28 | 441.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 446.45 | 442.88 | 441.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 439.75 | 441.68 | 441.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 439.75 | 441.68 | 441.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 10:15:00 | 418.75 | 436.93 | 439.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 414.85 | 414.66 | 418.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:15:00 | 410.60 | 414.66 | 418.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 14:00:00 | 412.40 | 411.91 | 415.08 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 410.25 | 411.79 | 414.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 407.70 | 411.79 | 414.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 412.45 | 408.13 | 410.58 | SL hit (close>ema400) qty=1.00 sl=410.58 alert=retest1 |

### Cycle 194 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 413.60 | 411.94 | 411.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 418.90 | 414.02 | 413.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 412.90 | 414.90 | 413.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 412.90 | 414.90 | 413.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 412.90 | 414.90 | 413.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:15:00 | 409.95 | 414.90 | 413.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 412.30 | 414.38 | 413.82 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 405.75 | 412.35 | 412.98 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 416.45 | 412.47 | 412.37 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 406.10 | 411.76 | 412.34 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 413.85 | 411.44 | 411.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 415.45 | 412.24 | 411.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 407.65 | 411.96 | 411.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 407.65 | 411.96 | 411.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 407.65 | 411.96 | 411.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 407.65 | 411.96 | 411.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 408.15 | 411.20 | 411.33 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 14:15:00 | 428.00 | 413.95 | 412.23 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-22 14:45:00 | 402.90 | 2024-04-23 10:15:00 | 417.60 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-04-23 09:45:00 | 404.00 | 2024-04-23 10:15:00 | 417.60 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-04-26 09:15:00 | 421.60 | 2024-04-29 12:15:00 | 414.65 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-04-30 11:45:00 | 413.35 | 2024-05-06 09:15:00 | 392.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 12:30:00 | 413.65 | 2024-05-06 09:15:00 | 392.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 13:00:00 | 413.95 | 2024-05-06 09:15:00 | 393.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 14:30:00 | 414.05 | 2024-05-06 09:15:00 | 393.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 10:30:00 | 413.85 | 2024-05-06 09:15:00 | 393.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 11:45:00 | 413.70 | 2024-05-06 09:15:00 | 393.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 11:45:00 | 413.35 | 2024-05-06 15:15:00 | 401.00 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2024-04-30 12:30:00 | 413.65 | 2024-05-06 15:15:00 | 401.00 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2024-04-30 13:00:00 | 413.95 | 2024-05-06 15:15:00 | 401.00 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-04-30 14:30:00 | 414.05 | 2024-05-06 15:15:00 | 401.00 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2024-05-02 10:30:00 | 413.85 | 2024-05-06 15:15:00 | 401.00 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2024-05-02 11:45:00 | 413.70 | 2024-05-06 15:15:00 | 401.00 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2024-06-14 11:15:00 | 400.60 | 2024-06-18 10:15:00 | 405.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-06-18 09:45:00 | 401.75 | 2024-06-18 10:15:00 | 405.45 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-06-26 09:15:00 | 449.55 | 2024-06-28 09:15:00 | 494.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-15 10:30:00 | 512.90 | 2024-07-24 09:15:00 | 564.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 12:30:00 | 523.85 | 2024-07-24 09:15:00 | 576.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 14:15:00 | 539.05 | 2024-08-12 11:15:00 | 540.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-08-07 14:30:00 | 538.60 | 2024-08-12 11:15:00 | 540.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-08-07 15:00:00 | 537.80 | 2024-08-12 11:15:00 | 540.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-08-08 09:15:00 | 535.85 | 2024-08-12 11:15:00 | 540.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-14 12:15:00 | 554.35 | 2024-08-19 12:15:00 | 550.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-08-14 12:45:00 | 554.95 | 2024-08-19 12:15:00 | 550.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-14 14:00:00 | 562.40 | 2024-08-19 12:15:00 | 550.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-30 13:15:00 | 622.30 | 2024-09-06 12:15:00 | 636.75 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2024-09-02 09:15:00 | 630.00 | 2024-09-06 12:15:00 | 636.75 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-09-02 13:30:00 | 626.15 | 2024-09-06 12:15:00 | 636.75 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-09-17 09:15:00 | 707.55 | 2024-09-19 09:15:00 | 688.95 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-09-17 10:30:00 | 704.95 | 2024-09-19 09:15:00 | 688.95 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-09-17 11:00:00 | 706.25 | 2024-09-19 09:15:00 | 688.95 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-09-18 09:30:00 | 714.25 | 2024-09-19 09:15:00 | 688.95 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2024-10-08 14:45:00 | 708.00 | 2024-10-09 13:15:00 | 718.10 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-10-09 11:45:00 | 707.90 | 2024-10-09 13:15:00 | 718.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-10-09 12:30:00 | 708.35 | 2024-10-09 13:15:00 | 718.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-10-09 13:00:00 | 708.20 | 2024-10-09 13:15:00 | 718.10 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-10-14 09:15:00 | 722.85 | 2024-10-17 10:15:00 | 721.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-10-18 14:15:00 | 720.40 | 2024-10-22 14:15:00 | 684.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:15:00 | 720.40 | 2024-10-23 12:15:00 | 698.95 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-11-04 09:15:00 | 662.75 | 2024-11-06 10:15:00 | 690.00 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2024-11-11 10:15:00 | 696.00 | 2024-11-13 09:15:00 | 683.40 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-11-14 13:30:00 | 650.30 | 2024-11-19 10:15:00 | 693.00 | STOP_HIT | 1.00 | -6.57% |
| SELL | retest2 | 2024-11-18 10:30:00 | 655.85 | 2024-11-19 10:15:00 | 693.00 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest2 | 2024-11-26 09:15:00 | 702.20 | 2024-11-26 11:15:00 | 695.55 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-11-26 10:45:00 | 699.80 | 2024-11-26 11:15:00 | 695.55 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-10 14:45:00 | 756.55 | 2024-12-13 09:15:00 | 743.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-12-19 09:15:00 | 743.85 | 2024-12-26 15:15:00 | 733.00 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2024-12-31 13:15:00 | 745.50 | 2025-01-06 13:15:00 | 752.75 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-01-20 14:45:00 | 532.35 | 2025-01-21 09:15:00 | 505.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 14:45:00 | 532.35 | 2025-01-22 09:15:00 | 479.12 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-03 10:45:00 | 511.55 | 2025-02-04 14:15:00 | 562.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-04 09:15:00 | 511.85 | 2025-02-04 14:15:00 | 563.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-04 09:45:00 | 513.50 | 2025-02-04 14:15:00 | 564.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-04 11:00:00 | 512.00 | 2025-02-04 14:15:00 | 563.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-07 11:15:00 | 551.00 | 2025-02-07 13:15:00 | 536.15 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-02-12 09:15:00 | 505.90 | 2025-02-13 14:15:00 | 524.95 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-02-12 12:00:00 | 514.00 | 2025-02-13 14:15:00 | 524.95 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-02-12 13:30:00 | 514.20 | 2025-02-13 14:15:00 | 524.95 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-02-18 09:15:00 | 497.10 | 2025-02-27 09:15:00 | 472.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 09:15:00 | 497.10 | 2025-02-28 15:15:00 | 468.55 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-03-13 09:15:00 | 418.30 | 2025-03-13 10:15:00 | 425.30 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-03-26 09:15:00 | 474.05 | 2025-03-27 09:15:00 | 469.75 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-03-26 09:45:00 | 475.00 | 2025-03-27 09:15:00 | 469.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-04-09 11:30:00 | 490.45 | 2025-04-23 10:15:00 | 522.95 | STOP_HIT | 1.00 | 6.63% |
| SELL | retest2 | 2025-05-22 10:45:00 | 549.50 | 2025-05-22 14:15:00 | 554.55 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-22 12:45:00 | 548.90 | 2025-05-22 14:15:00 | 554.55 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-22 13:30:00 | 547.85 | 2025-05-22 14:15:00 | 554.55 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-23 10:15:00 | 549.40 | 2025-05-23 12:15:00 | 553.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-28 09:15:00 | 564.45 | 2025-05-30 10:15:00 | 562.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-28 12:00:00 | 564.30 | 2025-05-30 10:15:00 | 562.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-06-05 09:15:00 | 553.25 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-06-05 12:00:00 | 554.10 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-06-05 15:15:00 | 552.85 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-06 10:45:00 | 553.80 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-06-17 11:30:00 | 520.25 | 2025-06-24 09:15:00 | 525.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-19 10:00:00 | 520.45 | 2025-06-24 09:15:00 | 525.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-20 15:00:00 | 518.00 | 2025-06-24 09:15:00 | 525.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-07 10:30:00 | 591.45 | 2025-07-08 09:15:00 | 566.60 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2025-07-07 11:30:00 | 589.95 | 2025-07-08 09:15:00 | 566.60 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-07-11 14:15:00 | 581.65 | 2025-07-18 15:15:00 | 589.90 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-08-06 12:30:00 | 594.60 | 2025-08-06 14:15:00 | 590.35 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-08 15:00:00 | 525.35 | 2025-08-14 13:15:00 | 538.50 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-08-12 09:15:00 | 525.30 | 2025-08-14 13:15:00 | 538.50 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-08-21 15:15:00 | 510.20 | 2025-08-22 09:15:00 | 514.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-22 13:45:00 | 510.40 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-08-25 09:15:00 | 510.55 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-25 14:15:00 | 510.40 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-08-26 11:30:00 | 501.65 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-08-26 13:15:00 | 501.50 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 518.25 | 2025-09-04 09:15:00 | 503.15 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-10 10:30:00 | 500.25 | 2025-09-10 14:15:00 | 506.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-10 11:15:00 | 500.20 | 2025-09-10 14:15:00 | 506.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-10 12:30:00 | 500.20 | 2025-09-10 14:15:00 | 506.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-30 10:30:00 | 455.00 | 2025-10-01 13:15:00 | 462.45 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-09-30 11:30:00 | 454.20 | 2025-10-01 13:15:00 | 462.45 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-10-09 09:30:00 | 483.85 | 2025-10-13 09:15:00 | 481.05 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-09 13:45:00 | 483.90 | 2025-10-13 09:15:00 | 481.05 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-09 15:15:00 | 483.95 | 2025-10-13 09:15:00 | 481.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-24 09:15:00 | 495.15 | 2025-11-06 15:15:00 | 515.00 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2025-10-24 15:00:00 | 495.40 | 2025-11-06 15:15:00 | 515.00 | STOP_HIT | 1.00 | 3.96% |
| SELL | retest2 | 2025-11-11 09:15:00 | 505.80 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-11-11 12:45:00 | 503.90 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-11-13 09:30:00 | 504.60 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-11-13 10:00:00 | 505.70 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-11-27 14:30:00 | 493.00 | 2025-12-03 09:15:00 | 495.55 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-11-27 15:00:00 | 493.50 | 2025-12-03 09:15:00 | 495.55 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-12-08 10:30:00 | 485.75 | 2025-12-09 09:15:00 | 461.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:30:00 | 485.75 | 2025-12-11 10:15:00 | 469.20 | STOP_HIT | 0.50 | 3.41% |
| BUY | retest2 | 2025-12-16 11:30:00 | 477.45 | 2025-12-17 12:15:00 | 473.35 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-29 09:30:00 | 496.40 | 2025-12-29 14:15:00 | 486.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-01 11:00:00 | 482.15 | 2026-01-02 09:15:00 | 490.75 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-01 13:00:00 | 481.85 | 2026-01-02 09:15:00 | 490.75 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-07 09:15:00 | 507.35 | 2026-01-09 13:15:00 | 504.75 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-13 12:45:00 | 491.25 | 2026-01-16 10:15:00 | 467.40 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-13 14:00:00 | 492.00 | 2026-01-16 11:15:00 | 466.69 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2026-01-14 09:15:00 | 485.35 | 2026-01-19 10:15:00 | 461.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 491.25 | 2026-01-19 14:15:00 | 461.45 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2026-01-13 14:00:00 | 492.00 | 2026-01-19 14:15:00 | 461.45 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2026-01-14 09:15:00 | 485.35 | 2026-01-19 14:15:00 | 461.45 | STOP_HIT | 0.50 | 4.92% |
| BUY | retest2 | 2026-02-12 14:00:00 | 432.20 | 2026-02-13 09:15:00 | 420.45 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-02-12 15:00:00 | 432.75 | 2026-02-13 09:15:00 | 420.45 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-02-17 13:00:00 | 425.75 | 2026-02-17 13:15:00 | 421.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-19 10:15:00 | 411.80 | 2026-02-25 10:15:00 | 407.10 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2026-03-05 10:15:00 | 391.20 | 2026-03-05 14:15:00 | 399.55 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-03-09 09:15:00 | 384.10 | 2026-03-11 09:15:00 | 397.55 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-03-13 09:15:00 | 382.30 | 2026-03-18 09:15:00 | 385.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-04-15 09:30:00 | 446.45 | 2026-04-16 11:15:00 | 439.75 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest1 | 2026-04-23 09:15:00 | 410.60 | 2026-04-27 09:15:00 | 412.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-23 14:00:00 | 412.40 | 2026-04-27 09:15:00 | 412.45 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-04-24 10:15:00 | 407.70 | 2026-04-27 13:15:00 | 413.60 | STOP_HIT | 1.00 | -1.45% |

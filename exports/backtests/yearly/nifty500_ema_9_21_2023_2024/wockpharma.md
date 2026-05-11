# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1611.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 234 |
| ALERT1 | 154 |
| ALERT2 | 152 |
| ALERT2_SKIP | 100 |
| ALERT3 | 360 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 112 |
| PARTIAL | 22 |
| TARGET_HIT | 11 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 87
- **Target hits / Stop hits / Partials:** 11 / 107 / 22
- **Avg / median % per leg:** 0.54% / -0.68%
- **Sum % (uncompounded):** 74.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 10 | 20.0% | 3 | 46 | 1 | -0.84% | -41.9% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.22% | 9.7% |
| BUY @ 3rd Alert (retest2) | 47 | 8 | 17.0% | 3 | 44 | 0 | -1.10% | -51.5% |
| SELL (all) | 90 | 43 | 47.8% | 8 | 61 | 21 | 1.30% | 116.8% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| SELL @ 3rd Alert (retest2) | 86 | 43 | 50.0% | 8 | 57 | 21 | 1.44% | 123.6% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 6 | 1 | 0.41% | 2.9% |
| retest2 (combined) | 133 | 51 | 38.3% | 11 | 101 | 21 | 0.54% | 72.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 14:15:00 | 174.35 | 174.60 | 174.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 12:15:00 | 173.20 | 174.13 | 174.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-16 13:15:00 | 174.25 | 174.16 | 174.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 13:15:00 | 174.25 | 174.16 | 174.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 13:15:00 | 174.25 | 174.16 | 174.36 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 10:15:00 | 172.25 | 169.75 | 169.55 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 12:15:00 | 169.65 | 170.75 | 170.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 14:15:00 | 169.40 | 170.36 | 170.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 09:15:00 | 171.40 | 170.36 | 170.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 171.40 | 170.36 | 170.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 171.40 | 170.36 | 170.56 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 14:15:00 | 175.00 | 169.77 | 169.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 10:15:00 | 185.50 | 178.85 | 175.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 14:15:00 | 231.55 | 232.89 | 223.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 225.85 | 229.03 | 226.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 225.85 | 229.03 | 226.45 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 13:15:00 | 222.10 | 225.25 | 225.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 217.10 | 222.63 | 224.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 229.40 | 223.39 | 224.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 11:15:00 | 229.40 | 223.39 | 224.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 229.40 | 223.39 | 224.08 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 12:15:00 | 233.50 | 225.41 | 224.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 12:15:00 | 242.00 | 234.79 | 232.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 13:15:00 | 235.30 | 235.69 | 234.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 15:15:00 | 234.70 | 235.50 | 234.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 15:15:00 | 234.70 | 235.50 | 234.46 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 15:15:00 | 244.50 | 245.54 | 245.56 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 246.50 | 245.72 | 245.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 11:15:00 | 247.15 | 246.00 | 245.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 11:15:00 | 246.50 | 247.40 | 246.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 11:15:00 | 246.50 | 247.40 | 246.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 246.50 | 247.40 | 246.71 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 244.75 | 246.05 | 246.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 15:15:00 | 243.50 | 245.54 | 245.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 15:15:00 | 236.25 | 235.95 | 238.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 237.45 | 236.25 | 238.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 237.45 | 236.25 | 238.65 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 10:15:00 | 237.50 | 236.41 | 236.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 12:15:00 | 246.35 | 238.65 | 237.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 12:15:00 | 241.75 | 242.12 | 240.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 14:15:00 | 241.65 | 242.14 | 240.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 241.65 | 242.14 | 240.55 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 235.20 | 239.70 | 239.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 11:15:00 | 234.30 | 238.62 | 239.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 237.75 | 234.69 | 235.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 237.75 | 234.69 | 235.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 237.75 | 234.69 | 235.79 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 14:15:00 | 243.30 | 237.53 | 236.83 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 11:15:00 | 239.05 | 240.10 | 240.17 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 13:15:00 | 244.20 | 240.84 | 240.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 09:15:00 | 254.00 | 244.07 | 242.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 244.75 | 247.72 | 244.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 13:15:00 | 244.75 | 247.72 | 244.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 244.75 | 247.72 | 244.84 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 15:15:00 | 255.80 | 260.22 | 260.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 11:15:00 | 252.95 | 257.67 | 259.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 244.10 | 243.86 | 247.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 10:15:00 | 235.50 | 232.08 | 234.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 235.50 | 232.08 | 234.33 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 10:15:00 | 236.80 | 233.97 | 233.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 11:15:00 | 238.00 | 234.77 | 234.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 232.70 | 235.11 | 234.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 232.70 | 235.11 | 234.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 232.70 | 235.11 | 234.73 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 230.70 | 234.23 | 234.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 227.70 | 231.55 | 232.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 15:15:00 | 230.20 | 230.18 | 231.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 233.50 | 230.85 | 231.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 233.50 | 230.85 | 231.47 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 235.15 | 232.08 | 231.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 12:15:00 | 236.50 | 232.96 | 232.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 237.35 | 238.64 | 236.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 15:15:00 | 237.90 | 238.49 | 236.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 237.90 | 238.49 | 236.54 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 244.30 | 253.15 | 253.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 240.25 | 247.89 | 251.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 243.25 | 243.20 | 246.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 12:15:00 | 250.10 | 244.58 | 247.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 250.10 | 244.58 | 247.15 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 253.50 | 249.18 | 248.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 258.70 | 254.86 | 252.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 252.55 | 254.88 | 253.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 12:15:00 | 252.55 | 254.88 | 253.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 12:15:00 | 252.55 | 254.88 | 253.02 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 15:15:00 | 247.50 | 251.24 | 251.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 15:15:00 | 241.55 | 245.78 | 248.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 231.70 | 230.84 | 234.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 232.45 | 230.88 | 232.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 232.45 | 230.88 | 232.85 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 10:15:00 | 233.20 | 233.13 | 233.12 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 230.30 | 232.62 | 232.90 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 238.40 | 233.25 | 233.04 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 231.30 | 234.60 | 234.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 228.30 | 233.34 | 234.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 231.60 | 231.49 | 232.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 11:15:00 | 233.35 | 231.95 | 232.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 233.35 | 231.95 | 232.82 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 232.70 | 230.51 | 230.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 236.30 | 232.93 | 231.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 13:15:00 | 250.30 | 252.73 | 248.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 14:15:00 | 248.85 | 251.95 | 248.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 248.85 | 251.95 | 248.58 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 15:15:00 | 246.75 | 247.90 | 247.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 243.75 | 246.93 | 247.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 246.25 | 244.52 | 245.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 246.25 | 244.52 | 245.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 246.25 | 244.52 | 245.49 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 13:15:00 | 249.55 | 246.40 | 246.23 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 242.40 | 245.96 | 246.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 235.95 | 243.26 | 244.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 220.90 | 220.62 | 225.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 13:15:00 | 223.40 | 221.94 | 224.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 13:15:00 | 223.40 | 221.94 | 224.36 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 13:15:00 | 226.55 | 224.87 | 224.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 15:15:00 | 227.80 | 225.78 | 225.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 226.70 | 227.58 | 226.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 226.70 | 227.58 | 226.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 226.70 | 227.58 | 226.65 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 223.70 | 226.17 | 226.28 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 233.40 | 226.71 | 226.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 10:15:00 | 237.55 | 228.88 | 227.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 10:15:00 | 235.25 | 235.43 | 232.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 14:15:00 | 233.70 | 234.64 | 232.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 233.70 | 234.64 | 232.76 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 13:15:00 | 331.00 | 334.76 | 335.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 10:15:00 | 328.60 | 332.00 | 333.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 329.40 | 328.07 | 330.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 329.40 | 328.07 | 330.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 329.40 | 328.07 | 330.53 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 339.50 | 332.53 | 332.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 348.10 | 337.12 | 334.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 352.10 | 354.45 | 350.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 10:15:00 | 348.75 | 353.31 | 350.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 348.75 | 353.31 | 350.07 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 344.35 | 348.44 | 348.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 14:15:00 | 341.55 | 344.61 | 346.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 339.50 | 338.18 | 341.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 339.50 | 338.18 | 341.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 339.50 | 338.18 | 341.16 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 392.85 | 349.99 | 346.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 404.40 | 377.14 | 362.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 14:15:00 | 403.25 | 403.50 | 390.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 411.00 | 415.97 | 411.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 411.00 | 415.97 | 411.69 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 406.25 | 411.57 | 411.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 387.55 | 406.76 | 409.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 399.80 | 398.03 | 403.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 11:15:00 | 399.80 | 398.03 | 403.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 399.80 | 398.03 | 403.14 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 411.30 | 404.37 | 404.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 417.00 | 409.17 | 406.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 416.25 | 416.94 | 413.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 411.00 | 415.75 | 413.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 411.00 | 415.75 | 413.30 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 13:15:00 | 410.30 | 413.26 | 413.38 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 09:15:00 | 424.20 | 414.92 | 414.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 442.00 | 425.67 | 420.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 486.40 | 490.73 | 483.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 10:15:00 | 486.40 | 490.73 | 483.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 486.40 | 490.73 | 483.91 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 470.20 | 482.25 | 482.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 466.65 | 479.13 | 481.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 475.40 | 470.16 | 475.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 475.40 | 470.16 | 475.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 475.40 | 470.16 | 475.15 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 472.15 | 467.94 | 467.89 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 466.75 | 467.89 | 467.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 460.50 | 466.11 | 467.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 436.85 | 431.52 | 440.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 436.85 | 431.52 | 440.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 436.85 | 431.52 | 440.03 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 14:15:00 | 459.00 | 444.54 | 444.18 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 09:15:00 | 443.40 | 448.44 | 448.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 10:15:00 | 438.25 | 446.40 | 447.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 439.50 | 435.16 | 440.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 439.50 | 435.16 | 440.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 439.50 | 435.16 | 440.29 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 442.95 | 441.63 | 441.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 11:15:00 | 450.45 | 443.40 | 442.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 15:15:00 | 455.60 | 457.90 | 453.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 10:15:00 | 452.50 | 456.76 | 453.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 452.50 | 456.76 | 453.44 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 447.90 | 451.81 | 452.03 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 454.55 | 452.36 | 452.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 10:15:00 | 464.40 | 454.77 | 453.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 15:15:00 | 457.00 | 458.81 | 456.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 455.35 | 458.12 | 456.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 455.35 | 458.12 | 456.26 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 450.00 | 454.36 | 454.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 12:15:00 | 448.75 | 452.35 | 453.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 09:15:00 | 450.30 | 448.65 | 451.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 450.30 | 448.65 | 451.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 450.30 | 448.65 | 451.09 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 462.65 | 451.79 | 450.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 09:15:00 | 485.35 | 461.08 | 457.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 460.50 | 467.22 | 463.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 460.50 | 467.22 | 463.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 460.50 | 467.22 | 463.34 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 13:15:00 | 457.40 | 460.76 | 461.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 450.65 | 458.05 | 459.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 10:15:00 | 439.90 | 437.33 | 442.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 439.90 | 437.33 | 442.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 439.90 | 437.33 | 442.85 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 447.55 | 445.72 | 445.50 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 12:15:00 | 442.90 | 444.91 | 445.16 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 472.25 | 450.13 | 447.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 10:15:00 | 487.60 | 457.62 | 451.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 15:15:00 | 560.55 | 562.40 | 546.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 553.60 | 560.64 | 546.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 553.60 | 560.64 | 546.94 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 15:15:00 | 575.00 | 583.99 | 584.45 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 11:15:00 | 594.00 | 584.52 | 584.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 13:15:00 | 605.00 | 590.47 | 587.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 11:15:00 | 612.00 | 612.68 | 605.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 605.00 | 610.43 | 605.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 605.00 | 610.43 | 605.35 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 12:15:00 | 590.00 | 602.70 | 602.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 583.00 | 595.30 | 599.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 14:15:00 | 560.10 | 559.75 | 567.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 570.00 | 560.86 | 566.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 570.00 | 560.86 | 566.65 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 533.55 | 522.08 | 521.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 09:15:00 | 549.75 | 529.45 | 525.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 540.00 | 541.71 | 534.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 10:15:00 | 535.00 | 540.37 | 534.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 535.00 | 540.37 | 534.86 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 09:15:00 | 535.00 | 543.35 | 543.54 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 10:15:00 | 557.85 | 543.55 | 542.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 09:15:00 | 580.00 | 558.53 | 550.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 09:15:00 | 592.00 | 594.99 | 583.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 10:15:00 | 598.00 | 595.59 | 584.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 598.00 | 595.59 | 584.85 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 585.95 | 589.49 | 589.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 12:15:00 | 579.90 | 584.49 | 586.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 15:15:00 | 585.00 | 584.45 | 586.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 09:15:00 | 583.00 | 584.16 | 585.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 583.00 | 584.16 | 585.86 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 11:15:00 | 593.00 | 583.66 | 583.56 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 581.90 | 584.09 | 584.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 14:15:00 | 574.90 | 580.36 | 582.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 556.00 | 554.45 | 561.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 574.60 | 555.77 | 558.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 574.60 | 555.77 | 558.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:45:00 | 574.60 | 555.77 | 558.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 574.60 | 559.53 | 559.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:30:00 | 574.60 | 559.53 | 559.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 574.60 | 562.55 | 561.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 10:15:00 | 591.00 | 574.34 | 568.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 10:15:00 | 580.00 | 581.83 | 576.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 10:15:00 | 580.00 | 581.83 | 576.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 580.00 | 581.83 | 576.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 10:30:00 | 577.95 | 581.83 | 576.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 11:15:00 | 580.00 | 581.46 | 576.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 11:45:00 | 578.00 | 581.46 | 576.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 13:15:00 | 580.00 | 580.94 | 577.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:30:00 | 580.25 | 576.57 | 575.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 563.80 | 574.02 | 574.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 10:15:00 | 563.80 | 574.02 | 574.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 12:15:00 | 563.10 | 570.62 | 573.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 570.00 | 569.96 | 571.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 570.00 | 569.96 | 571.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 570.00 | 569.96 | 571.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 572.00 | 569.96 | 571.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 566.00 | 569.15 | 571.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 11:45:00 | 567.00 | 569.15 | 571.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 12:15:00 | 571.00 | 569.52 | 571.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 12:45:00 | 571.90 | 569.52 | 571.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 567.00 | 569.26 | 570.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:30:00 | 568.00 | 569.26 | 570.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 569.00 | 569.33 | 570.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 573.00 | 569.33 | 570.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 568.00 | 569.06 | 570.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:30:00 | 571.45 | 569.06 | 570.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 571.00 | 569.45 | 570.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 571.00 | 569.45 | 570.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 568.50 | 569.26 | 570.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:30:00 | 572.00 | 569.26 | 570.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 572.00 | 569.81 | 570.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:00:00 | 572.00 | 569.81 | 570.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 568.00 | 569.45 | 570.17 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 586.50 | 573.27 | 571.81 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 566.40 | 573.99 | 574.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 559.90 | 568.43 | 571.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 15:15:00 | 569.00 | 561.71 | 565.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 15:15:00 | 569.00 | 561.71 | 565.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 569.00 | 561.71 | 565.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:45:00 | 548.00 | 558.53 | 564.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 11:00:00 | 550.00 | 556.82 | 562.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 12:45:00 | 551.00 | 553.98 | 560.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 520.60 | 537.09 | 548.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 522.50 | 537.09 | 548.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 523.45 | 537.09 | 548.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-07 14:15:00 | 541.00 | 536.58 | 545.81 | SL hit (close>ema200) qty=0.50 sl=536.58 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 542.85 | 537.06 | 536.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 560.65 | 541.78 | 538.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 545.00 | 548.54 | 544.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 545.00 | 548.54 | 544.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 545.00 | 547.83 | 544.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:00:00 | 545.00 | 547.83 | 544.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 545.95 | 547.45 | 545.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:45:00 | 543.20 | 547.45 | 545.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 542.00 | 546.36 | 544.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 549.00 | 546.36 | 544.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 546.40 | 546.37 | 544.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:15:00 | 550.00 | 546.37 | 544.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 13:45:00 | 549.65 | 546.84 | 545.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:30:00 | 550.00 | 547.26 | 545.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 553.20 | 547.26 | 545.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 553.20 | 548.45 | 546.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 550.00 | 548.45 | 546.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 544.95 | 547.75 | 546.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-17 14:15:00 | 544.00 | 545.68 | 545.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 14:15:00 | 544.00 | 545.68 | 545.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 15:15:00 | 539.50 | 544.45 | 545.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 545.00 | 544.56 | 545.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 545.00 | 544.56 | 545.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 545.00 | 544.56 | 545.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:30:00 | 548.85 | 544.56 | 545.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 545.00 | 544.65 | 545.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 544.00 | 544.65 | 545.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 12:15:00 | 560.00 | 547.72 | 546.56 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 537.00 | 544.47 | 545.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 15:15:00 | 535.00 | 541.50 | 543.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 10:15:00 | 541.40 | 540.17 | 542.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 10:15:00 | 541.40 | 540.17 | 542.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 541.40 | 540.17 | 542.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:45:00 | 538.00 | 540.17 | 542.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 543.00 | 540.87 | 542.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 545.00 | 540.87 | 542.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 542.00 | 541.09 | 542.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 541.50 | 541.09 | 542.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 542.00 | 541.28 | 542.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 545.00 | 541.28 | 542.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 547.00 | 542.06 | 542.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:15:00 | 548.00 | 542.06 | 542.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 546.00 | 542.85 | 542.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 549.00 | 542.85 | 542.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 550.10 | 544.30 | 543.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 554.00 | 546.24 | 544.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 558.00 | 559.13 | 554.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 11:15:00 | 565.00 | 560.36 | 556.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 565.00 | 560.36 | 556.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 559.25 | 560.36 | 556.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 560.00 | 561.21 | 557.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 564.70 | 561.21 | 557.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 549.15 | 558.80 | 557.08 | SL hit (close<static) qty=1.00 sl=556.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 553.55 | 556.11 | 556.13 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 15:15:00 | 561.90 | 556.96 | 556.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 565.20 | 558.61 | 557.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 556.50 | 561.60 | 560.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 556.50 | 561.60 | 560.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 556.50 | 561.60 | 560.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 559.90 | 561.60 | 560.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 555.00 | 560.28 | 559.61 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 544.25 | 557.07 | 558.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 12:15:00 | 539.35 | 553.53 | 556.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 540.00 | 539.67 | 545.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 540.00 | 539.67 | 545.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 536.15 | 535.39 | 542.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 523.00 | 537.23 | 540.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 519.95 | 530.61 | 536.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 496.85 | 514.64 | 524.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 493.95 | 514.64 | 524.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 12:15:00 | 513.95 | 511.14 | 520.48 | SL hit (close>ema200) qty=0.50 sl=511.14 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 535.00 | 524.72 | 523.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 545.00 | 532.83 | 528.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 568.70 | 569.99 | 560.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 579.75 | 569.99 | 560.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 584.95 | 587.67 | 581.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 587.20 | 587.67 | 581.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 583.40 | 586.14 | 581.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:30:00 | 580.80 | 586.14 | 581.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 589.35 | 586.78 | 582.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-13 14:15:00 | 578.00 | 584.44 | 582.10 | SL hit (close<ema400) qty=1.00 sl=582.10 alert=retest1 |

### Cycle 77 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 11:15:00 | 572.50 | 579.35 | 580.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 10:15:00 | 567.65 | 572.39 | 575.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 568.40 | 565.06 | 569.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 12:00:00 | 568.40 | 565.06 | 569.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 570.00 | 566.05 | 569.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:30:00 | 571.00 | 566.05 | 569.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 581.45 | 569.13 | 570.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 583.15 | 569.13 | 570.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 572.50 | 569.80 | 570.88 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 577.80 | 571.91 | 571.68 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 13:15:00 | 570.00 | 571.61 | 571.65 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 573.55 | 572.00 | 571.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 575.50 | 572.55 | 572.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 569.85 | 572.67 | 572.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 12:15:00 | 569.85 | 572.67 | 572.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 569.85 | 572.67 | 572.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 569.85 | 572.67 | 572.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 575.50 | 573.23 | 572.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 576.25 | 573.84 | 572.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-26 09:15:00 | 633.88 | 619.69 | 603.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 15:15:00 | 879.60 | 891.60 | 892.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 862.80 | 882.73 | 887.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 854.95 | 849.53 | 861.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 15:00:00 | 854.95 | 849.53 | 861.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 867.05 | 853.03 | 862.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 875.30 | 853.03 | 862.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 900.10 | 862.45 | 865.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 901.25 | 862.45 | 865.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 884.60 | 866.88 | 867.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:15:00 | 878.50 | 866.88 | 867.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 873.55 | 868.21 | 867.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 11:15:00 | 873.55 | 868.21 | 867.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 14:15:00 | 886.10 | 874.19 | 870.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 09:15:00 | 872.25 | 875.01 | 871.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 872.25 | 875.01 | 871.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 872.25 | 875.01 | 871.97 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 857.35 | 868.57 | 869.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 843.00 | 861.13 | 866.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 13:15:00 | 844.00 | 842.54 | 850.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 14:00:00 | 844.00 | 842.54 | 850.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 848.00 | 843.63 | 849.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 848.00 | 843.63 | 849.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 815.45 | 796.14 | 810.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 815.45 | 796.14 | 810.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 815.45 | 800.01 | 810.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 815.45 | 800.01 | 810.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 849.05 | 818.12 | 815.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 856.20 | 825.74 | 819.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 853.00 | 865.08 | 853.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 853.00 | 865.08 | 853.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 853.00 | 865.08 | 853.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 853.00 | 865.08 | 853.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 855.00 | 863.06 | 853.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 872.00 | 857.42 | 853.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:00:00 | 864.00 | 863.71 | 858.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 10:00:00 | 862.60 | 861.14 | 858.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:15:00 | 861.50 | 859.85 | 858.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 854.00 | 858.68 | 858.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-29 12:15:00 | 850.00 | 856.95 | 857.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 850.00 | 856.95 | 857.37 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 874.85 | 857.86 | 857.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 882.00 | 869.65 | 863.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 10:15:00 | 910.00 | 912.78 | 895.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 11:00:00 | 910.00 | 912.78 | 895.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 905.05 | 909.19 | 896.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 892.00 | 909.19 | 896.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 910.00 | 909.35 | 898.05 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 865.20 | 892.15 | 895.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 863.90 | 883.12 | 890.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 13:15:00 | 890.00 | 884.49 | 890.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 14:00:00 | 890.00 | 884.49 | 890.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 870.00 | 881.60 | 888.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:15:00 | 865.50 | 876.16 | 882.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:45:00 | 860.95 | 868.15 | 877.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 890.15 | 874.84 | 876.27 | SL hit (close>static) qty=1.00 sl=890.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 890.15 | 877.90 | 877.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 918.00 | 887.88 | 882.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 912.00 | 913.09 | 903.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 912.00 | 913.09 | 903.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 912.00 | 912.87 | 904.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 912.00 | 912.87 | 904.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 950.00 | 920.30 | 908.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 967.90 | 920.30 | 908.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:15:00 | 970.00 | 936.86 | 918.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 10:15:00 | 914.00 | 943.30 | 945.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 914.00 | 943.30 | 945.73 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 963.25 | 937.37 | 936.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 984.35 | 946.77 | 940.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 972.75 | 973.55 | 961.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:45:00 | 973.00 | 973.55 | 961.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 985.00 | 984.07 | 976.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 979.00 | 984.07 | 976.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 975.20 | 981.96 | 976.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 975.00 | 981.96 | 976.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 970.00 | 979.57 | 975.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:45:00 | 968.00 | 979.57 | 975.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 974.00 | 978.46 | 975.71 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 967.45 | 974.04 | 974.16 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 982.00 | 975.54 | 974.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 11:15:00 | 1018.00 | 984.04 | 978.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 1042.00 | 1048.86 | 1027.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1042.00 | 1048.86 | 1027.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1042.00 | 1048.86 | 1027.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:30:00 | 1078.40 | 1053.79 | 1036.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 11:45:00 | 1056.00 | 1060.63 | 1046.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 1060.00 | 1059.01 | 1049.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 1000.60 | 1046.65 | 1046.31 | SL hit (close<static) qty=1.00 sl=1021.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 1000.60 | 1037.44 | 1042.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 963.00 | 1005.15 | 1022.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 1021.00 | 1008.32 | 1022.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 1021.00 | 1008.32 | 1022.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1021.00 | 1008.32 | 1022.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1021.00 | 1008.32 | 1022.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1020.00 | 1010.66 | 1022.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:00:00 | 1020.00 | 1010.66 | 1022.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 1050.60 | 1018.65 | 1025.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:00:00 | 1050.60 | 1018.65 | 1025.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1050.60 | 1025.04 | 1027.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:30:00 | 1050.60 | 1025.04 | 1027.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 1050.60 | 1030.15 | 1029.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 1090.05 | 1045.40 | 1036.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 1051.95 | 1054.00 | 1044.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:45:00 | 1052.00 | 1054.00 | 1044.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1045.55 | 1052.31 | 1044.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 1049.00 | 1052.31 | 1044.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1050.00 | 1051.85 | 1045.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 1046.00 | 1051.85 | 1045.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1050.00 | 1051.48 | 1045.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 13:30:00 | 1070.00 | 1060.51 | 1052.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 14:00:00 | 1070.00 | 1060.51 | 1052.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 1069.95 | 1061.95 | 1056.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 1070.05 | 1063.57 | 1058.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1074.00 | 1080.89 | 1072.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1074.00 | 1080.89 | 1072.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 1065.05 | 1077.72 | 1072.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 1069.00 | 1077.72 | 1072.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 1066.50 | 1075.48 | 1071.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:30:00 | 1062.10 | 1075.48 | 1071.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 1068.95 | 1074.17 | 1071.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 1068.95 | 1074.17 | 1071.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-06 13:15:00 | 1034.55 | 1066.25 | 1067.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 1034.55 | 1066.25 | 1067.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1020.00 | 1047.87 | 1058.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 1030.00 | 1010.24 | 1024.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 13:15:00 | 1030.00 | 1010.24 | 1024.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 1030.00 | 1010.24 | 1024.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 1030.00 | 1010.24 | 1024.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 1042.90 | 1016.77 | 1025.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 1042.90 | 1016.77 | 1025.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1043.00 | 1022.02 | 1027.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 1044.60 | 1022.02 | 1027.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 1021.85 | 1023.26 | 1027.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:15:00 | 993.00 | 1017.00 | 1023.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 999.95 | 1014.20 | 1021.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 12:15:00 | 1024.00 | 1018.09 | 1018.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 1024.00 | 1018.09 | 1018.01 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 13:15:00 | 1013.05 | 1017.08 | 1017.56 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 14:15:00 | 1024.10 | 1018.48 | 1018.16 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 1010.00 | 1016.41 | 1017.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 998.15 | 1012.75 | 1015.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 990.00 | 989.22 | 997.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 990.00 | 989.22 | 997.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 990.00 | 989.22 | 997.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 999.00 | 989.22 | 997.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 986.25 | 988.62 | 996.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 10:15:00 | 974.90 | 987.89 | 995.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 10:45:00 | 981.00 | 985.81 | 993.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 12:15:00 | 1005.70 | 977.69 | 981.73 | SL hit (close>static) qty=1.00 sl=998.25 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 1005.70 | 987.78 | 985.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 1014.80 | 996.05 | 990.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 11:15:00 | 991.00 | 996.47 | 991.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 991.00 | 996.47 | 991.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 991.00 | 996.47 | 991.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 995.00 | 996.47 | 991.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 994.50 | 996.08 | 991.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 995.00 | 996.08 | 991.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 985.00 | 993.86 | 991.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:00:00 | 985.00 | 993.86 | 991.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1024.90 | 1000.07 | 994.19 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 997.50 | 1004.04 | 1004.08 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 1005.00 | 1004.23 | 1004.16 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 998.10 | 1003.00 | 1003.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 995.05 | 1001.41 | 1002.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 12:15:00 | 1010.50 | 999.91 | 1001.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 12:15:00 | 1010.50 | 999.91 | 1001.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1010.50 | 999.91 | 1001.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 1005.05 | 999.91 | 1001.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1011.65 | 1002.25 | 1002.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 1011.65 | 1002.25 | 1002.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 1009.20 | 1003.64 | 1003.02 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 1000.95 | 1002.50 | 1002.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 993.00 | 999.88 | 1001.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 10:15:00 | 993.00 | 990.64 | 994.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 993.00 | 990.64 | 994.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 993.00 | 990.64 | 994.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 993.00 | 990.64 | 994.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 994.05 | 991.32 | 994.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 995.00 | 991.32 | 994.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 995.95 | 992.25 | 994.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 995.45 | 992.25 | 994.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 987.05 | 991.21 | 993.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:45:00 | 993.45 | 991.21 | 993.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 978.10 | 988.59 | 992.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:45:00 | 971.00 | 984.10 | 989.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 969.50 | 984.10 | 989.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:30:00 | 967.65 | 980.85 | 987.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:45:00 | 965.00 | 970.51 | 977.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 964.95 | 966.70 | 972.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 935.00 | 965.58 | 969.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 922.45 | 955.91 | 964.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 921.02 | 955.91 | 964.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 919.27 | 955.91 | 964.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:30:00 | 931.00 | 948.68 | 959.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 13:00:00 | 930.00 | 948.68 | 959.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 946.80 | 945.32 | 956.31 | SL hit (close>ema200) qty=0.50 sl=945.32 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 981.55 | 954.96 | 953.23 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 949.00 | 956.51 | 957.40 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 996.75 | 963.52 | 960.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 1046.55 | 999.74 | 982.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 15:15:00 | 1090.00 | 1091.98 | 1071.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:15:00 | 1075.05 | 1091.98 | 1071.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1072.00 | 1087.99 | 1071.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 1063.10 | 1087.99 | 1071.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1071.00 | 1084.59 | 1071.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 1059.60 | 1084.59 | 1071.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1059.85 | 1079.64 | 1070.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 1067.05 | 1079.64 | 1070.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1064.95 | 1076.70 | 1069.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 13:30:00 | 1065.65 | 1073.06 | 1068.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 15:15:00 | 1051.75 | 1066.13 | 1066.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 1051.75 | 1066.13 | 1066.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1044.05 | 1061.71 | 1064.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 1054.00 | 1051.80 | 1058.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 1054.00 | 1051.80 | 1058.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1054.00 | 1051.80 | 1058.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 1058.80 | 1051.80 | 1058.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1069.55 | 1055.35 | 1059.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 1069.55 | 1055.35 | 1059.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1072.00 | 1058.68 | 1060.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 1072.65 | 1058.68 | 1060.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1073.40 | 1061.62 | 1061.47 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 1042.60 | 1063.52 | 1064.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1018.00 | 1044.32 | 1053.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1066.00 | 1045.08 | 1052.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 1066.00 | 1045.08 | 1052.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1066.00 | 1045.08 | 1052.40 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 1070.65 | 1057.56 | 1056.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 09:15:00 | 1124.15 | 1075.99 | 1066.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 1105.00 | 1108.98 | 1091.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 1105.00 | 1108.98 | 1091.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1105.00 | 1108.98 | 1091.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 1105.00 | 1108.98 | 1091.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 1081.00 | 1103.39 | 1090.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 1082.40 | 1103.39 | 1090.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 1099.85 | 1102.68 | 1091.80 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 1068.80 | 1083.80 | 1085.32 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 1111.80 | 1087.81 | 1086.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 1120.05 | 1094.26 | 1089.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 12:15:00 | 1115.00 | 1115.02 | 1105.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 13:00:00 | 1115.00 | 1115.02 | 1105.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 1098.55 | 1111.73 | 1105.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:45:00 | 1099.20 | 1111.73 | 1105.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1096.60 | 1108.70 | 1104.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 1096.60 | 1108.70 | 1104.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 1109.00 | 1108.76 | 1104.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 1109.65 | 1108.76 | 1104.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-01 17:15:00 | 1220.62 | 1203.73 | 1170.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1210.00 | 1256.21 | 1258.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 1206.35 | 1246.24 | 1254.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 1234.40 | 1225.03 | 1238.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1234.40 | 1225.03 | 1238.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1234.40 | 1225.03 | 1238.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 1231.50 | 1225.03 | 1238.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1248.35 | 1230.30 | 1238.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 1248.35 | 1230.30 | 1238.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 1250.00 | 1234.24 | 1239.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:00:00 | 1250.00 | 1234.24 | 1239.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 1238.95 | 1236.09 | 1239.66 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 1251.50 | 1242.08 | 1241.64 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 1228.70 | 1239.41 | 1240.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1221.00 | 1233.72 | 1237.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 1237.00 | 1183.19 | 1198.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 13:15:00 | 1237.00 | 1183.19 | 1198.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 1237.00 | 1183.19 | 1198.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 14:00:00 | 1237.00 | 1183.19 | 1198.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 1181.05 | 1182.76 | 1197.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 1156.50 | 1186.81 | 1197.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:00:00 | 1177.60 | 1185.49 | 1195.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:45:00 | 1177.75 | 1185.34 | 1194.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:15:00 | 1179.35 | 1185.34 | 1194.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1192.90 | 1160.67 | 1170.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 1192.90 | 1160.67 | 1170.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1192.90 | 1167.11 | 1172.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1192.90 | 1167.11 | 1172.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 1192.90 | 1176.40 | 1176.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1192.90 | 1176.40 | 1176.03 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 1164.90 | 1176.04 | 1176.66 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 1233.40 | 1182.71 | 1178.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1295.05 | 1235.10 | 1210.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 1275.95 | 1280.02 | 1253.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:00:00 | 1275.95 | 1280.02 | 1253.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1406.00 | 1419.27 | 1406.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:45:00 | 1408.60 | 1419.27 | 1406.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1399.65 | 1415.34 | 1405.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1424.40 | 1415.34 | 1405.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 1400.20 | 1426.21 | 1426.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 1400.20 | 1426.21 | 1426.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 13:15:00 | 1393.95 | 1416.42 | 1422.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 12:15:00 | 1405.50 | 1386.20 | 1401.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 12:15:00 | 1405.50 | 1386.20 | 1401.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1405.50 | 1386.20 | 1401.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:45:00 | 1401.10 | 1386.20 | 1401.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 1393.30 | 1387.62 | 1400.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:30:00 | 1407.40 | 1387.62 | 1400.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1394.00 | 1388.89 | 1400.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 1404.55 | 1388.89 | 1400.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1395.95 | 1391.92 | 1399.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 1406.00 | 1391.92 | 1399.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1411.35 | 1395.81 | 1400.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 1411.35 | 1395.81 | 1400.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1411.40 | 1398.93 | 1401.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:15:00 | 1403.35 | 1398.93 | 1401.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 1404.00 | 1400.00 | 1401.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 1401.00 | 1400.20 | 1401.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 1412.95 | 1402.75 | 1402.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 1412.95 | 1402.75 | 1402.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 1434.00 | 1409.00 | 1405.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 1419.30 | 1420.33 | 1413.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:45:00 | 1420.00 | 1420.33 | 1413.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1407.45 | 1417.30 | 1413.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 1407.45 | 1417.30 | 1413.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1399.00 | 1413.64 | 1411.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:00:00 | 1399.00 | 1413.64 | 1411.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1394.00 | 1409.71 | 1410.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1353.35 | 1391.62 | 1401.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1404.30 | 1387.13 | 1394.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 1404.30 | 1387.13 | 1394.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1404.30 | 1387.13 | 1394.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1404.30 | 1387.13 | 1394.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1387.15 | 1387.13 | 1393.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1514.00 | 1387.13 | 1393.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1536.60 | 1417.03 | 1406.71 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 1442.65 | 1466.26 | 1467.50 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 1478.60 | 1464.90 | 1464.86 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 10:15:00 | 1462.80 | 1464.61 | 1464.81 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 1471.60 | 1466.01 | 1465.43 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1454.25 | 1463.66 | 1464.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1436.90 | 1458.31 | 1461.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1474.85 | 1449.74 | 1452.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1474.85 | 1449.74 | 1452.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1474.85 | 1449.74 | 1452.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1474.85 | 1449.74 | 1452.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 10:15:00 | 1480.00 | 1455.80 | 1455.10 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 09:15:00 | 1453.10 | 1465.95 | 1466.47 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 1490.95 | 1470.33 | 1468.16 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1438.35 | 1462.81 | 1465.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 1428.00 | 1452.66 | 1460.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 1434.25 | 1427.39 | 1441.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 1434.25 | 1427.39 | 1441.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1434.25 | 1427.39 | 1441.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 1434.25 | 1427.39 | 1441.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1435.90 | 1429.09 | 1441.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:45:00 | 1438.05 | 1429.09 | 1441.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1436.45 | 1428.29 | 1437.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 1436.45 | 1428.29 | 1437.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1443.00 | 1431.23 | 1438.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 1446.50 | 1431.23 | 1438.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1466.50 | 1438.28 | 1440.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 1466.50 | 1438.28 | 1440.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 1462.00 | 1443.03 | 1442.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1523.65 | 1462.03 | 1453.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1484.85 | 1504.20 | 1485.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 1484.85 | 1504.20 | 1485.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1484.85 | 1504.20 | 1485.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1484.85 | 1504.20 | 1485.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1468.90 | 1497.14 | 1483.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:45:00 | 1470.85 | 1497.14 | 1483.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1471.40 | 1491.99 | 1482.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1465.35 | 1491.99 | 1482.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1455.00 | 1481.20 | 1479.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 1453.10 | 1481.20 | 1479.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1465.00 | 1475.59 | 1476.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 1458.25 | 1472.12 | 1475.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1489.65 | 1475.63 | 1476.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 1489.65 | 1475.63 | 1476.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1489.65 | 1475.63 | 1476.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 1482.50 | 1475.63 | 1476.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1477.40 | 1475.98 | 1476.56 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1492.30 | 1479.25 | 1477.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 13:15:00 | 1495.10 | 1482.42 | 1479.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1462.30 | 1478.91 | 1478.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1462.30 | 1478.91 | 1478.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1462.30 | 1478.91 | 1478.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 1462.30 | 1478.91 | 1478.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 1461.85 | 1475.50 | 1477.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 1444.20 | 1464.84 | 1471.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 13:15:00 | 1451.75 | 1450.63 | 1459.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 13:45:00 | 1449.95 | 1450.63 | 1459.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1409.00 | 1406.25 | 1425.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 13:30:00 | 1361.10 | 1388.58 | 1411.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:00:00 | 1365.00 | 1347.72 | 1358.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 15:15:00 | 1365.00 | 1362.47 | 1362.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 15:15:00 | 1365.00 | 1362.47 | 1362.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1376.30 | 1365.23 | 1363.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 14:15:00 | 1398.05 | 1400.00 | 1389.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 15:00:00 | 1398.05 | 1400.00 | 1389.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1387.00 | 1397.40 | 1389.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 1385.70 | 1397.40 | 1389.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1356.15 | 1389.15 | 1386.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1356.15 | 1389.15 | 1386.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1337.35 | 1378.79 | 1381.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1306.00 | 1343.12 | 1360.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1299.05 | 1295.48 | 1325.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1299.05 | 1295.48 | 1325.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 1359.65 | 1311.72 | 1322.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:00:00 | 1359.65 | 1311.72 | 1322.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1430.10 | 1335.40 | 1332.08 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 1264.50 | 1339.73 | 1347.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1200.80 | 1275.47 | 1308.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1305.50 | 1268.85 | 1295.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 1305.50 | 1268.85 | 1295.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1305.50 | 1268.85 | 1295.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 1295.40 | 1268.85 | 1295.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1303.45 | 1275.77 | 1296.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 1303.45 | 1275.77 | 1296.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1283.05 | 1277.23 | 1295.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 1304.50 | 1277.23 | 1295.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1302.20 | 1282.03 | 1294.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1302.20 | 1282.03 | 1294.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1326.35 | 1290.89 | 1297.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 1331.90 | 1290.89 | 1297.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1324.80 | 1297.67 | 1299.57 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1318.65 | 1301.87 | 1301.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1326.70 | 1309.58 | 1305.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 1299.00 | 1315.64 | 1310.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 12:15:00 | 1299.00 | 1315.64 | 1310.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 1299.00 | 1315.64 | 1310.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 1299.00 | 1315.64 | 1310.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1289.00 | 1310.32 | 1308.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1289.00 | 1310.32 | 1308.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 1286.20 | 1305.49 | 1306.84 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 1409.40 | 1322.58 | 1314.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 1484.30 | 1410.53 | 1370.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1642.65 | 1645.67 | 1612.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:30:00 | 1633.90 | 1645.67 | 1612.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1620.60 | 1640.45 | 1626.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:45:00 | 1662.00 | 1630.22 | 1625.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 15:15:00 | 1593.00 | 1618.35 | 1620.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 1593.00 | 1618.35 | 1620.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1529.95 | 1600.67 | 1612.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1450.00 | 1447.24 | 1493.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 1450.00 | 1447.24 | 1493.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1489.55 | 1459.27 | 1491.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1489.55 | 1459.27 | 1491.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1487.60 | 1464.94 | 1491.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 1508.70 | 1464.94 | 1491.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 1512.40 | 1474.43 | 1492.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 1512.40 | 1474.43 | 1492.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1515.60 | 1482.67 | 1495.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 1498.15 | 1482.67 | 1495.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1498.65 | 1488.42 | 1495.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:15:00 | 1490.00 | 1488.42 | 1495.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1469.50 | 1484.64 | 1493.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:15:00 | 1455.85 | 1484.64 | 1493.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 1383.06 | 1441.28 | 1467.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 1390.45 | 1389.68 | 1419.22 | SL hit (close>ema200) qty=0.50 sl=1389.68 alert=retest2 |

### Cycle 146 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 1217.75 | 1188.29 | 1186.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 1241.00 | 1198.83 | 1191.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 1217.30 | 1218.09 | 1205.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:15:00 | 1255.00 | 1218.09 | 1205.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 09:15:00 | 1317.75 | 1275.07 | 1247.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 1317.20 | 1322.03 | 1295.04 | SL hit (close<ema200) qty=0.50 sl=1322.03 alert=retest1 |

### Cycle 147 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 1262.60 | 1295.87 | 1297.91 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 1312.70 | 1297.01 | 1294.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 1317.75 | 1305.96 | 1300.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 1304.60 | 1305.68 | 1300.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 10:15:00 | 1304.60 | 1305.68 | 1300.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1304.60 | 1305.68 | 1300.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:00:00 | 1304.60 | 1305.68 | 1300.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1303.20 | 1305.19 | 1301.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:30:00 | 1297.00 | 1305.19 | 1301.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1296.70 | 1303.49 | 1300.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:30:00 | 1299.45 | 1303.49 | 1300.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1289.00 | 1300.59 | 1299.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:45:00 | 1290.00 | 1300.59 | 1299.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1288.05 | 1296.95 | 1298.07 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1320.95 | 1298.62 | 1297.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 1331.95 | 1308.59 | 1302.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 09:15:00 | 1496.25 | 1496.66 | 1464.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 09:30:00 | 1509.40 | 1496.66 | 1464.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1472.80 | 1490.00 | 1473.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 1472.80 | 1490.00 | 1473.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1477.50 | 1487.50 | 1473.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 1447.35 | 1487.50 | 1473.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1452.70 | 1480.54 | 1471.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 1458.90 | 1480.54 | 1471.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1436.00 | 1471.63 | 1468.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1433.00 | 1471.63 | 1468.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 1423.35 | 1461.97 | 1464.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1415.00 | 1426.83 | 1438.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 1428.00 | 1412.53 | 1423.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 15:15:00 | 1428.00 | 1412.53 | 1423.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1428.00 | 1412.53 | 1423.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1440.05 | 1412.53 | 1423.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1457.10 | 1421.44 | 1426.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 1457.10 | 1421.44 | 1426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1464.45 | 1430.05 | 1430.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 1464.45 | 1430.05 | 1430.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1446.00 | 1433.24 | 1431.52 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 1418.00 | 1429.58 | 1430.12 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 1444.00 | 1431.64 | 1430.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1496.35 | 1455.88 | 1448.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 1440.50 | 1452.80 | 1447.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 10:15:00 | 1440.50 | 1452.80 | 1447.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1440.50 | 1452.80 | 1447.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:30:00 | 1432.05 | 1452.80 | 1447.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 1404.35 | 1443.11 | 1443.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 1394.60 | 1433.41 | 1439.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1262.40 | 1241.40 | 1292.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1262.40 | 1241.40 | 1292.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1246.20 | 1212.21 | 1235.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:15:00 | 1278.10 | 1212.21 | 1235.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 1308.00 | 1231.37 | 1242.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 1308.00 | 1231.37 | 1242.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1319.80 | 1262.25 | 1255.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1394.80 | 1310.68 | 1282.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 1382.20 | 1383.70 | 1352.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:30:00 | 1367.30 | 1383.70 | 1352.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1418.20 | 1403.01 | 1392.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:45:00 | 1434.30 | 1420.54 | 1409.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1387.30 | 1439.35 | 1432.95 | SL hit (close<static) qty=1.00 sl=1388.90 alert=retest2 |

### Cycle 157 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 1405.00 | 1425.22 | 1427.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 1389.90 | 1410.62 | 1419.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1431.30 | 1414.76 | 1420.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 1431.30 | 1414.76 | 1420.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1431.30 | 1414.76 | 1420.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 1381.20 | 1399.75 | 1409.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 1387.00 | 1395.77 | 1405.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 1386.50 | 1395.77 | 1405.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:15:00 | 1382.90 | 1394.26 | 1403.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:15:00 | 1317.65 | 1354.78 | 1378.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:15:00 | 1317.17 | 1354.78 | 1378.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 12:15:00 | 1312.14 | 1345.97 | 1372.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 12:15:00 | 1313.76 | 1345.97 | 1372.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-05-06 11:15:00 | 1243.08 | 1282.78 | 1303.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 158 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1266.30 | 1253.06 | 1252.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1312.90 | 1271.81 | 1262.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1278.10 | 1284.97 | 1272.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 1278.70 | 1284.97 | 1272.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1275.00 | 1282.18 | 1273.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1280.50 | 1282.18 | 1273.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 1267.20 | 1278.64 | 1273.44 | SL hit (close<static) qty=1.00 sl=1270.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 1255.20 | 1269.09 | 1270.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1247.50 | 1262.89 | 1267.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 1275.20 | 1263.05 | 1266.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 11:15:00 | 1275.20 | 1263.05 | 1266.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1275.20 | 1263.05 | 1266.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 1282.00 | 1263.05 | 1266.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1273.00 | 1265.04 | 1266.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:15:00 | 1278.20 | 1265.04 | 1266.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1290.30 | 1270.09 | 1269.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 1307.30 | 1286.21 | 1277.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1343.70 | 1344.77 | 1320.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 1343.70 | 1344.77 | 1320.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1339.80 | 1350.69 | 1336.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1339.80 | 1350.69 | 1336.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1325.90 | 1345.73 | 1335.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 1317.90 | 1345.73 | 1335.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1324.90 | 1341.56 | 1334.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1343.30 | 1341.56 | 1334.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 1327.50 | 1337.39 | 1334.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1320.50 | 1333.90 | 1333.89 | SL hit (close<static) qty=1.00 sl=1320.70 alert=retest2 |

### Cycle 161 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 1320.90 | 1331.30 | 1332.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 1312.00 | 1325.79 | 1329.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 1339.30 | 1324.88 | 1326.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 1339.30 | 1324.88 | 1326.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1339.30 | 1324.88 | 1326.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 1339.30 | 1324.88 | 1326.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1333.60 | 1326.62 | 1327.56 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1340.40 | 1329.38 | 1328.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 15:15:00 | 1343.00 | 1333.80 | 1330.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1335.00 | 1335.19 | 1332.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 1335.00 | 1335.19 | 1332.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1330.90 | 1334.34 | 1332.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:30:00 | 1330.00 | 1334.34 | 1332.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1329.20 | 1333.31 | 1331.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 1325.70 | 1333.31 | 1331.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1325.00 | 1331.65 | 1331.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 1325.00 | 1331.65 | 1331.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1332.30 | 1332.52 | 1331.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1335.50 | 1332.52 | 1331.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1332.40 | 1332.50 | 1331.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 1321.50 | 1332.50 | 1331.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1333.20 | 1332.64 | 1331.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 1327.90 | 1332.64 | 1331.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1336.70 | 1333.45 | 1332.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 1332.90 | 1333.45 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1342.90 | 1350.68 | 1344.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1342.90 | 1350.68 | 1344.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1336.20 | 1347.78 | 1343.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 1336.20 | 1347.78 | 1343.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1335.00 | 1345.23 | 1342.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:30:00 | 1337.80 | 1344.88 | 1342.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 1343.50 | 1344.88 | 1342.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 1326.00 | 1341.11 | 1341.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1326.00 | 1341.11 | 1341.36 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 1344.80 | 1341.68 | 1341.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1413.00 | 1356.47 | 1348.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1463.80 | 1464.91 | 1436.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 1454.80 | 1464.91 | 1436.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1513.80 | 1523.41 | 1509.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:45:00 | 1510.20 | 1523.41 | 1509.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1503.10 | 1519.35 | 1509.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:45:00 | 1501.00 | 1519.35 | 1509.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1503.70 | 1516.22 | 1508.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 1501.80 | 1516.22 | 1508.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1506.00 | 1514.18 | 1508.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1489.20 | 1514.18 | 1508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1494.20 | 1510.18 | 1507.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 1534.10 | 1511.75 | 1508.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-11 11:15:00 | 1687.51 | 1595.11 | 1554.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 1720.00 | 1755.40 | 1755.75 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 1853.00 | 1774.92 | 1764.59 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 1740.00 | 1756.41 | 1758.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1716.40 | 1748.41 | 1754.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 1732.00 | 1702.42 | 1715.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 13:15:00 | 1732.00 | 1702.42 | 1715.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1732.00 | 1702.42 | 1715.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1732.00 | 1702.42 | 1715.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1731.60 | 1708.26 | 1716.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1731.60 | 1708.26 | 1716.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1725.10 | 1711.62 | 1717.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 1722.30 | 1716.66 | 1719.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1733.90 | 1722.23 | 1721.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1733.90 | 1722.23 | 1721.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1754.70 | 1728.72 | 1724.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1731.10 | 1749.19 | 1741.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 1731.10 | 1749.19 | 1741.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1731.10 | 1749.19 | 1741.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1731.10 | 1749.19 | 1741.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1750.00 | 1749.35 | 1742.40 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 13:15:00 | 1718.90 | 1736.54 | 1738.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 1685.00 | 1721.03 | 1730.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 10:15:00 | 1654.10 | 1650.18 | 1668.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:45:00 | 1656.50 | 1650.18 | 1668.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1701.50 | 1660.44 | 1671.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:45:00 | 1701.40 | 1660.44 | 1671.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 1705.30 | 1669.41 | 1674.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:45:00 | 1722.70 | 1669.41 | 1674.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1728.00 | 1681.13 | 1679.48 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1664.00 | 1680.25 | 1681.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1653.70 | 1673.11 | 1677.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 1681.00 | 1674.68 | 1677.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 1681.00 | 1674.68 | 1677.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1681.00 | 1674.68 | 1677.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 1681.00 | 1674.68 | 1677.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1676.00 | 1674.95 | 1677.25 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 12:15:00 | 1695.60 | 1679.08 | 1678.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 13:15:00 | 1704.00 | 1684.06 | 1681.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1726.90 | 1730.41 | 1714.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 1726.90 | 1730.41 | 1714.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1726.90 | 1730.41 | 1714.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1717.30 | 1730.41 | 1714.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1720.00 | 1729.93 | 1719.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1742.40 | 1729.93 | 1719.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1738.10 | 1731.56 | 1721.44 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 1697.00 | 1716.30 | 1718.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1681.50 | 1709.34 | 1714.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1686.20 | 1677.23 | 1691.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1686.20 | 1677.23 | 1691.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1686.20 | 1677.23 | 1691.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1694.40 | 1677.23 | 1691.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1692.80 | 1680.35 | 1692.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1692.80 | 1680.35 | 1692.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1717.30 | 1687.74 | 1694.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 1717.30 | 1687.74 | 1694.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1721.50 | 1694.49 | 1696.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 1721.50 | 1694.49 | 1696.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 1720.70 | 1699.73 | 1698.96 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1695.00 | 1700.55 | 1701.07 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1788.00 | 1716.99 | 1708.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1807.50 | 1769.81 | 1746.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1805.60 | 1818.06 | 1797.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 1810.00 | 1818.06 | 1797.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1780.50 | 1812.46 | 1810.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 1780.50 | 1812.46 | 1810.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1782.10 | 1806.39 | 1807.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1768.20 | 1783.08 | 1793.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 1733.50 | 1724.01 | 1742.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 1733.50 | 1724.01 | 1742.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1733.50 | 1724.01 | 1742.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1736.70 | 1724.01 | 1742.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1726.00 | 1720.78 | 1731.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 1732.80 | 1720.78 | 1731.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1710.00 | 1718.62 | 1729.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 1721.70 | 1718.62 | 1729.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1738.00 | 1697.26 | 1705.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 1762.40 | 1697.26 | 1705.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1713.40 | 1700.49 | 1705.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 1710.10 | 1701.83 | 1705.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:30:00 | 1707.50 | 1697.19 | 1699.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 1624.59 | 1638.33 | 1658.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 1622.12 | 1638.33 | 1658.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-05 15:15:00 | 1539.09 | 1558.58 | 1585.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 178 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 1509.60 | 1492.26 | 1491.95 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 1500.00 | 1509.45 | 1510.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 11:15:00 | 1495.00 | 1506.56 | 1508.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1496.20 | 1490.42 | 1495.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1496.20 | 1490.42 | 1495.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1496.20 | 1490.42 | 1495.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1496.60 | 1490.42 | 1495.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1503.70 | 1493.07 | 1496.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1508.70 | 1493.07 | 1496.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1491.00 | 1492.66 | 1495.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 1488.90 | 1492.66 | 1495.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:45:00 | 1490.40 | 1491.67 | 1495.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 13:15:00 | 1415.88 | 1427.24 | 1436.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 14:15:00 | 1414.45 | 1424.39 | 1434.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1429.30 | 1423.99 | 1430.92 | SL hit (close>ema200) qty=0.50 sl=1423.99 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1466.10 | 1432.34 | 1431.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 1479.40 | 1441.75 | 1435.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1475.80 | 1478.16 | 1464.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1475.80 | 1478.16 | 1464.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1461.00 | 1474.15 | 1467.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 1461.50 | 1474.15 | 1467.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1466.50 | 1472.62 | 1466.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 1464.80 | 1472.62 | 1466.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1481.40 | 1474.38 | 1468.30 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1456.60 | 1465.82 | 1466.75 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1517.50 | 1475.19 | 1470.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1544.50 | 1518.18 | 1500.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 10:15:00 | 1525.80 | 1532.16 | 1514.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:45:00 | 1537.60 | 1532.16 | 1514.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1535.40 | 1529.11 | 1520.02 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1511.10 | 1519.06 | 1519.92 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1567.90 | 1527.35 | 1523.31 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1517.00 | 1538.02 | 1538.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 1514.70 | 1524.69 | 1530.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 1475.30 | 1474.02 | 1486.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:00:00 | 1475.30 | 1474.02 | 1486.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1485.20 | 1476.25 | 1486.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 1495.30 | 1476.25 | 1486.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1486.00 | 1478.20 | 1486.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:45:00 | 1494.00 | 1478.20 | 1486.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1487.30 | 1480.02 | 1486.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1487.80 | 1480.02 | 1486.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1498.10 | 1483.64 | 1487.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1498.10 | 1483.64 | 1487.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1487.00 | 1484.31 | 1487.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1484.80 | 1484.97 | 1487.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1410.56 | 1462.90 | 1475.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-26 14:15:00 | 1336.32 | 1400.36 | 1437.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 186 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 1538.20 | 1452.35 | 1447.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 1585.00 | 1478.88 | 1459.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 15:15:00 | 1507.80 | 1514.01 | 1493.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1474.80 | 1514.01 | 1493.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1493.60 | 1509.92 | 1493.59 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 1466.70 | 1486.41 | 1488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 13:15:00 | 1465.50 | 1479.60 | 1484.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1410.40 | 1404.90 | 1419.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:15:00 | 1419.10 | 1404.90 | 1419.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1401.40 | 1404.20 | 1417.58 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1434.00 | 1415.96 | 1415.54 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1402.90 | 1415.44 | 1415.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1388.90 | 1410.13 | 1413.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1378.80 | 1377.51 | 1387.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1378.80 | 1377.51 | 1387.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1378.80 | 1377.51 | 1387.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1370.00 | 1377.21 | 1386.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 1371.00 | 1348.47 | 1349.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 1371.00 | 1352.98 | 1351.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1371.00 | 1352.98 | 1351.77 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1340.00 | 1353.60 | 1354.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 1335.00 | 1349.88 | 1352.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 1305.60 | 1305.54 | 1319.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 14:15:00 | 1298.40 | 1304.21 | 1315.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1308.80 | 1302.45 | 1310.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 1311.60 | 1302.45 | 1310.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1317.00 | 1305.36 | 1311.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1317.00 | 1305.36 | 1311.23 | SL hit (close>ema400) qty=1.00 sl=1311.23 alert=retest1 |

### Cycle 192 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1351.00 | 1300.57 | 1300.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1394.60 | 1341.90 | 1321.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 1369.40 | 1371.36 | 1351.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 13:30:00 | 1372.00 | 1371.36 | 1351.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1350.30 | 1365.16 | 1353.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1350.30 | 1365.16 | 1353.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1347.10 | 1361.55 | 1352.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1353.60 | 1361.20 | 1353.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:30:00 | 1355.60 | 1360.68 | 1353.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 1357.80 | 1359.60 | 1354.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1326.50 | 1351.44 | 1351.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 1326.50 | 1351.44 | 1351.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 1319.30 | 1345.01 | 1348.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1337.00 | 1336.86 | 1342.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 1340.00 | 1336.86 | 1342.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1327.10 | 1334.61 | 1340.78 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1355.10 | 1342.35 | 1340.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 1364.40 | 1350.72 | 1345.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1352.90 | 1355.53 | 1350.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 1352.90 | 1355.53 | 1350.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1352.90 | 1355.53 | 1350.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1351.80 | 1355.53 | 1350.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1354.10 | 1355.24 | 1350.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:30:00 | 1356.00 | 1355.24 | 1350.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1350.00 | 1354.19 | 1350.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1350.00 | 1354.19 | 1350.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1344.10 | 1352.18 | 1350.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1346.60 | 1352.18 | 1350.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1347.00 | 1349.37 | 1349.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 1350.60 | 1349.37 | 1349.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1350.90 | 1349.67 | 1349.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1345.30 | 1348.80 | 1349.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 1345.30 | 1348.80 | 1349.01 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1376.20 | 1353.79 | 1351.09 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1346.60 | 1355.12 | 1356.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 1332.60 | 1342.79 | 1348.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 1292.80 | 1291.88 | 1306.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 1292.80 | 1291.88 | 1306.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1309.80 | 1295.47 | 1306.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1282.50 | 1295.47 | 1306.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1310.80 | 1283.11 | 1285.40 | SL hit (close>static) qty=1.00 sl=1309.80 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1375.60 | 1271.47 | 1270.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 12:15:00 | 1460.50 | 1309.27 | 1287.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1438.00 | 1469.20 | 1418.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 1438.00 | 1469.20 | 1418.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1419.60 | 1449.03 | 1421.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 1419.60 | 1449.03 | 1421.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1411.80 | 1441.58 | 1420.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:45:00 | 1412.80 | 1441.58 | 1420.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1412.90 | 1431.21 | 1419.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 1403.50 | 1431.21 | 1419.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 1362.10 | 1409.03 | 1410.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 1344.90 | 1396.21 | 1404.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 1373.50 | 1361.92 | 1381.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1373.50 | 1361.92 | 1381.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1373.50 | 1361.92 | 1381.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1383.80 | 1361.92 | 1381.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1365.20 | 1362.58 | 1379.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 15:15:00 | 1348.60 | 1361.93 | 1374.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:30:00 | 1351.70 | 1338.22 | 1340.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1351.70 | 1340.34 | 1339.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 1351.70 | 1340.34 | 1339.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1357.30 | 1346.04 | 1342.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 14:15:00 | 1359.40 | 1361.97 | 1354.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 14:30:00 | 1359.50 | 1361.97 | 1354.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1360.10 | 1361.59 | 1355.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1340.40 | 1361.59 | 1355.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1347.10 | 1358.69 | 1354.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 1341.40 | 1358.69 | 1354.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1372.70 | 1361.50 | 1356.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 1375.60 | 1364.32 | 1358.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 1376.10 | 1367.27 | 1360.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 14:30:00 | 1375.90 | 1369.37 | 1362.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1430.50 | 1439.58 | 1439.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1430.50 | 1439.58 | 1439.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1424.00 | 1436.46 | 1438.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1398.30 | 1394.47 | 1410.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 1398.30 | 1394.47 | 1410.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1392.30 | 1394.04 | 1408.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1381.20 | 1394.04 | 1408.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1440.20 | 1401.04 | 1409.22 | SL hit (close>static) qty=1.00 sl=1414.90 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1445.00 | 1416.42 | 1415.20 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1408.70 | 1419.67 | 1419.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 14:15:00 | 1404.50 | 1414.75 | 1417.57 | Break + close below crossover candle low |

### Cycle 204 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 1477.00 | 1425.34 | 1421.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 12:15:00 | 1488.70 | 1458.47 | 1443.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 1477.60 | 1493.92 | 1477.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 1477.60 | 1493.92 | 1477.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1477.60 | 1493.92 | 1477.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1477.60 | 1493.92 | 1477.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1476.80 | 1490.49 | 1477.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 1477.90 | 1490.49 | 1477.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1471.90 | 1486.78 | 1477.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 1471.90 | 1486.78 | 1477.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1466.80 | 1482.78 | 1476.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 1464.80 | 1482.78 | 1476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1441.00 | 1467.45 | 1470.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 1429.70 | 1454.71 | 1463.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1402.50 | 1401.28 | 1425.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 1393.70 | 1401.28 | 1425.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1401.60 | 1391.10 | 1401.93 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1431.20 | 1403.77 | 1403.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1434.20 | 1409.85 | 1406.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 1423.50 | 1425.34 | 1417.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 1405.00 | 1425.34 | 1417.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1412.80 | 1422.83 | 1416.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 1397.20 | 1422.83 | 1416.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1432.00 | 1424.66 | 1418.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 1414.80 | 1424.66 | 1418.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1422.40 | 1424.21 | 1418.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 1415.00 | 1424.21 | 1418.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 1416.80 | 1423.56 | 1419.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 1416.80 | 1423.56 | 1419.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1417.60 | 1422.37 | 1419.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1408.20 | 1422.37 | 1419.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1404.40 | 1418.78 | 1418.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1404.40 | 1418.78 | 1418.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1405.10 | 1416.04 | 1416.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 1381.20 | 1409.07 | 1413.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1382.00 | 1362.32 | 1376.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1382.00 | 1362.32 | 1376.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1382.00 | 1362.32 | 1376.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1378.90 | 1362.32 | 1376.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1371.00 | 1364.06 | 1376.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1367.30 | 1364.06 | 1376.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:30:00 | 1362.00 | 1357.08 | 1361.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1366.10 | 1353.58 | 1354.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 1370.30 | 1355.90 | 1355.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 1370.30 | 1355.90 | 1355.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1379.90 | 1362.48 | 1358.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1376.10 | 1376.36 | 1369.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1376.10 | 1376.36 | 1369.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1376.10 | 1376.36 | 1369.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 1367.60 | 1376.36 | 1369.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1377.30 | 1376.55 | 1369.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1370.00 | 1376.55 | 1369.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1371.20 | 1375.48 | 1369.95 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1343.00 | 1364.14 | 1366.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1319.70 | 1355.25 | 1361.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 1321.50 | 1320.51 | 1337.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 1379.00 | 1320.51 | 1337.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1356.50 | 1327.71 | 1339.11 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 1356.90 | 1345.99 | 1344.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1379.90 | 1353.95 | 1348.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1384.00 | 1388.11 | 1370.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1384.00 | 1388.11 | 1370.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1369.60 | 1384.41 | 1370.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 1365.00 | 1384.41 | 1370.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1397.10 | 1386.95 | 1373.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1458.60 | 1390.79 | 1384.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 1414.20 | 1415.25 | 1404.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:30:00 | 1421.20 | 1407.21 | 1403.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:00:00 | 1408.00 | 1410.34 | 1406.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1404.30 | 1409.13 | 1405.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1404.30 | 1409.13 | 1405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1409.90 | 1409.29 | 1406.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:15:00 | 1406.40 | 1409.29 | 1406.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1405.00 | 1408.43 | 1406.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 1401.80 | 1408.43 | 1406.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1410.10 | 1408.76 | 1406.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1378.10 | 1397.48 | 1401.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 1393.00 | 1386.69 | 1392.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 10:15:00 | 1393.00 | 1386.69 | 1392.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1393.00 | 1386.69 | 1392.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 1393.00 | 1386.69 | 1392.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 1407.40 | 1390.83 | 1393.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 1407.40 | 1390.83 | 1393.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1413.50 | 1395.37 | 1395.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1416.10 | 1395.37 | 1395.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 13:15:00 | 1409.20 | 1398.13 | 1396.97 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1391.40 | 1395.95 | 1396.12 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 1397.40 | 1396.16 | 1396.11 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 13:15:00 | 1395.00 | 1395.93 | 1396.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 1389.00 | 1394.54 | 1395.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1399.10 | 1394.81 | 1395.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 1399.10 | 1394.81 | 1395.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1399.10 | 1394.81 | 1395.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1399.10 | 1394.81 | 1395.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1402.90 | 1396.43 | 1396.00 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 1393.50 | 1395.73 | 1395.82 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1398.30 | 1396.25 | 1396.04 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 15:15:00 | 1393.50 | 1395.70 | 1395.81 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1408.80 | 1398.32 | 1396.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 1422.00 | 1406.98 | 1401.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 1410.90 | 1417.38 | 1409.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:00:00 | 1410.90 | 1417.38 | 1409.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1398.60 | 1413.62 | 1408.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1398.60 | 1413.62 | 1408.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1405.10 | 1411.92 | 1408.55 | EMA400 retest candle locked (from upside) |

### Cycle 221 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1395.30 | 1406.40 | 1406.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1385.00 | 1402.12 | 1404.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 15:15:00 | 1370.00 | 1368.91 | 1379.56 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1353.90 | 1368.91 | 1379.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 12:00:00 | 1360.10 | 1363.67 | 1374.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1360.00 | 1361.69 | 1370.48 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1367.50 | 1362.58 | 1369.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 1365.30 | 1362.58 | 1369.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1367.00 | 1363.46 | 1369.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1363.30 | 1363.46 | 1369.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1363.10 | 1363.39 | 1368.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:30:00 | 1368.50 | 1363.39 | 1368.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1360.00 | 1362.71 | 1367.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:30:00 | 1356.00 | 1361.99 | 1367.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1382.30 | 1366.05 | 1368.40 | SL hit (close>ema400) qty=1.00 sl=1368.40 alert=retest1 |

### Cycle 222 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1391.60 | 1373.60 | 1371.58 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 1360.80 | 1370.10 | 1370.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1344.10 | 1360.83 | 1365.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1313.30 | 1298.52 | 1315.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1313.30 | 1298.52 | 1315.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1313.30 | 1298.52 | 1315.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 1313.30 | 1298.52 | 1315.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1310.20 | 1300.86 | 1315.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1298.40 | 1303.97 | 1313.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1322.10 | 1308.34 | 1312.99 | SL hit (close>static) qty=1.00 sl=1315.50 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1322.10 | 1299.52 | 1296.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 1327.80 | 1308.45 | 1301.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 1319.90 | 1321.00 | 1312.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:15:00 | 1318.10 | 1321.00 | 1312.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1308.40 | 1318.48 | 1311.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1308.40 | 1318.48 | 1311.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1304.00 | 1315.58 | 1311.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1289.00 | 1315.58 | 1311.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1307.50 | 1308.98 | 1308.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 1307.50 | 1308.98 | 1308.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 1305.60 | 1308.30 | 1308.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 1295.90 | 1305.82 | 1307.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1204.70 | 1197.55 | 1213.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1204.70 | 1197.55 | 1213.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1204.70 | 1197.55 | 1213.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1214.20 | 1197.55 | 1213.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1223.00 | 1202.64 | 1214.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1223.00 | 1202.64 | 1214.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1214.80 | 1205.07 | 1214.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 1210.50 | 1206.52 | 1213.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1195.00 | 1210.90 | 1214.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1149.97 | 1174.22 | 1188.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1135.25 | 1174.22 | 1188.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 12:15:00 | 1089.45 | 1139.29 | 1167.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 226 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1225.40 | 1153.65 | 1153.06 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 1164.00 | 1183.62 | 1184.36 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 1242.40 | 1195.38 | 1189.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 1272.90 | 1216.68 | 1200.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1245.40 | 1247.49 | 1224.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 11:15:00 | 1229.80 | 1243.57 | 1226.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 1229.80 | 1243.57 | 1226.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:30:00 | 1223.30 | 1243.57 | 1226.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1247.20 | 1244.30 | 1228.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:30:00 | 1230.00 | 1244.30 | 1228.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1242.90 | 1253.66 | 1239.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 1241.30 | 1253.66 | 1239.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1307.00 | 1276.30 | 1258.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1317.00 | 1276.30 | 1258.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1342.10 | 1294.39 | 1277.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 1319.30 | 1348.52 | 1345.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1331.90 | 1342.33 | 1343.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1331.90 | 1342.33 | 1343.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 13:15:00 | 1321.80 | 1335.93 | 1339.84 | Break + close below crossover candle low |

### Cycle 230 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 1369.20 | 1342.59 | 1342.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 1386.00 | 1365.57 | 1358.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 1406.90 | 1416.85 | 1403.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 1406.90 | 1416.85 | 1403.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1402.20 | 1413.92 | 1403.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1423.00 | 1413.92 | 1403.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 1411.40 | 1433.43 | 1433.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1411.40 | 1433.43 | 1433.47 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1444.20 | 1428.96 | 1428.29 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1413.40 | 1428.86 | 1428.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1405.90 | 1415.90 | 1421.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 1405.60 | 1399.87 | 1407.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:00:00 | 1405.60 | 1399.87 | 1407.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1402.00 | 1400.30 | 1407.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 1398.40 | 1400.30 | 1407.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 1397.10 | 1399.66 | 1406.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1432.80 | 1405.62 | 1407.77 | SL hit (close>static) qty=1.00 sl=1412.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1431.60 | 1410.82 | 1409.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1588.00 | 1452.81 | 1430.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1653.00 | 1673.52 | 1603.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:30:00 | 1652.00 | 1673.52 | 1603.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1624.90 | 1644.87 | 1620.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:30:00 | 1610.10 | 1644.87 | 1620.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1615.10 | 1638.92 | 1620.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1615.10 | 1638.92 | 1620.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1611.70 | 1633.47 | 1619.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 1611.70 | 1633.47 | 1619.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1608.50 | 1628.48 | 1618.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 1609.50 | 1628.48 | 1618.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-23 09:30:00 | 580.25 | 2024-04-23 10:15:00 | 563.80 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-05-06 09:45:00 | 548.00 | 2024-05-07 12:15:00 | 520.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 11:00:00 | 550.00 | 2024-05-07 12:15:00 | 522.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 12:45:00 | 551.00 | 2024-05-07 12:15:00 | 523.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 09:45:00 | 548.00 | 2024-05-07 14:15:00 | 541.00 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2024-05-06 11:00:00 | 550.00 | 2024-05-07 14:15:00 | 541.00 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2024-05-06 12:45:00 | 551.00 | 2024-05-07 14:15:00 | 541.00 | STOP_HIT | 0.50 | 1.81% |
| BUY | retest2 | 2024-05-16 10:15:00 | 550.00 | 2024-05-17 14:15:00 | 544.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-05-16 13:45:00 | 549.65 | 2024-05-17 14:15:00 | 544.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-05-16 14:30:00 | 550.00 | 2024-05-17 14:15:00 | 544.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-05-16 15:15:00 | 553.20 | 2024-05-17 14:15:00 | 544.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-05-28 09:15:00 | 564.70 | 2024-05-28 09:15:00 | 549.15 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-06-04 09:15:00 | 523.00 | 2024-06-05 09:15:00 | 496.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:30:00 | 519.95 | 2024-06-05 09:15:00 | 493.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 523.00 | 2024-06-05 12:15:00 | 513.95 | STOP_HIT | 0.50 | 1.73% |
| SELL | retest2 | 2024-06-04 10:30:00 | 519.95 | 2024-06-05 12:15:00 | 513.95 | STOP_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-06-11 09:15:00 | 579.75 | 2024-06-13 14:15:00 | 578.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-06-21 15:00:00 | 576.25 | 2024-06-26 09:15:00 | 633.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-11 11:15:00 | 878.50 | 2024-07-11 11:15:00 | 873.55 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-07-25 15:15:00 | 872.00 | 2024-07-29 12:15:00 | 850.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-07-26 13:00:00 | 864.00 | 2024-07-29 12:15:00 | 850.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-07-29 10:00:00 | 862.60 | 2024-07-29 12:15:00 | 850.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-07-29 11:15:00 | 861.50 | 2024-07-29 12:15:00 | 850.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-06 13:15:00 | 865.50 | 2024-08-07 13:15:00 | 890.15 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2024-08-06 14:45:00 | 860.95 | 2024-08-07 13:15:00 | 890.15 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-08-09 15:15:00 | 967.90 | 2024-08-14 10:15:00 | 914.00 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest2 | 2024-08-12 10:15:00 | 970.00 | 2024-08-14 10:15:00 | 914.00 | STOP_HIT | 1.00 | -5.77% |
| BUY | retest2 | 2024-08-27 13:30:00 | 1078.40 | 2024-08-29 10:15:00 | 1000.60 | STOP_HIT | 1.00 | -7.21% |
| BUY | retest2 | 2024-08-28 11:45:00 | 1056.00 | 2024-08-29 10:15:00 | 1000.60 | STOP_HIT | 1.00 | -5.25% |
| BUY | retest2 | 2024-08-28 15:15:00 | 1060.00 | 2024-08-29 10:15:00 | 1000.60 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2024-09-03 13:30:00 | 1070.00 | 2024-09-06 13:15:00 | 1034.55 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-09-03 14:00:00 | 1070.00 | 2024-09-06 13:15:00 | 1034.55 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-09-04 14:15:00 | 1069.95 | 2024-09-06 13:15:00 | 1034.55 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-09-04 14:45:00 | 1070.05 | 2024-09-06 13:15:00 | 1034.55 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-09-11 14:15:00 | 993.00 | 2024-09-13 12:15:00 | 1024.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2024-09-11 15:15:00 | 999.95 | 2024-09-13 12:15:00 | 1024.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-09-18 10:15:00 | 974.90 | 2024-09-19 12:15:00 | 1005.70 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-09-18 10:45:00 | 981.00 | 2024-09-19 12:15:00 | 1005.70 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-10-01 09:45:00 | 971.00 | 2024-10-07 10:15:00 | 922.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 969.50 | 2024-10-07 10:15:00 | 921.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:30:00 | 967.65 | 2024-10-07 10:15:00 | 919.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:45:00 | 971.00 | 2024-10-07 14:15:00 | 946.80 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2024-10-01 10:15:00 | 969.50 | 2024-10-07 14:15:00 | 946.80 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2024-10-01 11:30:00 | 967.65 | 2024-10-07 14:15:00 | 946.80 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2024-10-03 11:45:00 | 965.00 | 2024-10-08 09:15:00 | 916.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 11:45:00 | 965.00 | 2024-10-08 11:15:00 | 936.70 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2024-10-07 10:15:00 | 935.00 | 2024-10-09 09:15:00 | 981.55 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2024-10-07 12:30:00 | 931.00 | 2024-10-09 09:15:00 | 981.55 | STOP_HIT | 1.00 | -5.43% |
| SELL | retest2 | 2024-10-07 13:00:00 | 930.00 | 2024-10-09 09:15:00 | 981.55 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2024-10-07 15:15:00 | 925.00 | 2024-10-09 09:15:00 | 981.55 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest2 | 2024-10-17 13:30:00 | 1065.65 | 2024-10-17 15:15:00 | 1051.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-10-30 09:15:00 | 1109.65 | 2024-11-01 17:15:00 | 1220.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-14 09:15:00 | 1156.50 | 2024-11-19 12:15:00 | 1192.90 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-11-14 11:00:00 | 1177.60 | 2024-11-19 12:15:00 | 1192.90 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-11-14 11:45:00 | 1177.75 | 2024-11-19 12:15:00 | 1192.90 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-11-14 12:15:00 | 1179.35 | 2024-11-19 12:15:00 | 1192.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-12-04 09:15:00 | 1424.40 | 2024-12-06 11:15:00 | 1400.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-10 12:15:00 | 1403.35 | 2024-12-11 10:15:00 | 1412.95 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-12-10 14:15:00 | 1404.00 | 2024-12-11 10:15:00 | 1412.95 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-12-11 09:30:00 | 1401.00 | 2024-12-11 10:15:00 | 1412.95 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-01-13 13:30:00 | 1361.10 | 2025-01-16 15:15:00 | 1365.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-01-16 10:00:00 | 1365.00 | 2025-01-16 15:15:00 | 1365.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-02-07 13:45:00 | 1662.00 | 2025-02-07 15:15:00 | 1593.00 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-02-13 12:15:00 | 1455.85 | 2025-02-14 09:15:00 | 1383.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:15:00 | 1455.85 | 2025-02-17 10:15:00 | 1390.45 | STOP_HIT | 0.50 | 4.49% |
| BUY | retest1 | 2025-03-05 09:15:00 | 1255.00 | 2025-03-06 09:15:00 | 1317.75 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-05 09:15:00 | 1255.00 | 2025-03-07 11:15:00 | 1317.20 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2025-04-24 09:45:00 | 1434.30 | 2025-04-25 10:15:00 | 1387.30 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1381.20 | 2025-04-30 11:15:00 | 1317.65 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2025-04-29 11:45:00 | 1387.00 | 2025-04-30 11:15:00 | 1317.17 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1386.50 | 2025-04-30 12:15:00 | 1312.14 | PARTIAL | 0.50 | 5.36% |
| SELL | retest2 | 2025-04-29 13:15:00 | 1382.90 | 2025-04-30 12:15:00 | 1313.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1381.20 | 2025-05-06 11:15:00 | 1243.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-29 11:45:00 | 1387.00 | 2025-05-06 11:15:00 | 1248.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1386.50 | 2025-05-06 11:15:00 | 1247.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-29 13:15:00 | 1382.90 | 2025-05-06 11:15:00 | 1244.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-08 13:45:00 | 1246.90 | 2025-05-12 10:15:00 | 1284.80 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1234.10 | 2025-05-12 10:15:00 | 1284.80 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1280.50 | 2025-05-14 10:15:00 | 1267.20 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1343.30 | 2025-05-22 09:15:00 | 1320.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-21 12:15:00 | 1327.50 | 2025-05-22 09:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-29 09:30:00 | 1337.80 | 2025-05-29 10:15:00 | 1326.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-05-29 10:00:00 | 1343.50 | 2025-05-29 10:15:00 | 1326.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-06-10 09:30:00 | 1534.10 | 2025-06-11 11:15:00 | 1687.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-23 09:30:00 | 1722.30 | 2025-06-23 11:15:00 | 1733.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-28 11:30:00 | 1710.10 | 2025-08-01 15:15:00 | 1624.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 12:30:00 | 1707.50 | 2025-08-01 15:15:00 | 1622.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 11:30:00 | 1710.10 | 2025-08-05 15:15:00 | 1539.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-29 12:30:00 | 1707.50 | 2025-08-05 15:15:00 | 1536.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 1488.90 | 2025-09-01 13:15:00 | 1415.88 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-08-21 12:45:00 | 1490.40 | 2025-09-01 14:15:00 | 1414.45 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-08-21 12:15:00 | 1488.90 | 2025-09-02 11:15:00 | 1429.30 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-08-21 12:45:00 | 1490.40 | 2025-09-02 11:15:00 | 1429.30 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1484.80 | 2025-09-26 09:15:00 | 1410.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1484.80 | 2025-09-26 14:15:00 | 1336.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-15 12:15:00 | 1370.00 | 2025-10-21 14:15:00 | 1371.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-21 14:15:00 | 1371.00 | 2025-10-21 14:15:00 | 1371.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest1 | 2025-10-28 14:15:00 | 1298.40 | 2025-10-29 11:15:00 | 1317.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-29 15:00:00 | 1305.50 | 2025-11-03 09:15:00 | 1351.00 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-10-30 09:30:00 | 1299.00 | 2025-11-03 09:15:00 | 1351.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-11-06 11:30:00 | 1353.60 | 2025-11-07 09:15:00 | 1326.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-11-06 12:30:00 | 1355.60 | 2025-11-07 09:15:00 | 1326.50 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-11-06 15:00:00 | 1357.80 | 2025-11-07 09:15:00 | 1326.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-11-14 11:15:00 | 1350.60 | 2025-11-14 12:15:00 | 1345.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1350.90 | 2025-11-14 12:15:00 | 1345.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1282.50 | 2025-11-27 09:15:00 | 1310.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-11-27 11:45:00 | 1285.00 | 2025-12-01 11:15:00 | 1375.60 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2025-12-05 15:15:00 | 1348.60 | 2025-12-15 09:15:00 | 1351.70 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-12-10 10:30:00 | 1351.70 | 2025-12-15 09:15:00 | 1351.70 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-18 12:00:00 | 1375.60 | 2025-12-29 14:15:00 | 1430.50 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-12-18 13:30:00 | 1376.10 | 2025-12-29 14:15:00 | 1430.50 | STOP_HIT | 1.00 | 3.95% |
| BUY | retest2 | 2025-12-18 14:30:00 | 1375.90 | 2025-12-29 14:15:00 | 1430.50 | STOP_HIT | 1.00 | 3.97% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1381.20 | 2025-12-31 12:15:00 | 1440.20 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1367.30 | 2026-01-29 14:15:00 | 1370.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-01-27 10:30:00 | 1362.00 | 2026-01-29 14:15:00 | 1370.30 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1366.10 | 2026-01-29 14:15:00 | 1370.30 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1458.60 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2026-02-10 09:15:00 | 1414.20 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-02-10 14:30:00 | 1421.20 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-11 10:00:00 | 1408.00 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest1 | 2026-02-24 09:15:00 | 1353.90 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest1 | 2026-02-24 12:00:00 | 1360.10 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest1 | 2026-02-24 15:15:00 | 1360.00 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-25 13:30:00 | 1356.00 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1298.40 | 2026-03-06 09:15:00 | 1322.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1303.30 | 2026-03-10 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-03-06 14:45:00 | 1299.90 | 2026-03-10 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1264.70 | 2026-03-10 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2026-03-10 11:15:00 | 1313.40 | 2026-03-10 12:15:00 | 1322.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-03-10 11:45:00 | 1313.00 | 2026-03-10 12:15:00 | 1322.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-03-18 12:45:00 | 1210.50 | 2026-03-23 09:15:00 | 1149.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1195.00 | 2026-03-23 09:15:00 | 1135.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:45:00 | 1210.50 | 2026-03-23 12:15:00 | 1089.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1195.00 | 2026-03-24 09:15:00 | 1134.00 | STOP_HIT | 0.50 | 5.10% |
| BUY | retest2 | 2026-04-07 10:15:00 | 1317.00 | 2026-04-13 11:15:00 | 1331.90 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1342.10 | 2026-04-13 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-04-13 09:45:00 | 1319.30 | 2026-04-13 11:15:00 | 1331.90 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2026-04-22 09:15:00 | 1423.00 | 2026-04-24 11:15:00 | 1411.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1398.40 | 2026-05-04 09:15:00 | 1432.80 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-30 15:00:00 | 1397.10 | 2026-05-04 09:15:00 | 1432.80 | STOP_HIT | 1.00 | -2.56% |

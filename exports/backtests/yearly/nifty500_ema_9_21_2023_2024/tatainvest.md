# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 719.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 134 |
| ALERT2 | 130 |
| ALERT2_SKIP | 89 |
| ALERT3 | 250 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 109 |
| PARTIAL | 32 |
| TARGET_HIT | 4 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 80 / 64
- **Target hits / Stop hits / Partials:** 4 / 108 / 32
- **Avg / median % per leg:** 1.26% / 0.64%
- **Sum % (uncompounded):** 181.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 8 | 21.1% | 4 | 34 | 0 | -0.15% | -5.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.18% | -6.2% |
| BUY @ 3rd Alert (retest2) | 37 | 8 | 21.6% | 4 | 33 | 0 | 0.02% | 0.6% |
| SELL (all) | 106 | 72 | 67.9% | 0 | 74 | 32 | 1.76% | 186.6% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.50% | 18.0% |
| SELL @ 3rd Alert (retest2) | 102 | 68 | 66.7% | 0 | 72 | 30 | 1.65% | 168.6% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.37% | 11.8% |
| retest2 (combined) | 139 | 76 | 54.7% | 4 | 105 | 30 | 1.22% | 169.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 09:15:00 | 221.50 | 215.76 | 215.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 12:15:00 | 222.29 | 220.27 | 219.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 14:15:00 | 221.58 | 222.19 | 221.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 14:15:00 | 221.58 | 222.19 | 221.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 221.58 | 222.19 | 221.18 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 223.08 | 223.71 | 223.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 14:15:00 | 221.12 | 223.19 | 223.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 225.52 | 223.44 | 223.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 225.52 | 223.44 | 223.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 225.52 | 223.44 | 223.55 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 224.72 | 223.70 | 223.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 234.20 | 226.23 | 224.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 10:15:00 | 231.12 | 231.12 | 228.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 12:15:00 | 233.26 | 233.76 | 232.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 233.26 | 233.76 | 232.41 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 237.49 | 240.72 | 241.11 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 15:15:00 | 242.56 | 241.41 | 241.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 245.46 | 242.22 | 241.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 244.64 | 244.65 | 243.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 240.57 | 243.82 | 243.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 240.57 | 243.82 | 243.27 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 239.66 | 242.36 | 242.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 15:15:00 | 238.80 | 241.65 | 242.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 10:15:00 | 234.56 | 234.31 | 237.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 235.22 | 233.07 | 233.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 235.22 | 233.07 | 233.92 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 11:15:00 | 234.58 | 233.86 | 233.83 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 13:15:00 | 233.50 | 233.80 | 233.81 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 14:15:00 | 234.32 | 233.91 | 233.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 10:15:00 | 235.44 | 234.32 | 234.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 13:15:00 | 234.07 | 234.44 | 234.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 13:15:00 | 234.07 | 234.44 | 234.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 234.07 | 234.44 | 234.20 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 234.00 | 234.11 | 234.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 13:15:00 | 233.20 | 233.89 | 234.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 15:15:00 | 234.49 | 234.00 | 234.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 15:15:00 | 234.49 | 234.00 | 234.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 234.49 | 234.00 | 234.04 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 233.10 | 231.19 | 231.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 235.77 | 232.67 | 231.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 10:15:00 | 233.82 | 234.83 | 233.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 10:15:00 | 233.82 | 234.83 | 233.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 233.82 | 234.83 | 233.71 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 13:15:00 | 232.49 | 233.40 | 233.47 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 15:15:00 | 234.95 | 233.70 | 233.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 235.49 | 234.06 | 233.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 233.62 | 234.68 | 234.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 233.62 | 234.68 | 234.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 233.62 | 234.68 | 234.37 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 233.62 | 234.20 | 234.21 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 15:15:00 | 237.35 | 234.67 | 234.40 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 14:15:00 | 235.62 | 235.95 | 235.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 15:15:00 | 235.00 | 235.76 | 235.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 236.19 | 235.85 | 235.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 236.19 | 235.85 | 235.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 236.19 | 235.85 | 235.91 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 09:15:00 | 253.61 | 239.20 | 237.34 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 245.47 | 249.93 | 250.11 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 251.00 | 249.30 | 249.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 255.45 | 251.00 | 250.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 252.30 | 253.30 | 252.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 252.30 | 253.30 | 252.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 252.30 | 253.30 | 252.03 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 14:15:00 | 250.35 | 251.43 | 251.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 15:15:00 | 250.00 | 251.15 | 251.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 251.21 | 251.16 | 251.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 251.21 | 251.16 | 251.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 251.21 | 251.16 | 251.32 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 254.30 | 251.58 | 251.34 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 247.73 | 250.75 | 251.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 246.94 | 249.72 | 250.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 250.82 | 247.71 | 248.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 250.82 | 247.71 | 248.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 250.82 | 247.71 | 248.67 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 13:15:00 | 250.45 | 249.42 | 249.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 15:15:00 | 251.60 | 250.07 | 249.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 12:15:00 | 251.00 | 251.14 | 250.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 12:15:00 | 251.00 | 251.14 | 250.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 251.00 | 251.14 | 250.38 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 09:15:00 | 245.86 | 250.86 | 251.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 09:15:00 | 243.34 | 246.19 | 248.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 09:15:00 | 243.82 | 243.48 | 245.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 12:15:00 | 243.64 | 243.86 | 245.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 243.64 | 243.86 | 245.23 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 10:15:00 | 243.56 | 242.58 | 242.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 243.96 | 242.85 | 242.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 15:15:00 | 243.00 | 243.02 | 242.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 15:15:00 | 243.00 | 243.02 | 242.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 243.00 | 243.02 | 242.77 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 246.35 | 249.17 | 249.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 245.80 | 248.50 | 248.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 246.54 | 246.54 | 247.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 247.44 | 246.68 | 247.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 247.44 | 246.68 | 247.26 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 249.20 | 247.73 | 247.61 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 14:15:00 | 247.07 | 247.57 | 247.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 15:15:00 | 246.50 | 247.35 | 247.49 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 09:15:00 | 253.52 | 248.59 | 248.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 10:15:00 | 262.20 | 251.31 | 249.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 266.77 | 270.26 | 264.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 266.77 | 270.26 | 264.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 266.77 | 270.26 | 264.93 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 13:15:00 | 320.00 | 324.63 | 324.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 14:15:00 | 315.28 | 322.76 | 323.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 09:15:00 | 332.48 | 323.68 | 324.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 332.48 | 323.68 | 324.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 332.48 | 323.68 | 324.00 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 10:15:00 | 329.80 | 324.91 | 324.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 14:15:00 | 334.70 | 327.82 | 326.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 15:15:00 | 337.86 | 338.17 | 333.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 15:15:00 | 337.86 | 338.17 | 333.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 337.86 | 338.17 | 333.77 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 321.70 | 331.63 | 332.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 10:15:00 | 314.17 | 318.60 | 320.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 331.30 | 317.51 | 318.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 331.30 | 317.51 | 318.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 331.30 | 317.51 | 318.57 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 10:15:00 | 330.50 | 320.11 | 319.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 334.70 | 328.16 | 324.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 09:15:00 | 326.10 | 328.89 | 326.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 09:15:00 | 326.10 | 328.89 | 326.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 326.10 | 328.89 | 326.80 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 318.35 | 326.92 | 327.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 317.63 | 322.70 | 325.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 304.90 | 301.27 | 306.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 304.90 | 301.27 | 306.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 304.90 | 301.27 | 306.84 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 311.74 | 308.78 | 308.76 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 14:15:00 | 306.55 | 309.09 | 309.13 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 313.54 | 309.81 | 309.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 12:15:00 | 317.90 | 314.26 | 312.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 13:15:00 | 313.44 | 314.10 | 312.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 14:15:00 | 313.24 | 313.93 | 312.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 313.24 | 313.93 | 312.46 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-11-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 14:15:00 | 319.79 | 323.63 | 323.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 10:15:00 | 318.76 | 321.71 | 322.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 15:15:00 | 320.50 | 320.21 | 321.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 322.00 | 320.57 | 321.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 322.00 | 320.57 | 321.61 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 328.11 | 323.08 | 322.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 10:15:00 | 347.60 | 331.37 | 327.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 12:15:00 | 432.77 | 435.96 | 410.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 423.90 | 430.74 | 415.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 423.90 | 430.74 | 415.92 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 13:15:00 | 410.70 | 414.82 | 415.18 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 428.44 | 417.26 | 416.16 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 409.56 | 416.40 | 416.81 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 420.05 | 411.10 | 410.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 423.85 | 413.65 | 412.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 12:15:00 | 417.60 | 417.65 | 415.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 14:15:00 | 414.40 | 416.95 | 415.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 414.40 | 416.95 | 415.16 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 10:15:00 | 411.47 | 414.81 | 414.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 13:15:00 | 410.10 | 412.93 | 413.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 11:15:00 | 411.40 | 411.00 | 412.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 11:15:00 | 411.40 | 411.00 | 412.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 411.40 | 411.00 | 412.45 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 431.97 | 416.52 | 414.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 437.20 | 428.53 | 426.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 13:15:00 | 428.43 | 428.77 | 426.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 15:15:00 | 427.30 | 428.25 | 426.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 15:15:00 | 427.30 | 428.25 | 426.88 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 411.00 | 428.27 | 430.29 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 15:15:00 | 421.46 | 420.77 | 420.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 427.06 | 422.03 | 421.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 13:15:00 | 423.50 | 423.54 | 422.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 424.43 | 425.02 | 424.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 424.43 | 425.02 | 424.02 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 419.52 | 425.23 | 425.81 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 428.12 | 425.04 | 424.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 14:15:00 | 429.28 | 427.30 | 426.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 428.92 | 429.08 | 428.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 14:15:00 | 428.92 | 429.08 | 428.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 428.92 | 429.08 | 428.05 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 12:15:00 | 428.25 | 429.22 | 429.29 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 432.40 | 429.58 | 429.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 10:15:00 | 448.00 | 433.26 | 431.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 14:15:00 | 444.38 | 444.52 | 440.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 15:15:00 | 441.90 | 444.00 | 440.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 441.90 | 444.00 | 440.80 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 15:15:00 | 436.40 | 439.20 | 439.49 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 440.85 | 439.79 | 439.73 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 433.01 | 438.39 | 439.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 429.89 | 434.49 | 436.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 09:15:00 | 432.96 | 432.48 | 434.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 432.96 | 432.48 | 434.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 432.96 | 432.48 | 434.59 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 438.96 | 436.34 | 436.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 15:15:00 | 440.00 | 437.33 | 436.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 448.99 | 455.46 | 450.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 14:15:00 | 448.99 | 455.46 | 450.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 448.99 | 455.46 | 450.60 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 443.83 | 449.32 | 449.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 440.04 | 447.47 | 448.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 446.42 | 446.02 | 447.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 446.42 | 446.02 | 447.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 446.42 | 446.02 | 447.66 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 453.26 | 448.69 | 448.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 460.89 | 451.13 | 449.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 543.50 | 551.92 | 520.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 15:15:00 | 566.67 | 572.01 | 565.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 566.67 | 572.01 | 565.01 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 09:15:00 | 555.01 | 563.25 | 564.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 549.51 | 557.66 | 559.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 15:15:00 | 555.00 | 554.47 | 557.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 549.29 | 553.44 | 556.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 549.29 | 553.44 | 556.43 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 543.77 | 534.89 | 534.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 548.35 | 537.58 | 535.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 571.51 | 571.74 | 563.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 10:15:00 | 708.30 | 712.70 | 701.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 708.30 | 712.70 | 701.76 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 10:15:00 | 687.00 | 697.53 | 698.23 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 707.61 | 699.62 | 698.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 733.40 | 706.37 | 702.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 926.90 | 955.52 | 926.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 880.56 | 923.63 | 922.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 880.56 | 923.63 | 922.38 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 880.56 | 915.02 | 918.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 836.53 | 880.79 | 898.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 679.67 | 661.11 | 684.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 11:15:00 | 634.51 | 658.75 | 679.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 634.51 | 658.75 | 679.42 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 10:15:00 | 624.40 | 602.59 | 602.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 09:15:00 | 655.61 | 624.92 | 614.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 728.00 | 734.30 | 716.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 728.00 | 734.30 | 716.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 728.00 | 734.30 | 716.00 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 12:15:00 | 709.90 | 717.90 | 718.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 09:15:00 | 699.10 | 710.25 | 714.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 699.80 | 695.14 | 702.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-12 09:15:00 | 702.60 | 693.23 | 698.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 707.90 | 696.17 | 699.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 707.50 | 696.17 | 699.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 702.50 | 697.43 | 699.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:30:00 | 699.00 | 698.40 | 699.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 14:30:00 | 700.80 | 697.58 | 699.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 664.05 | 694.45 | 697.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 665.76 | 694.45 | 697.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 696.00 | 687.90 | 691.58 | SL hit (close>ema200) qty=0.50 sl=687.90 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 696.00 | 691.42 | 691.22 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 679.00 | 689.31 | 690.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 13:15:00 | 677.40 | 681.10 | 682.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 15:15:00 | 676.70 | 672.14 | 676.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 15:15:00 | 676.70 | 672.14 | 676.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 676.70 | 672.14 | 676.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 15:00:00 | 664.85 | 667.32 | 671.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 09:30:00 | 659.00 | 666.71 | 670.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 11:30:00 | 664.90 | 666.10 | 669.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 12:15:00 | 663.60 | 666.10 | 669.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 663.10 | 664.27 | 667.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:15:00 | 658.39 | 663.85 | 665.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 631.61 | 649.72 | 654.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 631.65 | 649.72 | 654.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 626.05 | 636.11 | 644.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 630.42 | 636.11 | 644.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 625.47 | 636.11 | 644.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 654.79 | 632.01 | 636.75 | SL hit (close>ema200) qty=0.50 sl=632.01 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 654.79 | 640.21 | 639.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-09 09:15:00 | 687.53 | 656.56 | 648.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-10 09:15:00 | 672.50 | 677.98 | 666.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 672.50 | 677.98 | 666.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 672.50 | 677.98 | 666.09 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 657.50 | 666.89 | 667.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 14:15:00 | 650.60 | 659.76 | 663.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 663.10 | 658.91 | 662.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 663.10 | 658.91 | 662.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 663.10 | 658.91 | 662.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 665.00 | 658.91 | 662.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 663.40 | 659.81 | 662.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 662.00 | 659.81 | 662.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 665.00 | 660.85 | 662.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:45:00 | 662.60 | 660.85 | 662.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 664.00 | 661.48 | 662.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:30:00 | 666.50 | 661.48 | 662.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 660.00 | 661.18 | 662.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 10:30:00 | 656.50 | 660.37 | 661.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 11:15:00 | 655.80 | 660.37 | 661.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:45:00 | 655.07 | 658.55 | 659.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 655.00 | 658.47 | 659.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 662.00 | 659.18 | 659.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:00:00 | 662.00 | 659.18 | 659.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 659.50 | 659.24 | 659.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 661.80 | 659.78 | 659.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 661.80 | 659.78 | 659.76 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 11:15:00 | 656.50 | 659.24 | 659.53 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 12:15:00 | 662.50 | 659.89 | 659.80 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 658.61 | 659.64 | 659.69 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 662.00 | 658.93 | 658.72 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 656.00 | 658.35 | 658.47 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 663.00 | 658.66 | 658.40 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 656.01 | 659.07 | 659.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 652.54 | 656.82 | 658.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 653.40 | 652.38 | 654.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:30:00 | 653.90 | 652.38 | 654.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 654.00 | 652.71 | 654.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 654.50 | 652.71 | 654.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 652.50 | 652.66 | 653.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 649.05 | 652.66 | 653.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 650.62 | 646.08 | 646.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 616.60 | 641.10 | 643.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 618.09 | 641.10 | 643.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 12:15:00 | 616.32 | 615.40 | 624.41 | SL hit (close>ema200) qty=0.50 sl=615.40 alert=retest2 |

### Cycle 77 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 640.00 | 628.98 | 627.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 648.17 | 643.61 | 638.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 645.90 | 645.92 | 641.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 645.90 | 645.92 | 641.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 641.75 | 645.16 | 642.30 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 641.47 | 641.67 | 641.70 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 13:15:00 | 644.22 | 642.01 | 641.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 645.75 | 642.75 | 642.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 678.97 | 684.90 | 673.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 678.97 | 684.90 | 673.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 678.97 | 684.90 | 673.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 678.01 | 684.90 | 673.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 679.17 | 683.75 | 674.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 674.53 | 683.75 | 674.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 675.00 | 680.86 | 675.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 14:00:00 | 675.00 | 680.86 | 675.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 673.00 | 679.29 | 675.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 673.00 | 679.29 | 675.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 677.50 | 678.24 | 675.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:15:00 | 676.91 | 678.24 | 675.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 678.00 | 678.19 | 675.55 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 668.00 | 673.63 | 674.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 666.52 | 671.55 | 672.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 679.40 | 668.46 | 670.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 679.40 | 668.46 | 670.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 679.40 | 668.46 | 670.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 678.00 | 668.46 | 670.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 686.27 | 672.02 | 671.53 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 665.10 | 674.41 | 675.16 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 682.50 | 676.03 | 675.82 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 671.40 | 675.10 | 675.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 667.92 | 673.67 | 674.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 656.50 | 652.25 | 657.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 656.50 | 652.25 | 657.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 656.50 | 652.25 | 657.48 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 661.83 | 659.17 | 658.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 667.00 | 660.95 | 659.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 662.10 | 665.15 | 662.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 662.10 | 665.15 | 662.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 662.10 | 665.15 | 662.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 662.10 | 665.15 | 662.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 659.57 | 664.03 | 662.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 659.01 | 664.03 | 662.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 659.90 | 663.20 | 662.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 662.38 | 663.20 | 662.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:00:00 | 661.31 | 662.50 | 662.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 13:15:00 | 660.44 | 661.87 | 661.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 13:15:00 | 660.44 | 661.87 | 661.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 653.31 | 659.59 | 660.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 656.90 | 656.54 | 658.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 14:15:00 | 656.90 | 656.54 | 658.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 656.90 | 656.54 | 658.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 656.90 | 656.54 | 658.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 652.50 | 655.86 | 657.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:45:00 | 650.74 | 654.06 | 656.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 14:45:00 | 650.69 | 652.86 | 655.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 10:15:00 | 650.82 | 652.31 | 654.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:00:00 | 650.90 | 652.03 | 654.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 645.00 | 650.16 | 652.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 644.50 | 650.16 | 652.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 13:00:00 | 644.00 | 647.83 | 650.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 644.86 | 646.53 | 649.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:00:00 | 644.51 | 646.13 | 648.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 662.46 | 646.49 | 646.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 662.46 | 646.49 | 646.58 | SL hit (close>static) qty=1.00 sl=654.90 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 653.03 | 647.80 | 647.17 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 642.45 | 648.88 | 649.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 633.01 | 641.68 | 644.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 628.90 | 623.44 | 627.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 628.90 | 623.44 | 627.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 628.90 | 623.44 | 627.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 628.90 | 623.44 | 627.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 643.00 | 627.35 | 628.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 642.51 | 627.35 | 628.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 644.31 | 630.74 | 630.11 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 630.30 | 634.08 | 634.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 13:15:00 | 630.04 | 631.71 | 632.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 14:15:00 | 637.00 | 632.77 | 633.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 14:15:00 | 637.00 | 632.77 | 633.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 637.00 | 632.77 | 633.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 637.00 | 632.77 | 633.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 15:15:00 | 636.10 | 633.43 | 633.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 639.30 | 634.61 | 633.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 634.30 | 634.58 | 634.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 11:15:00 | 634.30 | 634.58 | 634.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 634.30 | 634.58 | 634.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 634.38 | 634.58 | 634.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 636.97 | 635.05 | 634.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:30:00 | 637.10 | 635.05 | 634.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 634.20 | 634.88 | 634.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 634.20 | 634.88 | 634.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 634.46 | 634.80 | 634.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 634.46 | 634.80 | 634.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 636.00 | 635.04 | 634.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 631.36 | 635.04 | 634.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 632.50 | 634.53 | 634.29 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 632.18 | 634.06 | 634.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 630.80 | 633.41 | 633.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 606.90 | 606.02 | 614.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 606.90 | 606.02 | 614.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 606.90 | 606.02 | 614.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 13:15:00 | 599.99 | 602.19 | 603.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:45:00 | 599.96 | 601.32 | 603.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 13:45:00 | 599.61 | 601.49 | 602.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 606.90 | 599.51 | 598.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 606.90 | 599.51 | 598.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 617.93 | 605.40 | 602.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 14:15:00 | 626.50 | 626.51 | 620.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:45:00 | 626.39 | 626.51 | 620.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 623.26 | 625.63 | 623.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 625.33 | 625.63 | 623.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 617.80 | 623.96 | 622.92 | SL hit (close<static) qty=1.00 sl=622.60 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 614.65 | 622.10 | 622.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 613.30 | 616.59 | 618.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 15:15:00 | 620.01 | 616.82 | 618.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 15:15:00 | 620.01 | 616.82 | 618.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 620.01 | 616.82 | 618.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 627.68 | 616.82 | 618.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 664.60 | 626.38 | 622.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 12:15:00 | 720.00 | 657.23 | 638.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 716.40 | 737.22 | 703.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 15:00:00 | 716.40 | 737.22 | 703.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 700.35 | 723.22 | 705.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 694.57 | 723.22 | 705.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 715.69 | 721.71 | 706.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:45:00 | 758.00 | 726.18 | 713.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 737.69 | 731.57 | 718.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 13:45:00 | 737.86 | 733.21 | 721.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 740.00 | 731.28 | 722.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 728.42 | 729.98 | 723.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 717.84 | 721.18 | 721.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 717.84 | 721.18 | 721.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 714.73 | 719.06 | 720.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 733.00 | 719.57 | 720.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 733.00 | 719.57 | 720.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 733.00 | 719.57 | 720.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 733.00 | 719.57 | 720.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 723.44 | 720.34 | 720.37 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 725.16 | 721.31 | 720.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 729.10 | 722.88 | 721.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 730.00 | 730.05 | 726.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:30:00 | 730.26 | 730.05 | 726.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 725.50 | 729.14 | 726.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 732.26 | 729.14 | 726.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 715.04 | 726.32 | 725.70 | SL hit (close<static) qty=1.00 sl=723.60 alert=retest2 |

### Cycle 98 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 718.84 | 724.83 | 725.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 702.00 | 715.71 | 720.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 711.76 | 709.10 | 714.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 711.76 | 709.10 | 714.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 711.76 | 709.10 | 714.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 723.61 | 709.10 | 714.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 707.56 | 706.11 | 711.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 702.20 | 705.15 | 707.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:00:00 | 698.66 | 705.15 | 707.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 701.90 | 700.55 | 701.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:00:00 | 702.00 | 702.13 | 702.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 702.10 | 702.12 | 702.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 701.50 | 702.12 | 702.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 696.68 | 701.03 | 701.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 10:15:00 | 695.51 | 701.03 | 701.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 667.09 | 675.89 | 682.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 663.73 | 675.89 | 682.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 666.80 | 675.89 | 682.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 666.90 | 675.89 | 682.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 660.73 | 673.12 | 680.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 671.74 | 669.69 | 675.83 | SL hit (close>ema200) qty=0.50 sl=669.69 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 709.92 | 680.58 | 677.97 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 689.33 | 691.37 | 691.43 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 702.20 | 692.90 | 692.05 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 680.27 | 691.13 | 691.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 677.90 | 681.97 | 685.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 681.70 | 681.44 | 684.08 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 14:15:00 | 680.36 | 681.44 | 684.08 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 15:15:00 | 680.00 | 681.55 | 683.89 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 680.00 | 681.24 | 683.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 675.06 | 681.24 | 683.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 646.34 | 661.00 | 668.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 646.00 | 661.00 | 668.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 641.31 | 661.00 | 668.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 652.94 | 650.24 | 657.93 | SL hit (close>ema200) qty=0.50 sl=650.24 alert=retest1 |

### Cycle 103 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 664.70 | 658.72 | 658.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 720.44 | 671.47 | 664.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 691.39 | 697.14 | 682.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 691.39 | 697.14 | 682.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 693.50 | 703.28 | 695.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 692.70 | 703.28 | 695.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 692.62 | 701.15 | 695.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:45:00 | 695.25 | 697.77 | 695.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:15:00 | 695.00 | 697.04 | 694.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 691.75 | 695.65 | 694.67 | SL hit (close<static) qty=1.00 sl=692.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 692.50 | 697.46 | 698.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 690.42 | 694.97 | 696.78 | Break + close below crossover candle low |

### Cycle 105 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 733.48 | 693.27 | 691.88 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 686.02 | 698.55 | 698.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 680.30 | 691.21 | 694.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 689.20 | 688.85 | 693.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 689.20 | 688.85 | 693.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 687.06 | 688.49 | 692.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 682.87 | 686.88 | 690.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 648.73 | 666.76 | 676.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 659.09 | 657.11 | 665.46 | SL hit (close>ema200) qty=0.50 sl=657.11 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 675.70 | 662.51 | 661.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 682.20 | 674.27 | 669.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 668.07 | 681.50 | 677.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 668.07 | 681.50 | 677.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 668.07 | 681.50 | 677.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 669.99 | 681.50 | 677.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 667.72 | 678.74 | 676.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 667.72 | 678.74 | 676.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 671.00 | 674.62 | 674.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 667.80 | 673.26 | 674.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 668.53 | 667.30 | 670.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:45:00 | 668.01 | 667.30 | 670.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 672.55 | 668.79 | 670.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 669.64 | 668.79 | 670.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 673.17 | 671.22 | 671.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 673.17 | 671.22 | 671.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 676.18 | 672.21 | 671.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 672.00 | 673.72 | 672.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 12:15:00 | 672.00 | 673.72 | 672.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 672.00 | 673.72 | 672.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:45:00 | 672.52 | 673.72 | 672.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 670.16 | 673.01 | 672.49 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 667.64 | 671.30 | 671.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 664.88 | 670.02 | 671.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 667.34 | 667.16 | 669.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 667.34 | 667.16 | 669.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 684.72 | 670.67 | 670.76 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 678.10 | 672.16 | 671.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 09:15:00 | 704.27 | 679.08 | 674.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 14:15:00 | 686.99 | 687.16 | 681.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 15:00:00 | 686.99 | 687.16 | 681.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 682.00 | 685.94 | 681.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 682.00 | 685.94 | 681.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 685.64 | 685.88 | 681.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 693.65 | 685.88 | 681.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 13:00:00 | 687.70 | 686.62 | 683.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 689.20 | 686.29 | 683.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 670.04 | 683.73 | 682.87 | SL hit (close<static) qty=1.00 sl=680.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 669.91 | 680.97 | 681.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 661.59 | 671.56 | 676.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 677.53 | 665.99 | 669.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 677.53 | 665.99 | 669.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 677.53 | 665.99 | 669.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 677.53 | 665.99 | 669.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 673.63 | 667.52 | 669.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:15:00 | 671.00 | 669.71 | 670.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 670.09 | 669.79 | 670.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 667.56 | 670.29 | 670.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:45:00 | 671.08 | 669.98 | 670.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 664.77 | 668.94 | 669.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:45:00 | 667.80 | 668.94 | 669.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 663.39 | 666.90 | 668.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 666.58 | 666.90 | 668.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 657.93 | 653.89 | 657.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 666.78 | 659.97 | 659.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 666.78 | 659.97 | 659.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 15:15:00 | 668.50 | 661.67 | 660.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 666.01 | 666.33 | 664.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:45:00 | 665.73 | 666.33 | 664.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 670.90 | 668.66 | 666.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 668.61 | 668.66 | 666.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 668.04 | 668.52 | 666.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 668.04 | 668.52 | 666.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 667.82 | 668.38 | 667.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 667.82 | 668.38 | 667.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 665.01 | 667.70 | 666.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 668.20 | 667.70 | 666.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 668.95 | 667.32 | 666.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 12:15:00 | 664.38 | 666.51 | 666.51 | SL hit (close<static) qty=1.00 sl=664.98 alert=retest2 |

### Cycle 114 — SELL (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 13:15:00 | 662.31 | 665.67 | 666.12 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 672.58 | 667.33 | 666.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 696.75 | 674.83 | 670.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 685.80 | 685.87 | 680.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:45:00 | 684.32 | 685.87 | 680.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 689.62 | 686.77 | 682.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 701.00 | 688.80 | 685.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:30:00 | 697.27 | 690.87 | 688.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 686.57 | 688.38 | 688.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 686.57 | 688.38 | 688.48 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 12:15:00 | 692.23 | 689.02 | 688.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 713.50 | 695.42 | 691.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 14:15:00 | 698.00 | 699.71 | 695.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 15:00:00 | 698.00 | 699.71 | 695.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 694.38 | 698.53 | 695.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 694.38 | 698.53 | 695.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 693.83 | 697.59 | 695.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 697.73 | 697.59 | 695.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 690.60 | 696.19 | 695.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 690.60 | 696.19 | 695.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 690.84 | 694.42 | 694.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 687.71 | 692.61 | 693.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 689.00 | 686.43 | 689.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 689.00 | 686.43 | 689.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 689.00 | 686.43 | 689.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 689.00 | 686.43 | 689.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 688.35 | 686.82 | 688.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 689.32 | 686.82 | 688.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 686.05 | 686.67 | 688.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:45:00 | 685.56 | 686.54 | 688.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 685.89 | 685.45 | 687.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 685.65 | 685.45 | 687.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:15:00 | 679.83 | 686.40 | 687.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 677.30 | 684.58 | 686.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 671.00 | 677.53 | 682.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:30:00 | 670.41 | 675.88 | 680.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:30:00 | 671.18 | 674.72 | 679.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:30:00 | 670.61 | 673.07 | 677.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 651.28 | 660.24 | 665.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 651.60 | 660.24 | 665.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 651.37 | 660.24 | 665.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 720.98 | 667.44 | 664.74 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 720.98 | 667.44 | 664.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 734.18 | 680.79 | 671.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 14:15:00 | 678.98 | 689.68 | 679.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 14:15:00 | 678.98 | 689.68 | 679.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 678.98 | 689.68 | 679.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 678.98 | 689.68 | 679.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 680.20 | 687.78 | 679.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 692.00 | 687.78 | 679.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 679.75 | 684.15 | 684.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 679.75 | 684.15 | 684.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 676.21 | 681.56 | 683.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 09:15:00 | 683.84 | 679.95 | 681.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 683.84 | 679.95 | 681.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 683.84 | 679.95 | 681.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 700.32 | 679.95 | 681.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 686.31 | 681.22 | 682.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 686.51 | 681.22 | 682.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 683.82 | 682.61 | 682.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 689.12 | 684.49 | 683.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 13:15:00 | 685.50 | 685.56 | 684.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 14:00:00 | 685.50 | 685.56 | 684.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 682.60 | 685.04 | 684.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 686.72 | 685.04 | 684.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 686.67 | 685.85 | 684.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:15:00 | 686.90 | 685.85 | 684.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 15:00:00 | 689.60 | 687.91 | 687.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 687.02 | 687.73 | 687.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 683.00 | 687.73 | 687.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 682.56 | 686.70 | 686.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 684.09 | 686.70 | 686.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 677.60 | 684.88 | 685.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 677.60 | 684.88 | 685.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 672.29 | 680.12 | 683.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 662.50 | 662.30 | 667.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:15:00 | 662.54 | 662.30 | 667.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 622.95 | 619.91 | 623.94 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 629.55 | 624.80 | 624.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 15:15:00 | 633.00 | 628.79 | 626.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 14:15:00 | 626.90 | 628.85 | 627.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 14:15:00 | 626.90 | 628.85 | 627.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 626.90 | 628.85 | 627.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 626.90 | 628.85 | 627.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 627.10 | 628.50 | 627.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 631.57 | 628.50 | 627.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 622.97 | 627.40 | 627.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 622.97 | 627.40 | 627.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 619.49 | 625.81 | 626.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 613.63 | 620.89 | 623.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 614.99 | 614.33 | 618.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 614.99 | 614.33 | 618.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 622.80 | 616.13 | 618.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 622.80 | 616.13 | 618.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 623.51 | 617.61 | 618.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 621.48 | 617.61 | 618.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 622.20 | 618.53 | 619.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 619.50 | 618.62 | 619.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:00:00 | 619.00 | 618.62 | 619.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 13:15:00 | 625.21 | 619.94 | 619.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 625.21 | 619.94 | 619.75 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 612.70 | 618.51 | 619.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 612.67 | 616.08 | 617.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 588.75 | 579.85 | 588.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 588.75 | 579.85 | 588.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 588.75 | 579.85 | 588.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 588.75 | 579.85 | 588.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 588.31 | 581.54 | 588.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 588.31 | 581.54 | 588.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 583.87 | 582.01 | 588.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 588.29 | 582.01 | 588.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 589.21 | 584.02 | 588.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:45:00 | 589.80 | 584.02 | 588.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 593.50 | 585.92 | 588.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 593.50 | 585.92 | 588.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 594.50 | 587.63 | 589.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 595.21 | 587.63 | 589.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 601.53 | 590.41 | 590.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 607.78 | 601.15 | 597.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 599.27 | 602.02 | 599.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 599.27 | 602.02 | 599.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 599.27 | 602.02 | 599.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 599.27 | 602.02 | 599.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 600.07 | 601.63 | 599.26 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 591.60 | 597.25 | 597.96 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 604.73 | 597.66 | 597.04 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 597.50 | 598.63 | 598.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 592.00 | 596.78 | 597.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 530.17 | 529.45 | 539.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 14:30:00 | 531.52 | 529.45 | 539.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 131 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 594.75 | 536.95 | 532.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 634.95 | 592.53 | 567.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 609.51 | 613.04 | 589.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 609.51 | 613.04 | 589.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 586.68 | 606.96 | 590.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 586.68 | 606.96 | 590.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 592.46 | 604.06 | 590.76 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 581.60 | 588.00 | 588.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 575.80 | 582.97 | 585.69 | Break + close below crossover candle low |

### Cycle 133 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 622.10 | 589.68 | 588.20 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 573.57 | 594.18 | 596.61 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 617.50 | 593.76 | 593.71 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 582.91 | 591.59 | 592.73 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 611.00 | 595.05 | 593.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 626.00 | 618.13 | 611.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 630.52 | 631.87 | 623.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:30:00 | 633.07 | 631.87 | 623.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 622.36 | 628.64 | 624.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 622.36 | 628.64 | 624.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 622.50 | 627.41 | 624.31 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 615.74 | 621.92 | 622.34 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 618.76 | 615.56 | 615.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 618.80 | 616.20 | 615.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 622.32 | 623.91 | 620.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 622.32 | 623.91 | 620.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 622.32 | 623.91 | 620.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 622.32 | 623.91 | 620.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 622.86 | 623.70 | 621.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 631.09 | 623.12 | 621.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 631.34 | 635.68 | 635.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 631.34 | 635.68 | 635.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 628.91 | 633.58 | 634.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 633.93 | 630.01 | 631.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 633.93 | 630.01 | 631.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 633.93 | 630.01 | 631.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 633.93 | 630.01 | 631.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 632.99 | 630.60 | 631.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 634.74 | 630.60 | 631.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 634.00 | 631.28 | 632.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:15:00 | 635.00 | 631.28 | 632.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 635.00 | 632.03 | 632.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 643.43 | 632.03 | 632.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 646.00 | 634.82 | 633.58 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 629.96 | 633.86 | 634.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 623.08 | 627.45 | 630.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 626.61 | 624.87 | 627.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 626.61 | 624.87 | 627.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 626.61 | 624.87 | 627.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:30:00 | 625.88 | 624.87 | 627.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 627.50 | 625.40 | 627.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 628.98 | 625.40 | 627.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 626.07 | 625.53 | 627.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 618.40 | 626.44 | 626.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 587.48 | 608.87 | 616.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 600.42 | 598.28 | 606.53 | SL hit (close>ema200) qty=0.50 sl=598.28 alert=retest2 |

### Cycle 143 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 619.15 | 609.26 | 608.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 623.55 | 616.72 | 612.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 623.20 | 625.25 | 622.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 623.20 | 625.25 | 622.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 626.70 | 625.54 | 622.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 628.50 | 626.24 | 623.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 629.05 | 626.24 | 623.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 15:15:00 | 628.30 | 630.45 | 630.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 15:15:00 | 628.30 | 630.45 | 630.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 09:15:00 | 623.10 | 628.98 | 629.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 627.75 | 627.01 | 628.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 627.75 | 627.01 | 628.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 627.75 | 627.01 | 628.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:30:00 | 626.00 | 626.89 | 627.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 14:00:00 | 625.95 | 626.89 | 627.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 624.10 | 627.06 | 627.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 594.70 | 600.25 | 606.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 594.65 | 600.25 | 606.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 592.89 | 600.25 | 606.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 595.95 | 594.35 | 598.72 | SL hit (close>ema200) qty=0.50 sl=594.35 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 594.90 | 592.20 | 592.01 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 588.75 | 591.48 | 591.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 584.80 | 590.14 | 591.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 598.80 | 584.67 | 586.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 598.80 | 584.67 | 586.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 598.80 | 584.67 | 586.19 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 601.90 | 588.12 | 587.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 603.00 | 597.59 | 593.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 603.00 | 603.01 | 597.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 603.95 | 603.01 | 597.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 621.00 | 627.17 | 624.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 621.00 | 627.17 | 624.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 619.00 | 625.54 | 624.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:15:00 | 618.20 | 625.54 | 624.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 620.50 | 623.03 | 623.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 615.75 | 621.58 | 622.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 618.25 | 617.50 | 619.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 618.25 | 617.50 | 619.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 618.25 | 617.50 | 619.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 618.25 | 617.50 | 619.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 618.10 | 617.62 | 619.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 619.45 | 617.62 | 619.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 621.00 | 618.30 | 619.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 623.50 | 618.30 | 619.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 628.80 | 620.40 | 620.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 635.35 | 627.65 | 624.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 626.80 | 629.18 | 626.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 626.80 | 629.18 | 626.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 626.80 | 629.18 | 626.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 626.80 | 629.18 | 626.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 629.00 | 629.15 | 626.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 629.40 | 628.74 | 627.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 13:00:00 | 629.05 | 628.80 | 627.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:00:00 | 631.20 | 629.23 | 627.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 09:15:00 | 692.34 | 655.66 | 645.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 661.35 | 665.61 | 666.16 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 668.25 | 666.44 | 666.35 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 663.60 | 665.92 | 666.14 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 686.65 | 669.62 | 667.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 13:15:00 | 718.00 | 699.32 | 690.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 715.50 | 717.03 | 706.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 715.15 | 717.03 | 706.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 706.90 | 714.93 | 707.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 704.00 | 714.93 | 707.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 704.20 | 712.79 | 707.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 704.20 | 712.79 | 707.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 700.40 | 710.31 | 706.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 700.40 | 710.31 | 706.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 687.50 | 703.87 | 704.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 684.80 | 695.48 | 700.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 651.80 | 648.44 | 655.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:30:00 | 651.30 | 648.44 | 655.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 658.30 | 650.41 | 655.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 658.30 | 650.41 | 655.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 655.00 | 651.33 | 655.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 656.75 | 651.33 | 655.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 653.80 | 651.82 | 655.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 655.85 | 651.82 | 655.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 655.30 | 652.52 | 655.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 656.00 | 652.52 | 655.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 661.65 | 654.34 | 655.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 661.65 | 654.34 | 655.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 660.65 | 655.61 | 656.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 663.45 | 655.61 | 656.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 659.35 | 656.99 | 656.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 665.10 | 659.47 | 658.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 672.10 | 674.25 | 668.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 672.10 | 674.25 | 668.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 679.90 | 685.06 | 684.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 679.90 | 685.06 | 684.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 680.60 | 684.17 | 683.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 678.10 | 684.17 | 683.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 682.70 | 683.46 | 683.50 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 687.50 | 683.66 | 683.51 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 680.60 | 683.21 | 683.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 675.60 | 680.93 | 682.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 677.00 | 676.77 | 679.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 677.00 | 676.77 | 679.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 677.00 | 676.77 | 679.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 671.10 | 675.86 | 678.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 678.90 | 660.63 | 659.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 678.90 | 660.63 | 659.20 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 664.60 | 672.18 | 672.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 662.30 | 668.05 | 670.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 678.45 | 668.30 | 669.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 678.45 | 668.30 | 669.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 678.45 | 668.30 | 669.89 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 673.50 | 670.58 | 670.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 677.85 | 672.57 | 671.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 13:15:00 | 673.00 | 674.13 | 672.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 673.00 | 674.13 | 672.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 673.00 | 674.13 | 672.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 670.50 | 674.13 | 672.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 672.75 | 673.85 | 672.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 680.50 | 673.68 | 672.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-05 09:15:00 | 748.55 | 704.70 | 692.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 689.10 | 703.59 | 704.67 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 690.10 | 689.72 | 689.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 695.10 | 690.80 | 690.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 695.25 | 695.35 | 693.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 695.25 | 695.35 | 693.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 695.70 | 695.54 | 694.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 692.95 | 695.54 | 694.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 693.15 | 695.06 | 694.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 693.10 | 695.06 | 694.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 693.40 | 694.73 | 694.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:30:00 | 692.50 | 694.73 | 694.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 693.90 | 694.33 | 694.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 693.90 | 694.33 | 694.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 691.75 | 693.82 | 693.89 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 699.40 | 694.64 | 694.23 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 687.55 | 694.58 | 695.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 687.40 | 692.09 | 694.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 686.90 | 685.04 | 688.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 12:15:00 | 686.90 | 685.04 | 688.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 686.90 | 685.04 | 688.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 686.75 | 685.04 | 688.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 686.15 | 684.20 | 686.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:30:00 | 684.40 | 684.84 | 686.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:00:00 | 685.40 | 684.43 | 685.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:30:00 | 685.85 | 684.66 | 685.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 685.60 | 684.66 | 685.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 685.00 | 684.73 | 685.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 683.50 | 684.73 | 685.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 683.80 | 683.93 | 684.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 682.60 | 683.68 | 684.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 675.00 | 674.14 | 674.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 675.00 | 674.14 | 674.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 678.50 | 675.01 | 674.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 684.70 | 686.20 | 682.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 684.70 | 686.20 | 682.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 684.00 | 685.76 | 682.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 684.00 | 685.76 | 682.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 723.75 | 731.60 | 724.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 721.05 | 731.60 | 724.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 725.00 | 730.28 | 724.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 743.00 | 730.21 | 725.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-23 14:15:00 | 817.30 | 764.51 | 745.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 994.00 | 1009.93 | 1010.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 954.90 | 996.44 | 1004.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 913.50 | 909.25 | 925.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 10:15:00 | 940.50 | 909.25 | 925.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 916.00 | 910.60 | 925.08 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 12:15:00 | 938.10 | 928.15 | 927.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 990.60 | 940.64 | 932.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 980.00 | 1008.87 | 986.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 980.00 | 1008.87 | 986.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 980.00 | 1008.87 | 986.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 981.00 | 1008.87 | 986.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 979.00 | 1002.90 | 985.59 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 943.00 | 976.92 | 977.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 915.00 | 958.31 | 968.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 865.00 | 842.61 | 872.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:00:00 | 865.00 | 842.61 | 872.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 873.00 | 848.69 | 872.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 843.00 | 848.69 | 872.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 831.00 | 845.15 | 868.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 819.00 | 828.09 | 832.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 778.05 | 784.47 | 792.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 778.10 | 770.49 | 778.95 | SL hit (close>ema200) qty=0.50 sl=770.49 alert=retest2 |

### Cycle 171 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 783.70 | 777.55 | 777.25 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 772.15 | 778.60 | 778.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 761.80 | 773.54 | 775.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 762.70 | 762.12 | 766.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 762.70 | 762.12 | 766.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 759.65 | 762.14 | 765.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 757.75 | 761.47 | 764.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 785.00 | 741.11 | 742.35 | SL hit (close>static) qty=1.00 sl=766.85 alert=retest2 |

### Cycle 173 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 774.00 | 747.69 | 745.23 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 748.60 | 754.09 | 754.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 746.10 | 750.63 | 752.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 752.20 | 750.03 | 751.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 752.20 | 750.03 | 751.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 752.20 | 750.03 | 751.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 756.20 | 750.03 | 751.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 750.80 | 750.19 | 751.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 748.50 | 750.19 | 751.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 15:15:00 | 711.07 | 719.70 | 726.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 742.00 | 716.95 | 719.67 | SL hit (close>ema200) qty=0.50 sl=716.95 alert=retest2 |

### Cycle 175 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 739.70 | 721.50 | 721.49 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 709.85 | 720.31 | 721.62 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 724.40 | 717.82 | 717.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 728.15 | 721.95 | 720.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 716.90 | 720.94 | 720.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 716.90 | 720.94 | 720.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 716.90 | 720.94 | 720.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 718.25 | 720.94 | 720.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 718.10 | 720.37 | 719.97 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 717.25 | 719.21 | 719.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 712.10 | 717.79 | 718.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 713.00 | 710.47 | 713.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 713.00 | 710.47 | 713.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 707.50 | 704.61 | 707.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 707.85 | 704.61 | 707.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 705.05 | 704.70 | 707.42 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 718.45 | 709.62 | 708.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 725.30 | 714.25 | 711.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 717.45 | 719.04 | 715.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 717.45 | 719.04 | 715.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 716.30 | 717.77 | 715.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 715.50 | 717.77 | 715.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 715.60 | 717.34 | 715.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 718.60 | 717.34 | 715.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 716.55 | 717.09 | 715.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 716.30 | 716.52 | 715.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 710.80 | 714.79 | 715.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 710.80 | 714.79 | 715.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 708.65 | 713.56 | 714.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 709.60 | 708.97 | 711.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 709.60 | 708.97 | 711.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 709.60 | 708.97 | 711.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 711.95 | 708.97 | 711.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 696.25 | 694.74 | 699.22 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 701.65 | 699.67 | 699.50 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 696.70 | 699.17 | 699.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 694.65 | 697.91 | 698.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 695.90 | 693.88 | 695.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 695.90 | 693.88 | 695.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 695.90 | 693.88 | 695.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:15:00 | 696.80 | 693.88 | 695.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 695.45 | 694.19 | 695.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 694.75 | 694.67 | 695.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:15:00 | 694.80 | 695.03 | 695.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 13:15:00 | 697.85 | 695.60 | 696.06 | SL hit (close>static) qty=1.00 sl=697.00 alert=retest2 |

### Cycle 183 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 700.00 | 696.96 | 696.63 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 691.35 | 695.84 | 696.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 689.90 | 694.65 | 695.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 666.65 | 666.56 | 674.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 666.65 | 666.56 | 674.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 674.40 | 668.29 | 673.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 665.90 | 668.82 | 671.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 666.25 | 667.57 | 670.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 665.65 | 666.43 | 668.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 666.50 | 666.96 | 668.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 632.60 | 642.04 | 650.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 632.94 | 642.04 | 650.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 632.37 | 642.04 | 650.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 633.17 | 642.04 | 650.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 625.50 | 624.98 | 634.51 | SL hit (close>ema200) qty=0.50 sl=624.98 alert=retest2 |

### Cycle 185 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 622.40 | 614.78 | 614.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 627.70 | 617.36 | 615.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 607.50 | 615.39 | 614.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 607.50 | 615.39 | 614.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 607.50 | 615.39 | 614.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 607.50 | 615.39 | 614.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 605.55 | 613.42 | 614.01 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 625.00 | 615.75 | 614.62 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 608.85 | 616.80 | 617.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 604.90 | 613.00 | 615.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 610.65 | 602.54 | 607.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 610.65 | 602.54 | 607.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 610.65 | 602.54 | 607.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 622.40 | 602.54 | 607.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 620.00 | 606.03 | 608.52 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 627.35 | 610.29 | 610.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 634.70 | 619.97 | 615.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 624.25 | 631.51 | 626.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 624.25 | 631.51 | 626.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 624.25 | 631.51 | 626.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 624.25 | 631.51 | 626.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 621.10 | 629.43 | 625.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 621.10 | 629.43 | 625.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 622.50 | 628.04 | 625.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 622.00 | 628.04 | 625.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 613.75 | 622.69 | 623.62 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 641.05 | 624.88 | 623.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 654.55 | 630.82 | 626.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 661.25 | 663.30 | 653.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 658.15 | 663.30 | 653.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 650.50 | 657.91 | 655.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 649.90 | 657.91 | 655.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 650.25 | 656.38 | 654.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 649.55 | 656.38 | 654.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 648.85 | 653.53 | 653.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 647.10 | 651.85 | 652.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 632.00 | 631.47 | 636.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 630.30 | 631.47 | 636.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 636.55 | 632.48 | 636.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 636.55 | 632.48 | 636.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 629.40 | 631.87 | 635.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 633.65 | 631.87 | 635.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 634.90 | 632.47 | 635.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 636.50 | 632.47 | 635.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 635.95 | 633.17 | 635.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 636.50 | 633.17 | 635.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 636.50 | 633.84 | 635.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 638.60 | 633.84 | 635.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 639.05 | 634.88 | 636.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 639.05 | 634.88 | 636.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 638.10 | 635.52 | 636.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 655.35 | 635.52 | 636.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 642.00 | 637.58 | 637.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 644.00 | 640.90 | 639.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 682.80 | 685.73 | 668.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:15:00 | 735.30 | 685.73 | 668.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 695.50 | 703.69 | 690.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 689.85 | 700.92 | 690.51 | SL hit (close<ema400) qty=1.00 sl=690.51 alert=retest1 |

### Cycle 194 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 667.40 | 686.39 | 686.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 11:15:00 | 664.80 | 669.38 | 675.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 655.65 | 653.27 | 659.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 13:00:00 | 655.65 | 653.27 | 659.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 660.85 | 654.79 | 659.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 660.85 | 654.79 | 659.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 656.15 | 655.06 | 658.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:15:00 | 659.00 | 655.06 | 658.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 659.00 | 655.85 | 658.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 637.60 | 655.85 | 658.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 09:45:00 | 647.70 | 633.25 | 639.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 655.40 | 643.35 | 642.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 655.40 | 643.35 | 642.42 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 620.60 | 640.10 | 642.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 611.70 | 622.67 | 626.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 622.65 | 621.17 | 624.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:45:00 | 620.95 | 621.17 | 624.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 628.35 | 622.60 | 624.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 628.35 | 622.60 | 624.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 632.00 | 624.48 | 625.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 633.40 | 624.48 | 625.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 09:15:00 | 634.75 | 626.54 | 626.34 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 620.85 | 625.89 | 626.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 612.85 | 621.55 | 624.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 622.65 | 618.98 | 621.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 622.65 | 618.98 | 621.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 622.65 | 618.98 | 621.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 623.00 | 618.98 | 621.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 620.70 | 619.32 | 621.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 618.25 | 619.32 | 621.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 621.80 | 619.82 | 621.69 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 626.10 | 622.15 | 621.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 628.05 | 623.85 | 622.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 622.10 | 624.43 | 623.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 15:15:00 | 622.10 | 624.43 | 623.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 622.10 | 624.43 | 623.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 617.55 | 624.43 | 623.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 615.50 | 622.64 | 622.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 611.65 | 618.01 | 620.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 587.50 | 580.51 | 589.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 587.95 | 580.51 | 589.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 597.50 | 585.27 | 589.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 596.60 | 585.27 | 589.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 598.00 | 587.82 | 590.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 598.50 | 587.82 | 590.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 587.70 | 590.67 | 591.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 586.20 | 590.12 | 590.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 556.89 | 572.36 | 579.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 570.80 | 556.36 | 565.96 | SL hit (close>ema200) qty=0.50 sl=556.36 alert=retest2 |

### Cycle 201 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 580.95 | 571.36 | 570.39 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 560.60 | 569.20 | 569.50 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 580.80 | 570.82 | 569.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 585.20 | 573.70 | 571.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 15:15:00 | 708.05 | 711.73 | 692.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 09:45:00 | 706.95 | 710.69 | 694.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 706.50 | 711.10 | 707.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 707.50 | 711.10 | 707.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 712.05 | 711.29 | 708.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:00:00 | 716.90 | 712.88 | 709.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:45:00 | 716.85 | 721.52 | 721.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 717.50 | 720.72 | 720.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 717.50 | 720.72 | 720.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 711.90 | 718.52 | 719.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 714.00 | 708.01 | 712.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 714.00 | 708.01 | 712.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 714.00 | 708.01 | 712.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 714.20 | 708.01 | 712.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 717.40 | 709.89 | 712.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 717.40 | 709.89 | 712.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 712.50 | 712.71 | 713.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 713.05 | 712.71 | 713.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 714.10 | 712.98 | 713.37 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 726.65 | 715.72 | 714.58 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 10:15:00 | 718.00 | 719.67 | 719.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 713.95 | 717.94 | 718.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 09:15:00 | 720.10 | 717.48 | 718.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 720.10 | 717.48 | 718.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 720.10 | 717.48 | 718.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:45:00 | 726.10 | 717.48 | 718.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 718.90 | 717.77 | 718.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 721.95 | 717.77 | 718.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 723.00 | 718.81 | 718.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 729.00 | 720.83 | 719.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 721.90 | 727.75 | 726.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 721.90 | 727.75 | 726.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 721.90 | 727.75 | 726.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 721.90 | 727.75 | 726.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 722.00 | 726.60 | 725.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 720.50 | 726.60 | 725.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 720.90 | 724.61 | 724.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 717.90 | 723.27 | 724.22 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 12:30:00 | 699.00 | 2024-04-15 09:15:00 | 664.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 14:30:00 | 700.80 | 2024-04-15 09:15:00 | 665.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 12:30:00 | 699.00 | 2024-04-16 09:15:00 | 696.00 | STOP_HIT | 0.50 | 0.43% |
| SELL | retest2 | 2024-04-12 14:30:00 | 700.80 | 2024-04-16 09:15:00 | 696.00 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2024-04-26 15:00:00 | 664.85 | 2024-05-06 09:15:00 | 631.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-29 09:30:00 | 659.00 | 2024-05-06 09:15:00 | 631.65 | PARTIAL | 0.50 | 4.15% |
| SELL | retest2 | 2024-04-29 11:30:00 | 664.90 | 2024-05-07 09:15:00 | 626.05 | PARTIAL | 0.50 | 5.84% |
| SELL | retest2 | 2024-04-29 12:15:00 | 663.60 | 2024-05-07 09:15:00 | 630.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 10:15:00 | 658.39 | 2024-05-07 09:15:00 | 625.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 15:00:00 | 664.85 | 2024-05-08 09:15:00 | 654.79 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2024-04-29 09:30:00 | 659.00 | 2024-05-08 09:15:00 | 654.79 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2024-04-29 11:30:00 | 664.90 | 2024-05-08 09:15:00 | 654.79 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2024-04-29 12:15:00 | 663.60 | 2024-05-08 09:15:00 | 654.79 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2024-05-02 10:15:00 | 658.39 | 2024-05-08 09:15:00 | 654.79 | STOP_HIT | 0.50 | 0.55% |
| SELL | retest2 | 2024-05-15 10:30:00 | 656.50 | 2024-05-17 15:15:00 | 661.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-05-15 11:15:00 | 655.80 | 2024-05-17 15:15:00 | 661.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-05-16 13:45:00 | 655.07 | 2024-05-17 15:15:00 | 661.80 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-05-17 11:00:00 | 655.00 | 2024-05-17 15:15:00 | 661.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-05-30 09:15:00 | 649.05 | 2024-06-04 09:15:00 | 616.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:45:00 | 650.62 | 2024-06-04 09:15:00 | 618.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 09:15:00 | 649.05 | 2024-06-05 12:15:00 | 616.32 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2024-06-03 10:45:00 | 650.62 | 2024-06-05 12:15:00 | 616.32 | STOP_HIT | 0.50 | 5.27% |
| BUY | retest2 | 2024-07-04 09:15:00 | 662.38 | 2024-07-04 13:15:00 | 660.44 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-07-04 11:00:00 | 661.31 | 2024-07-04 13:15:00 | 660.44 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-07-08 11:45:00 | 650.74 | 2024-07-12 11:15:00 | 662.46 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-08 14:45:00 | 650.69 | 2024-07-12 11:15:00 | 662.46 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-07-09 10:15:00 | 650.82 | 2024-07-12 11:15:00 | 662.46 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-07-09 11:00:00 | 650.90 | 2024-07-12 11:15:00 | 662.46 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-10 10:15:00 | 644.50 | 2024-07-12 12:15:00 | 653.03 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-07-10 13:00:00 | 644.00 | 2024-07-12 12:15:00 | 653.03 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-07-11 09:15:00 | 644.86 | 2024-07-12 12:15:00 | 653.03 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-07-11 10:00:00 | 644.51 | 2024-07-12 12:15:00 | 653.03 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-08-12 13:15:00 | 599.99 | 2024-08-16 12:15:00 | 606.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-08-12 14:45:00 | 599.96 | 2024-08-16 12:15:00 | 606.90 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-13 13:45:00 | 599.61 | 2024-08-16 12:15:00 | 606.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-08-23 09:15:00 | 625.33 | 2024-08-23 10:15:00 | 617.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-08-30 09:45:00 | 758.00 | 2024-09-03 11:15:00 | 717.84 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest2 | 2024-08-30 11:45:00 | 737.69 | 2024-09-03 11:15:00 | 717.84 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-08-30 13:45:00 | 737.86 | 2024-09-03 11:15:00 | 717.84 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-09-02 09:15:00 | 740.00 | 2024-09-03 11:15:00 | 717.84 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-09-06 09:15:00 | 732.26 | 2024-09-06 09:15:00 | 715.04 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-09-11 13:30:00 | 702.20 | 2024-09-19 10:15:00 | 667.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 14:00:00 | 698.66 | 2024-09-19 10:15:00 | 663.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 11:30:00 | 701.90 | 2024-09-19 10:15:00 | 666.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 15:00:00 | 702.00 | 2024-09-19 10:15:00 | 666.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 10:15:00 | 695.51 | 2024-09-19 11:15:00 | 660.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 13:30:00 | 702.20 | 2024-09-20 09:15:00 | 671.74 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2024-09-11 14:00:00 | 698.66 | 2024-09-20 09:15:00 | 671.74 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2024-09-13 11:30:00 | 701.90 | 2024-09-20 09:15:00 | 671.74 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2024-09-13 15:00:00 | 702.00 | 2024-09-20 09:15:00 | 671.74 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2024-09-16 10:15:00 | 695.51 | 2024-09-20 09:15:00 | 671.74 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest1 | 2024-10-01 14:15:00 | 680.36 | 2024-10-07 10:15:00 | 646.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-01 15:15:00 | 680.00 | 2024-10-07 10:15:00 | 646.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 675.06 | 2024-10-07 10:15:00 | 641.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-01 14:15:00 | 680.36 | 2024-10-08 10:15:00 | 652.94 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest1 | 2024-10-01 15:15:00 | 680.00 | 2024-10-08 10:15:00 | 652.94 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2024-10-03 09:15:00 | 675.06 | 2024-10-08 10:15:00 | 652.94 | STOP_HIT | 0.50 | 3.28% |
| BUY | retest2 | 2024-10-14 13:45:00 | 695.25 | 2024-10-15 09:15:00 | 691.75 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-10-14 15:15:00 | 695.00 | 2024-10-15 09:15:00 | 691.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-10-15 10:30:00 | 697.00 | 2024-10-17 10:15:00 | 692.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-10-17 09:30:00 | 700.43 | 2024-10-17 10:15:00 | 692.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-10-23 14:00:00 | 682.87 | 2024-10-25 10:15:00 | 648.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:00:00 | 682.87 | 2024-10-28 10:15:00 | 659.09 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2024-11-06 10:15:00 | 669.64 | 2024-11-06 13:15:00 | 673.17 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-11-12 11:15:00 | 693.65 | 2024-11-13 09:15:00 | 670.04 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-11-12 13:00:00 | 687.70 | 2024-11-13 09:15:00 | 670.04 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-11-12 14:15:00 | 689.20 | 2024-11-13 09:15:00 | 670.04 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-11-18 13:15:00 | 671.00 | 2024-11-25 14:15:00 | 666.78 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2024-11-18 14:00:00 | 670.09 | 2024-11-25 14:15:00 | 666.78 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-11-19 12:15:00 | 667.56 | 2024-11-25 14:15:00 | 666.78 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-11-19 13:45:00 | 671.08 | 2024-11-25 14:15:00 | 666.78 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-11-29 09:15:00 | 668.20 | 2024-11-29 12:15:00 | 664.38 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-11-29 09:45:00 | 668.95 | 2024-11-29 12:15:00 | 664.38 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-12-06 09:15:00 | 701.00 | 2024-12-10 10:15:00 | 686.57 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-12-09 10:30:00 | 697.27 | 2024-12-10 10:15:00 | 686.57 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-12-16 13:45:00 | 685.56 | 2024-12-20 15:15:00 | 651.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 685.89 | 2024-12-20 15:15:00 | 651.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 685.65 | 2024-12-20 15:15:00 | 651.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 13:45:00 | 685.56 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 0.50 | -5.17% |
| SELL | retest2 | 2024-12-17 09:30:00 | 685.89 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 0.50 | -5.12% |
| SELL | retest2 | 2024-12-17 10:15:00 | 685.65 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 0.50 | -5.15% |
| SELL | retest2 | 2024-12-17 12:15:00 | 679.83 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest2 | 2024-12-18 09:45:00 | 671.00 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 1.00 | -7.45% |
| SELL | retest2 | 2024-12-18 10:30:00 | 670.41 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 1.00 | -7.54% |
| SELL | retest2 | 2024-12-18 11:30:00 | 671.18 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 1.00 | -7.42% |
| SELL | retest2 | 2024-12-18 14:30:00 | 670.61 | 2024-12-24 09:15:00 | 720.98 | STOP_HIT | 1.00 | -7.51% |
| BUY | retest2 | 2024-12-26 09:15:00 | 692.00 | 2024-12-30 10:15:00 | 679.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-01-02 09:15:00 | 686.72 | 2025-01-06 10:15:00 | 677.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-02 11:45:00 | 686.67 | 2025-01-06 10:15:00 | 677.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-01-02 12:15:00 | 686.90 | 2025-01-06 10:15:00 | 677.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-03 15:00:00 | 689.60 | 2025-01-06 10:15:00 | 677.60 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-01-23 12:30:00 | 619.50 | 2025-01-23 13:15:00 | 625.21 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-01-23 13:00:00 | 619.00 | 2025-01-23 13:15:00 | 625.21 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-03-21 09:15:00 | 631.09 | 2025-03-26 10:15:00 | 631.34 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-04-04 09:15:00 | 618.40 | 2025-04-07 09:15:00 | 587.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 618.40 | 2025-04-07 15:15:00 | 600.42 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2025-04-17 11:30:00 | 628.50 | 2025-04-22 15:15:00 | 628.30 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-04-17 12:00:00 | 629.05 | 2025-04-22 15:15:00 | 628.30 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-04-24 13:30:00 | 626.00 | 2025-04-30 14:15:00 | 594.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 14:00:00 | 625.95 | 2025-04-30 14:15:00 | 594.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 15:15:00 | 624.10 | 2025-04-30 14:15:00 | 592.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 13:30:00 | 626.00 | 2025-05-05 09:15:00 | 595.95 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2025-04-24 14:00:00 | 625.95 | 2025-05-05 09:15:00 | 595.95 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-04-24 15:15:00 | 624.10 | 2025-05-05 09:15:00 | 595.95 | STOP_HIT | 0.50 | 4.51% |
| BUY | retest2 | 2025-05-27 11:45:00 | 629.40 | 2025-05-30 09:15:00 | 692.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 13:00:00 | 629.05 | 2025-05-30 09:15:00 | 691.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 15:00:00 | 631.20 | 2025-06-04 14:15:00 | 661.35 | STOP_HIT | 1.00 | 4.78% |
| SELL | retest2 | 2025-07-07 11:15:00 | 671.10 | 2025-07-21 11:15:00 | 678.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-31 09:15:00 | 680.50 | 2025-08-05 09:15:00 | 748.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-01 11:30:00 | 684.40 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.37% |
| SELL | retest2 | 2025-09-02 11:00:00 | 685.40 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-09-02 11:30:00 | 685.85 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2025-09-02 12:00:00 | 685.60 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-09-02 13:15:00 | 683.50 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2025-09-03 10:15:00 | 683.80 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-09-03 13:00:00 | 682.60 | 2025-09-10 14:15:00 | 675.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-09-22 12:45:00 | 743.00 | 2025-09-23 14:15:00 | 817.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-28 15:15:00 | 819.00 | 2025-11-06 09:15:00 | 778.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 15:15:00 | 819.00 | 2025-11-07 10:15:00 | 778.10 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-11-20 15:15:00 | 757.75 | 2025-11-26 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-12-01 12:15:00 | 748.50 | 2025-12-04 15:15:00 | 711.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 12:15:00 | 748.50 | 2025-12-08 09:15:00 | 742.00 | STOP_HIT | 0.50 | 0.87% |
| BUY | retest2 | 2025-12-24 09:15:00 | 718.60 | 2025-12-24 15:15:00 | 710.80 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-12-24 11:15:00 | 716.55 | 2025-12-24 15:15:00 | 710.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-12-24 13:15:00 | 716.30 | 2025-12-24 15:15:00 | 710.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-01-07 11:30:00 | 694.75 | 2026-01-07 13:15:00 | 697.85 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-07 13:15:00 | 694.80 | 2026-01-07 13:15:00 | 697.85 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-01-14 10:00:00 | 665.90 | 2026-01-20 14:15:00 | 632.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:45:00 | 666.25 | 2026-01-20 14:15:00 | 632.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 665.65 | 2026-01-20 14:15:00 | 632.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 666.50 | 2026-01-20 14:15:00 | 633.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 665.90 | 2026-01-22 09:15:00 | 625.50 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2026-01-14 12:45:00 | 666.25 | 2026-01-22 09:15:00 | 625.50 | STOP_HIT | 0.50 | 6.12% |
| SELL | retest2 | 2026-01-16 09:30:00 | 665.65 | 2026-01-22 09:15:00 | 625.50 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-01-16 12:15:00 | 666.50 | 2026-01-22 09:15:00 | 625.50 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2026-01-23 11:45:00 | 612.15 | 2026-01-28 14:15:00 | 622.40 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-27 10:15:00 | 611.35 | 2026-01-28 14:15:00 | 622.40 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest1 | 2026-02-20 09:15:00 | 735.30 | 2026-02-23 10:15:00 | 689.85 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2026-03-02 09:15:00 | 637.60 | 2026-03-05 14:15:00 | 655.40 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2026-03-05 09:45:00 | 647.70 | 2026-03-05 14:15:00 | 655.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-27 09:15:00 | 586.20 | 2026-03-30 09:15:00 | 556.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 586.20 | 2026-04-01 09:15:00 | 570.80 | STOP_HIT | 0.50 | 2.63% |
| BUY | retest2 | 2026-04-21 11:00:00 | 716.90 | 2026-04-23 14:15:00 | 717.50 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2026-04-23 13:45:00 | 716.85 | 2026-04-23 14:15:00 | 717.50 | STOP_HIT | 1.00 | 0.09% |

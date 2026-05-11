# Sapphire Foods India Ltd. (SAPPHIRE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 183.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 228 |
| ALERT1 | 148 |
| ALERT2 | 145 |
| ALERT2_SKIP | 102 |
| ALERT3 | 302 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 108 |
| PARTIAL | 36 |
| TARGET_HIT | 6 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 80 / 69
- **Target hits / Stop hits / Partials:** 6 / 107 / 36
- **Avg / median % per leg:** 1.72% / 1.42%
- **Sum % (uncompounded):** 256.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 9 | 29.0% | 6 | 25 | 0 | 0.75% | 23.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.86% | -1.9% |
| BUY @ 3rd Alert (retest2) | 30 | 9 | 30.0% | 6 | 24 | 0 | 0.83% | 25.0% |
| SELL (all) | 118 | 71 | 60.2% | 0 | 82 | 36 | 1.98% | 233.6% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 0 | 6 | 2 | 1.92% | 15.4% |
| SELL @ 3rd Alert (retest2) | 110 | 65 | 59.1% | 0 | 76 | 34 | 1.98% | 218.2% |
| retest1 (combined) | 9 | 6 | 66.7% | 0 | 7 | 2 | 1.50% | 13.5% |
| retest2 (combined) | 140 | 74 | 52.9% | 6 | 100 | 34 | 1.74% | 243.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 12:15:00 | 259.27 | 257.38 | 257.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 13:15:00 | 260.17 | 257.94 | 257.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 14:15:00 | 259.31 | 259.74 | 258.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 15:15:00 | 260.96 | 259.98 | 259.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 260.96 | 259.98 | 259.13 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 15:15:00 | 259.60 | 261.01 | 261.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 15:15:00 | 258.29 | 259.58 | 260.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 264.21 | 260.50 | 260.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 264.21 | 260.50 | 260.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 264.21 | 260.50 | 260.61 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 265.36 | 261.48 | 261.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 13:15:00 | 267.11 | 265.00 | 263.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 268.73 | 269.16 | 267.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 10:15:00 | 268.73 | 269.07 | 267.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 268.73 | 269.07 | 267.43 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 09:15:00 | 263.51 | 266.82 | 266.94 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 12:15:00 | 272.16 | 266.01 | 265.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 14:15:00 | 277.20 | 268.72 | 267.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 286.20 | 286.29 | 281.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 286.00 | 287.15 | 285.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 286.00 | 287.15 | 285.63 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 09:15:00 | 280.50 | 285.51 | 285.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 10:15:00 | 279.28 | 284.26 | 285.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 272.40 | 271.10 | 274.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 272.40 | 271.10 | 274.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 272.40 | 271.10 | 274.23 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 14:15:00 | 281.08 | 275.55 | 275.42 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 12:15:00 | 276.60 | 277.29 | 277.31 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 14:15:00 | 279.22 | 277.67 | 277.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 11:15:00 | 282.24 | 278.94 | 278.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 277.23 | 278.60 | 278.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 12:15:00 | 277.23 | 278.60 | 278.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 12:15:00 | 277.23 | 278.60 | 278.09 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 14:15:00 | 274.53 | 277.39 | 277.60 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 280.45 | 277.54 | 277.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 10:15:00 | 282.46 | 278.52 | 277.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 282.35 | 282.57 | 280.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 12:15:00 | 281.33 | 282.33 | 280.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 281.33 | 282.33 | 280.91 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 275.44 | 280.10 | 280.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 14:15:00 | 272.89 | 276.72 | 278.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 277.24 | 276.04 | 277.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 277.24 | 276.04 | 277.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 277.24 | 276.04 | 277.64 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 280.00 | 278.22 | 278.14 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 12:15:00 | 278.59 | 280.61 | 280.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 13:15:00 | 275.73 | 279.63 | 280.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 09:15:00 | 275.19 | 274.86 | 276.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 275.19 | 274.86 | 276.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 275.19 | 274.86 | 276.45 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 13:15:00 | 281.60 | 277.47 | 277.25 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 277.40 | 278.63 | 278.78 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 279.27 | 278.62 | 278.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 14:15:00 | 280.07 | 279.04 | 278.77 | Break + close above crossover candle high |

### Cycle 18 — SELL (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 09:15:00 | 276.00 | 278.58 | 278.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 10:15:00 | 275.82 | 278.03 | 278.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 10:15:00 | 278.84 | 276.88 | 277.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 10:15:00 | 278.84 | 276.88 | 277.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 278.84 | 276.88 | 277.43 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 280.14 | 277.62 | 277.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 12:15:00 | 281.80 | 278.83 | 278.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 13:15:00 | 281.61 | 282.05 | 280.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 15:15:00 | 280.65 | 281.76 | 280.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 280.65 | 281.76 | 280.77 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 284.54 | 285.85 | 285.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 15:15:00 | 283.91 | 285.03 | 285.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 12:15:00 | 280.72 | 280.26 | 281.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 14:15:00 | 278.33 | 275.27 | 276.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 278.33 | 275.27 | 276.48 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 13:15:00 | 273.33 | 270.91 | 270.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 11:15:00 | 275.93 | 272.56 | 271.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 272.58 | 274.36 | 273.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 272.58 | 274.36 | 273.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 272.58 | 274.36 | 273.04 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 272.00 | 273.42 | 273.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 269.05 | 272.01 | 272.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 15:15:00 | 271.61 | 270.99 | 271.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 15:15:00 | 271.61 | 270.99 | 271.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 271.61 | 270.99 | 271.62 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 268.44 | 267.08 | 266.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 11:15:00 | 269.53 | 268.08 | 267.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 271.83 | 271.94 | 270.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 10:15:00 | 270.67 | 271.69 | 270.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 270.67 | 271.69 | 270.59 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 293.73 | 298.94 | 299.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 288.31 | 295.65 | 297.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 15:15:00 | 298.99 | 296.32 | 298.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 15:15:00 | 298.99 | 296.32 | 298.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 15:15:00 | 298.99 | 296.32 | 298.03 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 14:15:00 | 297.26 | 290.29 | 289.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 287.79 | 290.55 | 290.72 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 11:15:00 | 293.48 | 291.14 | 290.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 09:15:00 | 294.58 | 292.27 | 291.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 11:15:00 | 295.61 | 296.10 | 294.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 11:15:00 | 295.61 | 296.10 | 294.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 295.61 | 296.10 | 294.44 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 10:15:00 | 291.83 | 293.81 | 293.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 13:15:00 | 289.88 | 292.42 | 293.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 11:15:00 | 284.19 | 284.08 | 285.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 12:15:00 | 283.05 | 283.88 | 285.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 283.05 | 283.88 | 285.58 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 15:15:00 | 293.00 | 286.81 | 286.56 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 283.61 | 286.33 | 286.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 282.44 | 285.29 | 286.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 12:15:00 | 282.79 | 282.28 | 283.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 13:15:00 | 283.53 | 282.53 | 283.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 283.53 | 282.53 | 283.56 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 288.17 | 284.43 | 284.24 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 278.60 | 283.82 | 284.14 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 286.60 | 282.80 | 282.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 295.50 | 287.65 | 285.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 287.40 | 288.87 | 287.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 15:15:00 | 287.40 | 288.87 | 287.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 287.40 | 288.87 | 287.22 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 13:15:00 | 286.11 | 287.24 | 287.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 15:15:00 | 285.04 | 286.89 | 287.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 284.34 | 282.78 | 284.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 284.34 | 282.78 | 284.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 284.34 | 282.78 | 284.15 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 10:15:00 | 265.09 | 261.30 | 261.22 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 262.60 | 263.23 | 263.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 262.04 | 262.94 | 263.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 262.90 | 262.66 | 262.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 262.90 | 262.66 | 262.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 262.90 | 262.66 | 262.96 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 262.99 | 260.77 | 260.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 270.60 | 264.95 | 263.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 15:15:00 | 272.03 | 272.69 | 270.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 278.64 | 278.52 | 276.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 278.64 | 278.52 | 276.58 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 09:15:00 | 278.76 | 280.01 | 280.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 15:15:00 | 277.80 | 279.60 | 279.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 13:15:00 | 283.56 | 279.26 | 279.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 13:15:00 | 283.56 | 279.26 | 279.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 283.56 | 279.26 | 279.41 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 11:15:00 | 280.38 | 279.30 | 279.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 13:15:00 | 282.10 | 279.91 | 279.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 12:15:00 | 284.55 | 287.66 | 285.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 12:15:00 | 284.55 | 287.66 | 285.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 284.55 | 287.66 | 285.67 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 10:15:00 | 282.20 | 284.44 | 284.66 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 285.90 | 284.55 | 284.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 13:15:00 | 286.54 | 284.94 | 284.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 13:15:00 | 285.62 | 286.63 | 285.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 13:15:00 | 285.62 | 286.63 | 285.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 285.62 | 286.63 | 285.88 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 10:15:00 | 282.58 | 285.57 | 285.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 10:15:00 | 281.11 | 283.74 | 284.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 14:15:00 | 282.80 | 282.02 | 283.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 282.80 | 282.02 | 283.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 282.80 | 282.02 | 283.31 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 15:15:00 | 283.00 | 282.84 | 282.83 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 10:15:00 | 281.47 | 282.74 | 282.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 09:15:00 | 279.99 | 281.66 | 282.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 11:15:00 | 281.26 | 281.25 | 281.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 11:15:00 | 281.26 | 281.25 | 281.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 281.26 | 281.25 | 281.89 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 287.39 | 282.83 | 282.21 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 280.46 | 282.11 | 282.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 11:15:00 | 279.72 | 281.63 | 281.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 281.69 | 281.06 | 281.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 281.69 | 281.06 | 281.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 281.69 | 281.06 | 281.46 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 285.00 | 281.97 | 281.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 287.00 | 282.98 | 282.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 12:15:00 | 283.68 | 283.75 | 282.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 11:15:00 | 284.62 | 285.70 | 284.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 11:15:00 | 284.62 | 285.70 | 284.36 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 15:15:00 | 281.75 | 283.80 | 283.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 09:15:00 | 281.12 | 283.27 | 283.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-28 14:15:00 | 281.11 | 280.73 | 281.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 15:15:00 | 281.00 | 280.79 | 281.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 281.00 | 280.79 | 281.89 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 13:15:00 | 284.27 | 282.65 | 282.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 13:15:00 | 284.93 | 283.40 | 283.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 282.40 | 283.20 | 282.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 14:15:00 | 282.40 | 283.20 | 282.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 282.40 | 283.20 | 282.94 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 280.40 | 282.64 | 282.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 279.62 | 281.68 | 282.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 13:15:00 | 283.23 | 281.93 | 282.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 13:15:00 | 283.23 | 281.93 | 282.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 283.23 | 281.93 | 282.21 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 11:15:00 | 283.60 | 282.10 | 281.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 12:15:00 | 284.62 | 282.60 | 282.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 285.41 | 285.67 | 284.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 14:15:00 | 284.90 | 285.52 | 284.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 284.90 | 285.52 | 284.44 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 279.96 | 283.55 | 283.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 10:15:00 | 277.81 | 280.84 | 282.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 09:15:00 | 281.78 | 278.74 | 280.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 281.78 | 278.74 | 280.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 281.78 | 278.74 | 280.22 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 12:15:00 | 286.16 | 281.90 | 281.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 13:15:00 | 288.00 | 283.12 | 282.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 283.46 | 285.58 | 284.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 13:15:00 | 283.46 | 285.58 | 284.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 283.46 | 285.58 | 284.23 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 11:15:00 | 282.89 | 283.61 | 283.61 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 12:15:00 | 287.76 | 284.44 | 283.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 13:15:00 | 288.00 | 285.15 | 284.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 12:15:00 | 283.16 | 287.60 | 286.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 12:15:00 | 283.16 | 287.60 | 286.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 283.16 | 287.60 | 286.30 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 283.33 | 285.42 | 285.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 282.00 | 284.73 | 285.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 14:15:00 | 285.24 | 284.49 | 285.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 14:15:00 | 285.24 | 284.49 | 285.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 285.24 | 284.49 | 285.04 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 284.88 | 283.72 | 283.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 284.89 | 283.96 | 283.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 11:15:00 | 291.70 | 293.00 | 289.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 13:15:00 | 288.91 | 291.97 | 289.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 288.91 | 291.97 | 289.86 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 09:15:00 | 285.36 | 288.45 | 288.85 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 13:15:00 | 288.00 | 286.60 | 286.59 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 14:15:00 | 286.34 | 286.54 | 286.57 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 15:15:00 | 287.00 | 286.64 | 286.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 292.56 | 287.82 | 287.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 287.96 | 290.82 | 289.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 287.96 | 290.82 | 289.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 287.96 | 290.82 | 289.53 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 287.06 | 288.85 | 288.96 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 290.27 | 289.14 | 289.08 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 287.38 | 289.00 | 289.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 13:15:00 | 284.81 | 288.16 | 288.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 287.14 | 282.37 | 284.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 287.14 | 282.37 | 284.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 287.14 | 282.37 | 284.09 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 11:15:00 | 273.93 | 271.48 | 271.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 280.81 | 273.74 | 272.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 09:15:00 | 286.88 | 287.14 | 283.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 11:15:00 | 280.89 | 285.32 | 282.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 280.89 | 285.32 | 282.98 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 13:15:00 | 299.71 | 300.60 | 300.66 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 10:15:00 | 300.83 | 300.66 | 300.65 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 299.81 | 300.49 | 300.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 13:15:00 | 299.10 | 300.13 | 300.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 301.24 | 300.25 | 300.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 301.24 | 300.25 | 300.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 301.24 | 300.25 | 300.37 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 305.00 | 301.20 | 300.79 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 299.43 | 300.96 | 300.96 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 10:15:00 | 301.54 | 301.07 | 301.01 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 298.42 | 300.54 | 300.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 12:15:00 | 297.80 | 299.99 | 300.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 299.14 | 298.34 | 299.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 299.14 | 298.34 | 299.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 299.14 | 298.34 | 299.46 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 09:15:00 | 292.66 | 282.31 | 281.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 09:15:00 | 298.34 | 291.67 | 287.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 13:15:00 | 319.37 | 319.87 | 314.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 15:15:00 | 317.38 | 317.90 | 316.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 317.38 | 317.90 | 316.24 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 11:15:00 | 313.24 | 315.28 | 315.28 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 13:15:00 | 316.95 | 315.35 | 315.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 14:15:00 | 320.00 | 316.28 | 315.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 313.32 | 317.87 | 317.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 14:15:00 | 313.32 | 317.87 | 317.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 313.32 | 317.87 | 317.20 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 12:15:00 | 314.05 | 317.68 | 317.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-02 14:15:00 | 311.35 | 315.83 | 317.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 15:15:00 | 302.20 | 302.14 | 305.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 11:15:00 | 306.60 | 302.78 | 305.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 306.60 | 302.78 | 305.17 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 312.62 | 305.72 | 305.11 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 299.76 | 306.34 | 306.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 13:15:00 | 297.55 | 302.64 | 304.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 305.40 | 302.58 | 303.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 10:15:00 | 305.40 | 302.58 | 303.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 305.40 | 302.58 | 303.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 305.40 | 302.58 | 303.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 306.39 | 303.34 | 304.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 306.47 | 303.34 | 304.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 298.71 | 301.04 | 302.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 298.34 | 301.04 | 302.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:45:00 | 298.12 | 297.03 | 297.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 14:15:00 | 283.42 | 288.19 | 292.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 14:15:00 | 283.21 | 288.19 | 292.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 280.60 | 278.14 | 281.21 | SL hit (close>ema200) qty=0.50 sl=278.14 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 282.80 | 276.43 | 276.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 12:15:00 | 284.32 | 278.01 | 277.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 09:15:00 | 281.24 | 281.37 | 279.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 281.24 | 281.37 | 279.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 281.24 | 281.37 | 279.28 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 279.00 | 280.43 | 280.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 13:15:00 | 277.44 | 279.83 | 280.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 15:15:00 | 280.60 | 279.79 | 280.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 15:15:00 | 280.60 | 279.79 | 280.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 280.60 | 279.79 | 280.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 277.82 | 279.79 | 280.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 280.02 | 279.84 | 280.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 280.02 | 279.84 | 280.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 279.06 | 279.68 | 280.01 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 14:15:00 | 281.86 | 280.28 | 280.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 15:15:00 | 284.40 | 281.10 | 280.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 10:15:00 | 279.01 | 280.88 | 280.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 10:15:00 | 279.01 | 280.88 | 280.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 279.01 | 280.88 | 280.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 279.01 | 280.88 | 280.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 280.53 | 280.81 | 280.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:45:00 | 281.98 | 281.30 | 280.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 280.89 | 281.13 | 280.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 12:15:00 | 277.25 | 280.13 | 280.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 277.25 | 280.13 | 280.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 275.73 | 279.25 | 280.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 13:15:00 | 275.66 | 275.15 | 277.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 13:15:00 | 275.66 | 275.15 | 277.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 275.66 | 275.15 | 277.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:00:00 | 275.66 | 275.15 | 277.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 281.61 | 276.44 | 277.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 280.85 | 276.44 | 277.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-05-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 15:15:00 | 285.99 | 278.35 | 278.28 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 274.00 | 277.48 | 277.90 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 277.50 | 276.62 | 276.55 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 274.89 | 276.32 | 276.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 11:15:00 | 274.00 | 275.55 | 276.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 275.91 | 274.83 | 275.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 275.91 | 274.83 | 275.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 275.91 | 274.83 | 275.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:45:00 | 275.00 | 274.79 | 275.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:45:00 | 275.00 | 275.31 | 275.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 277.27 | 275.70 | 275.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 277.27 | 275.70 | 275.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 277.33 | 276.23 | 275.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 10:15:00 | 275.53 | 276.09 | 275.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 10:15:00 | 275.53 | 276.09 | 275.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 275.53 | 276.09 | 275.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:30:00 | 275.52 | 276.09 | 275.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 275.03 | 275.88 | 275.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:45:00 | 275.04 | 275.88 | 275.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 274.53 | 275.61 | 275.68 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 13:15:00 | 278.07 | 276.10 | 275.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 15:15:00 | 279.80 | 277.09 | 276.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 276.00 | 277.06 | 276.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 12:15:00 | 276.00 | 277.06 | 276.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 276.00 | 277.06 | 276.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 278.66 | 277.06 | 276.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:00:00 | 278.44 | 277.34 | 276.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 278.30 | 281.87 | 282.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 278.30 | 281.87 | 282.09 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 288.00 | 282.36 | 281.69 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 280.40 | 283.32 | 283.32 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 292.00 | 284.06 | 283.57 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 281.20 | 284.45 | 284.78 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 286.79 | 284.87 | 284.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 292.18 | 286.33 | 285.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 289.46 | 289.58 | 287.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 12:00:00 | 289.46 | 289.58 | 287.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 287.25 | 289.11 | 287.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 287.25 | 289.11 | 287.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 289.72 | 289.23 | 288.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 293.24 | 289.41 | 288.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-20 09:15:00 | 322.56 | 312.55 | 308.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 311.50 | 315.42 | 315.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 13:15:00 | 310.60 | 312.97 | 314.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 12:15:00 | 312.19 | 310.66 | 312.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 312.19 | 310.66 | 312.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 312.19 | 310.66 | 312.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 312.19 | 310.66 | 312.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 309.01 | 310.33 | 312.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 15:15:00 | 306.80 | 310.57 | 312.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 312.38 | 310.33 | 311.66 | SL hit (close>static) qty=1.00 sl=312.17 alert=retest2 |

### Cycle 97 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 314.00 | 312.57 | 312.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 315.51 | 313.37 | 312.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 14:15:00 | 314.41 | 314.68 | 313.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 15:00:00 | 314.41 | 314.68 | 313.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 314.99 | 314.74 | 313.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 311.80 | 314.59 | 313.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 314.47 | 314.57 | 313.96 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 09:15:00 | 312.92 | 313.58 | 313.66 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 316.60 | 313.96 | 313.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 319.05 | 316.28 | 315.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 12:15:00 | 317.76 | 318.84 | 317.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 12:15:00 | 317.76 | 318.84 | 317.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 317.76 | 318.84 | 317.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 317.76 | 318.84 | 317.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 316.59 | 318.39 | 317.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 316.59 | 318.39 | 317.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 316.45 | 318.00 | 317.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 315.70 | 318.00 | 317.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 316.62 | 317.73 | 317.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 317.18 | 317.62 | 317.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 313.60 | 316.81 | 316.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 310.01 | 315.45 | 316.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 319.02 | 314.79 | 315.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 319.02 | 314.79 | 315.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 319.02 | 314.79 | 315.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 319.02 | 314.79 | 315.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 318.66 | 315.56 | 315.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:15:00 | 317.90 | 315.56 | 315.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 12:00:00 | 317.00 | 315.85 | 315.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 302.00 | 304.85 | 306.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 301.15 | 304.85 | 306.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 305.54 | 304.91 | 306.50 | SL hit (close>ema200) qty=0.50 sl=304.91 alert=retest2 |

### Cycle 101 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 312.63 | 307.35 | 306.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 326.62 | 313.87 | 310.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 314.68 | 317.30 | 314.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 314.68 | 317.30 | 314.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 314.68 | 317.30 | 314.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 314.68 | 317.30 | 314.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 316.19 | 317.08 | 314.72 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 311.51 | 313.67 | 313.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 15:15:00 | 310.00 | 312.94 | 313.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 313.41 | 313.03 | 313.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 313.41 | 313.03 | 313.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 313.41 | 313.03 | 313.48 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 11:15:00 | 315.95 | 313.93 | 313.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 10:15:00 | 318.57 | 315.97 | 315.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 10:15:00 | 334.46 | 336.99 | 331.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 11:00:00 | 334.46 | 336.99 | 331.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 326.49 | 334.33 | 332.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 325.93 | 334.33 | 332.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 326.20 | 331.39 | 331.63 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 336.40 | 331.48 | 331.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 09:15:00 | 338.67 | 333.69 | 332.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 12:15:00 | 330.00 | 334.00 | 333.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 12:15:00 | 330.00 | 334.00 | 333.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 330.00 | 334.00 | 333.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 330.01 | 334.00 | 333.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 13:15:00 | 326.38 | 332.48 | 332.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 322.46 | 329.24 | 330.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 311.60 | 308.24 | 312.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 311.60 | 308.24 | 312.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 308.99 | 308.03 | 311.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:15:00 | 308.00 | 308.22 | 310.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 313.10 | 309.16 | 310.87 | SL hit (close>static) qty=1.00 sl=311.99 alert=retest2 |

### Cycle 107 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 313.43 | 311.51 | 311.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 328.96 | 325.07 | 323.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 12:15:00 | 323.69 | 325.14 | 323.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 12:15:00 | 323.69 | 325.14 | 323.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 323.69 | 325.14 | 323.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 323.69 | 325.14 | 323.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 324.79 | 325.07 | 324.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 325.00 | 325.07 | 324.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 325.16 | 324.86 | 324.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 322.37 | 324.37 | 323.94 | SL hit (close<static) qty=1.00 sl=323.20 alert=retest2 |

### Cycle 108 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 322.52 | 323.59 | 323.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 321.01 | 323.08 | 323.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 10:15:00 | 321.99 | 321.96 | 322.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 321.99 | 321.96 | 322.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 321.99 | 321.96 | 322.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 323.06 | 321.96 | 322.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 318.87 | 320.62 | 321.68 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 333.30 | 323.21 | 322.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 337.28 | 326.03 | 323.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 331.48 | 331.56 | 329.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 14:45:00 | 332.10 | 331.56 | 329.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 335.25 | 338.81 | 335.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 335.25 | 338.81 | 335.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 334.40 | 337.93 | 335.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 334.40 | 337.93 | 335.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 320.45 | 333.26 | 333.99 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 333.40 | 329.88 | 329.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 339.25 | 332.85 | 331.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 334.15 | 335.10 | 333.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 334.15 | 335.10 | 333.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 336.75 | 335.43 | 333.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 334.10 | 335.43 | 333.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 332.50 | 334.93 | 333.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 332.50 | 334.93 | 333.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 329.25 | 333.80 | 333.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 329.25 | 333.80 | 333.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 329.70 | 332.98 | 332.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 13:15:00 | 328.25 | 329.83 | 330.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 328.80 | 328.02 | 329.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 328.80 | 328.02 | 329.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 328.80 | 328.02 | 329.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:45:00 | 329.40 | 328.02 | 329.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 332.00 | 328.92 | 329.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 330.35 | 328.92 | 329.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 329.00 | 328.93 | 329.41 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 333.15 | 329.78 | 329.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 12:15:00 | 335.40 | 332.99 | 331.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 336.00 | 336.25 | 334.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 12:00:00 | 336.00 | 336.25 | 334.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 372.10 | 369.77 | 360.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 388.35 | 369.94 | 364.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 373.25 | 377.47 | 375.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 363.45 | 373.99 | 374.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 363.45 | 373.99 | 374.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 10:15:00 | 358.50 | 370.89 | 372.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 359.25 | 354.70 | 357.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 359.25 | 354.70 | 357.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 359.25 | 354.70 | 357.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:30:00 | 345.30 | 353.60 | 356.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 348.90 | 352.88 | 355.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 349.80 | 351.49 | 354.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 13:15:00 | 348.80 | 351.47 | 353.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 355.00 | 350.66 | 351.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 355.60 | 350.66 | 351.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 353.50 | 351.60 | 351.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 353.50 | 351.60 | 351.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-09 15:15:00 | 353.65 | 352.01 | 351.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 353.65 | 352.01 | 351.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 356.35 | 352.88 | 352.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 349.75 | 352.87 | 352.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 11:15:00 | 349.75 | 352.87 | 352.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 349.75 | 352.87 | 352.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 349.75 | 352.87 | 352.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 354.95 | 353.29 | 352.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:30:00 | 349.15 | 353.29 | 352.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 356.80 | 362.05 | 359.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 356.80 | 362.05 | 359.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 354.95 | 360.63 | 358.98 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 13:15:00 | 353.35 | 357.80 | 357.90 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 358.40 | 357.77 | 357.75 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 354.90 | 357.30 | 357.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 350.15 | 355.36 | 356.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 324.95 | 324.21 | 328.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:30:00 | 325.35 | 324.21 | 328.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 329.00 | 325.47 | 327.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 329.00 | 325.47 | 327.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 334.10 | 327.20 | 328.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 331.70 | 327.20 | 328.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 345.00 | 330.76 | 329.85 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 319.00 | 328.41 | 328.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 310.40 | 320.65 | 322.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 10:15:00 | 315.35 | 314.30 | 317.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:45:00 | 315.70 | 314.30 | 317.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 320.80 | 316.04 | 317.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 320.80 | 316.04 | 317.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 320.50 | 316.93 | 317.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:30:00 | 331.70 | 316.93 | 317.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 320.05 | 318.34 | 318.25 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 315.00 | 318.46 | 318.47 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 319.65 | 318.25 | 318.07 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 312.80 | 317.16 | 317.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 307.25 | 313.94 | 315.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 14:15:00 | 307.70 | 307.57 | 309.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 15:00:00 | 307.70 | 307.57 | 309.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 304.35 | 304.55 | 307.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 304.35 | 304.55 | 307.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 306.95 | 305.10 | 306.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:30:00 | 304.55 | 305.21 | 306.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:30:00 | 303.90 | 304.76 | 306.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 15:15:00 | 311.00 | 306.66 | 306.77 | SL hit (close>static) qty=1.00 sl=307.20 alert=retest2 |

### Cycle 125 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 306.10 | 303.43 | 303.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 320.10 | 307.43 | 305.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 324.95 | 325.40 | 322.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:45:00 | 324.95 | 325.40 | 322.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 325.85 | 326.73 | 325.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 325.85 | 326.73 | 325.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 326.25 | 326.63 | 325.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:30:00 | 329.80 | 327.25 | 325.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 331.10 | 340.75 | 341.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 331.10 | 340.75 | 341.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 10:15:00 | 329.65 | 338.53 | 340.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 315.80 | 313.40 | 317.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:00:00 | 315.80 | 313.40 | 317.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 317.20 | 314.56 | 317.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 317.20 | 314.56 | 317.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 317.50 | 315.15 | 317.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 314.60 | 315.15 | 317.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 314.25 | 314.97 | 316.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 311.95 | 314.97 | 316.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:00:00 | 311.40 | 313.34 | 314.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:30:00 | 312.35 | 313.13 | 314.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 311.05 | 311.27 | 311.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 15:15:00 | 312.90 | 311.60 | 311.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 15:15:00 | 312.90 | 311.60 | 311.49 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 309.50 | 311.18 | 311.31 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 315.55 | 312.05 | 311.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 11:15:00 | 320.95 | 313.83 | 312.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 321.70 | 324.30 | 320.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 10:00:00 | 321.70 | 324.30 | 320.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 324.35 | 324.31 | 321.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 323.60 | 324.31 | 321.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 321.40 | 323.38 | 321.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:30:00 | 321.40 | 323.38 | 321.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 321.40 | 322.99 | 321.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 321.70 | 322.99 | 321.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 320.75 | 322.54 | 321.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:30:00 | 320.55 | 322.54 | 321.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 324.00 | 322.83 | 321.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 324.05 | 322.83 | 321.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 10:30:00 | 326.30 | 323.67 | 322.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:30:00 | 325.05 | 328.40 | 327.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 325.00 | 326.68 | 326.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 325.00 | 326.68 | 326.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 324.15 | 326.18 | 326.51 | Break + close below crossover candle low |

### Cycle 131 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 336.80 | 328.30 | 327.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 345.80 | 336.34 | 332.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 15:15:00 | 340.10 | 340.67 | 337.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:15:00 | 359.60 | 340.67 | 337.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 345.65 | 349.87 | 344.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 345.65 | 349.87 | 344.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 349.30 | 349.75 | 344.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 343.70 | 349.75 | 344.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 352.85 | 356.69 | 351.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 351.30 | 356.69 | 351.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 357.40 | 357.37 | 354.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 352.65 | 357.37 | 354.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 355.05 | 356.91 | 354.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:15:00 | 352.90 | 356.91 | 354.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 352.90 | 356.11 | 354.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-07 15:15:00 | 352.90 | 356.11 | 354.22 | SL hit (close<ema400) qty=1.00 sl=354.22 alert=retest1 |

### Cycle 132 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 339.40 | 352.77 | 352.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 335.85 | 340.39 | 344.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 14:15:00 | 343.90 | 340.11 | 343.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 343.90 | 340.11 | 343.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 343.90 | 340.11 | 343.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 343.90 | 340.11 | 343.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 341.55 | 340.40 | 343.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 336.25 | 340.40 | 343.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 334.50 | 339.22 | 342.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:45:00 | 331.30 | 335.40 | 339.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 318.90 | 333.60 | 337.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 314.74 | 323.61 | 329.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 325.20 | 323.93 | 328.75 | SL hit (close>ema200) qty=0.50 sl=323.93 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 333.00 | 327.94 | 327.61 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 321.95 | 326.64 | 327.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 319.95 | 323.74 | 325.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 15:15:00 | 316.50 | 315.79 | 318.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-21 09:15:00 | 316.65 | 315.79 | 318.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 317.10 | 316.05 | 318.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 312.30 | 316.05 | 318.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 313.85 | 315.07 | 317.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:30:00 | 312.20 | 314.33 | 316.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 310.70 | 314.45 | 316.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 313.50 | 314.27 | 315.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:30:00 | 308.25 | 313.15 | 314.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 312.60 | 312.54 | 314.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:45:00 | 312.10 | 311.96 | 313.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 296.69 | 305.53 | 308.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 298.16 | 305.53 | 308.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 296.59 | 305.53 | 308.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 296.97 | 305.53 | 308.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 296.50 | 305.53 | 308.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 13:15:00 | 295.16 | 300.04 | 304.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 13:15:00 | 292.84 | 300.04 | 304.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 295.45 | 290.93 | 295.32 | SL hit (close>ema200) qty=0.50 sl=290.93 alert=retest2 |

### Cycle 135 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 310.95 | 294.88 | 292.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 316.20 | 299.15 | 294.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 10:15:00 | 322.00 | 322.90 | 314.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 11:15:00 | 320.15 | 322.90 | 314.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 317.90 | 321.45 | 315.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 315.40 | 321.45 | 315.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 324.50 | 321.03 | 316.46 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 300.00 | 314.10 | 315.60 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 323.80 | 316.69 | 316.09 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 310.00 | 314.61 | 315.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 308.85 | 313.46 | 314.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 302.20 | 299.48 | 302.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 302.20 | 299.48 | 302.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 302.20 | 299.48 | 302.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 302.20 | 299.48 | 302.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 302.35 | 300.05 | 302.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:15:00 | 305.90 | 300.05 | 302.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 302.55 | 300.55 | 302.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:30:00 | 302.30 | 300.89 | 302.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 302.25 | 300.89 | 302.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 302.05 | 300.96 | 302.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 287.19 | 298.53 | 300.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 287.14 | 298.53 | 300.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 286.95 | 298.53 | 300.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 297.75 | 296.28 | 298.01 | SL hit (close>ema200) qty=0.50 sl=296.28 alert=retest2 |

### Cycle 139 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 303.75 | 299.50 | 298.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 309.55 | 303.54 | 301.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 12:15:00 | 324.45 | 325.29 | 320.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 13:00:00 | 324.45 | 325.29 | 320.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 317.75 | 323.79 | 320.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:00:00 | 317.75 | 323.79 | 320.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 312.50 | 321.53 | 319.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 312.50 | 321.53 | 319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 318.85 | 319.96 | 319.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 318.85 | 319.96 | 319.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 319.45 | 319.86 | 319.12 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 316.65 | 318.52 | 318.60 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 321.15 | 319.05 | 318.83 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 315.55 | 318.70 | 318.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 311.80 | 317.32 | 318.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 310.10 | 307.88 | 311.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 310.10 | 307.88 | 311.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 310.10 | 307.88 | 311.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 310.10 | 307.88 | 311.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 311.60 | 308.63 | 311.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 305.95 | 308.63 | 311.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 316.50 | 308.51 | 310.23 | SL hit (close>static) qty=1.00 sl=315.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 326.65 | 314.01 | 312.55 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 13:15:00 | 308.50 | 314.87 | 315.32 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 339.75 | 318.31 | 316.65 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 303.75 | 321.00 | 322.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 11:15:00 | 298.15 | 313.79 | 319.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 323.55 | 314.42 | 317.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 323.55 | 314.42 | 317.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 323.55 | 314.42 | 317.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 323.55 | 314.42 | 317.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 325.55 | 316.65 | 318.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 322.80 | 316.65 | 318.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:45:00 | 322.70 | 316.94 | 318.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 09:15:00 | 306.66 | 312.68 | 315.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 09:15:00 | 306.56 | 312.68 | 315.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 302.00 | 300.40 | 304.59 | SL hit (close>ema200) qty=0.50 sl=300.40 alert=retest2 |

### Cycle 147 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 309.25 | 306.19 | 306.03 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 15:15:00 | 304.80 | 305.93 | 306.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-20 12:15:00 | 302.15 | 304.83 | 305.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 303.45 | 302.54 | 303.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 303.45 | 302.54 | 303.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 303.45 | 302.54 | 303.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 305.45 | 302.54 | 303.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 302.75 | 302.58 | 303.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:45:00 | 304.05 | 302.58 | 303.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 302.30 | 302.53 | 303.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:00:00 | 302.30 | 302.53 | 303.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 313.05 | 304.49 | 304.31 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 299.00 | 305.10 | 305.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 13:15:00 | 296.80 | 301.89 | 303.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 15:15:00 | 302.00 | 301.61 | 303.45 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:15:00 | 293.80 | 301.61 | 303.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 296.65 | 294.54 | 297.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:15:00 | 304.45 | 294.54 | 297.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 299.45 | 295.52 | 297.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-27 12:15:00 | 299.45 | 295.52 | 297.36 | SL hit (close>ema400) qty=1.00 sl=297.36 alert=retest1 |

### Cycle 151 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 298.80 | 295.67 | 295.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 300.50 | 298.10 | 296.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 301.75 | 301.94 | 299.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:15:00 | 300.75 | 301.94 | 299.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 303.00 | 302.15 | 299.99 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 296.90 | 298.89 | 298.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 278.30 | 293.99 | 296.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 292.55 | 290.19 | 293.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 292.55 | 290.19 | 293.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 292.55 | 290.19 | 293.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 292.55 | 290.19 | 293.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 291.10 | 290.37 | 293.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 294.05 | 290.37 | 293.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 290.70 | 290.43 | 292.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:45:00 | 288.75 | 290.34 | 292.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 290.00 | 290.75 | 291.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 289.65 | 290.31 | 291.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 304.15 | 294.01 | 292.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 304.15 | 294.01 | 292.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 305.10 | 296.23 | 293.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 307.80 | 309.08 | 304.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 15:00:00 | 307.80 | 309.08 | 304.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 304.35 | 307.95 | 304.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 09:45:00 | 305.45 | 307.95 | 304.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 307.25 | 307.81 | 304.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:00:00 | 307.90 | 307.83 | 304.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:30:00 | 307.60 | 307.85 | 306.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 307.80 | 307.84 | 306.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 14:15:00 | 302.50 | 306.30 | 306.06 | SL hit (close<static) qty=1.00 sl=303.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 15:15:00 | 303.20 | 305.68 | 305.80 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 314.45 | 307.44 | 306.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 315.20 | 308.99 | 307.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 313.15 | 317.14 | 314.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 313.15 | 317.14 | 314.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 313.15 | 317.14 | 314.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 313.15 | 317.14 | 314.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 316.05 | 316.92 | 314.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 320.90 | 316.92 | 314.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 321.20 | 327.17 | 327.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 321.20 | 327.17 | 327.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 14:15:00 | 320.40 | 323.11 | 325.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 10:15:00 | 322.75 | 322.01 | 323.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 10:45:00 | 322.45 | 322.01 | 323.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 320.70 | 321.75 | 323.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:45:00 | 319.50 | 320.97 | 322.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 318.20 | 320.65 | 322.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 323.80 | 321.28 | 322.61 | SL hit (close>static) qty=1.00 sl=323.75 alert=retest2 |

### Cycle 157 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 308.40 | 306.41 | 306.14 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 11:15:00 | 303.50 | 305.85 | 305.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 302.75 | 305.23 | 305.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 15:15:00 | 305.00 | 304.70 | 305.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 306.75 | 304.70 | 305.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 305.90 | 304.94 | 305.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 308.90 | 304.94 | 305.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 305.00 | 304.99 | 305.27 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 307.00 | 305.70 | 305.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 309.85 | 306.53 | 305.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 306.20 | 306.47 | 305.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 10:15:00 | 306.20 | 306.47 | 305.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 306.20 | 306.47 | 305.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 306.20 | 306.47 | 305.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 306.70 | 306.51 | 306.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:15:00 | 307.05 | 306.51 | 306.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:45:00 | 307.35 | 306.81 | 306.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 308.55 | 306.82 | 306.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 10:15:00 | 337.76 | 327.39 | 319.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 319.95 | 324.54 | 324.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 318.20 | 322.33 | 323.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 319.30 | 317.99 | 320.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 319.30 | 317.99 | 320.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 319.30 | 317.99 | 320.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 319.30 | 317.99 | 320.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 320.50 | 318.49 | 320.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 320.00 | 318.49 | 320.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 321.05 | 319.01 | 320.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 322.80 | 319.01 | 320.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 321.15 | 319.43 | 320.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 321.80 | 319.43 | 320.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 319.40 | 319.43 | 320.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 319.45 | 319.43 | 320.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 332.50 | 321.96 | 321.25 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 319.60 | 321.93 | 322.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 313.10 | 318.25 | 320.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 314.85 | 314.29 | 316.71 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:15:00 | 311.10 | 314.29 | 316.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 317.65 | 315.04 | 316.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 317.65 | 315.04 | 316.64 | SL hit (close>ema400) qty=1.00 sl=316.64 alert=retest1 |

### Cycle 163 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 318.55 | 317.21 | 317.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 321.15 | 318.24 | 317.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 13:15:00 | 318.50 | 318.52 | 318.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 318.50 | 318.52 | 318.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 318.50 | 318.52 | 318.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 317.95 | 318.52 | 318.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 318.00 | 318.41 | 318.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 318.00 | 318.41 | 318.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 318.00 | 318.33 | 318.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 319.65 | 318.33 | 318.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 314.90 | 317.65 | 317.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 314.90 | 317.65 | 317.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 313.50 | 315.96 | 316.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 316.70 | 314.30 | 315.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 316.70 | 314.30 | 315.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 316.70 | 314.30 | 315.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 316.70 | 314.30 | 315.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 316.95 | 314.83 | 315.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 315.90 | 315.03 | 315.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 316.05 | 316.19 | 316.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 322.65 | 317.46 | 316.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 322.65 | 317.46 | 316.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 13:15:00 | 326.90 | 321.58 | 319.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 331.10 | 331.17 | 327.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 331.10 | 331.17 | 327.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 325.85 | 329.76 | 327.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 325.85 | 329.76 | 327.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 326.40 | 329.09 | 327.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:30:00 | 325.85 | 329.09 | 327.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 328.00 | 328.87 | 327.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 323.60 | 328.87 | 327.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 324.35 | 327.96 | 327.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 324.85 | 327.96 | 327.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 322.50 | 326.87 | 326.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 322.50 | 326.87 | 326.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 321.20 | 325.74 | 326.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 319.75 | 324.54 | 325.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 318.65 | 318.48 | 320.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 320.10 | 318.25 | 319.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 320.10 | 318.25 | 319.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:30:00 | 320.60 | 318.25 | 319.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 320.90 | 318.78 | 319.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 321.85 | 319.37 | 319.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 320.30 | 319.56 | 320.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:30:00 | 318.20 | 319.72 | 320.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 321.00 | 320.15 | 320.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 321.00 | 320.15 | 320.15 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 11:15:00 | 320.00 | 320.15 | 320.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 12:15:00 | 316.00 | 319.32 | 319.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 11:15:00 | 320.00 | 318.53 | 319.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 320.00 | 318.53 | 319.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 320.00 | 318.53 | 319.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 320.00 | 318.53 | 319.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 321.40 | 319.10 | 319.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 321.40 | 319.10 | 319.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 321.60 | 319.60 | 319.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 322.00 | 320.08 | 319.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 326.50 | 327.17 | 324.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 326.05 | 327.17 | 324.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 325.40 | 326.82 | 324.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 324.80 | 326.82 | 324.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 324.40 | 326.33 | 324.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 324.40 | 326.33 | 324.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 324.50 | 325.97 | 324.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 320.00 | 325.97 | 324.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 324.50 | 325.67 | 324.74 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 12:15:00 | 322.25 | 324.03 | 324.14 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 328.10 | 324.63 | 324.38 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 326.45 | 328.26 | 328.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 322.60 | 327.13 | 327.75 | Break + close below crossover candle low |

### Cycle 173 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 349.40 | 326.65 | 325.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 10:15:00 | 354.35 | 332.19 | 327.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 14:15:00 | 336.90 | 338.76 | 332.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 336.90 | 338.76 | 332.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 329.30 | 336.07 | 332.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 329.30 | 336.07 | 332.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 331.60 | 335.17 | 332.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 329.60 | 335.17 | 332.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 330.00 | 334.14 | 332.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 330.00 | 334.14 | 332.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 330.00 | 333.31 | 332.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 15:00:00 | 337.35 | 333.65 | 332.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 335.45 | 335.88 | 335.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 335.45 | 335.88 | 335.93 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 338.00 | 336.30 | 336.12 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 333.95 | 335.94 | 335.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 329.10 | 334.57 | 335.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 15:15:00 | 332.00 | 331.48 | 333.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:45:00 | 332.35 | 331.42 | 333.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 330.85 | 331.31 | 332.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 332.80 | 331.31 | 332.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 331.50 | 330.97 | 332.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 331.15 | 330.97 | 332.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 332.90 | 331.25 | 332.07 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 333.35 | 332.51 | 332.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 335.65 | 333.40 | 332.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 344.25 | 344.28 | 340.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 15:15:00 | 339.55 | 343.03 | 340.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 339.55 | 343.03 | 340.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 334.25 | 343.03 | 340.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 334.65 | 341.36 | 340.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 335.00 | 341.36 | 340.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 336.50 | 340.38 | 339.93 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 336.10 | 339.53 | 339.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 334.00 | 338.42 | 339.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 14:15:00 | 334.00 | 333.69 | 335.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 334.00 | 333.69 | 335.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 334.00 | 333.69 | 335.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 334.00 | 333.69 | 335.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 337.15 | 334.38 | 335.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 348.00 | 334.38 | 335.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 346.10 | 336.73 | 336.71 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 330.90 | 336.39 | 336.68 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 342.95 | 337.70 | 337.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 15:15:00 | 344.70 | 339.10 | 337.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 337.45 | 339.29 | 338.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 337.45 | 339.29 | 338.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 337.45 | 339.29 | 338.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 337.45 | 339.29 | 338.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 337.75 | 338.98 | 338.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:30:00 | 337.60 | 338.98 | 338.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 340.15 | 339.22 | 338.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 342.40 | 340.07 | 338.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 332.20 | 338.25 | 338.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 332.20 | 338.25 | 338.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 329.30 | 336.46 | 337.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 333.85 | 333.16 | 335.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 329.85 | 333.16 | 335.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 326.60 | 331.84 | 334.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 325.80 | 330.05 | 331.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:00:00 | 325.60 | 328.52 | 330.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:00:00 | 325.45 | 327.44 | 329.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 15:15:00 | 309.51 | 313.23 | 315.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 15:15:00 | 309.32 | 313.23 | 315.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 309.18 | 311.44 | 314.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 306.60 | 305.33 | 308.91 | SL hit (close>ema200) qty=0.50 sl=305.33 alert=retest2 |

### Cycle 183 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 320.05 | 311.05 | 310.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 323.95 | 319.17 | 316.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 320.00 | 320.30 | 317.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:00:00 | 320.00 | 320.30 | 317.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 324.00 | 320.93 | 318.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 319.40 | 320.93 | 318.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 318.05 | 320.50 | 318.87 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 315.50 | 317.87 | 317.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 310.00 | 315.55 | 316.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 310.40 | 309.16 | 312.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 310.40 | 309.16 | 312.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 310.40 | 309.16 | 312.11 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 316.40 | 312.80 | 312.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 320.10 | 314.26 | 313.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 327.05 | 327.06 | 323.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:00:00 | 327.05 | 327.06 | 323.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 326.25 | 326.89 | 326.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 326.20 | 326.89 | 326.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 327.50 | 327.26 | 326.35 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 324.25 | 325.94 | 326.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 323.75 | 325.51 | 325.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 325.15 | 321.98 | 323.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 325.15 | 321.98 | 323.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 325.15 | 321.98 | 323.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 325.15 | 321.98 | 323.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 324.30 | 322.44 | 323.25 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 326.90 | 323.75 | 323.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 14:15:00 | 327.40 | 324.48 | 324.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 09:15:00 | 324.10 | 324.61 | 324.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 324.10 | 324.61 | 324.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 324.10 | 324.61 | 324.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:15:00 | 323.45 | 324.61 | 324.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 323.20 | 324.33 | 324.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 323.40 | 324.33 | 324.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 324.65 | 324.39 | 324.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 323.30 | 324.39 | 324.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 324.40 | 324.78 | 324.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 324.30 | 324.78 | 324.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 325.90 | 325.00 | 324.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 325.50 | 325.00 | 324.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 325.50 | 325.10 | 324.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 323.50 | 325.10 | 324.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 325.65 | 325.21 | 324.72 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 322.40 | 324.38 | 324.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 321.70 | 323.84 | 324.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 324.00 | 322.32 | 323.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 324.00 | 322.32 | 323.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 324.00 | 322.32 | 323.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 324.00 | 322.32 | 323.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 323.95 | 322.65 | 323.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 323.10 | 322.81 | 323.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 323.10 | 322.87 | 323.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:30:00 | 323.15 | 322.89 | 323.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:30:00 | 323.05 | 322.92 | 323.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 323.60 | 323.06 | 323.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 326.95 | 323.06 | 323.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 332.80 | 325.01 | 324.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 332.80 | 325.01 | 324.12 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 321.65 | 324.64 | 324.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 321.10 | 323.94 | 324.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 15:15:00 | 320.20 | 320.03 | 321.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 321.00 | 320.03 | 321.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 322.55 | 320.53 | 321.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:15:00 | 323.30 | 320.53 | 321.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 321.75 | 320.77 | 321.62 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 333.15 | 323.32 | 322.42 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 327.25 | 329.19 | 329.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 326.75 | 327.91 | 328.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 324.55 | 323.61 | 325.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:15:00 | 320.35 | 323.12 | 324.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 13:30:00 | 320.30 | 322.15 | 324.00 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 323.00 | 322.32 | 323.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 323.00 | 322.32 | 323.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 321.80 | 322.29 | 323.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 318.65 | 320.93 | 322.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 317.15 | 319.74 | 321.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 14:15:00 | 304.33 | 310.63 | 314.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 14:15:00 | 304.28 | 310.63 | 314.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 14:15:00 | 302.72 | 310.63 | 314.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 308.80 | 308.74 | 312.12 | SL hit (close>ema200) qty=0.50 sl=308.74 alert=retest1 |

### Cycle 193 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 283.65 | 278.91 | 278.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 288.20 | 282.09 | 280.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 288.05 | 288.17 | 285.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 15:00:00 | 288.05 | 288.17 | 285.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 286.40 | 287.56 | 286.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 286.00 | 287.56 | 286.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 286.50 | 287.35 | 286.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 286.50 | 287.35 | 286.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 286.50 | 287.18 | 286.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 293.95 | 288.98 | 287.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 290.85 | 291.01 | 289.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 287.30 | 288.90 | 288.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 287.30 | 288.90 | 288.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 285.40 | 288.20 | 288.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 15:15:00 | 288.85 | 287.79 | 288.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 15:15:00 | 288.85 | 287.79 | 288.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 288.85 | 287.79 | 288.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 292.40 | 287.79 | 288.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 290.65 | 288.36 | 288.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 294.40 | 288.36 | 288.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 284.85 | 287.66 | 288.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 284.25 | 286.96 | 287.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:15:00 | 283.85 | 286.14 | 287.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:00:00 | 283.95 | 285.70 | 286.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 283.35 | 284.36 | 285.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 287.40 | 282.68 | 284.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 287.40 | 282.68 | 284.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 287.60 | 283.66 | 284.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 287.60 | 283.66 | 284.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 290.20 | 285.67 | 285.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 290.20 | 285.67 | 285.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 292.90 | 287.12 | 285.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 298.20 | 298.23 | 293.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 10:15:00 | 295.95 | 298.23 | 293.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 290.60 | 296.70 | 293.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 290.60 | 296.70 | 293.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 292.70 | 295.90 | 293.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:00:00 | 293.90 | 295.50 | 293.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 284.85 | 292.72 | 292.51 | SL hit (close<static) qty=1.00 sl=290.30 alert=retest2 |

### Cycle 196 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 287.10 | 291.60 | 292.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 284.45 | 289.15 | 290.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 287.05 | 287.04 | 288.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 285.15 | 287.04 | 288.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 268.40 | 277.74 | 281.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 266.75 | 271.11 | 275.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 267.20 | 270.35 | 274.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 266.65 | 269.58 | 274.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:15:00 | 253.41 | 261.06 | 267.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:15:00 | 253.84 | 261.06 | 267.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:15:00 | 253.32 | 261.06 | 267.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 255.95 | 255.85 | 260.97 | SL hit (close>ema200) qty=0.50 sl=255.85 alert=retest2 |

### Cycle 197 — BUY (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 11:15:00 | 257.45 | 251.29 | 250.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 262.70 | 254.51 | 252.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 260.85 | 261.62 | 258.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 260.85 | 261.62 | 258.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 260.85 | 261.62 | 258.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 258.35 | 261.62 | 258.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 258.55 | 261.18 | 259.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 258.60 | 261.18 | 259.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 256.60 | 260.26 | 259.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 256.70 | 260.26 | 259.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 257.95 | 259.33 | 259.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 257.95 | 259.33 | 259.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 256.45 | 258.75 | 258.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 254.00 | 257.80 | 258.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 254.80 | 254.37 | 255.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 254.80 | 254.37 | 255.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 255.35 | 254.28 | 255.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 255.35 | 254.28 | 255.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 259.65 | 255.36 | 255.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 259.65 | 255.36 | 255.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 256.25 | 255.53 | 255.78 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 258.10 | 256.05 | 255.99 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 10:15:00 | 255.45 | 255.93 | 255.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 11:15:00 | 251.85 | 255.11 | 255.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 15:15:00 | 255.05 | 254.57 | 255.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 15:15:00 | 255.05 | 254.57 | 255.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 255.05 | 254.57 | 255.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 255.15 | 254.57 | 255.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 256.35 | 254.92 | 255.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:45:00 | 253.25 | 254.64 | 255.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 256.94 | 249.35 | 248.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 256.94 | 249.35 | 248.73 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 10:15:00 | 247.33 | 249.66 | 249.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 244.00 | 246.68 | 248.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 239.77 | 237.90 | 240.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 239.77 | 237.90 | 240.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 239.77 | 237.90 | 240.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 239.77 | 237.90 | 240.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 237.85 | 236.30 | 237.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 237.85 | 236.30 | 237.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 238.56 | 236.75 | 237.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 238.47 | 236.75 | 237.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 236.49 | 236.70 | 237.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 237.81 | 236.70 | 237.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 238.08 | 236.98 | 237.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 237.05 | 236.98 | 237.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 239.10 | 237.40 | 237.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 239.10 | 237.40 | 237.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 238.32 | 237.96 | 237.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 238.32 | 237.96 | 237.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 238.10 | 237.99 | 237.98 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 235.23 | 237.44 | 237.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 12:15:00 | 234.20 | 236.10 | 236.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 226.18 | 225.46 | 227.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 226.18 | 225.46 | 227.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 226.18 | 225.46 | 227.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 227.33 | 225.46 | 227.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 230.00 | 226.20 | 227.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 230.00 | 226.20 | 227.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 229.99 | 226.96 | 227.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 228.76 | 226.96 | 227.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 230.02 | 227.57 | 227.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 230.24 | 227.57 | 227.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 233.00 | 228.66 | 228.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 234.17 | 230.45 | 229.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 251.00 | 251.81 | 248.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 15:15:00 | 251.00 | 251.81 | 248.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 251.00 | 251.81 | 248.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 249.09 | 251.81 | 248.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 251.69 | 251.79 | 248.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 253.19 | 251.35 | 249.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 253.35 | 251.75 | 249.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 247.25 | 248.90 | 249.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 247.25 | 248.90 | 249.11 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 255.73 | 249.57 | 249.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 261.95 | 256.00 | 253.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 256.25 | 257.17 | 254.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 256.25 | 257.17 | 254.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 256.25 | 257.17 | 254.41 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 244.20 | 252.03 | 252.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 238.20 | 249.27 | 251.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 229.75 | 229.32 | 235.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 225.90 | 228.76 | 232.20 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:45:00 | 225.45 | 227.83 | 231.46 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 223.20 | 220.98 | 223.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 223.20 | 220.98 | 223.10 | SL hit (close>ema400) qty=1.00 sl=223.10 alert=retest1 |

### Cycle 209 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 227.15 | 223.81 | 223.52 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 221.80 | 224.04 | 224.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 221.45 | 222.73 | 223.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 184.65 | 183.61 | 188.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:30:00 | 184.60 | 183.95 | 188.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 189.00 | 185.43 | 188.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:45:00 | 189.00 | 185.43 | 188.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 188.75 | 186.10 | 188.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 187.30 | 188.00 | 188.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 187.95 | 185.46 | 186.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 187.60 | 186.55 | 186.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 188.80 | 187.18 | 187.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 188.80 | 187.18 | 187.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 189.72 | 188.01 | 187.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 10:15:00 | 188.00 | 188.01 | 187.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 10:15:00 | 188.00 | 188.01 | 187.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 188.00 | 188.01 | 187.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:30:00 | 189.50 | 188.01 | 187.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 188.00 | 188.01 | 187.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 188.00 | 188.01 | 187.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 188.21 | 188.05 | 187.62 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 183.69 | 187.17 | 187.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 182.32 | 186.20 | 186.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 187.00 | 185.17 | 186.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 187.00 | 185.17 | 186.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 187.00 | 185.17 | 186.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 187.00 | 185.17 | 186.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 188.00 | 185.73 | 186.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 189.55 | 185.73 | 186.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 187.02 | 185.99 | 186.31 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 189.10 | 186.61 | 186.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 190.32 | 187.35 | 186.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 188.00 | 188.05 | 187.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 190.98 | 188.05 | 187.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 192.63 | 188.96 | 187.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:00:00 | 195.21 | 190.38 | 188.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 09:15:00 | 214.73 | 200.20 | 194.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 215.06 | 219.46 | 219.63 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 221.49 | 219.86 | 219.73 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 218.43 | 219.60 | 219.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 216.46 | 218.88 | 219.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 13:15:00 | 215.17 | 214.62 | 216.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 215.17 | 214.62 | 216.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 218.07 | 215.31 | 216.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 218.07 | 215.31 | 216.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 219.00 | 216.05 | 216.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 219.54 | 216.05 | 216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 220.00 | 217.40 | 217.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 220.58 | 219.03 | 218.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 218.95 | 219.02 | 218.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 218.95 | 219.02 | 218.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 218.95 | 219.02 | 218.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 216.54 | 219.02 | 218.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 219.25 | 219.33 | 218.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 219.25 | 219.33 | 218.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 221.40 | 219.74 | 218.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 216.60 | 219.74 | 218.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 217.64 | 219.32 | 218.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 218.84 | 219.32 | 218.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 217.64 | 218.98 | 218.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 217.64 | 218.98 | 218.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 217.62 | 218.52 | 218.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 216.71 | 218.15 | 218.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 12:15:00 | 220.32 | 217.95 | 218.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 12:15:00 | 220.32 | 217.95 | 218.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 220.32 | 217.95 | 218.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 219.80 | 217.95 | 218.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 218.76 | 218.11 | 218.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 218.20 | 218.11 | 218.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 219.81 | 218.45 | 218.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 219.81 | 218.45 | 218.29 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 215.25 | 217.84 | 218.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 213.50 | 215.03 | 216.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 15:15:00 | 215.00 | 214.90 | 215.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 09:15:00 | 214.30 | 214.90 | 215.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 214.99 | 214.92 | 215.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 215.00 | 214.92 | 215.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 189.74 | 188.10 | 191.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 190.65 | 188.10 | 191.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 191.60 | 188.82 | 191.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 191.60 | 188.82 | 191.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 192.84 | 189.62 | 191.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 192.84 | 189.62 | 191.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 190.96 | 189.89 | 191.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 185.50 | 189.89 | 191.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 13:15:00 | 176.22 | 182.77 | 186.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 174.47 | 173.93 | 177.86 | SL hit (close>ema200) qty=0.50 sl=173.93 alert=retest2 |

### Cycle 221 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 165.10 | 164.29 | 164.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 168.43 | 165.12 | 164.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 166.78 | 169.56 | 167.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 166.78 | 169.56 | 167.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 166.78 | 169.56 | 167.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 167.44 | 169.56 | 167.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 165.40 | 168.73 | 167.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 165.40 | 168.73 | 167.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 163.08 | 166.32 | 166.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 162.66 | 164.96 | 165.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 157.00 | 155.99 | 158.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 157.00 | 155.99 | 158.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 158.35 | 156.46 | 158.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 158.35 | 156.46 | 158.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 158.41 | 156.85 | 158.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 158.41 | 156.85 | 158.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 158.18 | 157.12 | 158.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:15:00 | 158.62 | 157.12 | 158.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 158.62 | 157.42 | 158.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 163.78 | 157.42 | 158.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 162.51 | 158.44 | 158.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 162.70 | 158.44 | 158.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 162.90 | 159.33 | 159.22 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 153.73 | 158.33 | 158.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 152.16 | 155.85 | 157.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 152.26 | 152.18 | 154.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 152.26 | 152.18 | 154.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 152.26 | 152.18 | 154.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 150.56 | 152.18 | 154.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 150.43 | 151.83 | 154.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 149.24 | 153.45 | 154.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 143.03 | 151.27 | 153.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 142.91 | 151.27 | 153.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 141.78 | 151.27 | 153.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 151.20 | 150.12 | 151.93 | SL hit (close>ema200) qty=0.50 sl=150.12 alert=retest2 |

### Cycle 225 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 155.93 | 153.10 | 152.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 158.04 | 154.09 | 153.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 162.00 | 163.92 | 161.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 162.00 | 163.92 | 161.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 162.00 | 163.92 | 161.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 160.81 | 163.92 | 161.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 161.74 | 163.25 | 161.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 161.10 | 163.25 | 161.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 162.20 | 163.04 | 161.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 166.46 | 162.70 | 161.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 13:15:00 | 183.11 | 177.03 | 175.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 174.70 | 175.89 | 176.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 173.26 | 175.36 | 175.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 174.35 | 170.23 | 172.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 174.35 | 170.23 | 172.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 174.35 | 170.23 | 172.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 174.35 | 170.23 | 172.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 173.84 | 170.95 | 172.31 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 175.54 | 173.22 | 173.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 180.65 | 175.25 | 174.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 09:15:00 | 198.34 | 200.46 | 193.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:00:00 | 198.34 | 200.46 | 193.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 194.70 | 197.80 | 194.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 194.18 | 197.80 | 194.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 194.78 | 197.19 | 194.23 | EMA400 retest candle locked (from upside) |

### Cycle 228 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 187.55 | 192.89 | 192.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 185.68 | 191.45 | 192.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 187.72 | 185.86 | 187.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 187.72 | 185.86 | 187.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 187.72 | 185.86 | 187.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 187.43 | 185.86 | 187.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 188.01 | 186.29 | 187.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 187.77 | 186.29 | 187.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 186.78 | 186.39 | 187.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 185.12 | 186.54 | 187.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 10:00:00 | 184.18 | 186.54 | 187.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 10:15:00 | 298.34 | 2024-04-23 14:15:00 | 283.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-22 11:45:00 | 298.12 | 2024-04-23 14:15:00 | 283.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 10:15:00 | 298.34 | 2024-04-26 09:15:00 | 280.60 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2024-04-22 11:45:00 | 298.12 | 2024-04-26 09:15:00 | 280.60 | STOP_HIT | 0.50 | 5.88% |
| BUY | retest2 | 2024-05-08 12:45:00 | 281.98 | 2024-05-09 12:15:00 | 277.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-05-09 10:00:00 | 280.89 | 2024-05-09 12:15:00 | 277.25 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-05-16 10:45:00 | 275.00 | 2024-05-16 14:15:00 | 277.27 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-05-16 13:45:00 | 275.00 | 2024-05-16 14:15:00 | 277.27 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-05-21 09:15:00 | 278.66 | 2024-05-24 14:15:00 | 278.30 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-05-21 10:00:00 | 278.44 | 2024-05-24 14:15:00 | 278.30 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-06-07 09:15:00 | 293.24 | 2024-06-20 09:15:00 | 322.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-27 15:15:00 | 306.80 | 2024-06-28 09:15:00 | 312.38 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-07-09 11:15:00 | 317.90 | 2024-07-22 09:15:00 | 302.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 12:00:00 | 317.00 | 2024-07-22 09:15:00 | 301.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 11:15:00 | 317.90 | 2024-07-22 11:15:00 | 305.54 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2024-07-09 12:00:00 | 317.00 | 2024-07-22 11:15:00 | 305.54 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2024-08-16 11:15:00 | 308.00 | 2024-08-16 12:15:00 | 313.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-08-27 14:15:00 | 325.00 | 2024-08-28 09:15:00 | 322.37 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-08-28 09:15:00 | 325.16 | 2024-08-28 09:15:00 | 322.37 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-09-26 09:15:00 | 388.35 | 2024-09-30 09:15:00 | 363.45 | STOP_HIT | 1.00 | -6.41% |
| BUY | retest2 | 2024-09-27 15:15:00 | 373.25 | 2024-09-30 09:15:00 | 363.45 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-10-04 14:30:00 | 345.30 | 2024-10-09 15:15:00 | 353.65 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-10-07 09:15:00 | 348.90 | 2024-10-09 15:15:00 | 353.65 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-10-07 10:30:00 | 349.80 | 2024-10-09 15:15:00 | 353.65 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-10-07 13:15:00 | 348.80 | 2024-10-09 15:15:00 | 353.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-11-19 09:30:00 | 304.55 | 2024-11-19 15:15:00 | 311.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-11-19 10:30:00 | 303.90 | 2024-11-19 15:15:00 | 311.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-11-21 09:15:00 | 299.35 | 2024-11-25 11:15:00 | 306.10 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-12-03 09:30:00 | 329.80 | 2024-12-10 09:15:00 | 331.10 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-12-17 10:15:00 | 311.95 | 2024-12-20 15:15:00 | 312.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-12-18 10:00:00 | 311.40 | 2024-12-20 15:15:00 | 312.90 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-12-18 11:30:00 | 312.35 | 2024-12-20 15:15:00 | 312.90 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-12-20 15:00:00 | 311.05 | 2024-12-20 15:15:00 | 312.90 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-12-27 09:15:00 | 324.05 | 2024-12-31 13:15:00 | 325.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-12-27 10:30:00 | 326.30 | 2024-12-31 13:15:00 | 325.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-12-31 10:30:00 | 325.05 | 2024-12-31 13:15:00 | 325.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest1 | 2025-01-03 09:15:00 | 359.60 | 2025-01-07 15:15:00 | 352.90 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-01-10 13:45:00 | 331.30 | 2025-01-14 09:15:00 | 314.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:45:00 | 331.30 | 2025-01-14 10:15:00 | 325.20 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2025-01-13 09:15:00 | 318.90 | 2025-01-15 14:15:00 | 333.00 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2025-01-15 09:15:00 | 325.00 | 2025-01-15 14:15:00 | 333.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-01-21 10:15:00 | 312.30 | 2025-01-27 09:15:00 | 296.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 12:00:00 | 313.85 | 2025-01-27 09:15:00 | 298.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 14:30:00 | 312.20 | 2025-01-27 09:15:00 | 296.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 310.70 | 2025-01-27 09:15:00 | 296.97 | PARTIAL | 0.50 | 4.42% |
| SELL | retest2 | 2025-01-22 13:30:00 | 308.25 | 2025-01-27 09:15:00 | 296.50 | PARTIAL | 0.50 | 3.81% |
| SELL | retest2 | 2025-01-23 10:15:00 | 312.60 | 2025-01-27 13:15:00 | 295.16 | PARTIAL | 0.50 | 5.58% |
| SELL | retest2 | 2025-01-23 10:45:00 | 312.10 | 2025-01-27 13:15:00 | 292.84 | PARTIAL | 0.50 | 6.17% |
| SELL | retest2 | 2025-01-21 10:15:00 | 312.30 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-01-21 12:00:00 | 313.85 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2025-01-21 14:30:00 | 312.20 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2025-01-22 09:15:00 | 310.70 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-01-22 13:30:00 | 308.25 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2025-01-23 10:15:00 | 312.60 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 5.49% |
| SELL | retest2 | 2025-01-23 10:45:00 | 312.10 | 2025-01-29 09:15:00 | 295.45 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2025-02-13 12:30:00 | 302.30 | 2025-02-17 09:15:00 | 287.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 302.25 | 2025-02-17 09:15:00 | 287.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 302.05 | 2025-02-17 09:15:00 | 286.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:30:00 | 302.30 | 2025-02-17 15:15:00 | 297.75 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2025-02-13 13:00:00 | 302.25 | 2025-02-17 15:15:00 | 297.75 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2025-02-13 13:30:00 | 302.05 | 2025-02-17 15:15:00 | 297.75 | STOP_HIT | 0.50 | 1.42% |
| SELL | retest2 | 2025-03-03 09:15:00 | 305.95 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-03-12 09:15:00 | 322.80 | 2025-03-13 09:15:00 | 306.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:45:00 | 322.70 | 2025-03-13 09:15:00 | 306.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:15:00 | 322.80 | 2025-03-18 09:15:00 | 302.00 | STOP_HIT | 0.50 | 6.44% |
| SELL | retest2 | 2025-03-12 09:45:00 | 322.70 | 2025-03-18 09:15:00 | 302.00 | STOP_HIT | 0.50 | 6.41% |
| SELL | retest1 | 2025-03-26 09:15:00 | 293.80 | 2025-03-27 12:15:00 | 299.45 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-03-28 10:45:00 | 295.40 | 2025-04-02 11:15:00 | 298.80 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-04-01 09:45:00 | 293.60 | 2025-04-02 11:15:00 | 298.80 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-04-08 11:45:00 | 288.75 | 2025-04-11 09:15:00 | 304.15 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-04-09 09:15:00 | 290.00 | 2025-04-11 09:15:00 | 304.15 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2025-04-09 09:45:00 | 289.65 | 2025-04-11 09:15:00 | 304.15 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest2 | 2025-04-16 12:00:00 | 307.90 | 2025-04-17 14:15:00 | 302.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-04-17 09:30:00 | 307.60 | 2025-04-17 14:15:00 | 302.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-04-17 11:00:00 | 307.80 | 2025-04-17 14:15:00 | 302.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-04-23 11:15:00 | 320.90 | 2025-04-25 15:15:00 | 321.20 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-04-29 13:45:00 | 319.50 | 2025-04-30 09:15:00 | 323.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-04-30 09:15:00 | 318.20 | 2025-04-30 09:15:00 | 323.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-04-30 13:00:00 | 318.80 | 2025-05-07 13:15:00 | 302.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 09:15:00 | 316.15 | 2025-05-07 13:15:00 | 300.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 09:15:00 | 311.50 | 2025-05-08 15:15:00 | 297.44 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2025-05-07 12:45:00 | 313.10 | 2025-05-09 09:15:00 | 295.93 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2025-05-07 15:00:00 | 310.50 | 2025-05-09 09:15:00 | 294.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 13:00:00 | 318.80 | 2025-05-09 15:15:00 | 304.80 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2025-05-05 09:15:00 | 316.15 | 2025-05-09 15:15:00 | 304.80 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-05-07 09:15:00 | 311.50 | 2025-05-09 15:15:00 | 304.80 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2025-05-07 12:45:00 | 313.10 | 2025-05-09 15:15:00 | 304.80 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-05-07 15:00:00 | 310.50 | 2025-05-09 15:15:00 | 304.80 | STOP_HIT | 0.50 | 1.84% |
| BUY | retest2 | 2025-05-15 12:15:00 | 307.05 | 2025-05-19 10:15:00 | 337.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-15 12:45:00 | 307.35 | 2025-05-19 10:15:00 | 338.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-16 09:15:00 | 308.55 | 2025-05-20 09:15:00 | 339.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-05-29 10:15:00 | 311.10 | 2025-05-29 11:15:00 | 317.65 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-06-03 09:15:00 | 319.65 | 2025-06-03 09:15:00 | 314.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-04 11:45:00 | 315.90 | 2025-06-05 09:15:00 | 322.65 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-06-04 15:15:00 | 316.05 | 2025-06-05 09:15:00 | 322.65 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-06-13 12:30:00 | 318.20 | 2025-06-13 15:15:00 | 321.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-07 15:00:00 | 337.35 | 2025-07-10 13:15:00 | 335.45 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-24 14:30:00 | 342.40 | 2025-07-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-07-29 15:00:00 | 325.80 | 2025-08-05 15:15:00 | 309.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:00:00 | 325.60 | 2025-08-05 15:15:00 | 309.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 13:00:00 | 325.45 | 2025-08-06 09:15:00 | 309.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 15:00:00 | 325.80 | 2025-08-07 10:15:00 | 306.60 | STOP_HIT | 0.50 | 5.89% |
| SELL | retest2 | 2025-07-30 10:00:00 | 325.60 | 2025-08-07 10:15:00 | 306.60 | STOP_HIT | 0.50 | 5.84% |
| SELL | retest2 | 2025-07-30 13:00:00 | 325.45 | 2025-08-07 10:15:00 | 306.60 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2025-09-03 12:15:00 | 323.10 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-03 13:00:00 | 323.10 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-03 13:30:00 | 323.15 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-09-03 14:30:00 | 323.05 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest1 | 2025-09-18 11:15:00 | 320.35 | 2025-09-23 14:15:00 | 304.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-18 13:30:00 | 320.30 | 2025-09-23 14:15:00 | 304.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:30:00 | 318.65 | 2025-09-23 14:15:00 | 302.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-18 11:15:00 | 320.35 | 2025-09-24 11:15:00 | 308.80 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest1 | 2025-09-18 13:30:00 | 320.30 | 2025-09-24 11:15:00 | 308.80 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-09-19 12:30:00 | 318.65 | 2025-09-24 11:15:00 | 308.80 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-09-19 15:00:00 | 317.15 | 2025-09-26 11:15:00 | 301.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:00:00 | 317.15 | 2025-10-01 12:15:00 | 294.15 | STOP_HIT | 0.50 | 7.25% |
| BUY | retest2 | 2025-10-21 13:45:00 | 293.95 | 2025-10-24 12:15:00 | 287.30 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-10-23 15:15:00 | 290.85 | 2025-10-24 12:15:00 | 287.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-27 11:45:00 | 284.25 | 2025-10-29 14:15:00 | 290.20 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-10-27 14:15:00 | 283.85 | 2025-10-29 14:15:00 | 290.20 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-10-27 15:00:00 | 283.95 | 2025-10-29 14:15:00 | 290.20 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-10-28 13:30:00 | 283.35 | 2025-10-29 14:15:00 | 290.20 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-10-31 13:00:00 | 293.90 | 2025-10-31 14:15:00 | 284.85 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-10 09:45:00 | 266.75 | 2025-11-11 11:15:00 | 253.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 10:30:00 | 267.20 | 2025-11-11 11:15:00 | 253.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 11:45:00 | 266.65 | 2025-11-11 11:15:00 | 253.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 09:45:00 | 266.75 | 2025-11-12 11:15:00 | 255.95 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-11-10 10:30:00 | 267.20 | 2025-11-12 11:15:00 | 255.95 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-11-10 11:45:00 | 266.65 | 2025-11-12 11:15:00 | 255.95 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-11-27 12:45:00 | 253.25 | 2025-12-03 14:15:00 | 256.94 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-12-29 14:15:00 | 253.19 | 2025-12-30 14:15:00 | 247.25 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-12-29 15:00:00 | 253.35 | 2025-12-30 14:15:00 | 247.25 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest1 | 2026-01-08 10:15:00 | 225.90 | 2026-01-12 13:15:00 | 223.20 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest1 | 2026-01-08 10:45:00 | 225.45 | 2026-01-12 13:15:00 | 223.20 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 187.30 | 2026-01-30 13:15:00 | 188.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-30 10:15:00 | 187.95 | 2026-01-30 13:15:00 | 188.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-30 12:15:00 | 187.60 | 2026-01-30 13:15:00 | 188.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-04 12:00:00 | 195.21 | 2026-02-05 09:15:00 | 214.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-23 14:15:00 | 218.20 | 2026-02-23 14:15:00 | 219.81 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-03-09 09:15:00 | 185.50 | 2026-03-09 13:15:00 | 176.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 185.50 | 2026-03-11 10:15:00 | 174.47 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2026-04-01 10:15:00 | 150.56 | 2026-04-02 09:15:00 | 143.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 11:00:00 | 150.43 | 2026-04-02 09:15:00 | 142.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-02 09:15:00 | 149.24 | 2026-04-02 09:15:00 | 141.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 150.56 | 2026-04-02 12:15:00 | 151.20 | STOP_HIT | 0.50 | -0.43% |
| SELL | retest2 | 2026-04-01 11:00:00 | 150.43 | 2026-04-02 12:15:00 | 151.20 | STOP_HIT | 0.50 | -0.51% |
| SELL | retest2 | 2026-04-02 09:15:00 | 149.24 | 2026-04-02 12:15:00 | 151.20 | STOP_HIT | 0.50 | -1.31% |
| SELL | retest2 | 2026-04-06 09:45:00 | 150.31 | 2026-04-06 12:15:00 | 155.93 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2026-04-10 09:15:00 | 166.46 | 2026-04-21 13:15:00 | 183.11 | TARGET_HIT | 1.00 | 10.00% |

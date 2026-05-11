# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 537.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 251 |
| ALERT1 | 162 |
| ALERT2 | 159 |
| ALERT2_SKIP | 102 |
| ALERT3 | 354 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 130 |
| PARTIAL | 26 |
| TARGET_HIT | 7 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 160 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 93
- **Target hits / Stop hits / Partials:** 7 / 127 / 26
- **Avg / median % per leg:** 0.86% / -0.67%
- **Sum % (uncompounded):** 137.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 13 | 17.6% | 5 | 69 | 0 | -0.53% | -39.0% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.02% | -0.0% |
| BUY @ 3rd Alert (retest2) | 72 | 12 | 16.7% | 5 | 67 | 0 | -0.54% | -38.9% |
| SELL (all) | 86 | 54 | 62.8% | 2 | 58 | 26 | 2.05% | 176.3% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.63% | 10.9% |
| SELL @ 3rd Alert (retest2) | 83 | 51 | 61.4% | 2 | 56 | 25 | 1.99% | 165.4% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 4 | 1 | 2.17% | 10.9% |
| retest2 (combined) | 155 | 63 | 40.6% | 7 | 123 | 25 | 0.82% | 126.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 273.60 | 270.99 | 270.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 09:15:00 | 277.00 | 272.71 | 271.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 09:15:00 | 273.15 | 277.30 | 275.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 273.15 | 277.30 | 275.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 273.15 | 277.30 | 275.15 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 276.90 | 280.34 | 280.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 14:15:00 | 276.20 | 278.70 | 279.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 15:15:00 | 280.75 | 279.11 | 279.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 15:15:00 | 280.75 | 279.11 | 279.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 280.75 | 279.11 | 279.75 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 282.45 | 280.37 | 280.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 14:15:00 | 283.95 | 281.78 | 281.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 09:15:00 | 281.95 | 282.17 | 281.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 281.95 | 282.17 | 281.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 281.95 | 282.17 | 281.35 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 14:15:00 | 278.45 | 280.61 | 280.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 15:15:00 | 277.70 | 280.03 | 280.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 13:15:00 | 280.30 | 279.63 | 280.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 13:15:00 | 280.30 | 279.63 | 280.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 280.30 | 279.63 | 280.11 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 283.90 | 280.50 | 280.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 14:15:00 | 284.30 | 281.33 | 280.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 281.50 | 283.14 | 282.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 281.50 | 283.14 | 282.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 281.50 | 283.14 | 282.36 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 13:15:00 | 291.00 | 292.54 | 292.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 14:15:00 | 290.10 | 292.06 | 292.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 277.00 | 274.73 | 279.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 277.00 | 274.73 | 279.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 277.00 | 274.73 | 279.38 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 12:15:00 | 282.90 | 279.63 | 279.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 15:15:00 | 285.65 | 281.99 | 280.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 10:15:00 | 281.70 | 282.23 | 280.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 11:15:00 | 282.50 | 282.28 | 281.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 282.50 | 282.28 | 281.08 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 10:15:00 | 279.30 | 280.65 | 280.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 11:15:00 | 277.15 | 279.95 | 280.40 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 286.55 | 280.31 | 280.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 10:15:00 | 290.95 | 282.44 | 281.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 288.00 | 289.71 | 286.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 13:15:00 | 287.70 | 289.31 | 286.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 287.70 | 289.31 | 286.93 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 285.20 | 288.60 | 288.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 10:15:00 | 283.15 | 285.96 | 287.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 284.00 | 283.55 | 285.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 284.00 | 283.55 | 285.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 284.00 | 283.55 | 285.27 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 288.10 | 286.07 | 285.93 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 10:15:00 | 284.15 | 285.81 | 285.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 12:15:00 | 282.60 | 284.92 | 285.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 282.25 | 281.69 | 282.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 282.25 | 281.69 | 282.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 282.25 | 281.69 | 282.93 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 289.00 | 282.99 | 282.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 10:15:00 | 289.95 | 284.38 | 283.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 282.00 | 288.48 | 286.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 282.00 | 288.48 | 286.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 282.00 | 288.48 | 286.48 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 14:15:00 | 284.20 | 285.40 | 285.50 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 09:15:00 | 288.45 | 285.92 | 285.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 09:15:00 | 298.15 | 291.49 | 289.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 14:15:00 | 295.00 | 295.38 | 292.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 11:15:00 | 290.35 | 294.19 | 292.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 290.35 | 294.19 | 292.92 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 294.05 | 297.37 | 297.80 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 14:15:00 | 299.20 | 297.75 | 297.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 303.20 | 298.88 | 298.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 10:15:00 | 302.90 | 303.53 | 301.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 298.30 | 302.49 | 301.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 298.30 | 302.49 | 301.12 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 15:15:00 | 301.30 | 303.34 | 303.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 299.10 | 302.49 | 303.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 306.80 | 302.10 | 302.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 306.80 | 302.10 | 302.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 306.80 | 302.10 | 302.36 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 310.95 | 303.87 | 303.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 11:15:00 | 314.50 | 309.10 | 306.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 10:15:00 | 309.65 | 310.92 | 308.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 10:15:00 | 309.65 | 310.92 | 308.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 309.65 | 310.92 | 308.82 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 306.35 | 308.70 | 308.95 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 10:15:00 | 311.90 | 308.91 | 308.87 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 12:15:00 | 307.90 | 308.86 | 308.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 10:15:00 | 306.50 | 307.82 | 308.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 300.50 | 298.08 | 301.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 13:15:00 | 298.75 | 295.45 | 297.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 13:15:00 | 298.75 | 295.45 | 297.31 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 11:15:00 | 300.40 | 295.39 | 295.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 12:15:00 | 300.70 | 296.45 | 295.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 297.60 | 297.87 | 296.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 10:15:00 | 297.60 | 297.87 | 296.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 297.60 | 297.87 | 296.84 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 294.40 | 296.49 | 296.57 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 11:15:00 | 302.05 | 297.49 | 297.00 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 14:15:00 | 295.90 | 298.04 | 298.04 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 301.15 | 298.27 | 298.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 14:15:00 | 309.00 | 301.08 | 299.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 14:15:00 | 314.50 | 315.64 | 311.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 317.20 | 316.84 | 314.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 317.20 | 316.84 | 314.62 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 311.50 | 315.73 | 315.78 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 320.85 | 314.69 | 314.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 11:15:00 | 323.50 | 316.45 | 315.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 15:15:00 | 322.65 | 323.55 | 321.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 326.10 | 325.60 | 323.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 326.10 | 325.60 | 323.47 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 338.80 | 342.86 | 343.03 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 10:15:00 | 347.40 | 342.96 | 342.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 12:15:00 | 348.05 | 344.71 | 343.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 14:15:00 | 347.70 | 348.71 | 347.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 14:15:00 | 347.70 | 348.71 | 347.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 347.70 | 348.71 | 347.04 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 334.55 | 345.21 | 346.13 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 341.40 | 338.25 | 338.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 343.50 | 339.50 | 338.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 341.45 | 341.57 | 340.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 13:15:00 | 343.40 | 341.93 | 340.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 13:15:00 | 343.40 | 341.93 | 340.48 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 13:15:00 | 337.75 | 340.36 | 340.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 10:15:00 | 336.65 | 339.03 | 339.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 09:15:00 | 336.90 | 336.78 | 338.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 336.90 | 336.78 | 338.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 336.90 | 336.78 | 338.15 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 15:15:00 | 332.00 | 330.76 | 330.64 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 11:15:00 | 329.10 | 330.48 | 330.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 13:15:00 | 327.50 | 329.55 | 330.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 09:15:00 | 331.80 | 329.69 | 329.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 331.80 | 329.69 | 329.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 331.80 | 329.69 | 329.99 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 332.15 | 330.46 | 330.25 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 328.30 | 330.81 | 330.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 12:15:00 | 326.95 | 330.04 | 330.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 14:15:00 | 326.30 | 325.14 | 327.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 15:15:00 | 326.20 | 325.35 | 327.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 326.20 | 325.35 | 327.11 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 330.00 | 327.07 | 326.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 15:15:00 | 335.85 | 331.52 | 329.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 331.05 | 331.43 | 329.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 331.05 | 331.43 | 329.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 331.05 | 331.43 | 329.45 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 326.75 | 328.71 | 328.72 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 331.15 | 328.77 | 328.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 333.50 | 330.02 | 329.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 342.65 | 343.74 | 340.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 10:15:00 | 340.60 | 343.12 | 340.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 340.60 | 343.12 | 340.47 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 11:15:00 | 338.90 | 339.54 | 339.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 12:15:00 | 337.45 | 339.12 | 339.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 338.80 | 338.53 | 339.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 338.80 | 338.53 | 339.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 338.80 | 338.53 | 339.00 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 12:15:00 | 341.00 | 338.58 | 338.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 09:15:00 | 343.00 | 340.13 | 339.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 13:15:00 | 358.05 | 359.29 | 353.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 15:15:00 | 356.00 | 358.11 | 353.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 356.00 | 358.11 | 353.62 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 337.40 | 349.73 | 350.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 332.35 | 346.26 | 348.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 15:15:00 | 330.00 | 328.37 | 335.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 12:15:00 | 328.60 | 326.21 | 331.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 12:15:00 | 328.60 | 326.21 | 331.69 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 340.40 | 334.15 | 333.59 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 10:15:00 | 329.60 | 333.83 | 334.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 09:15:00 | 328.10 | 330.54 | 332.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 09:15:00 | 338.35 | 329.43 | 330.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 338.35 | 329.43 | 330.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 338.35 | 329.43 | 330.35 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 10:15:00 | 342.35 | 332.01 | 331.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 348.00 | 340.14 | 337.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 12:15:00 | 339.75 | 340.77 | 338.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 12:15:00 | 339.75 | 340.77 | 338.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 339.75 | 340.77 | 338.63 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 14:15:00 | 336.20 | 338.56 | 338.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 11:15:00 | 335.00 | 337.32 | 337.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 14:15:00 | 338.00 | 337.27 | 337.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 14:15:00 | 338.00 | 337.27 | 337.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 338.00 | 337.27 | 337.77 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 343.45 | 338.46 | 338.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 10:15:00 | 348.00 | 340.37 | 339.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 14:15:00 | 343.00 | 344.14 | 341.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 344.05 | 348.08 | 345.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 344.05 | 348.08 | 345.71 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 10:15:00 | 346.85 | 349.05 | 349.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 12:15:00 | 344.95 | 347.90 | 348.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 342.30 | 342.25 | 344.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 10:15:00 | 345.75 | 342.95 | 344.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 345.75 | 342.95 | 344.33 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 12:15:00 | 350.80 | 345.59 | 345.36 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 343.25 | 346.10 | 346.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 15:15:00 | 342.00 | 344.43 | 345.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 344.80 | 344.51 | 345.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 344.80 | 344.51 | 345.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 344.80 | 344.51 | 345.06 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 354.05 | 342.56 | 342.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 11:15:00 | 364.50 | 346.95 | 344.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 13:15:00 | 363.50 | 364.96 | 358.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 14:15:00 | 370.50 | 369.27 | 366.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 370.50 | 369.27 | 366.65 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 10:15:00 | 372.95 | 374.50 | 374.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 369.85 | 373.25 | 374.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 14:15:00 | 374.30 | 372.96 | 373.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 14:15:00 | 374.30 | 372.96 | 373.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 374.30 | 372.96 | 373.72 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 378.90 | 374.01 | 374.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 382.95 | 376.56 | 375.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 378.20 | 378.24 | 376.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 12:15:00 | 380.80 | 382.66 | 381.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 380.80 | 382.66 | 381.02 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 11:15:00 | 377.35 | 380.47 | 380.56 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 15:15:00 | 383.00 | 380.63 | 380.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 13:15:00 | 384.65 | 382.33 | 381.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 15:15:00 | 381.00 | 382.55 | 381.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 15:15:00 | 381.00 | 382.55 | 381.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 381.00 | 382.55 | 381.77 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 377.25 | 387.22 | 387.25 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2023-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 15:15:00 | 386.00 | 383.78 | 383.56 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2023-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 12:15:00 | 380.50 | 383.14 | 383.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 378.45 | 380.86 | 381.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 09:15:00 | 385.55 | 381.65 | 382.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 385.55 | 381.65 | 382.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 385.55 | 381.65 | 382.08 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 10:15:00 | 385.75 | 382.47 | 382.42 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 380.30 | 383.66 | 384.04 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 15:15:00 | 385.00 | 382.81 | 382.58 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 12:15:00 | 379.65 | 382.46 | 382.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 378.10 | 380.24 | 381.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 379.25 | 377.86 | 379.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 379.25 | 377.86 | 379.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 379.25 | 377.86 | 379.24 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 13:15:00 | 388.55 | 380.93 | 380.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 392.25 | 388.42 | 385.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 14:15:00 | 387.50 | 390.61 | 387.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 14:15:00 | 387.50 | 390.61 | 387.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 387.50 | 390.61 | 387.85 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 384.65 | 392.76 | 393.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 378.30 | 387.34 | 390.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 13:15:00 | 389.55 | 385.69 | 388.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 13:15:00 | 389.55 | 385.69 | 388.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 13:15:00 | 389.55 | 385.69 | 388.64 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 15:15:00 | 403.85 | 391.58 | 390.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 12:15:00 | 406.40 | 396.95 | 393.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 11:15:00 | 400.40 | 400.40 | 397.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 13:15:00 | 398.40 | 399.82 | 397.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 13:15:00 | 398.40 | 399.82 | 397.54 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 397.00 | 400.15 | 400.20 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 09:15:00 | 404.20 | 400.96 | 400.56 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 15:15:00 | 395.50 | 400.83 | 400.98 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 403.00 | 401.26 | 401.16 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 399.80 | 400.97 | 401.04 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 402.20 | 401.22 | 401.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 14:15:00 | 407.55 | 402.85 | 401.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 15:15:00 | 406.00 | 406.01 | 404.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 409.20 | 408.10 | 406.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 409.20 | 408.10 | 406.38 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 09:15:00 | 404.95 | 409.14 | 409.41 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 13:15:00 | 410.50 | 408.62 | 408.43 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 11:15:00 | 407.00 | 408.43 | 408.43 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 12:15:00 | 408.80 | 408.50 | 408.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 413.95 | 409.67 | 409.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 11:15:00 | 408.95 | 411.20 | 410.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 11:15:00 | 408.95 | 411.20 | 410.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 408.95 | 411.20 | 410.13 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 409.00 | 409.71 | 409.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 406.60 | 409.09 | 409.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 11:15:00 | 400.25 | 400.24 | 402.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 11:15:00 | 400.25 | 400.24 | 402.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 400.25 | 400.24 | 402.92 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 404.90 | 401.49 | 401.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 09:15:00 | 413.15 | 403.82 | 402.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 15:15:00 | 407.45 | 408.15 | 405.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 09:15:00 | 405.75 | 407.67 | 405.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 405.75 | 407.67 | 405.74 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 428.90 | 430.88 | 430.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 14:15:00 | 427.90 | 430.28 | 430.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 15:15:00 | 432.70 | 430.77 | 430.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 15:15:00 | 432.70 | 430.77 | 430.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 432.70 | 430.77 | 430.87 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 430.00 | 425.55 | 425.16 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 11:15:00 | 424.00 | 426.29 | 426.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 12:15:00 | 422.80 | 425.59 | 426.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 14:15:00 | 426.55 | 425.31 | 425.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 14:15:00 | 426.55 | 425.31 | 425.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 426.55 | 425.31 | 425.89 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 15:15:00 | 432.00 | 426.65 | 426.44 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 420.80 | 425.48 | 425.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 10:15:00 | 414.25 | 423.23 | 424.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 15:15:00 | 420.00 | 418.82 | 421.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 10:15:00 | 420.80 | 419.26 | 421.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 420.80 | 419.26 | 421.32 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 10:15:00 | 389.20 | 384.25 | 383.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 11:15:00 | 391.85 | 385.77 | 384.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 11:15:00 | 402.65 | 403.02 | 399.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 416.35 | 417.91 | 415.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 416.35 | 417.91 | 415.84 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 418.15 | 421.99 | 422.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 417.45 | 419.25 | 420.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 10:15:00 | 416.25 | 415.30 | 417.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 10:15:00 | 416.25 | 415.30 | 417.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 416.25 | 415.30 | 417.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:45:00 | 416.55 | 415.30 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 417.15 | 415.79 | 417.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:45:00 | 416.50 | 415.79 | 417.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 412.75 | 415.18 | 416.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 15:15:00 | 410.00 | 415.18 | 416.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 389.50 | 405.12 | 408.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 406.00 | 405.13 | 407.56 | SL hit (close>ema200) qty=0.50 sl=405.13 alert=retest2 |

### Cycle 87 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 409.95 | 406.70 | 406.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 14:15:00 | 412.90 | 409.34 | 408.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 13:15:00 | 413.65 | 414.36 | 412.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:15:00 | 413.60 | 414.36 | 412.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 410.95 | 413.68 | 412.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 15:00:00 | 410.95 | 413.68 | 412.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 412.60 | 413.46 | 412.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 414.65 | 413.46 | 412.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:45:00 | 413.05 | 412.93 | 412.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 12:15:00 | 413.70 | 412.93 | 412.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 13:00:00 | 413.15 | 412.97 | 412.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 13:15:00 | 410.20 | 412.42 | 412.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-04-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 13:15:00 | 410.20 | 412.42 | 412.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 14:15:00 | 409.45 | 411.82 | 412.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 410.55 | 410.12 | 411.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 12:00:00 | 410.55 | 410.12 | 411.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 410.80 | 410.26 | 411.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:30:00 | 411.30 | 410.26 | 411.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 411.25 | 410.45 | 411.10 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 10:15:00 | 412.65 | 411.37 | 411.36 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 11:15:00 | 409.65 | 411.02 | 411.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 12:15:00 | 409.45 | 410.71 | 411.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 12:15:00 | 415.80 | 410.15 | 410.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 12:15:00 | 415.80 | 410.15 | 410.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 415.80 | 410.15 | 410.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:45:00 | 416.40 | 410.15 | 410.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 412.10 | 410.54 | 410.49 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 10:15:00 | 409.50 | 411.06 | 411.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 403.85 | 408.42 | 409.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 406.95 | 404.77 | 406.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 406.95 | 404.77 | 406.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 406.95 | 404.77 | 406.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 406.95 | 404.77 | 406.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 407.20 | 405.25 | 406.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:45:00 | 402.80 | 405.74 | 406.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 09:30:00 | 403.60 | 405.74 | 406.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 10:00:00 | 404.35 | 405.74 | 406.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 12:15:00 | 412.00 | 406.91 | 406.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 12:15:00 | 412.00 | 406.91 | 406.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 13:15:00 | 415.95 | 408.72 | 407.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 401.95 | 408.83 | 407.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 401.95 | 408.83 | 407.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 401.95 | 408.83 | 407.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:00:00 | 401.95 | 408.83 | 407.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 398.40 | 406.74 | 407.08 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 411.70 | 407.89 | 407.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 419.75 | 411.11 | 409.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 10:15:00 | 421.55 | 421.60 | 416.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:15:00 | 419.70 | 421.60 | 416.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 418.00 | 419.93 | 417.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 418.00 | 419.93 | 417.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 418.95 | 419.74 | 417.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 422.20 | 419.74 | 417.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:00:00 | 421.20 | 420.29 | 418.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 416.20 | 423.79 | 424.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 416.20 | 423.79 | 424.22 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 421.20 | 420.24 | 420.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 14:15:00 | 427.75 | 421.74 | 420.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 12:15:00 | 422.25 | 423.61 | 422.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 422.25 | 423.61 | 422.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 422.25 | 423.61 | 422.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 422.10 | 423.61 | 422.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 422.75 | 423.44 | 422.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 422.75 | 423.44 | 422.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 424.35 | 423.62 | 422.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:30:00 | 423.00 | 423.62 | 422.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 420.00 | 423.12 | 422.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 425.50 | 423.12 | 422.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 418.35 | 422.31 | 422.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 418.35 | 422.31 | 422.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 417.80 | 420.88 | 421.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 12:15:00 | 421.40 | 420.98 | 421.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:00:00 | 421.40 | 420.98 | 421.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 420.75 | 420.94 | 421.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 421.30 | 420.94 | 421.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 419.00 | 420.55 | 421.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:45:00 | 421.10 | 420.55 | 421.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 417.55 | 419.64 | 420.83 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 425.05 | 421.68 | 421.24 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 10:15:00 | 418.00 | 420.69 | 420.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 11:15:00 | 416.05 | 419.76 | 420.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 12:15:00 | 420.00 | 419.81 | 420.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 12:15:00 | 420.00 | 419.81 | 420.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 420.00 | 419.81 | 420.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 13:00:00 | 420.00 | 419.81 | 420.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 435.20 | 422.89 | 421.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 441.50 | 426.61 | 423.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 423.30 | 428.23 | 424.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 423.30 | 428.23 | 424.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 423.30 | 428.23 | 424.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 420.20 | 428.23 | 424.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 404.00 | 423.39 | 423.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 404.00 | 423.39 | 423.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 403.05 | 419.32 | 421.22 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 429.20 | 418.86 | 418.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 442.60 | 423.61 | 420.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 446.55 | 449.40 | 441.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 09:45:00 | 445.25 | 449.40 | 441.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 445.00 | 447.69 | 445.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 13:00:00 | 445.00 | 447.69 | 445.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 443.85 | 446.92 | 445.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 443.85 | 446.92 | 445.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 440.35 | 444.68 | 444.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 444.55 | 444.68 | 444.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 443.25 | 444.25 | 444.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 443.25 | 444.25 | 444.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 13:15:00 | 442.60 | 443.72 | 444.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 14:15:00 | 444.05 | 443.79 | 444.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 14:15:00 | 444.05 | 443.79 | 444.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 444.05 | 443.79 | 444.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:30:00 | 444.95 | 443.79 | 444.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 443.00 | 443.63 | 443.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 447.15 | 443.63 | 443.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 449.05 | 444.71 | 444.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 13:15:00 | 453.15 | 447.31 | 445.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 467.65 | 470.25 | 463.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 470.05 | 470.25 | 463.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 466.00 | 468.98 | 465.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 14:45:00 | 465.70 | 468.98 | 465.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 468.80 | 468.54 | 465.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:45:00 | 471.80 | 468.98 | 466.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:45:00 | 473.15 | 474.54 | 472.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 15:15:00 | 477.00 | 481.01 | 481.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 477.00 | 481.01 | 481.47 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 487.10 | 482.23 | 481.98 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 13:15:00 | 482.00 | 482.77 | 482.82 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 486.05 | 483.04 | 482.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 494.05 | 485.25 | 483.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 09:15:00 | 497.65 | 498.50 | 494.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 10:15:00 | 494.45 | 498.50 | 494.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 494.60 | 497.72 | 494.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 494.35 | 497.72 | 494.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 494.45 | 497.06 | 494.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:30:00 | 494.85 | 497.06 | 494.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 494.50 | 496.55 | 494.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:30:00 | 494.10 | 496.55 | 494.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 494.10 | 496.06 | 494.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 494.55 | 496.06 | 494.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 492.30 | 495.31 | 494.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 492.30 | 495.31 | 494.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 494.00 | 495.05 | 494.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 495.10 | 495.05 | 494.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 495.10 | 494.57 | 494.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 496.00 | 494.64 | 494.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:45:00 | 495.05 | 495.05 | 494.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 492.45 | 494.99 | 494.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 492.95 | 494.99 | 494.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 492.00 | 494.39 | 494.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 492.00 | 494.39 | 494.38 | SL hit (close<static) qty=1.00 sl=492.05 alert=retest2 |

### Cycle 110 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 493.25 | 494.16 | 494.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 15:15:00 | 490.95 | 492.91 | 493.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 13:15:00 | 487.90 | 487.19 | 490.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 14:15:00 | 488.95 | 487.54 | 489.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 488.95 | 487.54 | 489.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 488.95 | 487.54 | 489.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 490.00 | 488.03 | 489.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 489.80 | 488.03 | 489.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 487.95 | 488.02 | 489.75 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 492.55 | 490.84 | 490.62 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 488.15 | 490.31 | 490.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 487.40 | 489.03 | 489.70 | Break + close below crossover candle low |

### Cycle 113 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 494.90 | 490.20 | 490.17 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 10:15:00 | 487.70 | 490.05 | 490.17 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 492.55 | 490.28 | 490.23 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 489.85 | 490.19 | 490.20 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 490.60 | 490.23 | 490.21 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 483.70 | 488.92 | 489.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 11:15:00 | 483.05 | 486.97 | 488.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 489.95 | 485.44 | 486.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 489.95 | 485.44 | 486.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 489.95 | 485.44 | 486.90 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 489.40 | 487.62 | 487.49 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 481.90 | 487.32 | 487.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 475.40 | 483.35 | 485.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 15:15:00 | 479.50 | 475.07 | 479.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 15:15:00 | 479.50 | 475.07 | 479.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 479.50 | 475.07 | 479.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 466.55 | 470.20 | 475.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:00:00 | 464.45 | 470.20 | 475.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:00:00 | 466.05 | 468.05 | 472.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 461.30 | 467.97 | 471.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 478.40 | 468.82 | 470.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 478.40 | 468.82 | 470.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 479.10 | 470.88 | 471.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 479.10 | 470.88 | 471.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-24 11:15:00 | 477.90 | 472.28 | 471.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 477.90 | 472.28 | 471.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 480.40 | 474.88 | 473.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 476.25 | 476.56 | 474.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:45:00 | 480.90 | 477.12 | 475.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:15:00 | 483.60 | 477.10 | 475.81 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 482.15 | 485.52 | 483.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-29 13:15:00 | 482.15 | 485.52 | 483.47 | SL hit (close<ema400) qty=1.00 sl=483.47 alert=retest1 |

### Cycle 122 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 503.35 | 510.47 | 510.49 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 14:15:00 | 511.95 | 510.76 | 510.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 15:15:00 | 515.55 | 511.72 | 511.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 10:15:00 | 514.15 | 514.16 | 512.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:00:00 | 514.15 | 514.16 | 512.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 514.40 | 514.21 | 512.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 514.25 | 514.21 | 512.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 515.05 | 515.40 | 513.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 531.30 | 515.40 | 513.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 518.55 | 521.86 | 518.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 12:45:00 | 519.80 | 517.49 | 517.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 13:15:00 | 514.00 | 516.79 | 517.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 514.00 | 516.79 | 517.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 510.00 | 515.44 | 516.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 09:15:00 | 519.80 | 515.87 | 516.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 519.80 | 515.87 | 516.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 519.80 | 515.87 | 516.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 519.80 | 515.87 | 516.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 521.55 | 517.01 | 516.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 15:15:00 | 522.95 | 519.98 | 518.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 518.15 | 519.61 | 518.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 518.15 | 519.61 | 518.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 518.15 | 519.61 | 518.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:15:00 | 518.90 | 519.61 | 518.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 523.95 | 520.48 | 519.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 525.90 | 520.48 | 519.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 510.05 | 517.93 | 518.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 510.05 | 517.93 | 518.31 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 520.95 | 516.93 | 516.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 522.60 | 518.45 | 517.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 519.80 | 521.23 | 519.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 519.80 | 521.23 | 519.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 520.00 | 520.98 | 519.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:30:00 | 519.35 | 521.60 | 519.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 520.50 | 521.86 | 520.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:15:00 | 520.00 | 521.86 | 520.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 525.60 | 522.61 | 521.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:15:00 | 521.25 | 522.61 | 521.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 521.25 | 522.34 | 521.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:30:00 | 527.00 | 523.07 | 521.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:45:00 | 527.55 | 523.94 | 522.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 13:15:00 | 528.35 | 524.91 | 522.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:15:00 | 548.00 | 526.85 | 524.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 542.00 | 529.88 | 526.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:30:00 | 571.25 | 541.79 | 534.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-23 10:15:00 | 579.70 | 550.03 | 538.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 589.30 | 595.34 | 595.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 586.55 | 593.58 | 594.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 574.00 | 571.77 | 577.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 574.00 | 571.77 | 577.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 574.00 | 571.77 | 577.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:30:00 | 564.85 | 570.56 | 576.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 563.30 | 564.27 | 569.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 563.95 | 564.27 | 569.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 15:15:00 | 567.90 | 568.64 | 570.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 567.90 | 568.49 | 570.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 572.55 | 568.49 | 570.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 574.65 | 569.72 | 570.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:45:00 | 573.50 | 569.72 | 570.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 582.70 | 572.32 | 571.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 582.70 | 572.32 | 571.54 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 11:15:00 | 565.00 | 575.75 | 575.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 550.00 | 568.32 | 572.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 535.75 | 535.69 | 542.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 11:00:00 | 535.75 | 535.69 | 542.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 542.40 | 537.03 | 542.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 542.40 | 537.03 | 542.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 536.05 | 536.83 | 541.93 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 549.80 | 541.72 | 541.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 14:15:00 | 553.70 | 544.11 | 542.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 550.10 | 551.61 | 549.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:00:00 | 550.10 | 551.61 | 549.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 551.80 | 551.65 | 549.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 550.95 | 551.65 | 549.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 544.80 | 550.28 | 549.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 544.80 | 550.28 | 549.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 535.00 | 547.22 | 547.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 531.40 | 544.06 | 546.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 552.25 | 532.39 | 537.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 552.25 | 532.39 | 537.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 552.25 | 532.39 | 537.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 552.25 | 532.39 | 537.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 562.50 | 538.41 | 539.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 562.50 | 538.41 | 539.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 564.00 | 543.53 | 541.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 597.55 | 566.44 | 555.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 09:15:00 | 609.10 | 611.10 | 601.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 10:00:00 | 609.10 | 611.10 | 601.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 601.80 | 608.41 | 603.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 601.80 | 608.41 | 603.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 603.65 | 607.45 | 603.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 596.45 | 607.45 | 603.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 599.40 | 605.84 | 603.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 595.95 | 605.84 | 603.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 594.80 | 603.63 | 602.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 594.80 | 603.63 | 602.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 597.25 | 601.18 | 601.70 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 607.40 | 602.28 | 601.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 610.05 | 604.93 | 603.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 598.50 | 606.75 | 605.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 598.50 | 606.75 | 605.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 598.50 | 606.75 | 605.23 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 599.65 | 604.09 | 604.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 595.10 | 600.30 | 602.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 608.35 | 601.62 | 602.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 608.35 | 601.62 | 602.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 608.35 | 601.62 | 602.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 608.35 | 601.62 | 602.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 600.75 | 601.45 | 602.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:30:00 | 595.80 | 600.02 | 601.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:45:00 | 585.50 | 594.97 | 598.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 566.01 | 589.27 | 595.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 556.23 | 572.49 | 583.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 579.35 | 568.21 | 574.67 | SL hit (close>ema200) qty=0.50 sl=568.21 alert=retest2 |

### Cycle 137 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 591.30 | 579.43 | 578.02 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 573.15 | 578.46 | 578.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 568.70 | 574.22 | 576.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 571.45 | 571.22 | 573.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 571.45 | 571.22 | 573.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 565.70 | 570.08 | 572.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:15:00 | 561.80 | 570.08 | 572.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:45:00 | 564.55 | 565.30 | 569.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:00:00 | 557.85 | 562.83 | 566.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:15:00 | 536.32 | 547.79 | 556.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 14:15:00 | 533.71 | 540.27 | 549.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 15:15:00 | 529.96 | 538.42 | 547.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 10:15:00 | 538.00 | 537.11 | 545.65 | SL hit (close>ema200) qty=0.50 sl=537.11 alert=retest2 |

### Cycle 139 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 504.35 | 496.35 | 496.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 504.65 | 501.50 | 499.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 10:15:00 | 500.15 | 501.23 | 499.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 10:15:00 | 500.15 | 501.23 | 499.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 500.15 | 501.23 | 499.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 500.15 | 501.23 | 499.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 510.75 | 505.43 | 502.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 515.55 | 509.19 | 505.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 518.60 | 511.82 | 507.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 499.85 | 510.66 | 507.80 | SL hit (close<static) qty=1.00 sl=499.95 alert=retest2 |

### Cycle 140 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 502.30 | 506.02 | 506.32 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 11:15:00 | 514.40 | 506.78 | 506.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 520.20 | 510.92 | 508.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 526.45 | 526.58 | 520.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:30:00 | 524.00 | 526.58 | 520.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 521.60 | 524.90 | 521.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 521.60 | 524.90 | 521.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 519.40 | 523.80 | 521.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 517.40 | 523.80 | 521.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 518.50 | 522.74 | 521.38 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 511.15 | 519.03 | 519.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 508.25 | 515.81 | 518.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 514.35 | 512.90 | 515.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 10:45:00 | 513.85 | 512.90 | 515.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 505.95 | 511.51 | 514.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:30:00 | 504.85 | 510.07 | 513.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 479.61 | 487.85 | 494.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 492.70 | 487.23 | 493.15 | SL hit (close>ema200) qty=0.50 sl=487.23 alert=retest2 |

### Cycle 143 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 493.50 | 491.78 | 491.56 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 13:15:00 | 489.00 | 491.22 | 491.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 14:15:00 | 483.70 | 489.72 | 490.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 483.85 | 482.08 | 484.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 483.85 | 482.08 | 484.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 483.85 | 482.08 | 484.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 483.85 | 482.08 | 484.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 483.45 | 482.43 | 484.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:45:00 | 484.00 | 482.43 | 484.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 484.20 | 482.79 | 484.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 484.20 | 482.79 | 484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 484.65 | 483.34 | 484.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 487.85 | 483.34 | 484.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 480.25 | 482.73 | 483.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 12:00:00 | 478.30 | 481.84 | 483.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 486.65 | 480.78 | 482.38 | SL hit (close>static) qty=1.00 sl=486.00 alert=retest2 |

### Cycle 145 — BUY (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 15:15:00 | 486.00 | 483.42 | 483.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 495.90 | 485.92 | 484.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 491.50 | 496.72 | 493.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 491.50 | 496.72 | 493.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 491.50 | 496.72 | 493.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 491.50 | 496.72 | 493.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 494.80 | 496.33 | 494.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 509.60 | 495.42 | 494.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 498.75 | 503.33 | 503.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 498.75 | 503.33 | 503.39 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 504.55 | 503.03 | 502.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 508.10 | 504.04 | 503.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 13:15:00 | 521.90 | 521.93 | 515.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 14:00:00 | 521.90 | 521.93 | 515.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 517.95 | 521.23 | 516.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 514.50 | 521.23 | 516.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 532.90 | 523.56 | 517.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 516.55 | 523.56 | 517.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 537.30 | 542.04 | 539.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 530.30 | 542.04 | 539.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 526.15 | 538.86 | 537.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 527.45 | 538.86 | 537.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 525.70 | 536.23 | 536.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 520.70 | 526.40 | 529.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 514.55 | 514.17 | 517.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 14:00:00 | 514.55 | 514.17 | 517.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 519.00 | 515.14 | 517.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 518.75 | 515.14 | 517.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 517.00 | 515.51 | 517.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 515.60 | 515.51 | 517.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 513.40 | 515.09 | 517.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:15:00 | 511.15 | 515.09 | 517.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 511.45 | 514.83 | 516.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:30:00 | 509.40 | 513.20 | 515.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:15:00 | 485.59 | 494.69 | 497.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:15:00 | 485.88 | 494.69 | 497.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:15:00 | 483.93 | 492.97 | 496.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 488.75 | 488.72 | 491.93 | SL hit (close>ema200) qty=0.50 sl=488.72 alert=retest2 |

### Cycle 149 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 511.00 | 495.69 | 493.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 522.40 | 506.27 | 499.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 513.70 | 514.16 | 509.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:45:00 | 514.35 | 514.16 | 509.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 509.70 | 513.04 | 509.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 509.70 | 513.04 | 509.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 509.70 | 512.37 | 509.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 509.70 | 512.37 | 509.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 505.85 | 511.07 | 509.51 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 501.90 | 508.09 | 508.45 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 540.05 | 513.92 | 510.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 10:15:00 | 544.80 | 535.59 | 528.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 13:15:00 | 533.55 | 535.99 | 530.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 14:00:00 | 533.55 | 535.99 | 530.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 535.00 | 535.08 | 531.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 530.35 | 534.13 | 531.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 526.30 | 532.57 | 530.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:00:00 | 526.30 | 532.57 | 530.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 523.80 | 530.81 | 530.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 523.80 | 530.81 | 530.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 521.65 | 528.98 | 529.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 518.00 | 523.34 | 525.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 529.30 | 519.93 | 523.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 529.30 | 519.93 | 523.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 529.30 | 519.93 | 523.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 529.30 | 519.93 | 523.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 538.25 | 523.59 | 524.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 538.25 | 523.59 | 524.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 541.65 | 527.20 | 526.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 561.80 | 540.22 | 533.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 586.00 | 586.16 | 576.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 586.00 | 586.16 | 576.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 575.10 | 587.16 | 583.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 575.10 | 587.16 | 583.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 575.45 | 584.82 | 582.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 575.45 | 584.82 | 582.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 577.85 | 581.30 | 581.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 564.85 | 576.32 | 578.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 572.40 | 571.91 | 575.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 572.40 | 571.91 | 575.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 566.55 | 570.84 | 574.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 569.15 | 570.84 | 574.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 574.25 | 570.19 | 572.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 574.25 | 570.19 | 572.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 581.45 | 572.44 | 573.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 569.20 | 571.29 | 573.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 572.80 | 570.25 | 572.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 544.16 | 563.51 | 568.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 540.74 | 558.80 | 565.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 544.00 | 539.16 | 547.31 | SL hit (close>ema200) qty=0.50 sl=539.16 alert=retest2 |

### Cycle 155 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 566.60 | 544.99 | 544.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 575.55 | 551.10 | 546.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 560.00 | 561.78 | 555.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 12:45:00 | 560.80 | 561.78 | 555.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 555.10 | 562.35 | 558.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:00:00 | 572.20 | 560.72 | 558.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 569.50 | 574.33 | 572.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 562.00 | 570.78 | 571.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 562.00 | 570.78 | 571.65 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 579.65 | 572.87 | 572.30 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 558.30 | 570.28 | 571.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 555.85 | 565.56 | 568.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 537.65 | 537.10 | 545.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 537.65 | 537.10 | 545.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 548.55 | 539.39 | 546.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 547.05 | 539.39 | 546.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 533.40 | 538.19 | 544.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:15:00 | 532.50 | 538.19 | 544.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:30:00 | 532.50 | 539.94 | 542.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 530.80 | 539.94 | 542.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 523.10 | 538.75 | 542.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 505.88 | 518.74 | 527.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 505.88 | 518.74 | 527.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 504.26 | 518.74 | 527.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 516.25 | 516.15 | 523.43 | SL hit (close>ema200) qty=0.50 sl=516.15 alert=retest2 |

### Cycle 159 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 559.00 | 520.91 | 518.61 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 525.95 | 531.48 | 531.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 517.35 | 527.04 | 529.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 503.15 | 501.03 | 508.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 503.15 | 501.03 | 508.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 485.15 | 476.57 | 485.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 481.20 | 476.57 | 485.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 493.60 | 479.97 | 486.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 493.60 | 479.97 | 486.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 487.10 | 481.40 | 486.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 484.35 | 481.40 | 486.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 11:15:00 | 496.70 | 487.88 | 488.55 | SL hit (close>static) qty=1.00 sl=495.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 497.15 | 489.73 | 489.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 500.75 | 491.94 | 490.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 516.20 | 518.44 | 511.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 522.70 | 518.44 | 511.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 519.00 | 521.34 | 516.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:30:00 | 513.80 | 521.34 | 516.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 531.50 | 522.86 | 518.02 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 505.15 | 516.65 | 517.20 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 522.05 | 512.85 | 512.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 529.45 | 523.82 | 519.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 13:15:00 | 524.45 | 529.03 | 525.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 13:15:00 | 524.45 | 529.03 | 525.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 524.45 | 529.03 | 525.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:00:00 | 524.45 | 529.03 | 525.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 530.05 | 529.24 | 525.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:30:00 | 528.35 | 529.24 | 525.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 526.20 | 528.84 | 526.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 526.20 | 528.84 | 526.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 523.50 | 527.77 | 525.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 524.70 | 527.77 | 525.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 525.55 | 524.92 | 524.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:00:00 | 525.55 | 524.92 | 524.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 535.00 | 526.93 | 525.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:45:00 | 536.25 | 528.36 | 526.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 536.65 | 528.36 | 526.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 536.50 | 531.87 | 528.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:30:00 | 536.80 | 532.37 | 529.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 532.30 | 535.53 | 532.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 532.30 | 535.53 | 532.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 539.50 | 536.32 | 533.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 15:15:00 | 541.20 | 536.32 | 533.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 542.35 | 537.84 | 534.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 531.35 | 536.55 | 534.69 | SL hit (close<static) qty=1.00 sl=532.30 alert=retest2 |

### Cycle 164 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 527.70 | 534.78 | 535.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 526.55 | 533.13 | 534.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 538.95 | 533.80 | 534.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 538.95 | 533.80 | 534.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 538.95 | 533.80 | 534.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 538.95 | 533.80 | 534.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 540.00 | 535.04 | 535.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 539.00 | 535.04 | 535.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 541.30 | 536.29 | 535.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 14:15:00 | 544.95 | 539.13 | 537.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 10:15:00 | 540.00 | 540.11 | 538.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 540.00 | 540.11 | 538.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 539.70 | 540.02 | 538.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:30:00 | 537.20 | 540.02 | 538.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 539.00 | 539.82 | 538.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 537.65 | 539.82 | 538.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 537.50 | 539.36 | 538.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:45:00 | 538.05 | 539.36 | 538.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 539.25 | 539.33 | 538.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:15:00 | 535.40 | 539.33 | 538.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 535.40 | 538.55 | 538.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 536.95 | 538.55 | 538.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 535.45 | 537.93 | 538.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 513.60 | 531.05 | 534.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 490.80 | 480.21 | 486.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 490.80 | 480.21 | 486.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 490.80 | 480.21 | 486.26 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 495.10 | 489.79 | 489.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 495.80 | 490.99 | 489.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 12:15:00 | 516.75 | 517.13 | 513.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 13:00:00 | 516.75 | 517.13 | 513.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 518.20 | 517.15 | 515.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 515.10 | 517.15 | 515.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 518.40 | 517.40 | 515.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:45:00 | 517.20 | 517.40 | 515.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 519.60 | 520.75 | 518.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:45:00 | 519.90 | 520.75 | 518.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 518.25 | 520.25 | 518.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 518.25 | 520.25 | 518.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 516.15 | 519.43 | 518.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 516.15 | 519.43 | 518.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 520.80 | 519.70 | 518.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 515.50 | 519.70 | 518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 506.15 | 516.99 | 517.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 498.20 | 513.23 | 515.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 506.55 | 506.16 | 509.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 10:15:00 | 506.55 | 506.16 | 509.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 506.55 | 506.16 | 509.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 506.55 | 506.16 | 509.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 509.05 | 506.74 | 509.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 509.20 | 506.74 | 509.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 510.00 | 507.39 | 509.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:00:00 | 510.00 | 507.39 | 509.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 509.95 | 507.90 | 509.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:45:00 | 510.05 | 507.90 | 509.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 510.10 | 508.34 | 509.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:45:00 | 510.40 | 508.34 | 509.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 508.30 | 508.33 | 509.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 515.20 | 508.33 | 509.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 512.85 | 509.24 | 510.04 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 513.00 | 510.59 | 510.55 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 508.60 | 510.19 | 510.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 13:15:00 | 504.80 | 509.11 | 509.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 491.40 | 490.68 | 495.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 15:00:00 | 491.40 | 490.68 | 495.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 492.60 | 491.00 | 494.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 492.20 | 491.00 | 494.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 495.55 | 491.91 | 494.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 495.90 | 491.91 | 494.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 495.00 | 492.53 | 494.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 489.05 | 493.30 | 494.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:30:00 | 491.10 | 485.98 | 486.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 492.90 | 487.37 | 487.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 492.90 | 487.37 | 487.09 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 474.10 | 484.64 | 485.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 466.40 | 479.81 | 483.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 488.65 | 472.94 | 476.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 488.65 | 472.94 | 476.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 488.65 | 472.94 | 476.60 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 485.00 | 478.74 | 478.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 487.10 | 482.31 | 480.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 503.95 | 503.98 | 499.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:45:00 | 502.75 | 503.98 | 499.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 514.20 | 505.75 | 501.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 517.70 | 505.75 | 501.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 500.50 | 508.33 | 507.82 | SL hit (close<static) qty=1.00 sl=501.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 500.25 | 506.71 | 507.13 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 508.00 | 505.88 | 505.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 12:15:00 | 510.55 | 506.82 | 506.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 538.10 | 538.59 | 529.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:30:00 | 539.00 | 538.59 | 529.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 526.75 | 537.00 | 533.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 526.75 | 537.00 | 533.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 525.70 | 534.74 | 532.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 525.50 | 534.74 | 532.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 528.35 | 531.67 | 531.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 528.35 | 531.67 | 531.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 526.80 | 530.70 | 531.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 524.40 | 529.44 | 530.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 514.50 | 513.93 | 517.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 10:15:00 | 515.60 | 513.93 | 517.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 515.85 | 514.32 | 517.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 515.95 | 514.32 | 517.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 515.60 | 514.57 | 517.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 517.40 | 514.57 | 517.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 515.75 | 514.96 | 516.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 515.75 | 514.96 | 516.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 518.10 | 515.59 | 516.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 531.65 | 515.59 | 516.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 536.00 | 519.67 | 518.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 550.00 | 535.13 | 528.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 570.70 | 571.80 | 564.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:30:00 | 570.15 | 571.80 | 564.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 567.30 | 571.62 | 567.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 567.30 | 571.62 | 567.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 565.75 | 570.45 | 567.63 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 552.45 | 563.68 | 564.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 546.10 | 557.60 | 561.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 543.15 | 533.56 | 538.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 543.15 | 533.56 | 538.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 543.15 | 533.56 | 538.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 542.00 | 533.56 | 538.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 541.60 | 535.17 | 538.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 535.70 | 538.25 | 538.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 508.92 | 522.37 | 530.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 512.00 | 507.27 | 511.64 | SL hit (close>ema200) qty=0.50 sl=507.27 alert=retest2 |

### Cycle 179 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 518.70 | 512.24 | 511.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 522.90 | 515.43 | 513.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 512.25 | 518.81 | 516.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 512.25 | 518.81 | 516.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 512.25 | 518.81 | 516.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 512.00 | 518.81 | 516.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 521.00 | 519.25 | 517.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:15:00 | 521.50 | 519.25 | 517.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 521.20 | 519.88 | 518.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 524.95 | 520.08 | 518.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 524.80 | 521.03 | 519.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 519.30 | 521.21 | 519.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 519.30 | 521.21 | 519.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 520.10 | 520.99 | 519.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 518.05 | 520.99 | 519.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 519.05 | 520.60 | 519.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 518.60 | 520.60 | 519.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 519.75 | 520.43 | 519.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 519.60 | 520.43 | 519.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 518.50 | 520.05 | 519.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 516.05 | 519.12 | 519.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 516.05 | 519.12 | 519.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 514.95 | 518.28 | 518.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 521.40 | 518.91 | 519.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 10:15:00 | 521.40 | 518.91 | 519.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 521.40 | 518.91 | 519.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 521.75 | 518.91 | 519.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 518.25 | 518.77 | 519.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:15:00 | 522.00 | 518.77 | 519.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 522.70 | 519.56 | 519.44 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 514.40 | 518.73 | 519.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 512.35 | 517.45 | 518.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 519.25 | 517.81 | 518.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 519.25 | 517.81 | 518.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 519.25 | 517.81 | 518.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 521.75 | 517.81 | 518.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 516.50 | 517.55 | 518.39 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 526.50 | 519.34 | 519.13 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 516.05 | 519.36 | 519.36 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 13:15:00 | 523.30 | 519.79 | 519.50 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 518.10 | 519.37 | 519.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 517.20 | 518.93 | 519.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 519.95 | 518.80 | 519.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 519.95 | 518.80 | 519.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 519.95 | 518.80 | 519.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:00:00 | 511.00 | 516.72 | 517.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 517.35 | 514.42 | 514.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 517.35 | 514.42 | 514.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 518.85 | 516.54 | 515.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 517.35 | 518.20 | 516.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 517.35 | 518.20 | 516.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 517.35 | 518.20 | 516.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 517.20 | 518.20 | 516.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 516.40 | 517.84 | 516.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 515.50 | 517.84 | 516.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 521.05 | 518.48 | 517.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 515.85 | 518.48 | 517.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 531.20 | 531.73 | 527.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 527.50 | 531.73 | 527.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 529.00 | 530.68 | 528.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 524.35 | 530.68 | 528.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 526.95 | 529.93 | 528.34 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 520.95 | 526.48 | 527.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 519.50 | 525.08 | 526.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 518.00 | 517.68 | 520.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 518.00 | 517.68 | 520.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 518.00 | 517.68 | 520.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 509.00 | 514.08 | 516.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:45:00 | 510.80 | 511.20 | 513.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:00:00 | 509.35 | 512.48 | 513.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 512.15 | 510.29 | 510.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 511.40 | 510.70 | 511.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 511.60 | 510.70 | 511.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 483.55 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 485.26 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 483.88 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 486.54 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 495.70 | 491.80 | 495.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 495.70 | 491.80 | 495.54 | SL hit (close>ema200) qty=0.50 sl=491.80 alert=retest2 |

### Cycle 189 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 469.10 | 462.60 | 462.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 474.00 | 464.88 | 463.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 513.35 | 514.30 | 508.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 513.35 | 514.30 | 508.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 506.85 | 511.96 | 508.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 506.85 | 511.96 | 508.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 507.20 | 511.01 | 508.33 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 503.75 | 507.13 | 507.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 499.50 | 504.89 | 506.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 508.65 | 504.45 | 505.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 11:15:00 | 508.65 | 504.45 | 505.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 508.65 | 504.45 | 505.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 508.65 | 504.45 | 505.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 508.40 | 505.24 | 505.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 508.35 | 505.24 | 505.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 509.30 | 506.05 | 505.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 514.40 | 508.23 | 507.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 509.85 | 509.99 | 508.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 509.85 | 509.99 | 508.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 509.85 | 509.99 | 508.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 507.90 | 509.99 | 508.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 506.55 | 509.30 | 508.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 506.55 | 509.30 | 508.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 511.80 | 509.80 | 508.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:30:00 | 512.25 | 509.80 | 508.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 508.50 | 510.01 | 508.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 504.50 | 510.01 | 508.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 507.25 | 509.46 | 508.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 506.60 | 509.46 | 508.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 505.75 | 508.44 | 508.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 505.75 | 508.44 | 508.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 505.30 | 507.82 | 508.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 498.15 | 505.88 | 507.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 504.90 | 504.42 | 506.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 504.90 | 504.42 | 506.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 504.90 | 504.42 | 506.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 504.90 | 504.42 | 506.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 506.50 | 504.84 | 506.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 506.50 | 504.84 | 506.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 503.75 | 504.62 | 506.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 502.50 | 504.62 | 506.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:45:00 | 502.40 | 504.24 | 505.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:30:00 | 502.65 | 503.93 | 505.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:00:00 | 502.70 | 503.93 | 505.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 497.35 | 501.66 | 503.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 506.70 | 502.11 | 502.63 | SL hit (close>static) qty=1.00 sl=506.65 alert=retest2 |

### Cycle 193 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 507.35 | 503.71 | 503.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 512.45 | 506.40 | 504.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 508.00 | 508.60 | 506.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 508.00 | 508.60 | 506.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 506.20 | 508.12 | 506.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 509.70 | 508.12 | 506.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:00:00 | 508.75 | 508.25 | 506.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:30:00 | 508.60 | 507.81 | 506.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 509.05 | 507.33 | 506.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 508.55 | 508.97 | 508.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 508.10 | 508.97 | 508.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 504.35 | 508.04 | 507.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 504.35 | 508.04 | 507.71 | SL hit (close<static) qty=1.00 sl=505.15 alert=retest2 |

### Cycle 194 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 505.00 | 507.43 | 507.47 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 515.15 | 506.85 | 506.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 524.05 | 510.99 | 508.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 515.90 | 517.82 | 514.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 515.90 | 517.82 | 514.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 518.90 | 517.30 | 515.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 520.65 | 517.10 | 515.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 511.30 | 515.45 | 514.71 | SL hit (close<static) qty=1.00 sl=514.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 510.65 | 513.77 | 514.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 14:15:00 | 509.75 | 512.97 | 513.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 508.35 | 507.91 | 509.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 508.35 | 507.91 | 509.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 508.35 | 507.91 | 509.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 509.55 | 507.91 | 509.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 510.40 | 508.41 | 509.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 510.40 | 508.41 | 509.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 511.10 | 508.95 | 510.05 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 513.35 | 510.94 | 510.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 515.90 | 512.26 | 511.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 513.55 | 515.16 | 513.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 513.55 | 515.16 | 513.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 513.55 | 515.16 | 513.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 520.60 | 517.38 | 514.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 533.60 | 535.91 | 536.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 533.60 | 535.91 | 536.03 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 538.00 | 536.33 | 536.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 577.65 | 544.59 | 539.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 575.00 | 577.35 | 567.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 575.00 | 577.35 | 567.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 577.55 | 577.17 | 568.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 586.75 | 575.49 | 571.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 568.95 | 571.30 | 571.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 568.95 | 571.30 | 571.41 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 576.35 | 571.55 | 571.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 579.00 | 573.04 | 571.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 584.00 | 585.72 | 581.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 586.40 | 585.72 | 581.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 578.15 | 584.21 | 580.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 578.15 | 584.21 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 580.50 | 583.47 | 580.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 582.20 | 581.40 | 580.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 580.80 | 585.00 | 585.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 580.80 | 585.00 | 585.35 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 586.20 | 585.57 | 585.57 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 584.25 | 585.31 | 585.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 582.45 | 584.25 | 584.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 587.10 | 584.38 | 584.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 587.10 | 584.38 | 584.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 587.10 | 584.38 | 584.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 587.50 | 584.38 | 584.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 587.50 | 585.01 | 585.06 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 587.75 | 585.55 | 585.30 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 580.05 | 585.51 | 585.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 565.00 | 581.41 | 583.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 565.60 | 563.10 | 569.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 15:00:00 | 565.60 | 563.10 | 569.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 565.10 | 562.70 | 567.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 565.10 | 562.70 | 567.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 567.00 | 563.56 | 567.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 568.65 | 563.56 | 567.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 565.30 | 563.91 | 567.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 565.00 | 564.12 | 567.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 570.40 | 565.88 | 567.52 | SL hit (close>static) qty=1.00 sl=567.60 alert=retest2 |

### Cycle 207 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 584.00 | 569.50 | 569.02 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 566.55 | 572.81 | 573.60 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 576.50 | 574.34 | 574.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 577.70 | 575.01 | 574.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 578.00 | 578.10 | 576.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 578.00 | 578.10 | 576.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 578.00 | 578.10 | 576.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 578.00 | 578.10 | 576.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 576.85 | 577.85 | 576.43 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 566.55 | 574.27 | 575.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 564.25 | 569.42 | 572.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 15:15:00 | 565.00 | 564.72 | 567.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 567.50 | 564.72 | 567.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 569.95 | 565.77 | 567.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 572.25 | 565.77 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 568.45 | 566.31 | 567.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 569.90 | 566.31 | 567.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 568.90 | 566.82 | 567.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 568.40 | 566.82 | 567.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 566.80 | 566.82 | 567.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 567.60 | 566.82 | 567.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 566.70 | 566.67 | 567.45 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 568.85 | 567.84 | 567.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 575.90 | 569.45 | 568.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 565.50 | 569.60 | 568.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 565.50 | 569.60 | 568.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 565.50 | 569.60 | 568.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 565.50 | 569.60 | 568.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 570.60 | 569.80 | 568.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 566.20 | 569.80 | 568.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 568.30 | 569.50 | 568.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 568.30 | 569.50 | 568.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 568.00 | 569.20 | 568.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:15:00 | 569.65 | 569.10 | 568.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 570.90 | 568.88 | 568.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 575.50 | 571.93 | 570.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 569.90 | 571.84 | 570.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 563.15 | 570.10 | 569.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 563.15 | 570.10 | 569.96 | SL hit (close<static) qty=1.00 sl=567.05 alert=retest2 |

### Cycle 212 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 564.00 | 568.88 | 569.42 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 578.45 | 570.79 | 570.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 583.50 | 574.93 | 572.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 590.45 | 591.07 | 585.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 11:15:00 | 586.75 | 590.07 | 585.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 586.75 | 590.07 | 585.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 587.60 | 590.07 | 585.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 590.15 | 590.08 | 586.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 610.90 | 584.68 | 584.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 579.65 | 583.68 | 584.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 579.65 | 583.68 | 584.14 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 590.70 | 585.59 | 584.92 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 581.00 | 584.42 | 584.57 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 588.85 | 585.20 | 584.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 602.85 | 591.18 | 588.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 605.35 | 607.62 | 603.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 605.35 | 607.62 | 603.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 605.00 | 607.09 | 603.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 607.10 | 607.09 | 603.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 602.25 | 606.12 | 603.09 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 594.60 | 602.41 | 603.44 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 608.85 | 602.35 | 602.26 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 601.00 | 602.80 | 602.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 591.75 | 600.59 | 601.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 599.15 | 598.42 | 600.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 12:00:00 | 599.15 | 598.42 | 600.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 600.45 | 598.82 | 600.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:00:00 | 600.45 | 598.82 | 600.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 599.00 | 598.86 | 600.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:30:00 | 600.45 | 598.86 | 600.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 595.50 | 598.19 | 599.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:30:00 | 599.55 | 598.19 | 599.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 591.50 | 590.96 | 593.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:30:00 | 590.70 | 590.96 | 593.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 593.35 | 591.44 | 593.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 595.35 | 591.44 | 593.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 594.40 | 592.03 | 593.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 601.90 | 592.03 | 593.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 605.75 | 594.77 | 594.47 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 588.05 | 593.52 | 594.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 585.60 | 591.19 | 593.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 587.80 | 583.64 | 586.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 587.80 | 583.64 | 586.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 587.80 | 583.64 | 586.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 587.80 | 583.64 | 586.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 586.60 | 584.24 | 586.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 582.90 | 583.86 | 586.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 589.35 | 584.92 | 584.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 589.35 | 584.92 | 584.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 601.50 | 588.24 | 586.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 12:15:00 | 591.65 | 592.02 | 589.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:45:00 | 591.05 | 592.02 | 589.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 588.35 | 591.29 | 589.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 588.35 | 591.29 | 589.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 589.05 | 590.84 | 589.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 591.95 | 590.12 | 589.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 585.50 | 589.19 | 588.93 | SL hit (close<static) qty=1.00 sl=587.35 alert=retest2 |

### Cycle 224 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 580.20 | 588.03 | 588.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 565.30 | 578.27 | 581.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 576.80 | 575.23 | 579.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:15:00 | 579.00 | 575.23 | 579.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 582.95 | 576.78 | 579.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 582.95 | 576.78 | 579.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 581.75 | 577.77 | 579.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:15:00 | 581.00 | 577.77 | 579.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 594.60 | 581.65 | 581.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 598.15 | 589.85 | 587.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 598.60 | 598.68 | 594.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:45:00 | 598.50 | 598.68 | 594.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 594.85 | 598.83 | 596.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 594.85 | 598.83 | 596.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 595.85 | 598.23 | 596.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:15:00 | 593.00 | 598.23 | 596.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 593.00 | 597.19 | 596.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 591.80 | 597.19 | 596.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 586.45 | 595.04 | 595.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 581.95 | 587.70 | 591.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 591.50 | 588.03 | 590.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 591.50 | 588.03 | 590.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 591.50 | 588.03 | 590.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 592.35 | 588.03 | 590.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 593.70 | 589.16 | 591.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 593.25 | 589.16 | 591.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 593.10 | 589.95 | 591.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 592.05 | 589.95 | 591.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 595.25 | 591.67 | 591.82 | SL hit (close>static) qty=1.00 sl=595.15 alert=retest2 |

### Cycle 227 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 596.45 | 592.62 | 592.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 601.00 | 595.49 | 593.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 600.35 | 601.16 | 598.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 600.35 | 601.16 | 598.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 599.25 | 600.78 | 598.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 599.10 | 600.78 | 598.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 596.50 | 599.92 | 598.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 596.50 | 599.92 | 598.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 597.30 | 599.40 | 598.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:45:00 | 597.05 | 599.40 | 598.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 597.25 | 598.97 | 598.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 596.15 | 598.97 | 598.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 596.00 | 598.37 | 597.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 596.05 | 597.90 | 597.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 592.85 | 596.89 | 597.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 591.00 | 595.14 | 596.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 15:15:00 | 585.00 | 584.09 | 588.32 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:45:00 | 573.65 | 581.79 | 586.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 572.90 | 568.50 | 572.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 572.90 | 568.50 | 572.47 | SL hit (close>ema400) qty=1.00 sl=572.47 alert=retest1 |

### Cycle 229 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 577.20 | 573.47 | 573.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 581.15 | 577.29 | 575.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 603.00 | 603.10 | 595.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 603.00 | 603.10 | 595.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 597.95 | 601.28 | 596.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 596.85 | 601.28 | 596.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 598.25 | 600.02 | 597.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 597.65 | 600.02 | 597.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 599.20 | 599.86 | 597.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 600.00 | 599.86 | 597.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 603.75 | 600.64 | 598.32 | EMA400 retest candle locked (from upside) |

### Cycle 230 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 591.10 | 596.57 | 597.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 588.35 | 594.15 | 596.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 579.85 | 576.75 | 582.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 580.00 | 576.75 | 582.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 578.80 | 577.64 | 581.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:00:00 | 576.00 | 577.31 | 581.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 575.20 | 575.69 | 578.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 547.20 | 562.05 | 567.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 546.44 | 562.05 | 567.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 518.40 | 532.16 | 542.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 231 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 563.40 | 540.93 | 538.86 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 533.40 | 540.51 | 540.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 529.20 | 536.91 | 538.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 539.70 | 537.47 | 538.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 539.70 | 537.47 | 538.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 539.70 | 537.47 | 538.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 539.70 | 537.47 | 538.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 540.35 | 538.04 | 539.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 542.30 | 538.04 | 539.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 537.80 | 537.31 | 538.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 537.80 | 537.31 | 538.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 540.90 | 538.03 | 538.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:30:00 | 539.75 | 538.03 | 538.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 540.95 | 538.61 | 538.93 | EMA400 retest candle locked (from downside) |

### Cycle 233 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 543.30 | 539.79 | 539.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 544.60 | 540.75 | 539.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 563.50 | 563.91 | 556.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 563.50 | 563.91 | 556.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 563.50 | 563.91 | 556.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 563.50 | 563.91 | 556.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 545.00 | 559.33 | 556.85 | EMA400 retest candle locked (from upside) |

### Cycle 234 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 547.80 | 555.37 | 555.38 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 560.40 | 555.39 | 555.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 573.90 | 559.10 | 556.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 578.05 | 578.67 | 573.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 578.05 | 578.67 | 573.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 584.90 | 589.83 | 585.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:45:00 | 583.75 | 589.83 | 585.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 584.00 | 588.66 | 585.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:30:00 | 585.95 | 587.93 | 585.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 586.75 | 587.93 | 585.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 585.80 | 587.60 | 585.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:45:00 | 585.85 | 586.38 | 585.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 582.15 | 585.53 | 584.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 598.70 | 585.53 | 584.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 595.35 | 589.07 | 587.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 593.95 | 593.96 | 591.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 596.05 | 592.73 | 592.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 599.30 | 594.05 | 592.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 602.65 | 595.79 | 593.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 582.05 | 591.92 | 593.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 236 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 582.05 | 591.92 | 593.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 578.05 | 589.14 | 591.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 595.00 | 588.59 | 590.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 595.00 | 588.59 | 590.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 595.00 | 588.59 | 590.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 595.00 | 588.59 | 590.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 595.00 | 589.87 | 590.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 588.00 | 589.87 | 590.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 583.00 | 577.49 | 576.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 237 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 583.00 | 577.49 | 576.99 | EMA200 above EMA400 |

### Cycle 238 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 573.15 | 576.72 | 576.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 572.10 | 575.79 | 576.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 574.80 | 574.33 | 575.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:15:00 | 561.55 | 574.33 | 575.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 533.47 | 557.70 | 565.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 529.20 | 528.84 | 538.75 | SL hit (close>ema200) qty=0.50 sl=528.84 alert=retest1 |

### Cycle 239 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 489.05 | 485.36 | 484.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 499.65 | 488.80 | 486.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 496.90 | 500.83 | 495.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 496.90 | 500.83 | 495.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 496.90 | 500.83 | 495.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 501.00 | 500.86 | 495.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 489.15 | 499.86 | 499.26 | SL hit (close<static) qty=1.00 sl=494.35 alert=retest2 |

### Cycle 240 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 488.30 | 497.55 | 498.26 | EMA200 below EMA400 |

### Cycle 241 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 522.30 | 500.82 | 498.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 15:15:00 | 525.85 | 505.82 | 500.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 519.30 | 525.07 | 515.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:45:00 | 520.05 | 525.07 | 515.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 515.80 | 523.22 | 515.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 513.15 | 523.22 | 515.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 515.60 | 521.69 | 515.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 515.80 | 521.69 | 515.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 515.65 | 520.48 | 515.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:30:00 | 516.80 | 520.09 | 515.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 542.00 | 520.29 | 516.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 504.25 | 518.64 | 516.23 | SL hit (close<static) qty=1.00 sl=513.05 alert=retest2 |

### Cycle 242 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 504.85 | 513.18 | 514.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 492.10 | 506.60 | 509.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 501.65 | 501.04 | 505.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 501.65 | 501.04 | 505.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 498.40 | 500.21 | 503.75 | EMA400 retest candle locked (from downside) |

### Cycle 243 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 508.30 | 505.54 | 505.24 | EMA200 above EMA400 |

### Cycle 244 — SELL (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 13:15:00 | 501.60 | 504.74 | 504.95 | EMA200 below EMA400 |

### Cycle 245 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 507.35 | 505.43 | 505.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 525.70 | 509.48 | 507.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 522.20 | 524.39 | 517.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 520.05 | 524.39 | 517.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 518.15 | 522.55 | 518.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:45:00 | 518.50 | 522.55 | 518.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 517.50 | 521.54 | 518.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 517.45 | 521.54 | 518.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 516.70 | 520.57 | 518.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 514.85 | 520.57 | 518.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 514.35 | 517.72 | 517.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 516.20 | 517.72 | 517.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 516.60 | 517.49 | 517.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:45:00 | 514.00 | 517.49 | 517.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 516.90 | 517.38 | 517.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 514.70 | 517.38 | 517.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 246 — SELL (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 13:15:00 | 515.45 | 516.99 | 517.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 15:15:00 | 513.00 | 515.71 | 516.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 15:15:00 | 513.95 | 512.44 | 513.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 09:15:00 | 516.30 | 512.44 | 513.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 518.55 | 513.66 | 514.40 | EMA400 retest candle locked (from downside) |

### Cycle 247 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 519.50 | 515.60 | 515.20 | EMA200 above EMA400 |

### Cycle 248 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 513.95 | 515.07 | 515.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 505.05 | 512.97 | 514.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 519.45 | 511.11 | 512.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 519.45 | 511.11 | 512.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 519.45 | 511.11 | 512.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 519.45 | 511.11 | 512.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 249 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 527.35 | 514.36 | 513.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 537.25 | 518.93 | 516.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 10:15:00 | 542.30 | 542.36 | 537.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:00:00 | 542.30 | 542.36 | 537.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 539.80 | 542.25 | 539.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 539.80 | 542.25 | 539.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 538.80 | 541.56 | 539.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 538.10 | 541.56 | 539.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 543.00 | 541.85 | 539.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 538.90 | 541.85 | 539.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 540.45 | 541.57 | 539.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:45:00 | 540.70 | 541.57 | 539.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 540.30 | 541.13 | 540.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 539.65 | 541.13 | 540.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 538.95 | 540.70 | 539.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 533.65 | 540.70 | 539.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 250 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 528.75 | 538.31 | 538.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 09:15:00 | 526.25 | 530.12 | 533.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 14:15:00 | 527.35 | 526.48 | 530.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 527.35 | 526.48 | 530.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 516.60 | 520.75 | 524.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 514.55 | 519.70 | 523.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 514.40 | 520.64 | 522.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 514.80 | 519.75 | 522.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:00:00 | 515.50 | 518.90 | 521.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 520.85 | 518.62 | 520.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 520.85 | 518.62 | 520.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 520.00 | 518.90 | 520.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 525.85 | 518.90 | 520.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 524.25 | 519.97 | 520.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 519.95 | 519.93 | 520.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 519.80 | 520.04 | 520.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:00:00 | 514.45 | 518.93 | 520.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 12:00:00 | 518.15 | 519.16 | 519.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 519.50 | 519.23 | 519.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 519.50 | 519.23 | 519.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 516.00 | 518.58 | 519.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 522.00 | 519.90 | 519.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 251 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 522.00 | 519.90 | 519.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 525.75 | 521.43 | 520.52 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 15:15:00 | 410.00 | 2024-04-19 09:15:00 | 389.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-15 15:15:00 | 410.00 | 2024-04-19 12:15:00 | 406.00 | STOP_HIT | 0.50 | 0.98% |
| BUY | retest2 | 2024-04-26 09:15:00 | 414.65 | 2024-04-26 13:15:00 | 410.20 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-04-26 11:45:00 | 413.05 | 2024-04-26 13:15:00 | 410.20 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-04-26 12:15:00 | 413.70 | 2024-04-26 13:15:00 | 410.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-04-26 13:00:00 | 413.15 | 2024-04-26 13:15:00 | 410.20 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-05-09 12:45:00 | 402.80 | 2024-05-10 12:15:00 | 412.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-05-10 09:30:00 | 403.60 | 2024-05-10 12:15:00 | 412.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-05-10 10:00:00 | 404.35 | 2024-05-10 12:15:00 | 412.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-05-16 09:15:00 | 422.20 | 2024-05-23 09:15:00 | 416.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-05-16 15:00:00 | 421.20 | 2024-05-23 09:15:00 | 416.20 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-05-29 10:15:00 | 425.50 | 2024-05-30 09:15:00 | 418.35 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-06-12 09:15:00 | 444.55 | 2024-06-12 11:15:00 | 443.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-06-20 10:45:00 | 471.80 | 2024-06-26 15:15:00 | 477.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2024-06-24 09:45:00 | 473.15 | 2024-06-26 15:15:00 | 477.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-07-04 09:15:00 | 495.10 | 2024-07-05 10:15:00 | 492.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-04 10:30:00 | 495.10 | 2024-07-05 10:15:00 | 492.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-04 13:30:00 | 496.00 | 2024-07-05 10:15:00 | 492.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-07-04 14:45:00 | 495.05 | 2024-07-05 10:15:00 | 492.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-07-22 12:30:00 | 466.55 | 2024-07-24 11:15:00 | 477.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-07-22 13:00:00 | 464.45 | 2024-07-24 11:15:00 | 477.90 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-07-23 11:00:00 | 466.05 | 2024-07-24 11:15:00 | 477.90 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-07-23 12:15:00 | 461.30 | 2024-07-24 11:15:00 | 477.90 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest1 | 2024-07-25 10:45:00 | 480.90 | 2024-07-29 13:15:00 | 482.15 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest1 | 2024-07-26 09:15:00 | 483.60 | 2024-07-29 13:15:00 | 482.15 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-07-30 09:45:00 | 488.25 | 2024-08-06 13:15:00 | 503.35 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2024-07-30 11:30:00 | 486.65 | 2024-08-06 13:15:00 | 503.35 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2024-07-30 12:15:00 | 486.55 | 2024-08-06 13:15:00 | 503.35 | STOP_HIT | 1.00 | 3.45% |
| BUY | retest2 | 2024-07-30 13:30:00 | 486.20 | 2024-08-06 13:15:00 | 503.35 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2024-08-05 10:15:00 | 520.70 | 2024-08-06 13:15:00 | 503.35 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-08-08 09:15:00 | 531.30 | 2024-08-09 13:15:00 | 514.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-08-08 15:00:00 | 518.55 | 2024-08-09 13:15:00 | 514.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-08-09 12:45:00 | 519.80 | 2024-08-09 13:15:00 | 514.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-08-13 11:15:00 | 525.90 | 2024-08-13 14:15:00 | 510.05 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-08-21 09:30:00 | 527.00 | 2024-08-23 10:15:00 | 579.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 10:45:00 | 527.55 | 2024-08-23 10:15:00 | 580.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 13:15:00 | 528.35 | 2024-08-23 10:15:00 | 581.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-22 10:15:00 | 548.00 | 2024-08-23 13:15:00 | 602.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 09:30:00 | 571.25 | 2024-08-23 14:15:00 | 628.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-02 10:30:00 | 564.85 | 2024-09-04 10:15:00 | 582.70 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-09-03 09:30:00 | 563.30 | 2024-09-04 10:15:00 | 582.70 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-09-03 10:15:00 | 563.95 | 2024-09-04 10:15:00 | 582.70 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-09-03 15:15:00 | 567.90 | 2024-09-04 10:15:00 | 582.70 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-10-04 13:30:00 | 595.80 | 2024-10-07 10:15:00 | 566.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 09:45:00 | 585.50 | 2024-10-08 09:15:00 | 556.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 13:30:00 | 595.80 | 2024-10-09 09:15:00 | 579.35 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2024-10-07 09:45:00 | 585.50 | 2024-10-09 09:15:00 | 579.35 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2024-10-15 10:15:00 | 561.80 | 2024-10-17 10:15:00 | 536.32 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2024-10-15 14:45:00 | 564.55 | 2024-10-17 14:15:00 | 533.71 | PARTIAL | 0.50 | 5.46% |
| SELL | retest2 | 2024-10-16 11:00:00 | 557.85 | 2024-10-17 15:15:00 | 529.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 10:15:00 | 561.80 | 2024-10-18 10:15:00 | 538.00 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2024-10-15 14:45:00 | 564.55 | 2024-10-18 10:15:00 | 538.00 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2024-10-16 11:00:00 | 557.85 | 2024-10-18 10:15:00 | 538.00 | STOP_HIT | 0.50 | 3.56% |
| BUY | retest2 | 2024-10-31 15:00:00 | 515.55 | 2024-11-04 09:15:00 | 499.85 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-11-01 18:00:00 | 518.60 | 2024-11-04 09:15:00 | 499.85 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-11-11 12:30:00 | 504.85 | 2024-11-13 14:15:00 | 479.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 12:30:00 | 504.85 | 2024-11-14 09:15:00 | 492.70 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-11-25 12:00:00 | 478.30 | 2024-11-25 14:15:00 | 486.65 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-12-02 09:15:00 | 509.60 | 2024-12-05 10:15:00 | 498.75 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-12-20 10:15:00 | 511.15 | 2024-12-30 09:15:00 | 485.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 13:00:00 | 511.45 | 2024-12-30 09:15:00 | 485.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:30:00 | 509.40 | 2024-12-30 10:15:00 | 483.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 10:15:00 | 511.15 | 2024-12-31 11:15:00 | 488.75 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2024-12-20 13:00:00 | 511.45 | 2024-12-31 11:15:00 | 488.75 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2024-12-20 14:30:00 | 509.40 | 2024-12-31 11:15:00 | 488.75 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-01-24 09:45:00 | 569.20 | 2025-01-27 09:15:00 | 544.16 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-01-24 12:00:00 | 572.80 | 2025-01-27 10:15:00 | 540.74 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-01-24 09:45:00 | 569.20 | 2025-01-28 14:15:00 | 544.00 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-01-24 12:00:00 | 572.80 | 2025-01-28 14:15:00 | 544.00 | STOP_HIT | 0.50 | 5.03% |
| BUY | retest2 | 2025-02-04 10:00:00 | 572.20 | 2025-02-07 10:15:00 | 562.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-02-06 12:15:00 | 569.50 | 2025-02-07 10:15:00 | 562.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-02-12 14:15:00 | 532.50 | 2025-02-17 09:15:00 | 505.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:30:00 | 532.50 | 2025-02-17 09:15:00 | 505.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 530.80 | 2025-02-17 09:15:00 | 504.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 14:15:00 | 532.50 | 2025-02-17 13:15:00 | 516.25 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-02-13 14:30:00 | 532.50 | 2025-02-17 13:15:00 | 516.25 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-02-13 15:00:00 | 530.80 | 2025-02-17 13:15:00 | 516.25 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-02-14 09:15:00 | 523.10 | 2025-02-18 12:15:00 | 496.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 523.10 | 2025-02-18 15:15:00 | 505.00 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-02-18 12:45:00 | 497.90 | 2025-02-19 09:15:00 | 540.50 | STOP_HIT | 1.00 | -8.56% |
| SELL | retest2 | 2025-03-04 09:15:00 | 484.35 | 2025-03-04 11:15:00 | 496.70 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-03-21 09:45:00 | 536.25 | 2025-03-25 11:15:00 | 531.35 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-03-21 10:15:00 | 536.65 | 2025-03-25 11:15:00 | 531.35 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-03-21 12:00:00 | 536.50 | 2025-03-27 11:15:00 | 527.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-03-21 14:30:00 | 536.80 | 2025-03-27 11:15:00 | 527.70 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-03-24 15:15:00 | 541.20 | 2025-03-27 11:15:00 | 527.70 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-03-25 09:30:00 | 542.35 | 2025-03-27 11:15:00 | 527.70 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-03-25 14:45:00 | 540.00 | 2025-03-27 11:15:00 | 527.70 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-03-26 09:15:00 | 545.65 | 2025-03-27 11:15:00 | 527.70 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-05-06 09:45:00 | 489.05 | 2025-05-08 11:15:00 | 492.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-05-08 10:30:00 | 491.10 | 2025-05-08 11:15:00 | 492.90 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-05-19 10:15:00 | 517.70 | 2025-05-20 13:15:00 | 500.50 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-06-18 15:15:00 | 535.70 | 2025-06-19 13:15:00 | 508.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 15:15:00 | 535.70 | 2025-06-24 09:15:00 | 512.00 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2025-07-01 11:15:00 | 521.50 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-01 14:30:00 | 521.20 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-02 09:15:00 | 524.95 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-07-02 10:00:00 | 524.80 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-11 10:00:00 | 511.00 | 2025-07-14 13:15:00 | 517.35 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-07-25 09:30:00 | 509.00 | 2025-08-01 13:15:00 | 483.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 09:45:00 | 510.80 | 2025-08-01 13:15:00 | 485.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 15:00:00 | 509.35 | 2025-08-01 13:15:00 | 483.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 512.15 | 2025-08-01 13:15:00 | 486.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 509.00 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-07-28 09:45:00 | 510.80 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-07-28 15:00:00 | 509.35 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2025-07-30 10:15:00 | 512.15 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-08-05 09:15:00 | 491.40 | 2025-08-06 10:15:00 | 466.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:15:00 | 491.40 | 2025-08-11 14:15:00 | 460.15 | STOP_HIT | 0.50 | 6.36% |
| SELL | retest2 | 2025-08-28 12:15:00 | 502.50 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-28 12:45:00 | 502.40 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-28 13:30:00 | 502.65 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-28 14:00:00 | 502.70 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-03 09:15:00 | 509.70 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-03 10:00:00 | 508.75 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-03 12:30:00 | 508.60 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-04 09:15:00 | 509.05 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-10 09:30:00 | 520.65 | 2025-09-10 11:15:00 | 511.30 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-17 10:30:00 | 520.60 | 2025-09-23 14:15:00 | 533.60 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-09-29 10:00:00 | 586.75 | 2025-09-30 13:15:00 | 568.95 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-10-07 09:30:00 | 582.20 | 2025-10-08 15:15:00 | 580.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-10-15 14:00:00 | 565.00 | 2025-10-15 15:15:00 | 570.40 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-30 14:15:00 | 569.65 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-30 15:15:00 | 570.90 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-31 11:45:00 | 575.50 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-10-31 13:45:00 | 569.90 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-07 09:15:00 | 610.90 | 2025-11-07 09:15:00 | 579.65 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2025-12-01 11:45:00 | 582.90 | 2025-12-02 13:15:00 | 589.35 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-04 10:30:00 | 591.95 | 2025-12-04 11:15:00 | 585.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-19 12:15:00 | 592.05 | 2025-12-19 13:15:00 | 595.25 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-12-29 09:45:00 | 573.65 | 2025-12-31 09:15:00 | 572.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2026-01-13 11:00:00 | 576.00 | 2026-01-19 09:15:00 | 547.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 575.20 | 2026-01-19 09:15:00 | 546.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:00:00 | 576.00 | 2026-01-21 10:15:00 | 518.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 575.20 | 2026-01-21 10:15:00 | 517.68 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-09 09:30:00 | 585.95 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-02-09 10:15:00 | 586.75 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-09 11:45:00 | 585.80 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-09 14:45:00 | 585.85 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-10 09:15:00 | 598.70 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-11 09:15:00 | 595.35 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-02-12 09:30:00 | 593.95 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-13 09:15:00 | 596.05 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-02-13 12:15:00 | 602.65 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2026-02-17 09:15:00 | 588.00 | 2026-02-25 15:15:00 | 583.00 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest1 | 2026-02-27 09:15:00 | 561.55 | 2026-03-02 09:15:00 | 533.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-27 09:15:00 | 561.55 | 2026-03-05 09:15:00 | 529.20 | STOP_HIT | 0.50 | 5.76% |
| SELL | retest2 | 2026-03-06 09:30:00 | 525.05 | 2026-03-09 09:15:00 | 498.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:30:00 | 525.05 | 2026-03-10 09:15:00 | 509.20 | STOP_HIT | 0.50 | 3.02% |
| BUY | retest2 | 2026-03-19 11:00:00 | 501.00 | 2026-03-23 09:15:00 | 489.15 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-03-27 13:30:00 | 516.80 | 2026-03-30 09:15:00 | 504.25 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-27 14:45:00 | 542.00 | 2026-03-30 09:15:00 | 504.25 | STOP_HIT | 1.00 | -6.96% |
| SELL | retest2 | 2026-04-29 10:30:00 | 514.55 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-30 09:15:00 | 514.40 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-30 10:15:00 | 514.80 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-30 11:00:00 | 515.50 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-04 11:45:00 | 519.95 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-05-04 13:15:00 | 519.80 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-04 14:00:00 | 514.45 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-05-05 12:00:00 | 518.15 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -0.74% |

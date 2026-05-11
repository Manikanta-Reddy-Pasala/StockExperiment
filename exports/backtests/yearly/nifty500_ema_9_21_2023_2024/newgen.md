# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 506.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 138 |
| ALERT2 | 134 |
| ALERT2_SKIP | 95 |
| ALERT3 | 277 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 110 |
| PARTIAL | 15 |
| TARGET_HIT | 13 |
| STOP_HIT | 100 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 83
- **Target hits / Stop hits / Partials:** 13 / 100 / 15
- **Avg / median % per leg:** 0.23% / -1.24%
- **Sum % (uncompounded):** 29.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 14 | 35.9% | 10 | 28 | 1 | 1.40% | 54.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 2.98% | 6.0% |
| BUY @ 3rd Alert (retest2) | 37 | 12 | 32.4% | 10 | 27 | 0 | 1.31% | 48.6% |
| SELL (all) | 89 | 31 | 34.8% | 3 | 72 | 14 | -0.28% | -25.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.01% | -6.0% |
| SELL @ 3rd Alert (retest2) | 87 | 31 | 35.6% | 3 | 70 | 14 | -0.22% | -19.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.01% | -0.1% |
| retest2 (combined) | 124 | 43 | 34.7% | 13 | 97 | 14 | 0.24% | 29.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 13:15:00 | 289.60 | 292.09 | 292.24 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 300.80 | 293.00 | 292.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 09:15:00 | 303.55 | 297.05 | 295.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 09:15:00 | 302.52 | 304.29 | 300.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 302.52 | 304.29 | 300.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 302.52 | 304.29 | 300.91 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 14:15:00 | 310.00 | 311.05 | 311.17 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 313.43 | 311.36 | 311.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 09:15:00 | 321.45 | 314.57 | 312.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 14:15:00 | 333.98 | 336.43 | 333.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 15:15:00 | 334.18 | 335.98 | 333.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 334.18 | 335.98 | 333.17 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 328.93 | 335.45 | 336.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 13:15:00 | 325.60 | 332.34 | 334.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 15:15:00 | 325.30 | 325.24 | 328.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 332.05 | 326.60 | 328.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 332.05 | 326.60 | 328.84 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 12:15:00 | 327.88 | 324.80 | 324.70 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 15:15:00 | 323.55 | 325.24 | 325.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 09:15:00 | 320.00 | 324.19 | 324.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 11:15:00 | 323.85 | 323.61 | 324.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 11:15:00 | 323.85 | 323.61 | 324.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 323.85 | 323.61 | 324.41 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 332.00 | 324.30 | 324.27 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 321.90 | 324.29 | 324.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 10:15:00 | 321.27 | 323.68 | 324.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 320.35 | 318.48 | 320.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 12:15:00 | 323.00 | 319.39 | 320.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 323.00 | 319.39 | 320.85 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 320.85 | 318.25 | 317.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 330.75 | 321.60 | 319.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 10:15:00 | 334.63 | 335.13 | 329.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 14:15:00 | 331.70 | 333.67 | 330.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 331.70 | 333.67 | 330.64 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 326.90 | 329.56 | 329.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 12:15:00 | 325.48 | 326.91 | 327.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 09:15:00 | 326.25 | 325.61 | 326.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 326.25 | 325.61 | 326.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 326.25 | 325.61 | 326.72 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 322.50 | 315.38 | 314.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 14:15:00 | 323.52 | 319.09 | 316.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 340.90 | 342.67 | 335.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 12:15:00 | 359.00 | 350.25 | 343.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 359.00 | 350.25 | 343.71 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 12:15:00 | 406.13 | 411.34 | 411.52 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 427.20 | 411.93 | 411.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 10:15:00 | 433.30 | 416.20 | 413.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 422.70 | 422.98 | 419.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 420.73 | 423.01 | 420.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 420.73 | 423.01 | 420.95 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 411.35 | 419.45 | 420.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 11:15:00 | 408.50 | 416.03 | 418.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 426.28 | 414.32 | 416.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 426.28 | 414.32 | 416.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 426.28 | 414.32 | 416.09 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 421.48 | 417.55 | 417.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 423.38 | 419.99 | 418.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 11:15:00 | 418.40 | 419.92 | 418.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 11:15:00 | 418.40 | 419.92 | 418.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 418.40 | 419.92 | 418.85 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 13:15:00 | 454.35 | 454.80 | 454.83 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 15:15:00 | 454.95 | 454.86 | 454.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 09:15:00 | 457.53 | 455.39 | 455.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 480.45 | 480.50 | 472.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 14:15:00 | 472.50 | 477.77 | 473.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 472.50 | 477.77 | 473.63 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 11:15:00 | 459.98 | 471.26 | 471.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 447.20 | 461.50 | 466.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 14:15:00 | 459.93 | 457.57 | 462.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 15:15:00 | 464.00 | 458.85 | 462.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 464.00 | 458.85 | 462.64 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 13:15:00 | 446.23 | 442.75 | 442.69 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 14:15:00 | 420.20 | 438.24 | 440.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 15:15:00 | 418.50 | 434.29 | 438.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 12:15:00 | 414.65 | 413.34 | 421.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 13:15:00 | 420.83 | 414.83 | 421.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 420.83 | 414.83 | 421.06 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 12:15:00 | 429.00 | 423.85 | 423.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 430.15 | 425.11 | 424.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 10:15:00 | 442.93 | 444.85 | 440.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 439.40 | 443.83 | 442.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 439.40 | 443.83 | 442.05 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 11:15:00 | 436.00 | 440.87 | 440.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 13:15:00 | 432.78 | 438.08 | 439.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 425.00 | 423.40 | 428.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 427.15 | 424.79 | 427.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 427.15 | 424.79 | 427.81 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 440.00 | 430.04 | 429.55 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 13:15:00 | 427.23 | 434.87 | 434.98 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 15:15:00 | 434.90 | 431.23 | 431.05 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 09:15:00 | 429.25 | 430.84 | 430.89 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 10:15:00 | 434.90 | 431.65 | 431.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 09:15:00 | 445.60 | 437.33 | 434.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 10:15:00 | 452.93 | 454.51 | 447.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 442.90 | 450.91 | 447.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 442.90 | 450.91 | 447.72 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 437.40 | 447.13 | 447.31 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 445.20 | 442.14 | 442.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 15:15:00 | 450.08 | 443.73 | 442.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 12:15:00 | 448.18 | 452.20 | 449.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 12:15:00 | 448.18 | 452.20 | 449.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 12:15:00 | 448.18 | 452.20 | 449.52 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 14:15:00 | 520.63 | 547.79 | 550.94 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 11:15:00 | 556.95 | 552.12 | 551.76 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 545.00 | 550.69 | 551.14 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 13:15:00 | 557.05 | 551.96 | 551.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-25 14:15:00 | 562.75 | 554.12 | 552.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-26 09:15:00 | 555.00 | 556.04 | 553.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 09:15:00 | 555.00 | 556.04 | 553.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 555.00 | 556.04 | 553.91 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 565.10 | 578.79 | 580.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 558.10 | 564.44 | 568.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 563.50 | 563.38 | 567.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 563.50 | 563.38 | 567.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 563.50 | 563.38 | 567.01 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 13:15:00 | 579.00 | 569.56 | 568.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 11:15:00 | 584.40 | 576.56 | 573.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 13:15:00 | 575.60 | 578.12 | 574.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 15:15:00 | 577.50 | 577.74 | 574.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 577.50 | 577.74 | 574.88 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 12:15:00 | 585.55 | 595.01 | 595.87 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 10:15:00 | 602.00 | 594.64 | 594.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 11:15:00 | 610.00 | 598.08 | 596.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 09:15:00 | 633.13 | 656.84 | 646.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 633.13 | 656.84 | 646.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 633.13 | 656.84 | 646.84 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 11:15:00 | 640.00 | 648.02 | 648.75 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 672.00 | 650.43 | 649.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 696.00 | 672.53 | 662.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 680.00 | 686.31 | 676.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 680.00 | 686.31 | 676.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 680.00 | 686.31 | 676.63 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 14:15:00 | 709.75 | 712.62 | 712.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 09:15:00 | 706.00 | 710.50 | 711.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 709.98 | 702.99 | 706.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 709.98 | 702.99 | 706.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 709.98 | 702.99 | 706.34 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 15:15:00 | 709.50 | 707.96 | 707.82 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 10:15:00 | 705.00 | 707.70 | 707.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 700.03 | 705.73 | 706.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 15:15:00 | 705.00 | 704.00 | 705.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 699.50 | 703.10 | 705.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 699.50 | 703.10 | 705.06 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 709.00 | 704.30 | 704.11 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 15:15:00 | 700.50 | 703.44 | 703.80 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 711.25 | 705.00 | 704.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 10:15:00 | 714.38 | 706.88 | 705.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 13:15:00 | 721.40 | 722.39 | 717.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 710.53 | 720.51 | 717.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 710.53 | 720.51 | 717.62 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 14:15:00 | 713.00 | 715.89 | 716.09 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 722.30 | 717.43 | 716.77 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 699.25 | 714.21 | 715.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 697.00 | 709.94 | 713.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 710.00 | 708.82 | 711.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 11:15:00 | 710.00 | 708.82 | 711.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 710.00 | 708.82 | 711.87 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 716.30 | 711.59 | 711.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 15:15:00 | 723.50 | 715.25 | 713.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 764.98 | 765.17 | 756.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 10:15:00 | 759.98 | 764.82 | 758.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 759.98 | 764.82 | 758.19 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 09:15:00 | 784.00 | 796.92 | 798.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 780.00 | 787.14 | 791.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 780.00 | 777.14 | 783.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 780.00 | 777.14 | 783.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 780.00 | 777.14 | 783.39 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 817.30 | 787.38 | 785.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 09:15:00 | 830.00 | 813.53 | 802.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 869.00 | 878.27 | 856.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 13:15:00 | 859.00 | 872.82 | 860.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 13:15:00 | 859.00 | 872.82 | 860.41 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 11:15:00 | 840.00 | 855.08 | 855.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 14:15:00 | 834.90 | 846.13 | 851.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 847.00 | 844.36 | 849.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 847.00 | 844.36 | 849.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 847.00 | 844.36 | 849.33 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 855.00 | 827.57 | 824.33 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 10:15:00 | 827.10 | 838.55 | 839.23 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 844.95 | 839.45 | 839.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 14:15:00 | 850.00 | 841.56 | 840.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 15:15:00 | 840.25 | 841.30 | 840.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 15:15:00 | 840.25 | 841.30 | 840.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 840.25 | 841.30 | 840.36 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 845.00 | 848.36 | 848.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 810.00 | 839.15 | 844.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 11:15:00 | 832.50 | 830.54 | 837.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 13:15:00 | 841.00 | 833.50 | 837.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 841.00 | 833.50 | 837.85 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 758.85 | 747.67 | 747.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 13:15:00 | 768.75 | 751.89 | 749.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 15:15:00 | 781.80 | 782.56 | 770.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 768.15 | 779.67 | 770.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 768.15 | 779.67 | 770.20 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 761.10 | 766.02 | 766.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 15:15:00 | 748.75 | 762.57 | 764.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 09:15:00 | 762.65 | 762.58 | 764.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 762.65 | 762.58 | 764.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 762.65 | 762.58 | 764.32 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 12:15:00 | 771.50 | 765.97 | 765.55 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 15:15:00 | 740.00 | 760.82 | 763.33 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 775.00 | 763.95 | 762.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 10:15:00 | 783.50 | 767.86 | 764.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 10:15:00 | 821.20 | 823.75 | 806.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 822.40 | 827.69 | 816.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 822.40 | 827.69 | 816.89 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 794.40 | 810.80 | 811.80 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 850.05 | 816.92 | 813.57 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 804.20 | 837.06 | 841.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 803.00 | 830.25 | 837.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 13:15:00 | 746.55 | 742.03 | 763.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 14:15:00 | 765.15 | 746.66 | 763.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 765.15 | 746.66 | 763.90 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 760.00 | 737.96 | 737.03 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 728.00 | 735.97 | 736.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 10:15:00 | 718.05 | 732.38 | 734.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 722.20 | 717.24 | 721.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 10:15:00 | 722.20 | 717.24 | 721.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 722.20 | 717.24 | 721.45 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 743.70 | 725.48 | 723.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 747.15 | 736.49 | 731.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 12:15:00 | 792.95 | 793.36 | 785.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 14:15:00 | 793.65 | 799.60 | 794.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 793.65 | 799.60 | 794.35 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 09:15:00 | 790.95 | 792.62 | 792.63 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 15:15:00 | 796.90 | 792.35 | 792.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 807.05 | 795.29 | 793.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 13:15:00 | 798.70 | 798.92 | 796.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 14:15:00 | 799.00 | 798.93 | 796.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 799.00 | 798.93 | 796.47 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 12:15:00 | 803.65 | 814.50 | 814.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 799.45 | 808.34 | 811.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 765.90 | 765.63 | 779.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:00:00 | 765.90 | 765.63 | 779.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 761.40 | 761.24 | 772.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:30:00 | 765.05 | 761.24 | 772.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 747.95 | 760.71 | 769.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 744.05 | 760.71 | 769.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 11:15:00 | 745.30 | 758.19 | 767.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 12:45:00 | 745.60 | 754.21 | 764.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 743.00 | 754.21 | 764.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 744.85 | 740.30 | 747.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:30:00 | 744.50 | 740.30 | 747.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 737.00 | 738.81 | 743.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 752.90 | 744.87 | 744.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 752.90 | 744.87 | 744.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 12:15:00 | 755.80 | 748.01 | 745.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 801.00 | 805.91 | 789.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 801.00 | 805.91 | 789.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 997.00 | 1001.72 | 982.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 985.50 | 1001.72 | 982.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 987.35 | 998.84 | 982.78 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 979.90 | 989.36 | 989.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 963.70 | 984.23 | 987.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 985.00 | 971.17 | 978.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 985.00 | 971.17 | 978.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 985.00 | 971.17 | 978.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:00:00 | 985.00 | 971.17 | 978.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 989.65 | 974.87 | 979.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 989.65 | 974.87 | 979.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 979.80 | 975.85 | 979.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 15:15:00 | 977.95 | 980.27 | 981.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 929.05 | 970.39 | 976.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-14 13:15:00 | 961.25 | 950.34 | 957.29 | SL hit (close>ema200) qty=0.50 sl=950.34 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 990.65 | 961.28 | 960.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1006.05 | 984.35 | 974.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 982.30 | 987.31 | 979.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 982.30 | 987.31 | 979.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 982.30 | 987.31 | 979.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:15:00 | 996.00 | 984.88 | 981.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 11:30:00 | 991.05 | 989.18 | 985.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 10:15:00 | 974.95 | 984.56 | 984.12 | SL hit (close<static) qty=1.00 sl=976.50 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 965.55 | 980.75 | 982.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 958.05 | 976.21 | 980.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 880.75 | 874.51 | 883.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 12:15:00 | 880.75 | 874.51 | 883.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 880.75 | 874.51 | 883.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:00:00 | 880.75 | 874.51 | 883.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 869.10 | 858.04 | 864.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 869.10 | 858.04 | 864.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 884.00 | 863.23 | 866.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 879.90 | 863.23 | 866.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 904.40 | 871.46 | 869.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 916.55 | 880.48 | 874.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 11:15:00 | 887.90 | 890.43 | 881.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 12:00:00 | 887.90 | 890.43 | 881.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 861.40 | 885.79 | 883.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 855.95 | 885.79 | 883.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 840.15 | 876.66 | 879.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 836.90 | 868.71 | 875.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 863.20 | 862.49 | 870.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 15:00:00 | 863.20 | 862.49 | 870.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 888.80 | 867.75 | 872.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 852.20 | 864.87 | 870.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 13:15:00 | 880.00 | 873.02 | 872.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 880.00 | 873.02 | 872.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 903.00 | 882.09 | 877.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 15:15:00 | 885.00 | 888.95 | 883.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 15:15:00 | 885.00 | 888.95 | 883.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 885.00 | 888.95 | 883.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 923.70 | 888.95 | 883.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 11:15:00 | 927.85 | 932.24 | 932.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 927.85 | 932.24 | 932.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 14:15:00 | 924.55 | 929.73 | 931.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 918.00 | 917.90 | 922.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 13:15:00 | 925.15 | 919.98 | 922.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 925.15 | 919.98 | 922.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:00:00 | 925.15 | 919.98 | 922.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 922.00 | 920.39 | 922.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:15:00 | 930.00 | 920.39 | 922.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 930.00 | 922.31 | 923.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 941.15 | 922.31 | 923.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 09:15:00 | 946.00 | 927.05 | 925.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 11:15:00 | 961.90 | 938.51 | 931.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 12:15:00 | 949.00 | 956.22 | 947.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 12:15:00 | 949.00 | 956.22 | 947.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 949.00 | 956.22 | 947.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:45:00 | 949.50 | 956.22 | 947.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 949.90 | 954.95 | 947.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 949.90 | 954.95 | 947.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 951.40 | 954.24 | 947.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 951.40 | 954.24 | 947.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 1001.95 | 994.20 | 983.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 1001.95 | 994.20 | 983.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 996.20 | 996.65 | 986.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 996.20 | 996.65 | 986.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 997.40 | 1006.52 | 998.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 997.40 | 1006.52 | 998.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 996.95 | 1004.61 | 997.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 996.95 | 1004.61 | 997.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1000.00 | 1003.69 | 998.12 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 959.00 | 988.56 | 992.25 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 989.75 | 982.59 | 981.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 1002.95 | 986.66 | 983.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 996.50 | 997.55 | 991.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 11:45:00 | 995.55 | 997.55 | 991.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 986.25 | 995.29 | 991.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 986.25 | 995.29 | 991.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 994.65 | 995.16 | 991.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 1003.25 | 993.09 | 991.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:00:00 | 1000.00 | 999.43 | 996.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 1000.65 | 998.54 | 996.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 12:15:00 | 984.00 | 996.34 | 995.63 | SL hit (close<static) qty=1.00 sl=985.55 alert=retest2 |

### Cycle 83 — SELL (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 13:15:00 | 973.00 | 991.67 | 993.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 14:15:00 | 967.90 | 986.92 | 991.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 950.45 | 949.60 | 959.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 11:00:00 | 950.45 | 949.60 | 959.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 949.10 | 949.97 | 956.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 949.10 | 949.97 | 956.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 940.00 | 948.30 | 954.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 934.55 | 948.30 | 954.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:15:00 | 936.15 | 947.17 | 953.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 14:15:00 | 962.00 | 951.02 | 953.20 | SL hit (close>static) qty=1.00 sl=958.40 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 15:15:00 | 980.60 | 956.94 | 955.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 998.85 | 969.68 | 963.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 1016.05 | 1036.92 | 1025.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 1016.05 | 1036.92 | 1025.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1016.05 | 1036.92 | 1025.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1005.25 | 1036.92 | 1025.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1018.95 | 1033.33 | 1024.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:30:00 | 1016.05 | 1033.33 | 1024.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1037.90 | 1034.24 | 1026.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:15:00 | 997.00 | 1034.24 | 1026.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1013.05 | 1030.00 | 1024.89 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1002.00 | 1022.33 | 1022.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 1000.00 | 1010.82 | 1016.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 1012.55 | 1010.05 | 1014.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 1012.55 | 1010.05 | 1014.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1012.55 | 1010.05 | 1014.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 1012.55 | 1010.05 | 1014.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1013.40 | 1010.72 | 1014.20 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 1040.30 | 1020.61 | 1017.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 15:15:00 | 1075.00 | 1044.70 | 1032.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1044.60 | 1068.95 | 1056.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1044.60 | 1068.95 | 1056.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1044.60 | 1068.95 | 1056.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 1044.60 | 1068.95 | 1056.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 1063.20 | 1067.80 | 1057.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:45:00 | 1063.90 | 1066.25 | 1057.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:00:00 | 1067.90 | 1064.59 | 1058.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:30:00 | 1072.25 | 1064.25 | 1059.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 10:15:00 | 1039.10 | 1065.98 | 1068.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 10:15:00 | 1039.10 | 1065.98 | 1068.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 12:15:00 | 1032.05 | 1054.83 | 1062.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 1064.20 | 1040.32 | 1045.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 1064.20 | 1040.32 | 1045.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1064.20 | 1040.32 | 1045.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 1061.65 | 1040.32 | 1045.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1061.95 | 1044.65 | 1046.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:00:00 | 1055.30 | 1046.78 | 1047.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 1060.00 | 1049.42 | 1048.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 12:15:00 | 1060.00 | 1049.42 | 1048.77 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 1039.25 | 1047.67 | 1048.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 1032.20 | 1044.58 | 1047.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 973.40 | 971.64 | 990.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 969.25 | 971.40 | 988.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:45:00 | 966.00 | 967.78 | 985.66 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 978.20 | 968.62 | 982.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:30:00 | 983.60 | 968.62 | 982.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 982.80 | 971.45 | 982.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 984.20 | 971.45 | 982.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 976.00 | 972.36 | 982.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 981.90 | 972.36 | 982.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 980.30 | 975.67 | 981.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 996.70 | 980.42 | 982.63 | SL hit (close>ema400) qty=1.00 sl=982.63 alert=retest1 |

### Cycle 90 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 1016.10 | 987.55 | 985.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 15:15:00 | 1020.00 | 1006.51 | 996.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 1046.60 | 1050.93 | 1034.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 1046.60 | 1050.93 | 1034.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1047.80 | 1051.53 | 1041.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 1046.20 | 1051.53 | 1041.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1043.00 | 1049.82 | 1041.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1043.00 | 1049.82 | 1041.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1039.70 | 1047.80 | 1041.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1039.70 | 1047.80 | 1041.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1031.60 | 1044.56 | 1040.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1026.55 | 1042.42 | 1040.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 1022.10 | 1038.35 | 1038.38 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1073.40 | 1041.57 | 1038.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 1086.80 | 1056.27 | 1046.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 10:15:00 | 1072.25 | 1072.37 | 1059.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 11:00:00 | 1072.25 | 1072.37 | 1059.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1098.00 | 1100.05 | 1088.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:30:00 | 1088.05 | 1100.05 | 1088.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 1090.35 | 1098.11 | 1088.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:45:00 | 1089.15 | 1098.11 | 1088.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1081.05 | 1093.95 | 1088.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 1081.05 | 1093.95 | 1088.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1086.65 | 1092.49 | 1088.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:15:00 | 1092.00 | 1092.49 | 1088.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 1073.00 | 1084.56 | 1085.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 1073.00 | 1084.56 | 1085.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 1065.35 | 1076.79 | 1081.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 14:15:00 | 1074.30 | 1072.03 | 1076.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 14:15:00 | 1074.30 | 1072.03 | 1076.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1074.30 | 1072.03 | 1076.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 09:45:00 | 1065.00 | 1071.47 | 1075.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 1064.70 | 1051.00 | 1052.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 1062.60 | 1055.38 | 1054.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1062.60 | 1055.38 | 1054.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 1074.85 | 1063.66 | 1060.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 1082.75 | 1087.12 | 1076.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:45:00 | 1082.50 | 1087.12 | 1076.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1076.60 | 1085.01 | 1076.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 1070.30 | 1085.01 | 1076.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1078.00 | 1083.61 | 1076.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:30:00 | 1072.00 | 1083.61 | 1076.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1080.15 | 1082.92 | 1077.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:30:00 | 1078.25 | 1082.92 | 1077.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1083.90 | 1083.11 | 1077.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1097.80 | 1083.11 | 1077.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 1071.00 | 1080.72 | 1079.95 | SL hit (close<static) qty=1.00 sl=1072.80 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 1065.50 | 1077.67 | 1078.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 1062.30 | 1073.21 | 1076.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 1072.25 | 1066.66 | 1071.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1072.25 | 1066.66 | 1071.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1072.25 | 1066.66 | 1071.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:00:00 | 1072.25 | 1066.66 | 1071.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1093.85 | 1072.10 | 1073.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:30:00 | 1091.10 | 1072.10 | 1073.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1082.50 | 1074.18 | 1074.41 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 1081.80 | 1075.70 | 1075.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 1089.95 | 1081.40 | 1078.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 13:15:00 | 1083.35 | 1084.89 | 1081.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 14:00:00 | 1083.35 | 1084.89 | 1081.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1096.30 | 1087.18 | 1082.57 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 1076.05 | 1080.50 | 1080.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 15:15:00 | 1076.00 | 1079.98 | 1080.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 15:15:00 | 1074.10 | 1073.35 | 1076.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 15:15:00 | 1074.10 | 1073.35 | 1076.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1074.10 | 1073.35 | 1076.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 1082.95 | 1073.35 | 1076.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1065.30 | 1071.74 | 1075.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:30:00 | 1055.00 | 1068.32 | 1073.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 1086.35 | 1071.39 | 1071.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 1086.35 | 1071.39 | 1071.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 12:15:00 | 1098.40 | 1079.37 | 1075.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 1170.00 | 1171.44 | 1148.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 12:30:00 | 1186.75 | 1182.00 | 1161.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 13:15:00 | 1246.09 | 1190.50 | 1167.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 1198.05 | 1202.15 | 1179.56 | SL hit (close<ema200) qty=0.50 sl=1202.15 alert=retest1 |

### Cycle 99 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 1322.65 | 1354.99 | 1358.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 12:15:00 | 1313.50 | 1336.51 | 1348.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 15:15:00 | 1307.00 | 1303.41 | 1318.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 15:15:00 | 1307.00 | 1303.41 | 1318.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1307.00 | 1303.41 | 1318.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 1280.45 | 1303.41 | 1318.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 13:00:00 | 1285.60 | 1298.87 | 1306.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 11:45:00 | 1283.90 | 1274.01 | 1287.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 1286.00 | 1266.92 | 1278.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1284.85 | 1270.51 | 1278.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 1261.85 | 1270.46 | 1278.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 1328.85 | 1281.90 | 1280.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 1328.85 | 1281.90 | 1280.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 10:15:00 | 1373.90 | 1321.43 | 1303.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1350.10 | 1351.72 | 1329.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:45:00 | 1349.50 | 1351.72 | 1329.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1354.55 | 1355.40 | 1343.07 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 13:15:00 | 1268.70 | 1335.48 | 1337.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 1239.00 | 1256.69 | 1278.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 1264.40 | 1254.60 | 1273.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 11:00:00 | 1264.40 | 1254.60 | 1273.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1270.45 | 1261.14 | 1268.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 1258.70 | 1261.99 | 1268.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 1257.30 | 1260.22 | 1266.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 1241.65 | 1262.16 | 1266.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 1195.77 | 1231.79 | 1249.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 1194.43 | 1231.79 | 1249.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 1179.57 | 1223.20 | 1244.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 1246.55 | 1215.28 | 1232.44 | SL hit (close>ema200) qty=0.50 sl=1215.28 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 13:15:00 | 1257.30 | 1240.57 | 1240.51 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 1227.80 | 1238.01 | 1239.36 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 1258.80 | 1242.43 | 1241.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 10:15:00 | 1273.80 | 1248.71 | 1244.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 1242.85 | 1256.79 | 1251.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 1242.85 | 1256.79 | 1251.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1242.85 | 1256.79 | 1251.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 1242.85 | 1256.79 | 1251.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 1241.75 | 1253.78 | 1250.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 1236.20 | 1253.78 | 1250.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 1250.05 | 1253.04 | 1250.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:45:00 | 1245.00 | 1253.04 | 1250.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 1247.55 | 1251.94 | 1250.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 13:30:00 | 1250.50 | 1251.15 | 1250.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 1230.95 | 1247.11 | 1248.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 1230.95 | 1247.11 | 1248.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1215.25 | 1238.80 | 1244.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 1224.90 | 1213.47 | 1222.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 1224.90 | 1213.47 | 1222.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1224.90 | 1213.47 | 1222.01 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 1245.95 | 1226.60 | 1226.55 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 1223.40 | 1225.96 | 1226.26 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 1233.80 | 1227.53 | 1226.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 1247.00 | 1231.18 | 1228.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 14:15:00 | 1285.65 | 1286.01 | 1270.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 1285.65 | 1286.01 | 1270.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1277.80 | 1283.12 | 1274.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 1277.80 | 1283.12 | 1274.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 1275.35 | 1281.56 | 1274.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:30:00 | 1272.20 | 1281.56 | 1274.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1280.90 | 1281.43 | 1274.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:30:00 | 1289.95 | 1283.14 | 1277.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:45:00 | 1290.00 | 1284.55 | 1278.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 1275.95 | 1286.36 | 1286.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1275.95 | 1286.36 | 1286.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1268.30 | 1282.75 | 1285.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1272.00 | 1271.33 | 1277.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 09:45:00 | 1270.80 | 1271.33 | 1277.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1289.65 | 1274.99 | 1278.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 1286.95 | 1274.99 | 1278.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1281.20 | 1276.23 | 1279.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 1276.30 | 1276.23 | 1279.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 1303.20 | 1283.68 | 1281.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 1303.20 | 1283.68 | 1281.73 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 1263.55 | 1280.46 | 1282.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1215.80 | 1267.53 | 1276.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 1236.65 | 1223.11 | 1240.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 1236.65 | 1223.11 | 1240.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1244.45 | 1229.31 | 1240.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:45:00 | 1244.90 | 1229.31 | 1240.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1235.45 | 1230.53 | 1240.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 1242.00 | 1230.53 | 1240.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1104.15 | 1094.87 | 1129.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:30:00 | 1137.45 | 1094.87 | 1129.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1100.05 | 1090.97 | 1100.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 1100.05 | 1090.97 | 1100.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1096.90 | 1092.16 | 1099.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:30:00 | 1095.65 | 1092.16 | 1099.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 1116.00 | 1096.93 | 1101.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:00:00 | 1116.00 | 1096.93 | 1101.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 1117.10 | 1100.96 | 1102.74 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 1117.10 | 1104.19 | 1104.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1140.15 | 1115.90 | 1109.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 14:15:00 | 1144.70 | 1145.46 | 1135.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 15:00:00 | 1144.70 | 1145.46 | 1135.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 1145.00 | 1145.37 | 1136.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 1131.30 | 1145.37 | 1136.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1119.50 | 1140.19 | 1134.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 1124.20 | 1140.19 | 1134.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1115.90 | 1135.33 | 1132.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1112.50 | 1135.33 | 1132.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1125.00 | 1130.52 | 1131.10 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 15:15:00 | 1134.00 | 1131.45 | 1131.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 1167.10 | 1138.58 | 1134.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 1156.90 | 1162.93 | 1151.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1156.90 | 1162.93 | 1151.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1156.90 | 1162.93 | 1151.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:30:00 | 1178.15 | 1166.59 | 1155.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:30:00 | 1176.85 | 1168.27 | 1157.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:15:00 | 1176.00 | 1168.27 | 1157.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1187.95 | 1171.01 | 1160.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-04 09:15:00 | 1295.97 | 1213.90 | 1189.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1383.75 | 1398.36 | 1398.37 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1415.75 | 1397.47 | 1397.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 1430.90 | 1413.53 | 1405.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 15:15:00 | 1491.50 | 1493.49 | 1472.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 09:15:00 | 1485.00 | 1493.49 | 1472.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1518.85 | 1498.56 | 1476.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 1549.80 | 1511.14 | 1492.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-26 10:15:00 | 1704.78 | 1619.98 | 1592.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1670.00 | 1691.28 | 1693.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 1619.35 | 1642.10 | 1654.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 11:15:00 | 1610.20 | 1606.83 | 1626.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 12:00:00 | 1610.20 | 1606.83 | 1626.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 1606.10 | 1609.31 | 1624.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1599.95 | 1615.12 | 1624.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 1632.00 | 1591.60 | 1593.26 | SL hit (close>static) qty=1.00 sl=1629.95 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 1645.00 | 1602.28 | 1597.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 1783.05 | 1655.43 | 1624.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 10:15:00 | 1651.75 | 1664.81 | 1637.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 11:00:00 | 1651.75 | 1664.81 | 1637.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 1626.70 | 1651.96 | 1639.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 1626.70 | 1651.96 | 1639.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 1626.00 | 1646.77 | 1638.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 1610.55 | 1646.77 | 1638.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1575.00 | 1623.74 | 1629.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1569.90 | 1605.09 | 1619.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 1598.70 | 1591.49 | 1604.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 11:15:00 | 1598.70 | 1591.49 | 1604.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 1598.70 | 1591.49 | 1604.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:00:00 | 1598.70 | 1591.49 | 1604.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1600.00 | 1593.20 | 1604.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:45:00 | 1609.60 | 1593.20 | 1604.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1574.75 | 1589.51 | 1601.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:00:00 | 1558.55 | 1583.31 | 1597.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1488.55 | 1581.32 | 1595.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-21 09:15:00 | 1402.69 | 1545.92 | 1578.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 1074.00 | 1067.68 | 1067.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 1093.55 | 1072.85 | 1069.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 1066.85 | 1078.49 | 1073.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 1066.85 | 1078.49 | 1073.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1066.85 | 1078.49 | 1073.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 1070.00 | 1078.49 | 1073.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1082.85 | 1079.36 | 1074.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:45:00 | 1088.40 | 1080.73 | 1075.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:15:00 | 1087.85 | 1080.73 | 1075.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:45:00 | 1089.90 | 1083.94 | 1077.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 1089.95 | 1081.55 | 1077.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1080.40 | 1081.32 | 1078.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1084.80 | 1081.32 | 1078.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1081.20 | 1081.29 | 1078.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 1067.20 | 1081.29 | 1078.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 1082.00 | 1081.44 | 1078.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:45:00 | 1081.05 | 1081.44 | 1078.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 1078.75 | 1080.90 | 1078.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 1079.60 | 1080.90 | 1078.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 1081.45 | 1081.01 | 1078.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 15:15:00 | 1087.70 | 1081.77 | 1079.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 11:45:00 | 1096.20 | 1103.72 | 1100.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 13:15:00 | 1067.40 | 1093.01 | 1095.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1067.40 | 1093.01 | 1095.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 1057.00 | 1079.88 | 1088.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 994.05 | 984.83 | 1007.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 986.00 | 984.83 | 1007.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 990.30 | 985.92 | 1006.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1003.10 | 985.92 | 1006.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 970.60 | 978.30 | 987.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 966.65 | 978.30 | 987.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 964.90 | 974.87 | 985.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 964.80 | 961.74 | 967.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:00:00 | 965.75 | 960.76 | 965.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 965.20 | 961.65 | 965.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:15:00 | 965.60 | 961.65 | 965.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 968.85 | 963.09 | 965.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 968.85 | 963.09 | 965.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 977.95 | 966.06 | 966.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:45:00 | 987.90 | 966.06 | 966.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-18 15:15:00 | 976.80 | 968.21 | 967.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 976.80 | 968.21 | 967.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 1009.25 | 976.42 | 971.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 995.15 | 1010.26 | 995.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 995.15 | 1010.26 | 995.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 995.15 | 1010.26 | 995.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:00:00 | 995.15 | 1010.26 | 995.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1000.80 | 1008.37 | 996.38 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 967.40 | 992.03 | 992.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 950.35 | 974.06 | 982.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 971.50 | 971.12 | 979.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 11:45:00 | 973.80 | 971.12 | 979.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 983.75 | 965.43 | 968.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:45:00 | 981.10 | 965.43 | 968.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 981.75 | 968.69 | 969.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:45:00 | 983.05 | 968.69 | 969.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 11:15:00 | 983.20 | 971.60 | 970.75 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 939.85 | 971.34 | 971.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 10:15:00 | 935.75 | 964.22 | 968.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 10:15:00 | 945.00 | 941.62 | 951.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 10:15:00 | 945.00 | 941.62 | 951.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 945.00 | 941.62 | 951.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:15:00 | 938.45 | 947.26 | 950.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 938.95 | 945.59 | 949.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 974.55 | 951.38 | 950.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 974.55 | 951.38 | 950.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 980.00 | 957.10 | 953.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 997.65 | 998.27 | 983.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 997.65 | 998.27 | 983.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 994.00 | 997.26 | 991.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 987.95 | 997.26 | 991.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 977.50 | 993.30 | 989.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 977.50 | 993.30 | 989.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 970.05 | 988.65 | 988.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 970.05 | 988.65 | 988.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 969.25 | 984.77 | 986.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 965.70 | 980.96 | 984.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 949.25 | 944.33 | 953.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 13:00:00 | 949.25 | 944.33 | 953.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 943.40 | 944.37 | 952.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:45:00 | 948.80 | 944.37 | 952.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 942.80 | 944.05 | 951.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 938.95 | 944.05 | 951.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 940.00 | 943.44 | 950.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 939.20 | 940.52 | 947.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:30:00 | 940.00 | 943.66 | 946.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 943.00 | 942.50 | 944.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 944.50 | 942.50 | 944.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 945.35 | 943.07 | 944.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:15:00 | 946.50 | 943.07 | 944.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 940.85 | 942.63 | 944.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:45:00 | 938.10 | 940.61 | 943.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:30:00 | 938.70 | 925.22 | 931.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 11:45:00 | 936.55 | 928.99 | 932.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 12:45:00 | 938.45 | 931.40 | 932.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 963.05 | 939.15 | 936.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 14:15:00 | 963.05 | 939.15 | 936.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 1023.20 | 959.29 | 946.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 1024.15 | 1026.39 | 1005.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 14:45:00 | 1025.50 | 1026.39 | 1005.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1010.00 | 1022.52 | 1007.46 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 996.00 | 1002.18 | 1002.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 995.05 | 1000.75 | 1002.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 1030.00 | 994.16 | 997.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 1030.00 | 994.16 | 997.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1030.00 | 994.16 | 997.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 1030.00 | 994.16 | 997.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1014.35 | 998.20 | 998.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 1012.00 | 998.20 | 998.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 1013.85 | 1001.82 | 1000.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 1013.85 | 1001.82 | 1000.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 1015.05 | 1005.81 | 1002.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 09:15:00 | 1002.95 | 1005.24 | 1002.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1002.95 | 1005.24 | 1002.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1002.95 | 1005.24 | 1002.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 1002.95 | 1005.24 | 1002.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1005.25 | 1005.24 | 1002.68 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 988.00 | 999.45 | 1000.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 980.05 | 994.27 | 997.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 14:15:00 | 991.75 | 991.20 | 995.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 14:15:00 | 991.75 | 991.20 | 995.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 991.75 | 991.20 | 995.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 991.75 | 991.20 | 995.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 989.40 | 990.84 | 994.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 983.55 | 990.84 | 994.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 1000.15 | 990.84 | 992.85 | SL hit (close>static) qty=1.00 sl=995.35 alert=retest2 |

### Cycle 132 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 1001.00 | 994.73 | 994.38 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 991.40 | 993.89 | 994.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 990.20 | 992.77 | 993.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 13:15:00 | 993.00 | 992.82 | 993.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 13:15:00 | 993.00 | 992.82 | 993.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 993.00 | 992.82 | 993.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:00:00 | 993.00 | 992.82 | 993.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 995.65 | 993.38 | 993.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 995.65 | 993.38 | 993.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 990.00 | 992.71 | 993.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 972.20 | 992.71 | 993.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 13:15:00 | 923.59 | 954.31 | 972.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 874.98 | 919.76 | 951.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 899.85 | 889.67 | 888.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 907.60 | 896.82 | 892.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 924.00 | 933.86 | 925.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 924.00 | 933.86 | 925.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 924.00 | 933.86 | 925.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 924.00 | 933.86 | 925.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 926.85 | 932.46 | 925.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:30:00 | 923.30 | 932.46 | 925.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 928.00 | 930.16 | 926.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:15:00 | 924.00 | 930.16 | 926.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 924.00 | 928.93 | 926.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 930.50 | 928.93 | 926.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-24 09:15:00 | 1023.55 | 995.63 | 971.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 15:15:00 | 1025.40 | 1038.70 | 1039.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 986.00 | 1014.59 | 1025.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1070.55 | 1021.05 | 1026.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1070.55 | 1021.05 | 1026.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1070.55 | 1021.05 | 1026.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1070.55 | 1021.05 | 1026.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 1107.50 | 1038.34 | 1033.65 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 1005.00 | 1029.81 | 1032.39 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 1039.20 | 1032.57 | 1032.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 1065.30 | 1040.63 | 1036.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 1060.50 | 1066.87 | 1054.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 15:15:00 | 1060.50 | 1066.87 | 1054.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 1060.50 | 1066.87 | 1054.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 1082.85 | 1066.87 | 1054.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1077.30 | 1068.95 | 1056.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 1091.50 | 1078.08 | 1067.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:00:00 | 1091.40 | 1081.96 | 1070.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1027.60 | 1068.72 | 1070.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1027.60 | 1068.72 | 1070.00 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 1080.80 | 1070.40 | 1069.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1118.20 | 1080.20 | 1074.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1139.35 | 1139.51 | 1121.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 15:15:00 | 1132.05 | 1137.14 | 1128.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 1132.05 | 1137.14 | 1128.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1144.90 | 1137.14 | 1128.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-20 10:15:00 | 1259.39 | 1187.22 | 1167.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 1250.00 | 1254.99 | 1255.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 1247.25 | 1250.47 | 1252.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1248.20 | 1225.61 | 1234.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1248.20 | 1225.61 | 1234.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1248.20 | 1225.61 | 1234.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1250.40 | 1225.61 | 1234.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1248.75 | 1230.24 | 1235.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1248.70 | 1230.24 | 1235.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 1247.90 | 1238.55 | 1238.32 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 1233.95 | 1237.63 | 1237.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1225.60 | 1234.72 | 1236.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1231.00 | 1229.56 | 1232.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 1231.00 | 1229.56 | 1232.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1231.00 | 1229.56 | 1232.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1231.00 | 1229.56 | 1232.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1231.20 | 1229.89 | 1232.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1234.10 | 1229.89 | 1232.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1226.30 | 1229.17 | 1232.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 1218.90 | 1226.80 | 1230.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 12:45:00 | 1222.30 | 1225.14 | 1226.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 1269.00 | 1233.49 | 1230.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1269.00 | 1233.49 | 1230.06 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 1220.00 | 1230.15 | 1230.94 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 1234.90 | 1231.53 | 1231.19 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 1226.00 | 1230.00 | 1230.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 1222.00 | 1227.35 | 1228.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1174.20 | 1169.23 | 1182.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 1174.20 | 1169.23 | 1182.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1240.10 | 1183.40 | 1187.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1240.10 | 1183.40 | 1187.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1227.00 | 1192.12 | 1190.90 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1186.00 | 1206.56 | 1208.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 13:15:00 | 1179.50 | 1190.15 | 1196.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1147.90 | 1133.89 | 1148.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1147.90 | 1133.89 | 1148.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1147.90 | 1133.89 | 1148.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 1147.90 | 1133.89 | 1148.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1145.20 | 1136.15 | 1148.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:30:00 | 1145.00 | 1136.15 | 1148.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1151.90 | 1139.30 | 1148.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 1151.90 | 1139.30 | 1148.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1150.60 | 1141.56 | 1148.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 1149.80 | 1141.56 | 1148.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1148.80 | 1143.01 | 1148.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 1150.60 | 1143.01 | 1148.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1150.80 | 1144.57 | 1148.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 1154.90 | 1144.57 | 1148.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1154.90 | 1146.63 | 1149.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 1189.50 | 1146.63 | 1149.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 1224.60 | 1162.23 | 1156.19 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 1144.20 | 1166.16 | 1168.62 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 14:15:00 | 1170.30 | 1167.86 | 1167.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 15:15:00 | 1175.00 | 1169.29 | 1168.51 | Break + close above crossover candle high |

### Cycle 153 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1160.00 | 1167.43 | 1167.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 1153.00 | 1162.36 | 1165.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 1135.00 | 1131.71 | 1137.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 1135.00 | 1131.71 | 1137.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1135.00 | 1131.71 | 1137.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 1135.70 | 1131.71 | 1137.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1134.00 | 1123.32 | 1128.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1134.00 | 1123.32 | 1128.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1132.90 | 1125.24 | 1128.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1117.10 | 1129.09 | 1129.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 1090.90 | 1088.76 | 1088.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 1090.90 | 1088.76 | 1088.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1094.00 | 1089.81 | 1089.00 | Break + close above crossover candle high |

### Cycle 155 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1027.90 | 1079.07 | 1084.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 972.70 | 1036.67 | 1060.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 866.90 | 860.29 | 873.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 866.90 | 860.29 | 873.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 866.90 | 860.29 | 873.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 869.00 | 860.29 | 873.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 857.70 | 854.25 | 860.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 863.80 | 854.25 | 860.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 859.40 | 855.28 | 860.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 864.90 | 855.28 | 860.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 860.20 | 856.26 | 860.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 854.00 | 856.91 | 859.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 853.50 | 856.23 | 859.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 855.50 | 852.79 | 855.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 909.05 | 849.63 | 843.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 909.05 | 849.63 | 843.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 12:15:00 | 934.00 | 876.96 | 857.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 917.00 | 929.74 | 906.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 910.25 | 929.74 | 906.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 904.40 | 924.67 | 906.75 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 887.75 | 901.65 | 902.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 885.50 | 895.99 | 899.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 872.10 | 871.43 | 880.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 872.10 | 871.43 | 880.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 872.10 | 871.43 | 880.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:45:00 | 863.25 | 869.13 | 876.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:15:00 | 858.00 | 868.28 | 875.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 857.30 | 864.55 | 872.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 917.25 | 869.40 | 869.44 | SL hit (close>static) qty=1.00 sl=886.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 919.20 | 879.36 | 873.96 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 875.75 | 882.34 | 882.55 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 895.45 | 881.54 | 881.22 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 877.25 | 881.01 | 881.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 871.10 | 876.94 | 879.13 | Break + close below crossover candle low |

### Cycle 162 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 914.75 | 879.40 | 877.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 933.45 | 890.21 | 882.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 899.10 | 906.87 | 896.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 899.10 | 906.87 | 896.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 899.10 | 906.87 | 896.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 895.10 | 906.87 | 896.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 899.35 | 905.36 | 896.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 897.00 | 905.36 | 896.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 904.40 | 905.17 | 897.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 897.70 | 905.17 | 897.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 897.85 | 903.71 | 897.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 897.85 | 903.71 | 897.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 898.65 | 902.70 | 897.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 905.00 | 899.48 | 896.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 887.05 | 897.05 | 896.92 | SL hit (close<static) qty=1.00 sl=896.50 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 894.40 | 897.22 | 897.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 888.80 | 894.59 | 895.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 891.20 | 890.08 | 893.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 891.20 | 890.08 | 893.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 891.20 | 890.08 | 893.06 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 894.55 | 892.24 | 892.17 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 891.40 | 892.34 | 892.36 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 914.85 | 896.84 | 894.40 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 890.35 | 898.52 | 899.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 885.05 | 895.83 | 897.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 888.00 | 881.40 | 884.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 888.00 | 881.40 | 884.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 888.00 | 881.40 | 884.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 893.00 | 881.40 | 884.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 888.75 | 882.87 | 885.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 888.75 | 882.87 | 885.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 887.70 | 883.84 | 885.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:30:00 | 886.85 | 883.84 | 885.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 893.45 | 885.76 | 886.03 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 890.80 | 886.77 | 886.46 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 883.45 | 886.10 | 886.19 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 889.00 | 886.73 | 886.46 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 885.25 | 886.67 | 886.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 884.55 | 886.12 | 886.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 887.00 | 886.30 | 886.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 887.00 | 886.30 | 886.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 887.00 | 886.30 | 886.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 887.00 | 886.30 | 886.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 885.30 | 886.10 | 886.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:30:00 | 884.00 | 885.83 | 886.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 883.85 | 885.44 | 885.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 883.50 | 885.58 | 885.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 884.25 | 885.46 | 885.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 884.85 | 885.34 | 885.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 885.50 | 885.34 | 885.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 884.30 | 885.13 | 885.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 893.25 | 885.13 | 885.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 886.25 | 885.36 | 885.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 887.25 | 885.90 | 885.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 887.25 | 885.90 | 885.85 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 884.85 | 885.69 | 885.76 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 932.65 | 894.94 | 889.93 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 898.60 | 907.16 | 907.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 891.10 | 899.65 | 903.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 917.50 | 886.96 | 889.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 917.50 | 886.96 | 889.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 917.50 | 886.96 | 889.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 917.00 | 886.96 | 889.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 917.60 | 893.08 | 891.79 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 890.00 | 894.62 | 895.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 887.80 | 893.26 | 894.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 888.35 | 880.84 | 883.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 888.35 | 880.84 | 883.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 888.35 | 880.84 | 883.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 888.35 | 880.84 | 883.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 887.60 | 882.19 | 883.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 910.00 | 882.19 | 883.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 903.00 | 886.35 | 885.69 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 883.15 | 890.83 | 891.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 879.00 | 882.64 | 884.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 872.00 | 871.85 | 875.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 876.95 | 871.85 | 875.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 877.35 | 872.95 | 875.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 872.10 | 873.26 | 875.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:15:00 | 872.50 | 873.26 | 875.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 868.05 | 872.70 | 874.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 903.15 | 864.88 | 865.55 | SL hit (close>static) qty=1.00 sl=881.90 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 899.60 | 871.83 | 868.64 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 873.95 | 876.23 | 876.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 870.85 | 874.90 | 875.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 878.45 | 875.23 | 875.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 878.45 | 875.23 | 875.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 878.45 | 875.23 | 875.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 878.85 | 875.23 | 875.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 879.55 | 876.09 | 876.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 878.20 | 876.09 | 876.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 879.80 | 876.83 | 876.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 883.25 | 878.12 | 877.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 891.80 | 898.72 | 891.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 15:15:00 | 891.80 | 898.72 | 891.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 891.80 | 898.72 | 891.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 890.60 | 898.72 | 891.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 891.80 | 897.33 | 891.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 904.00 | 894.68 | 893.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 14:15:00 | 994.40 | 929.31 | 911.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 970.10 | 977.54 | 977.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 957.00 | 972.47 | 975.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 961.50 | 951.21 | 959.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 961.50 | 951.21 | 959.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 961.50 | 951.21 | 959.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 961.50 | 951.21 | 959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 947.80 | 950.53 | 958.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 944.90 | 950.42 | 957.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 943.40 | 943.47 | 950.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 963.30 | 953.32 | 952.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 963.30 | 953.32 | 952.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 971.00 | 959.10 | 955.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 977.35 | 977.58 | 968.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 980.35 | 977.58 | 968.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 972.60 | 975.50 | 969.40 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 941.15 | 963.64 | 965.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 929.90 | 944.56 | 953.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 929.60 | 923.01 | 930.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 929.60 | 923.01 | 930.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 929.60 | 923.01 | 930.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 929.50 | 923.01 | 930.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 927.35 | 923.88 | 929.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 926.10 | 925.10 | 929.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 918.35 | 924.90 | 928.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 09:15:00 | 879.79 | 890.90 | 900.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 872.43 | 883.48 | 894.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 881.00 | 879.42 | 888.60 | SL hit (close>ema200) qty=0.50 sl=879.42 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 879.90 | 877.90 | 877.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 891.60 | 880.64 | 879.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 890.50 | 892.66 | 887.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:30:00 | 899.90 | 892.66 | 887.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 887.35 | 891.60 | 887.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 887.35 | 891.60 | 887.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 883.75 | 890.03 | 887.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 883.75 | 890.03 | 887.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 884.85 | 888.99 | 887.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 881.25 | 888.99 | 887.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 879.95 | 887.19 | 886.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 878.05 | 887.19 | 886.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 879.00 | 885.55 | 885.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 866.95 | 881.83 | 884.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 847.45 | 846.41 | 857.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:45:00 | 841.40 | 846.41 | 857.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 855.20 | 848.26 | 856.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 857.75 | 848.26 | 856.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 847.30 | 848.07 | 855.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 856.65 | 848.07 | 855.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 847.80 | 840.17 | 846.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 847.80 | 840.17 | 846.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 846.85 | 841.51 | 846.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 845.10 | 843.97 | 846.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 856.55 | 847.61 | 848.12 | SL hit (close>static) qty=1.00 sl=853.60 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 853.35 | 848.76 | 848.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 862.60 | 857.70 | 854.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 856.50 | 858.90 | 856.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 856.50 | 858.90 | 856.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 856.50 | 858.90 | 856.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 856.50 | 858.90 | 856.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 854.15 | 857.95 | 855.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 854.15 | 857.95 | 855.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 855.00 | 857.36 | 855.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 854.30 | 857.36 | 855.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 851.95 | 856.28 | 855.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 851.95 | 856.28 | 855.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 849.05 | 854.83 | 854.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 844.90 | 852.85 | 853.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 839.95 | 838.75 | 843.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 839.95 | 838.75 | 843.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 836.65 | 837.17 | 840.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 831.00 | 834.66 | 839.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 871.60 | 837.22 | 837.38 | SL hit (close>static) qty=1.00 sl=844.40 alert=retest2 |

### Cycle 190 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 856.50 | 841.08 | 839.12 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 845.35 | 848.86 | 849.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 843.20 | 847.27 | 848.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 835.00 | 831.28 | 836.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 835.00 | 831.28 | 836.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 826.00 | 830.22 | 835.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 835.00 | 830.22 | 835.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 836.10 | 831.62 | 835.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 836.10 | 831.62 | 835.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 841.75 | 833.65 | 835.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 842.70 | 833.65 | 835.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 845.15 | 837.35 | 837.09 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 833.10 | 837.85 | 838.33 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 843.20 | 838.23 | 838.07 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 835.80 | 837.67 | 837.83 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 842.00 | 838.54 | 838.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 12:15:00 | 843.20 | 839.47 | 838.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 834.80 | 838.53 | 838.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 834.80 | 838.53 | 838.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 834.80 | 838.53 | 838.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 834.80 | 838.53 | 838.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 833.00 | 837.43 | 837.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 832.05 | 836.35 | 837.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 841.15 | 829.32 | 831.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 841.15 | 829.32 | 831.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 841.15 | 829.32 | 831.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 841.15 | 829.32 | 831.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 837.85 | 831.02 | 832.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:45:00 | 833.75 | 833.06 | 833.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 838.60 | 834.16 | 833.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 838.60 | 834.16 | 833.63 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 828.20 | 833.14 | 833.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 825.05 | 829.50 | 831.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 780.60 | 779.47 | 790.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 781.20 | 779.47 | 790.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 587.80 | 583.50 | 588.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 587.80 | 583.50 | 588.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 579.20 | 582.64 | 587.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 571.05 | 579.99 | 585.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 13:15:00 | 542.50 | 550.01 | 556.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 554.95 | 538.46 | 544.04 | SL hit (close>ema200) qty=0.50 sl=538.46 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 565.15 | 547.95 | 547.64 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 539.90 | 550.50 | 551.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 537.65 | 547.93 | 550.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 501.45 | 475.89 | 487.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 501.45 | 475.89 | 487.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 501.45 | 475.89 | 487.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 501.45 | 475.89 | 487.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 519.00 | 484.51 | 489.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 519.00 | 484.51 | 489.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 528.30 | 500.39 | 496.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 593.90 | 535.46 | 519.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 582.95 | 592.78 | 563.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:45:00 | 581.30 | 592.78 | 563.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 561.35 | 577.46 | 566.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 561.35 | 577.46 | 566.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 550.00 | 571.97 | 564.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 546.15 | 571.97 | 564.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 539.20 | 560.75 | 560.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 539.60 | 560.75 | 560.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 539.95 | 556.59 | 558.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 12:15:00 | 533.80 | 552.03 | 556.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 517.50 | 507.90 | 518.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 517.50 | 507.90 | 518.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 517.50 | 507.90 | 518.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 528.50 | 507.90 | 518.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 515.35 | 509.39 | 518.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:00:00 | 511.20 | 510.41 | 517.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 509.65 | 512.12 | 515.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 485.64 | 507.71 | 511.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 484.17 | 507.71 | 511.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 494.80 | 493.87 | 501.11 | SL hit (close>ema200) qty=0.50 sl=493.87 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 461.70 | 448.31 | 448.01 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 444.70 | 451.68 | 451.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 441.10 | 448.38 | 450.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 446.55 | 445.77 | 448.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 446.55 | 445.77 | 448.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 446.55 | 445.77 | 448.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 447.90 | 445.77 | 448.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 445.30 | 445.68 | 447.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 443.80 | 445.68 | 447.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 443.70 | 445.38 | 447.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 444.50 | 444.98 | 447.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 421.61 | 432.49 | 439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 421.51 | 432.49 | 439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 422.27 | 432.49 | 439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 429.50 | 426.57 | 432.74 | SL hit (close>ema200) qty=0.50 sl=426.57 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 443.80 | 436.28 | 435.40 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 429.35 | 435.00 | 435.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 420.00 | 431.52 | 433.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 431.85 | 414.94 | 420.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 431.85 | 414.94 | 420.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 431.85 | 414.94 | 420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 431.85 | 414.94 | 420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 427.70 | 417.49 | 421.19 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 430.85 | 424.29 | 423.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 432.35 | 426.91 | 425.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 418.85 | 425.29 | 424.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 418.85 | 425.29 | 424.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 418.85 | 425.29 | 424.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 427.40 | 424.93 | 424.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 450.00 | 453.73 | 453.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 450.00 | 453.73 | 453.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 436.90 | 450.37 | 452.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 14:15:00 | 446.40 | 445.97 | 448.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 446.40 | 445.97 | 448.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 210 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 470.90 | 450.80 | 450.62 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 468.45 | 475.08 | 475.71 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 480.50 | 475.18 | 475.07 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 469.60 | 475.79 | 476.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 462.65 | 469.08 | 472.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 486.25 | 470.64 | 472.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 486.25 | 470.64 | 472.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 486.25 | 470.64 | 472.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 489.05 | 470.64 | 472.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 491.00 | 474.71 | 473.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 504.00 | 480.57 | 476.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 487.35 | 488.51 | 483.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 487.35 | 488.51 | 483.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 483.20 | 487.72 | 484.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 483.20 | 487.72 | 484.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 483.00 | 486.78 | 484.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 483.00 | 486.78 | 484.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 483.80 | 486.18 | 484.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 490.60 | 486.18 | 484.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 478.10 | 484.15 | 484.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 478.10 | 484.15 | 484.37 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 493.25 | 484.68 | 484.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 515.55 | 490.85 | 487.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 10:15:00 | 497.50 | 498.84 | 493.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:45:00 | 495.00 | 498.84 | 493.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 494.40 | 497.43 | 493.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:15:00 | 492.80 | 497.43 | 493.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 496.45 | 497.23 | 493.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 495.40 | 497.23 | 493.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 494.95 | 496.78 | 494.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 494.95 | 496.78 | 494.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 489.40 | 495.34 | 493.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 489.40 | 495.34 | 493.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 487.40 | 493.75 | 493.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 487.75 | 493.75 | 493.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 488.30 | 492.66 | 492.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 485.15 | 490.28 | 491.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 491.50 | 490.15 | 491.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 15:15:00 | 491.50 | 490.15 | 491.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 491.50 | 490.15 | 491.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 496.00 | 490.15 | 491.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 487.10 | 489.54 | 490.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 484.50 | 488.07 | 489.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 484.80 | 487.48 | 489.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 500.25 | 492.36 | 491.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 500.25 | 492.36 | 491.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 506.70 | 499.69 | 495.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 10:15:00 | 744.05 | 2024-04-24 10:15:00 | 752.90 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-04-18 11:15:00 | 745.30 | 2024-04-24 10:15:00 | 752.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-04-18 12:45:00 | 745.60 | 2024-04-24 10:15:00 | 752.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-04-18 13:15:00 | 743.00 | 2024-04-24 10:15:00 | 752.90 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-05-10 15:15:00 | 977.95 | 2024-05-13 09:15:00 | 929.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-10 15:15:00 | 977.95 | 2024-05-14 13:15:00 | 961.25 | STOP_HIT | 0.50 | 1.71% |
| BUY | retest2 | 2024-05-17 13:15:00 | 996.00 | 2024-05-21 10:15:00 | 974.95 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-05-18 11:30:00 | 991.05 | 2024-05-21 10:15:00 | 974.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-06-05 09:30:00 | 852.20 | 2024-06-05 13:15:00 | 880.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-06-07 09:15:00 | 923.70 | 2024-06-13 11:15:00 | 927.85 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2024-07-03 09:15:00 | 1003.25 | 2024-07-04 12:15:00 | 984.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-07-04 10:00:00 | 1000.00 | 2024-07-04 12:15:00 | 984.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-04 11:15:00 | 1000.65 | 2024-07-04 12:15:00 | 984.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-07-10 10:15:00 | 934.55 | 2024-07-10 14:15:00 | 962.00 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-07-10 11:15:00 | 936.15 | 2024-07-10 14:15:00 | 962.00 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-07-25 11:45:00 | 1063.90 | 2024-07-30 10:15:00 | 1039.10 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-07-25 14:00:00 | 1067.90 | 2024-07-30 10:15:00 | 1039.10 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-07-26 09:30:00 | 1072.25 | 2024-07-30 10:15:00 | 1039.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-08-01 12:00:00 | 1055.30 | 2024-08-01 12:15:00 | 1060.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-06 13:30:00 | 969.25 | 2024-08-08 09:15:00 | 996.70 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest1 | 2024-08-06 14:45:00 | 966.00 | 2024-08-08 09:15:00 | 996.70 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-08-21 15:15:00 | 1092.00 | 2024-08-22 12:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-26 09:45:00 | 1065.00 | 2024-08-28 14:15:00 | 1062.60 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-08-28 12:30:00 | 1064.70 | 2024-08-28 14:15:00 | 1062.60 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1097.80 | 2024-09-04 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-09-11 10:30:00 | 1055.00 | 2024-09-12 10:15:00 | 1086.35 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest1 | 2024-09-17 12:30:00 | 1186.75 | 2024-09-17 13:15:00 | 1246.09 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-09-17 12:30:00 | 1186.75 | 2024-09-18 09:15:00 | 1198.05 | STOP_HIT | 0.50 | 0.95% |
| BUY | retest2 | 2024-09-19 11:15:00 | 1217.85 | 2024-09-20 09:15:00 | 1339.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 12:00:00 | 1245.00 | 2024-09-26 09:15:00 | 1369.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1280.45 | 2024-10-10 09:15:00 | 1328.85 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2024-10-07 13:00:00 | 1285.60 | 2024-10-10 09:15:00 | 1328.85 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-10-08 11:45:00 | 1283.90 | 2024-10-10 09:15:00 | 1328.85 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-10-09 10:15:00 | 1286.00 | 2024-10-10 09:15:00 | 1328.85 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-10-09 11:45:00 | 1261.85 | 2024-10-10 09:15:00 | 1328.85 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2024-10-21 12:00:00 | 1258.70 | 2024-10-22 11:15:00 | 1195.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 1257.30 | 2024-10-22 11:15:00 | 1194.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1241.65 | 2024-10-22 12:15:00 | 1179.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 1258.70 | 2024-10-23 09:15:00 | 1246.55 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2024-10-21 12:30:00 | 1257.30 | 2024-10-23 09:15:00 | 1246.55 | STOP_HIT | 0.50 | 0.86% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1241.65 | 2024-10-23 09:15:00 | 1246.55 | STOP_HIT | 0.50 | -0.39% |
| SELL | retest2 | 2024-10-23 12:00:00 | 1255.75 | 2024-10-23 13:15:00 | 1257.30 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-10-25 13:30:00 | 1250.50 | 2024-10-25 14:15:00 | 1230.95 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-11-06 09:30:00 | 1289.95 | 2024-11-08 10:15:00 | 1275.95 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-11-06 10:45:00 | 1290.00 | 2024-11-08 10:15:00 | 1275.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-11-11 12:15:00 | 1276.30 | 2024-11-12 09:15:00 | 1303.20 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-12-02 12:30:00 | 1178.15 | 2024-12-04 09:15:00 | 1295.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 13:30:00 | 1176.85 | 2024-12-04 09:15:00 | 1294.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 14:15:00 | 1176.00 | 2024-12-04 09:15:00 | 1293.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 09:15:00 | 1187.95 | 2024-12-04 09:15:00 | 1306.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-12 14:15:00 | 1412.05 | 2024-12-13 09:15:00 | 1380.15 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-12-20 09:15:00 | 1549.80 | 2024-12-26 10:15:00 | 1704.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1599.95 | 2025-01-15 10:15:00 | 1632.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-01-20 15:00:00 | 1558.55 | 2025-01-21 09:15:00 | 1402.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1488.55 | 2025-01-21 09:15:00 | 1414.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1488.55 | 2025-01-21 10:15:00 | 1339.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-03 11:45:00 | 1088.40 | 2025-02-06 13:15:00 | 1067.40 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-02-03 12:15:00 | 1087.85 | 2025-02-06 13:15:00 | 1067.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-02-03 12:45:00 | 1089.90 | 2025-02-06 13:15:00 | 1067.40 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-02-04 09:15:00 | 1089.95 | 2025-02-06 13:15:00 | 1067.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-02-04 15:15:00 | 1087.70 | 2025-02-06 13:15:00 | 1067.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-02-06 11:45:00 | 1096.20 | 2025-02-06 13:15:00 | 1067.40 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-02-14 10:15:00 | 966.65 | 2025-02-18 15:15:00 | 976.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-02-14 10:45:00 | 964.90 | 2025-02-18 15:15:00 | 976.80 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-02-18 09:15:00 | 964.80 | 2025-02-18 15:15:00 | 976.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-02-18 12:00:00 | 965.75 | 2025-02-18 15:15:00 | 976.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-03-04 12:15:00 | 938.45 | 2025-03-05 09:15:00 | 974.55 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-03-04 13:00:00 | 938.95 | 2025-03-05 09:15:00 | 974.55 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-03-13 09:15:00 | 938.95 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-03-13 10:15:00 | 940.00 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-03-13 13:00:00 | 939.20 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-03-17 12:30:00 | 940.00 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-03-18 12:45:00 | 938.10 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-20 09:30:00 | 938.70 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-03-20 11:45:00 | 936.55 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-03-20 12:45:00 | 938.45 | 2025-03-20 14:15:00 | 963.05 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-03-27 11:15:00 | 1012.00 | 2025-03-27 13:15:00 | 1013.85 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-04-02 09:15:00 | 983.55 | 2025-04-02 13:15:00 | 1000.15 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-04-04 09:15:00 | 972.20 | 2025-04-04 13:15:00 | 923.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 972.20 | 2025-04-07 09:15:00 | 874.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 930.50 | 2025-04-24 09:15:00 | 1023.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-08 09:15:00 | 1091.50 | 2025-05-09 09:15:00 | 1027.60 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest2 | 2025-05-08 11:00:00 | 1091.40 | 2025-05-09 09:15:00 | 1027.60 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1144.90 | 2025-05-20 10:15:00 | 1259.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-03 12:00:00 | 1218.90 | 2025-06-05 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-06-04 12:45:00 | 1222.30 | 2025-06-05 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-07-10 09:15:00 | 1117.10 | 2025-07-16 15:15:00 | 1090.90 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-07-30 13:45:00 | 854.00 | 2025-08-05 10:15:00 | 909.05 | STOP_HIT | 1.00 | -6.45% |
| SELL | retest2 | 2025-07-30 15:00:00 | 853.50 | 2025-08-05 10:15:00 | 909.05 | STOP_HIT | 1.00 | -6.51% |
| SELL | retest2 | 2025-07-31 13:45:00 | 855.50 | 2025-08-05 10:15:00 | 909.05 | STOP_HIT | 1.00 | -6.26% |
| SELL | retest2 | 2025-08-12 12:45:00 | 863.25 | 2025-08-14 09:15:00 | 917.25 | STOP_HIT | 1.00 | -6.26% |
| SELL | retest2 | 2025-08-12 15:15:00 | 858.00 | 2025-08-14 09:15:00 | 917.25 | STOP_HIT | 1.00 | -6.91% |
| SELL | retest2 | 2025-08-13 09:30:00 | 857.30 | 2025-08-14 09:15:00 | 917.25 | STOP_HIT | 1.00 | -6.99% |
| BUY | retest2 | 2025-08-28 09:15:00 | 905.00 | 2025-08-28 13:15:00 | 887.05 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-28 15:00:00 | 902.50 | 2025-08-29 09:15:00 | 891.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-15 13:30:00 | 884.00 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-15 15:00:00 | 883.85 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-16 12:45:00 | 883.50 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-16 13:45:00 | 884.25 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-10 12:30:00 | 872.10 | 2025-10-15 09:15:00 | 903.15 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-10-10 13:15:00 | 872.50 | 2025-10-15 09:15:00 | 903.15 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-10-13 09:15:00 | 868.05 | 2025-10-15 09:15:00 | 903.15 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-10-28 09:15:00 | 904.00 | 2025-10-28 14:15:00 | 994.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-10 09:15:00 | 944.90 | 2025-11-11 12:15:00 | 963.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-10 14:45:00 | 943.40 | 2025-11-11 12:15:00 | 963.30 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-11-19 11:45:00 | 926.10 | 2025-11-25 09:15:00 | 879.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:00:00 | 918.35 | 2025-11-25 12:15:00 | 872.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 11:45:00 | 926.10 | 2025-11-26 09:15:00 | 881.00 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-11-19 15:00:00 | 918.35 | 2025-11-26 09:15:00 | 881.00 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-12-11 14:30:00 | 845.10 | 2025-12-12 09:15:00 | 856.55 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-19 10:30:00 | 831.00 | 2025-12-22 09:15:00 | 871.60 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-01-07 13:45:00 | 833.75 | 2026-01-07 14:15:00 | 838.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-02-01 15:15:00 | 571.05 | 2026-02-05 13:15:00 | 542.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 15:15:00 | 571.05 | 2026-02-09 09:15:00 | 554.95 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-02-26 13:00:00 | 511.20 | 2026-03-02 09:15:00 | 485.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 509.65 | 2026-03-02 09:15:00 | 484.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:00:00 | 511.20 | 2026-03-04 09:15:00 | 494.80 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2026-02-27 10:45:00 | 509.65 | 2026-03-04 09:15:00 | 494.80 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-03-20 12:15:00 | 443.80 | 2026-03-23 12:15:00 | 421.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:45:00 | 443.70 | 2026-03-23 12:15:00 | 421.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 444.50 | 2026-03-23 12:15:00 | 422.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 443.80 | 2026-03-24 11:15:00 | 429.50 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2026-03-20 12:45:00 | 443.70 | 2026-03-24 11:15:00 | 429.50 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-03-20 13:30:00 | 444.50 | 2026-03-24 11:15:00 | 429.50 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2026-04-02 11:45:00 | 427.40 | 2026-04-10 15:15:00 | 450.00 | STOP_HIT | 1.00 | 5.29% |
| BUY | retest2 | 2026-04-29 09:15:00 | 490.60 | 2026-04-29 15:15:00 | 478.10 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-05-06 12:30:00 | 484.50 | 2026-05-07 10:15:00 | 500.25 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-05-06 13:45:00 | 484.80 | 2026-05-07 10:15:00 | 500.25 | STOP_HIT | 1.00 | -3.19% |

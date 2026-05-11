# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1282.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 223 |
| ALERT1 | 144 |
| ALERT2 | 142 |
| ALERT2_SKIP | 103 |
| ALERT3 | 286 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 139 |
| PARTIAL | 11 |
| TARGET_HIT | 12 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 111
- **Target hits / Stop hits / Partials:** 12 / 128 / 11
- **Avg / median % per leg:** 0.14% / -1.11%
- **Sum % (uncompounded):** 20.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 15 | 21.4% | 9 | 61 | 0 | -0.07% | -4.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 70 | 15 | 21.4% | 9 | 61 | 0 | -0.07% | -4.9% |
| SELL (all) | 81 | 25 | 30.9% | 3 | 67 | 11 | 0.32% | 25.7% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL @ 3rd Alert (retest2) | 80 | 24 | 30.0% | 2 | 67 | 11 | 0.20% | 15.7% |
| retest1 (combined) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| retest2 (combined) | 150 | 39 | 26.0% | 11 | 128 | 11 | 0.07% | 10.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 230.50 | 228.06 | 227.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 12:15:00 | 232.75 | 229.49 | 228.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 14:15:00 | 230.05 | 231.57 | 230.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 14:15:00 | 230.05 | 231.57 | 230.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 14:15:00 | 230.05 | 231.57 | 230.61 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 13:15:00 | 271.70 | 272.45 | 272.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 14:15:00 | 268.90 | 271.74 | 272.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 272.25 | 271.40 | 271.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 272.25 | 271.40 | 271.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 272.25 | 271.40 | 271.96 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 270.20 | 267.72 | 267.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 273.90 | 268.93 | 268.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 269.25 | 269.43 | 268.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 12:15:00 | 267.90 | 269.12 | 268.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 267.90 | 269.12 | 268.43 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 265.45 | 267.67 | 267.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 11:15:00 | 263.20 | 266.77 | 267.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 15:15:00 | 262.90 | 262.54 | 264.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 266.70 | 263.37 | 264.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 266.70 | 263.37 | 264.28 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 269.25 | 263.82 | 263.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 11:15:00 | 270.55 | 265.16 | 264.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 266.75 | 268.01 | 266.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 266.75 | 268.01 | 266.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 266.75 | 268.01 | 266.23 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 300.00 | 301.53 | 301.55 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 15:15:00 | 301.85 | 301.60 | 301.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 09:15:00 | 304.70 | 302.22 | 301.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 14:15:00 | 303.10 | 303.22 | 302.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 14:15:00 | 303.10 | 303.22 | 302.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 303.10 | 303.22 | 302.61 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 11:15:00 | 315.70 | 318.61 | 318.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 14:15:00 | 313.75 | 316.79 | 317.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 314.20 | 312.96 | 314.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 314.20 | 312.96 | 314.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 314.20 | 312.96 | 314.73 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 15:15:00 | 319.00 | 315.81 | 315.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 10:15:00 | 321.50 | 317.59 | 316.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 12:15:00 | 317.05 | 317.94 | 316.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 12:15:00 | 317.05 | 317.94 | 316.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 12:15:00 | 317.05 | 317.94 | 316.79 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 09:15:00 | 327.80 | 328.73 | 328.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 11:15:00 | 321.65 | 326.82 | 327.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 09:15:00 | 332.30 | 326.82 | 327.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 332.30 | 326.82 | 327.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 332.30 | 326.82 | 327.31 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 11:15:00 | 332.45 | 328.45 | 328.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 14:15:00 | 334.60 | 330.14 | 328.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 10:15:00 | 331.60 | 331.62 | 330.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 11:15:00 | 334.50 | 332.20 | 330.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 334.50 | 332.20 | 330.42 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 13:15:00 | 329.95 | 333.44 | 333.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 15:15:00 | 327.15 | 331.58 | 332.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 10:15:00 | 334.00 | 331.99 | 332.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 10:15:00 | 334.00 | 331.99 | 332.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 334.00 | 331.99 | 332.62 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 323.80 | 319.00 | 318.77 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 318.70 | 321.63 | 321.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 317.60 | 320.31 | 321.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 319.65 | 319.29 | 320.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 14:15:00 | 320.25 | 319.48 | 320.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 320.25 | 319.48 | 320.30 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 327.10 | 321.03 | 320.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 10:15:00 | 332.65 | 323.35 | 321.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 11:15:00 | 330.50 | 331.06 | 327.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 13:15:00 | 327.50 | 329.84 | 327.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 327.50 | 329.84 | 327.70 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 14:15:00 | 333.00 | 334.62 | 334.71 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 09:15:00 | 342.50 | 336.00 | 335.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 344.00 | 340.19 | 337.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 10:15:00 | 372.10 | 372.14 | 362.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 366.10 | 375.70 | 368.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 366.10 | 375.70 | 368.68 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 355.40 | 365.29 | 365.84 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 378.90 | 366.16 | 364.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 12:15:00 | 379.65 | 370.32 | 366.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 12:15:00 | 395.50 | 396.13 | 391.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 391.60 | 396.81 | 394.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 391.60 | 396.81 | 394.29 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 10:15:00 | 386.55 | 392.24 | 392.88 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 397.25 | 390.11 | 390.06 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 390.30 | 393.48 | 393.91 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 397.10 | 394.05 | 393.94 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 15:15:00 | 392.00 | 393.53 | 393.72 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 395.30 | 393.88 | 393.86 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 392.65 | 393.76 | 393.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 12:15:00 | 388.30 | 392.67 | 393.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 386.70 | 383.38 | 386.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 386.70 | 383.38 | 386.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 386.70 | 383.38 | 386.62 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 394.60 | 387.76 | 387.43 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 381.20 | 388.54 | 388.56 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 390.25 | 386.78 | 386.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 411.40 | 392.35 | 389.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 13:15:00 | 447.25 | 448.06 | 439.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 438.90 | 447.90 | 442.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 438.90 | 447.90 | 442.61 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 439.45 | 442.35 | 442.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 436.55 | 440.85 | 441.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 14:15:00 | 439.70 | 438.98 | 440.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 14:15:00 | 439.70 | 438.98 | 440.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 439.70 | 438.98 | 440.57 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 416.40 | 409.62 | 409.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 10:15:00 | 421.70 | 412.04 | 410.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 13:15:00 | 413.60 | 415.25 | 412.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 14:15:00 | 435.85 | 436.26 | 431.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 435.85 | 436.26 | 431.17 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 534.55 | 542.63 | 543.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 11:15:00 | 531.25 | 538.30 | 540.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 530.00 | 522.05 | 527.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 530.00 | 522.05 | 527.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 530.00 | 522.05 | 527.29 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 531.90 | 526.23 | 525.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 538.15 | 531.85 | 529.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 11:15:00 | 531.95 | 532.21 | 530.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 12:15:00 | 536.05 | 532.98 | 530.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 536.05 | 532.98 | 530.55 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 12:15:00 | 538.45 | 546.63 | 547.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 13:15:00 | 535.05 | 544.32 | 546.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 09:15:00 | 542.25 | 540.59 | 543.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 10:15:00 | 543.40 | 541.15 | 543.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 543.40 | 541.15 | 543.84 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 12:15:00 | 547.00 | 540.05 | 539.28 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 14:15:00 | 535.95 | 539.91 | 540.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 14:15:00 | 532.30 | 536.89 | 538.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 538.05 | 536.55 | 537.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 538.05 | 536.55 | 537.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 538.05 | 536.55 | 537.90 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 11:15:00 | 541.85 | 535.68 | 534.90 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 530.35 | 535.30 | 535.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 514.80 | 530.27 | 533.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 520.15 | 516.11 | 521.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 520.15 | 516.11 | 521.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 520.15 | 516.11 | 521.23 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 530.60 | 523.08 | 523.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 12:15:00 | 539.70 | 528.81 | 526.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 543.25 | 546.75 | 542.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 543.25 | 546.75 | 542.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 543.25 | 546.75 | 542.31 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 15:15:00 | 556.00 | 559.77 | 560.08 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 564.35 | 560.68 | 560.47 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 10:15:00 | 557.85 | 560.12 | 560.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 13:15:00 | 556.40 | 559.06 | 559.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 560.00 | 559.08 | 559.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 15:15:00 | 560.00 | 559.08 | 559.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 560.00 | 559.08 | 559.59 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 568.35 | 560.93 | 560.39 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 14:15:00 | 558.45 | 559.95 | 560.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 09:15:00 | 556.00 | 557.79 | 558.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 10:15:00 | 558.10 | 557.85 | 558.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 10:15:00 | 558.10 | 557.85 | 558.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 558.10 | 557.85 | 558.73 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 13:15:00 | 539.65 | 537.04 | 536.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-17 09:15:00 | 562.15 | 543.14 | 539.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 13:15:00 | 592.10 | 592.28 | 580.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 12:15:00 | 590.00 | 592.01 | 585.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 590.00 | 592.01 | 585.90 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 14:15:00 | 587.40 | 590.23 | 590.35 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 593.30 | 590.85 | 590.62 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 09:15:00 | 577.80 | 588.24 | 589.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 13:15:00 | 574.75 | 582.09 | 585.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 15:15:00 | 581.95 | 581.03 | 584.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 584.90 | 581.81 | 584.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 584.90 | 581.81 | 584.65 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 12:15:00 | 585.55 | 584.10 | 583.98 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 13:15:00 | 581.90 | 583.66 | 583.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 578.60 | 582.65 | 583.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 15:15:00 | 587.35 | 583.59 | 583.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 15:15:00 | 587.35 | 583.59 | 583.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 587.35 | 583.59 | 583.69 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 15:15:00 | 590.00 | 583.29 | 582.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 09:15:00 | 593.10 | 589.46 | 586.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 14:15:00 | 591.50 | 591.67 | 589.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 15:15:00 | 591.50 | 591.63 | 589.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 591.50 | 591.63 | 589.26 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 12:15:00 | 587.95 | 594.73 | 595.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 14:15:00 | 585.00 | 592.03 | 594.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 547.40 | 547.38 | 557.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 556.70 | 548.43 | 553.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 556.70 | 548.43 | 553.46 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 560.45 | 556.33 | 555.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 568.40 | 558.74 | 556.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 10:15:00 | 556.50 | 563.25 | 561.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 10:15:00 | 556.50 | 563.25 | 561.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 556.50 | 563.25 | 561.12 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 12:15:00 | 548.50 | 558.51 | 559.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 13:15:00 | 545.50 | 555.90 | 557.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 09:15:00 | 554.10 | 551.98 | 555.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 554.10 | 551.98 | 555.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 554.10 | 551.98 | 555.35 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 552.15 | 541.26 | 539.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 564.50 | 547.77 | 543.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 15:15:00 | 563.95 | 564.26 | 557.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 11:15:00 | 560.95 | 564.74 | 559.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 560.95 | 564.74 | 559.35 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 541.05 | 554.82 | 556.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 535.00 | 548.28 | 553.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 536.80 | 526.00 | 534.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 536.80 | 526.00 | 534.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 536.80 | 526.00 | 534.51 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 543.00 | 536.48 | 536.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 10:15:00 | 544.70 | 538.99 | 537.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 13:15:00 | 546.50 | 547.19 | 543.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 13:15:00 | 546.50 | 547.19 | 543.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 546.50 | 547.19 | 543.88 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 534.50 | 541.98 | 542.53 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 549.85 | 542.50 | 542.06 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 530.70 | 540.63 | 541.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 528.00 | 538.55 | 540.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 502.45 | 499.28 | 510.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 12:15:00 | 510.70 | 501.56 | 510.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 510.70 | 501.56 | 510.54 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 512.35 | 509.26 | 508.84 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 13:15:00 | 506.80 | 508.32 | 508.46 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 515.00 | 509.65 | 509.05 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 500.50 | 508.38 | 508.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 11:15:00 | 498.50 | 506.40 | 507.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 503.20 | 502.89 | 505.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 11:15:00 | 503.25 | 502.96 | 504.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 503.25 | 502.96 | 504.89 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 518.10 | 507.52 | 506.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 528.00 | 511.62 | 508.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 527.65 | 532.07 | 525.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 527.65 | 532.07 | 525.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 527.65 | 532.07 | 525.55 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 09:15:00 | 517.85 | 522.72 | 523.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 516.00 | 519.99 | 521.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 518.60 | 518.52 | 520.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 518.60 | 518.52 | 520.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 518.60 | 518.52 | 520.63 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 539.85 | 521.30 | 520.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 550.85 | 532.54 | 526.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 574.40 | 574.88 | 565.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 10:15:00 | 576.00 | 581.16 | 577.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 576.00 | 581.16 | 577.52 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 566.15 | 575.76 | 575.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 10:15:00 | 562.90 | 570.45 | 573.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 568.85 | 567.13 | 569.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 568.85 | 567.13 | 569.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 568.85 | 567.13 | 569.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 571.85 | 567.13 | 569.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 573.70 | 568.45 | 570.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:45:00 | 573.45 | 568.45 | 570.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 573.30 | 569.42 | 570.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:00:00 | 573.30 | 569.42 | 570.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 572.40 | 569.90 | 570.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:00:00 | 572.40 | 569.90 | 570.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 566.30 | 569.18 | 570.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:30:00 | 576.95 | 569.18 | 570.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 572.00 | 569.74 | 570.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 543.05 | 569.74 | 570.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 10:15:00 | 562.15 | 561.00 | 563.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 14:15:00 | 534.04 | 549.57 | 554.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 541.90 | 533.95 | 540.94 | SL hit (close>ema200) qty=0.50 sl=533.95 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 552.80 | 545.70 | 545.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 554.40 | 547.99 | 546.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 576.05 | 576.50 | 570.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 12:15:00 | 570.00 | 575.31 | 571.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 570.00 | 575.31 | 571.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:00:00 | 570.00 | 575.31 | 571.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 572.50 | 574.75 | 571.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:30:00 | 569.90 | 574.75 | 571.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 570.60 | 573.92 | 571.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 570.60 | 573.92 | 571.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 573.00 | 573.73 | 571.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 574.90 | 573.73 | 571.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 11:00:00 | 573.65 | 573.91 | 572.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 11:30:00 | 574.80 | 574.24 | 572.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 14:15:00 | 569.90 | 573.50 | 572.61 | SL hit (close<static) qty=1.00 sl=570.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 566.95 | 571.26 | 571.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 12:15:00 | 565.00 | 569.32 | 570.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 09:15:00 | 582.00 | 570.52 | 570.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 582.00 | 570.52 | 570.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 582.00 | 570.52 | 570.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:45:00 | 584.00 | 570.52 | 570.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 10:15:00 | 586.45 | 573.71 | 572.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 11:15:00 | 599.10 | 578.79 | 574.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 11:15:00 | 590.25 | 596.15 | 587.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 12:00:00 | 590.25 | 596.15 | 587.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 589.00 | 594.72 | 587.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:45:00 | 586.75 | 594.72 | 587.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 590.50 | 593.88 | 588.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:45:00 | 584.90 | 593.88 | 588.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 601.50 | 595.40 | 589.26 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 13:15:00 | 582.45 | 586.88 | 587.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 574.10 | 582.91 | 585.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 14:15:00 | 581.50 | 579.48 | 582.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 14:15:00 | 581.50 | 579.48 | 582.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 581.50 | 579.48 | 582.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 15:00:00 | 581.50 | 579.48 | 582.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 583.50 | 580.28 | 582.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 588.55 | 581.72 | 582.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 588.30 | 583.03 | 583.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 588.30 | 583.03 | 583.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 591.00 | 584.63 | 584.03 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 579.55 | 585.02 | 585.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 576.50 | 583.17 | 584.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 09:15:00 | 567.95 | 561.62 | 568.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 567.95 | 561.62 | 568.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 567.95 | 561.62 | 568.99 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 584.35 | 574.41 | 573.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 591.45 | 580.52 | 576.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 12:15:00 | 602.70 | 604.13 | 599.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 12:15:00 | 602.70 | 604.13 | 599.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 602.70 | 604.13 | 599.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 602.70 | 604.13 | 599.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 615.00 | 606.52 | 601.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 618.35 | 606.82 | 602.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:30:00 | 622.80 | 616.52 | 608.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 14:45:00 | 615.55 | 615.34 | 611.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 597.40 | 610.44 | 609.73 | SL hit (close<static) qty=1.00 sl=599.60 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 589.90 | 606.34 | 607.93 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 608.55 | 604.38 | 604.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 610.90 | 606.78 | 605.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 15:15:00 | 606.10 | 608.23 | 607.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 15:15:00 | 606.10 | 608.23 | 607.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 606.10 | 608.23 | 607.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 606.15 | 608.23 | 607.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 597.40 | 606.07 | 606.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 592.90 | 602.17 | 604.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 13:15:00 | 600.65 | 600.51 | 603.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 14:00:00 | 600.65 | 600.51 | 603.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 600.30 | 600.46 | 602.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:00:00 | 599.40 | 600.30 | 602.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:30:00 | 598.55 | 598.93 | 600.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 598.90 | 598.93 | 600.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:45:00 | 599.90 | 598.89 | 600.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 598.80 | 598.87 | 600.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-30 12:15:00 | 606.05 | 600.64 | 600.88 | SL hit (close>static) qty=1.00 sl=604.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 15:15:00 | 603.90 | 601.40 | 601.16 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 562.00 | 593.52 | 597.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 10:15:00 | 554.00 | 585.62 | 593.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 504.00 | 495.33 | 510.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 504.00 | 495.33 | 510.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 504.00 | 495.33 | 510.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 507.60 | 495.33 | 510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 510.50 | 498.36 | 510.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 509.20 | 498.36 | 510.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 507.65 | 500.22 | 510.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:30:00 | 507.20 | 500.96 | 509.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 15:15:00 | 518.00 | 505.46 | 509.73 | SL hit (close>static) qty=1.00 sl=513.20 alert=retest2 |

### Cycle 81 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 11:15:00 | 530.05 | 515.61 | 513.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 536.20 | 523.56 | 518.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 522.50 | 524.78 | 519.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 522.50 | 524.78 | 519.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 522.50 | 524.78 | 519.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 15:00:00 | 536.20 | 529.08 | 526.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:45:00 | 536.00 | 533.72 | 529.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:15:00 | 536.00 | 533.72 | 529.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:15:00 | 536.25 | 538.86 | 538.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 538.40 | 538.77 | 538.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:45:00 | 535.00 | 538.77 | 538.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 534.55 | 537.92 | 538.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 534.55 | 537.92 | 538.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 532.25 | 536.79 | 537.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 520.65 | 516.00 | 519.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 520.65 | 516.00 | 519.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 520.65 | 516.00 | 519.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:15:00 | 523.35 | 516.00 | 519.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 523.65 | 517.53 | 519.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:30:00 | 523.80 | 517.53 | 519.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 535.70 | 523.99 | 522.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 550.20 | 534.93 | 528.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 554.00 | 555.15 | 546.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 12:45:00 | 553.65 | 555.15 | 546.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 553.40 | 554.03 | 548.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 548.60 | 554.03 | 548.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 551.70 | 553.30 | 549.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 548.90 | 553.30 | 549.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 550.50 | 552.74 | 549.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:30:00 | 550.05 | 552.74 | 549.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 550.30 | 552.25 | 549.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 550.30 | 552.25 | 549.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 548.60 | 551.52 | 549.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 548.00 | 551.52 | 549.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 549.70 | 551.16 | 549.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 547.45 | 551.16 | 549.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 551.45 | 551.22 | 549.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:30:00 | 555.00 | 551.96 | 550.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 13:45:00 | 554.10 | 552.36 | 550.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-04 12:15:00 | 610.50 | 599.58 | 586.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 637.60 | 648.53 | 650.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 635.85 | 646.00 | 648.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 646.00 | 642.76 | 645.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 646.00 | 642.76 | 645.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 646.00 | 642.76 | 645.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:45:00 | 641.60 | 642.76 | 645.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 646.95 | 643.60 | 646.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 640.40 | 645.00 | 646.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 14:15:00 | 653.20 | 647.67 | 646.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 14:15:00 | 653.20 | 647.67 | 646.92 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 632.45 | 645.48 | 646.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 629.30 | 642.24 | 644.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 642.90 | 637.63 | 640.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 642.90 | 637.63 | 640.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 642.90 | 637.63 | 640.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 637.85 | 637.63 | 640.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 640.20 | 638.15 | 640.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 645.50 | 638.15 | 640.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 638.70 | 638.26 | 640.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 640.20 | 638.26 | 640.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 643.00 | 639.21 | 640.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 643.00 | 639.21 | 640.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 639.80 | 639.32 | 640.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 642.20 | 639.32 | 640.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 653.85 | 642.23 | 641.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 15:15:00 | 658.40 | 645.46 | 643.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 642.85 | 644.94 | 643.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 642.85 | 644.94 | 643.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 642.85 | 644.94 | 643.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 642.85 | 644.94 | 643.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 647.35 | 645.42 | 643.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:15:00 | 642.45 | 645.42 | 643.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 646.30 | 645.60 | 643.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:30:00 | 644.35 | 645.60 | 643.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 641.75 | 644.83 | 643.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 638.00 | 644.83 | 643.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 649.00 | 645.66 | 644.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 642.15 | 645.66 | 644.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 642.10 | 644.95 | 643.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 642.10 | 644.95 | 643.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 645.00 | 644.96 | 644.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 652.00 | 644.96 | 644.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:30:00 | 648.00 | 658.60 | 654.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:00:00 | 648.75 | 656.63 | 654.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:45:00 | 647.20 | 654.75 | 653.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 13:15:00 | 642.75 | 652.35 | 652.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 642.75 | 652.35 | 652.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 638.20 | 649.52 | 651.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 650.80 | 647.67 | 650.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 650.80 | 647.67 | 650.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 650.80 | 647.67 | 650.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 652.85 | 647.67 | 650.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 651.75 | 648.48 | 650.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 651.75 | 648.48 | 650.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 650.70 | 648.93 | 650.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:00:00 | 647.20 | 648.58 | 650.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 09:45:00 | 649.10 | 647.17 | 648.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 655.65 | 646.81 | 646.88 | SL hit (close>static) qty=1.00 sl=652.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 10:15:00 | 656.70 | 648.79 | 647.78 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 640.00 | 646.48 | 647.21 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 12:15:00 | 648.95 | 646.31 | 646.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 13:15:00 | 657.85 | 648.62 | 647.22 | Break + close above crossover candle high |

### Cycle 92 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 624.85 | 645.21 | 646.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 613.00 | 634.11 | 640.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 630.05 | 628.92 | 636.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 15:00:00 | 630.05 | 628.92 | 636.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 653.25 | 634.13 | 637.32 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 652.00 | 640.72 | 639.95 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 630.70 | 639.65 | 639.80 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 648.00 | 640.45 | 640.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 656.90 | 649.74 | 645.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 11:15:00 | 649.40 | 649.68 | 646.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 12:00:00 | 649.40 | 649.68 | 646.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 648.30 | 649.98 | 647.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 648.30 | 649.98 | 647.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 649.55 | 649.90 | 647.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 649.85 | 649.90 | 647.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 10:15:00 | 644.90 | 648.74 | 647.29 | SL hit (close<static) qty=1.00 sl=645.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 658.70 | 669.87 | 671.02 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 680.60 | 670.73 | 670.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 694.40 | 677.93 | 673.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 707.50 | 710.51 | 700.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 707.50 | 710.51 | 700.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 703.80 | 708.05 | 701.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:15:00 | 704.00 | 708.05 | 701.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 711.05 | 708.65 | 702.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 717.70 | 709.15 | 703.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:45:00 | 713.50 | 709.99 | 704.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 11:00:00 | 713.35 | 710.66 | 704.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 12:00:00 | 713.35 | 711.20 | 705.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 724.75 | 726.87 | 724.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 723.30 | 726.87 | 724.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 724.15 | 726.33 | 724.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:15:00 | 723.10 | 726.33 | 724.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 718.10 | 724.68 | 724.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 718.10 | 724.68 | 724.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 721.80 | 724.11 | 723.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-27 14:15:00 | 717.00 | 722.68 | 723.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 717.00 | 722.68 | 723.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 707.50 | 719.54 | 721.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 12:15:00 | 716.90 | 716.41 | 719.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 12:15:00 | 716.90 | 716.41 | 719.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 716.90 | 716.41 | 719.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:30:00 | 715.50 | 716.41 | 719.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 708.20 | 709.11 | 712.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 704.15 | 709.40 | 711.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 668.94 | 682.57 | 688.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 683.30 | 682.72 | 688.35 | SL hit (close>ema200) qty=0.50 sl=682.72 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 691.90 | 686.81 | 686.73 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 681.00 | 686.82 | 687.02 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 688.60 | 685.82 | 685.76 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 15:15:00 | 685.00 | 685.66 | 685.69 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 686.10 | 685.75 | 685.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 13:15:00 | 690.00 | 687.39 | 686.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 686.20 | 687.49 | 686.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 15:15:00 | 686.20 | 687.49 | 686.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 686.20 | 687.49 | 686.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 691.05 | 687.49 | 686.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 683.60 | 686.71 | 686.50 | SL hit (close<static) qty=1.00 sl=686.20 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 676.65 | 684.70 | 685.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 674.00 | 680.09 | 681.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 682.15 | 680.51 | 681.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 14:15:00 | 682.15 | 680.51 | 681.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 682.15 | 680.51 | 681.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 682.15 | 680.51 | 681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 681.65 | 680.73 | 681.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 680.40 | 680.73 | 681.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 671.70 | 678.93 | 680.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 670.70 | 678.93 | 680.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 685.10 | 665.96 | 668.54 | SL hit (close>static) qty=1.00 sl=684.10 alert=retest2 |

### Cycle 105 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 683.90 | 672.85 | 671.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 689.40 | 678.71 | 674.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 704.95 | 706.82 | 697.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:45:00 | 704.75 | 706.82 | 697.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 733.55 | 750.12 | 742.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 733.55 | 750.12 | 742.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 735.35 | 747.17 | 742.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:15:00 | 734.50 | 747.17 | 742.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 734.75 | 744.68 | 741.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 733.60 | 744.68 | 741.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 736.85 | 741.90 | 740.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:15:00 | 734.60 | 741.90 | 740.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 733.00 | 740.12 | 740.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 733.00 | 740.12 | 740.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 728.85 | 737.86 | 739.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 722.25 | 729.98 | 733.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 11:15:00 | 696.85 | 695.06 | 706.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 12:00:00 | 696.85 | 695.06 | 706.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 707.70 | 697.76 | 703.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 707.70 | 697.76 | 703.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 703.60 | 698.92 | 703.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 701.35 | 699.70 | 703.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 701.05 | 699.97 | 703.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:45:00 | 697.55 | 699.66 | 702.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 709.60 | 703.24 | 703.74 | SL hit (close>static) qty=1.00 sl=708.80 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 708.00 | 701.70 | 701.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 712.20 | 704.85 | 702.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 710.55 | 712.29 | 709.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 710.55 | 712.29 | 709.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 710.55 | 712.29 | 709.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:00:00 | 710.55 | 712.29 | 709.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 722.00 | 715.24 | 711.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 715.90 | 715.24 | 711.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 715.00 | 722.80 | 718.69 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 15:15:00 | 714.50 | 716.84 | 716.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 708.90 | 715.25 | 716.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 14:15:00 | 714.00 | 713.76 | 715.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 15:00:00 | 714.00 | 713.76 | 715.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 703.60 | 711.74 | 713.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 695.30 | 711.74 | 713.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:00:00 | 699.85 | 709.36 | 712.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 12:45:00 | 702.05 | 706.07 | 710.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 716.95 | 706.69 | 705.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 716.95 | 706.69 | 705.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 10:15:00 | 728.80 | 711.11 | 707.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 702.60 | 714.41 | 711.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 702.60 | 714.41 | 711.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 702.60 | 714.41 | 711.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 702.60 | 714.41 | 711.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 696.15 | 710.76 | 710.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 698.00 | 710.76 | 710.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 12:15:00 | 705.85 | 709.11 | 709.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 703.65 | 708.02 | 708.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 712.00 | 705.75 | 707.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 712.00 | 705.75 | 707.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 712.00 | 705.75 | 707.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:45:00 | 713.20 | 705.75 | 707.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 718.95 | 708.39 | 708.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:45:00 | 717.80 | 708.39 | 708.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 719.05 | 710.52 | 709.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 727.20 | 717.49 | 714.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 712.70 | 717.73 | 714.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 712.70 | 717.73 | 714.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 712.70 | 717.73 | 714.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:30:00 | 712.90 | 717.73 | 714.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 720.00 | 718.19 | 715.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 09:45:00 | 725.10 | 719.70 | 717.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:00:00 | 723.90 | 725.40 | 723.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 729.10 | 726.97 | 723.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-07 12:15:00 | 797.61 | 777.36 | 761.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 768.50 | 771.83 | 771.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 761.00 | 769.66 | 770.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 13:15:00 | 685.95 | 683.39 | 697.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 14:00:00 | 685.95 | 683.39 | 697.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 698.00 | 687.14 | 692.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 698.00 | 687.14 | 692.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 686.95 | 687.10 | 691.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 681.50 | 687.10 | 691.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 13:45:00 | 685.05 | 684.31 | 687.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 700.70 | 689.94 | 689.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 700.70 | 689.94 | 689.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 720.90 | 702.46 | 696.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 14:15:00 | 773.50 | 775.72 | 767.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 15:00:00 | 773.50 | 775.72 | 767.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 778.00 | 782.03 | 777.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 778.00 | 782.03 | 777.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 781.90 | 782.00 | 777.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 784.35 | 781.75 | 778.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 14:30:00 | 788.70 | 785.16 | 782.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 784.55 | 786.12 | 783.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:00:00 | 784.05 | 785.70 | 783.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 784.95 | 785.55 | 783.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 14:00:00 | 789.95 | 785.31 | 784.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 783.95 | 795.53 | 795.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 783.95 | 795.53 | 795.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 779.55 | 792.34 | 794.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 14:15:00 | 798.25 | 791.25 | 793.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 14:15:00 | 798.25 | 791.25 | 793.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 798.25 | 791.25 | 793.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 798.25 | 791.25 | 793.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 794.00 | 791.80 | 793.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 789.00 | 791.80 | 793.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 792.80 | 789.37 | 790.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 789.00 | 789.31 | 790.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 794.15 | 790.75 | 790.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 794.15 | 790.75 | 790.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 799.95 | 792.59 | 791.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 13:15:00 | 792.10 | 797.90 | 795.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 13:15:00 | 792.10 | 797.90 | 795.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 792.10 | 797.90 | 795.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 792.10 | 797.90 | 795.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 792.95 | 796.91 | 795.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 787.90 | 796.91 | 795.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 786.65 | 793.83 | 793.91 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 13:15:00 | 795.30 | 793.81 | 793.78 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 787.15 | 792.48 | 793.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 773.55 | 788.18 | 791.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 779.05 | 778.39 | 784.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 779.05 | 778.39 | 784.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 770.85 | 769.76 | 776.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 770.85 | 769.76 | 776.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 772.00 | 770.21 | 775.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 780.90 | 770.21 | 775.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 778.45 | 771.86 | 776.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 785.30 | 771.86 | 776.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 770.05 | 771.49 | 775.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 767.50 | 772.62 | 774.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 780.15 | 773.66 | 774.78 | SL hit (close>static) qty=1.00 sl=778.45 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 785.50 | 776.03 | 775.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 12:15:00 | 789.10 | 778.64 | 776.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 789.50 | 794.10 | 789.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 789.50 | 794.10 | 789.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 789.50 | 794.10 | 789.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 789.50 | 794.10 | 789.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 798.25 | 794.93 | 790.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 15:00:00 | 813.55 | 799.31 | 793.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 803.85 | 799.56 | 796.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:45:00 | 802.95 | 799.54 | 797.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 799.95 | 809.97 | 810.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 799.95 | 809.97 | 810.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 784.55 | 804.89 | 808.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 791.50 | 787.63 | 796.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 791.50 | 787.63 | 796.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 791.50 | 787.63 | 796.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 796.45 | 787.63 | 796.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 798.25 | 789.61 | 794.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 798.25 | 789.61 | 794.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 795.00 | 790.69 | 794.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 802.00 | 792.96 | 794.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 800.85 | 794.54 | 795.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:15:00 | 792.80 | 795.23 | 795.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 753.16 | 772.53 | 781.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 713.52 | 735.85 | 753.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 762.60 | 743.61 | 742.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 782.80 | 764.09 | 754.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 764.75 | 767.87 | 759.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 15:00:00 | 764.75 | 767.87 | 759.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 776.50 | 769.84 | 761.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 780.00 | 769.84 | 761.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:00:00 | 778.25 | 771.52 | 763.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 784.85 | 776.99 | 769.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 759.50 | 784.24 | 785.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 759.50 | 784.24 | 785.99 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 786.90 | 782.82 | 782.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 793.00 | 784.85 | 783.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 794.70 | 796.61 | 791.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 794.70 | 796.61 | 791.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 790.10 | 795.31 | 791.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 789.85 | 795.31 | 791.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 783.00 | 792.85 | 790.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 764.10 | 792.85 | 790.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 744.05 | 783.09 | 786.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 728.60 | 772.19 | 781.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 744.00 | 723.30 | 737.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 744.00 | 723.30 | 737.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 744.00 | 723.30 | 737.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 744.00 | 723.30 | 737.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 735.85 | 725.81 | 737.06 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 743.15 | 740.64 | 740.55 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 11:15:00 | 733.10 | 739.13 | 739.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 12:15:00 | 728.05 | 736.92 | 738.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 731.15 | 730.16 | 734.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 731.15 | 730.16 | 734.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 731.15 | 730.16 | 734.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:45:00 | 735.25 | 730.16 | 734.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 741.75 | 732.47 | 735.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 741.75 | 732.47 | 735.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 742.95 | 734.57 | 735.83 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 743.10 | 737.85 | 737.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 10:15:00 | 747.90 | 740.04 | 738.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 736.85 | 739.40 | 738.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 736.85 | 739.40 | 738.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 736.85 | 739.40 | 738.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 735.35 | 739.40 | 738.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 726.95 | 736.91 | 737.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 724.15 | 733.66 | 735.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 726.80 | 721.23 | 726.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 726.80 | 721.23 | 726.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 726.80 | 721.23 | 726.52 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 732.80 | 727.39 | 726.71 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 712.00 | 725.58 | 726.17 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 735.00 | 727.47 | 726.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 11:15:00 | 744.00 | 730.77 | 728.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 11:15:00 | 779.20 | 782.51 | 766.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 12:00:00 | 779.20 | 782.51 | 766.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 766.85 | 776.26 | 769.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-11 10:15:00 | 775.05 | 776.26 | 769.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-11 14:00:00 | 774.25 | 779.55 | 773.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:30:00 | 772.85 | 775.98 | 773.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 12:45:00 | 775.90 | 776.08 | 774.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 768.45 | 774.56 | 773.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:00:00 | 768.45 | 774.56 | 773.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 777.65 | 775.17 | 774.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 779.45 | 775.94 | 774.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 763.20 | 781.63 | 781.10 | SL hit (close<static) qty=1.00 sl=765.15 alert=retest2 |

### Cycle 132 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 747.95 | 774.90 | 778.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 741.25 | 768.17 | 774.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 735.05 | 734.70 | 748.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 735.05 | 734.70 | 748.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 717.30 | 731.27 | 744.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:15:00 | 713.90 | 728.47 | 742.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 755.05 | 733.01 | 731.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 755.05 | 733.01 | 731.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 760.50 | 738.51 | 733.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 763.85 | 772.65 | 761.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 763.85 | 772.65 | 761.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 763.85 | 772.65 | 761.81 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 15:15:00 | 756.00 | 761.08 | 761.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 749.85 | 758.83 | 760.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 756.85 | 754.81 | 757.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 756.85 | 754.81 | 757.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 756.85 | 754.81 | 757.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 756.85 | 754.81 | 757.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 753.00 | 754.45 | 756.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 742.95 | 754.45 | 756.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 705.80 | 729.64 | 741.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 11:15:00 | 668.66 | 711.85 | 730.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 738.20 | 720.79 | 720.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 750.75 | 726.78 | 722.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 794.15 | 794.60 | 783.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 794.15 | 794.60 | 783.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 793.50 | 794.38 | 783.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 800.65 | 794.69 | 786.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 806.10 | 808.61 | 808.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 11:15:00 | 806.10 | 808.61 | 808.85 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 823.50 | 811.54 | 810.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 851.90 | 820.41 | 814.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 12:15:00 | 864.10 | 874.06 | 861.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 13:00:00 | 864.10 | 874.06 | 861.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 829.30 | 862.73 | 860.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 829.30 | 862.73 | 860.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 830.45 | 856.27 | 857.48 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 867.60 | 845.78 | 844.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 872.90 | 856.93 | 850.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 862.00 | 862.52 | 856.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 862.00 | 862.52 | 856.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 848.05 | 861.26 | 857.87 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 850.45 | 855.76 | 855.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 848.80 | 853.88 | 854.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 852.55 | 850.88 | 853.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 852.55 | 850.88 | 853.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 852.55 | 850.88 | 853.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 852.55 | 850.88 | 853.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 855.85 | 851.87 | 853.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:30:00 | 853.05 | 851.87 | 853.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 853.40 | 852.18 | 853.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 15:00:00 | 850.15 | 851.76 | 852.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:30:00 | 847.50 | 850.66 | 852.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 14:15:00 | 862.90 | 849.41 | 850.37 | SL hit (close>static) qty=1.00 sl=857.25 alert=retest2 |

### Cycle 141 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 859.95 | 851.52 | 851.24 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 826.00 | 846.42 | 848.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 818.25 | 840.78 | 846.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 819.70 | 816.28 | 829.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-07 09:15:00 | 778.40 | 816.28 | 829.82 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 700.56 | 809.83 | 825.65 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 143 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 780.00 | 768.52 | 767.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 10:15:00 | 789.25 | 778.73 | 773.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 779.20 | 787.43 | 781.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 779.20 | 787.43 | 781.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 779.20 | 787.43 | 781.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 779.20 | 787.43 | 781.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 781.00 | 786.14 | 781.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 786.20 | 781.52 | 780.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 789.70 | 782.03 | 780.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 785.35 | 782.03 | 780.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:00:00 | 787.90 | 783.20 | 781.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 789.80 | 786.40 | 783.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:30:00 | 789.00 | 786.40 | 783.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 777.55 | 786.62 | 785.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-22 14:15:00 | 777.55 | 786.62 | 785.32 | SL hit (close<static) qty=1.00 sl=778.80 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 15:15:00 | 775.15 | 784.33 | 784.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 10:15:00 | 767.70 | 779.53 | 782.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 793.00 | 776.06 | 778.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 793.00 | 776.06 | 778.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 793.00 | 776.06 | 778.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 793.70 | 776.06 | 778.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 789.90 | 778.83 | 779.31 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 790.95 | 781.26 | 780.36 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 758.40 | 779.56 | 780.45 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 784.90 | 774.66 | 774.46 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 767.00 | 774.18 | 774.47 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 780.95 | 773.86 | 773.84 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 766.25 | 773.33 | 773.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 763.00 | 771.26 | 772.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 773.00 | 771.61 | 772.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 773.00 | 771.61 | 772.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 773.00 | 771.61 | 772.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 10:00:00 | 751.50 | 757.77 | 761.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 763.40 | 762.62 | 762.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 763.40 | 762.62 | 762.56 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 761.00 | 762.30 | 762.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 749.90 | 759.82 | 761.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 752.30 | 736.60 | 743.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 752.30 | 736.60 | 743.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 752.30 | 736.60 | 743.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 757.25 | 736.60 | 743.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 760.00 | 748.70 | 747.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 761.50 | 751.26 | 749.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 779.55 | 783.51 | 776.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 779.55 | 783.51 | 776.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 779.55 | 783.51 | 776.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 779.55 | 783.51 | 776.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 785.05 | 783.11 | 778.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 787.85 | 782.03 | 778.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:45:00 | 786.10 | 784.23 | 780.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:45:00 | 787.20 | 785.58 | 781.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 788.20 | 785.99 | 782.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 789.55 | 787.65 | 784.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 789.40 | 787.65 | 784.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 783.00 | 786.94 | 784.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 783.00 | 786.94 | 784.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 778.20 | 785.19 | 783.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 777.00 | 785.19 | 783.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 772.00 | 782.55 | 782.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 772.00 | 782.55 | 782.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 10:15:00 | 771.05 | 777.57 | 780.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 774.45 | 773.20 | 776.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 774.45 | 773.20 | 776.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 774.45 | 773.20 | 776.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 774.45 | 773.20 | 776.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 767.00 | 771.96 | 775.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:30:00 | 764.40 | 769.25 | 773.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:00:00 | 764.60 | 769.25 | 773.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 764.65 | 767.95 | 772.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 763.05 | 766.53 | 769.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 767.00 | 765.05 | 767.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 766.55 | 765.05 | 767.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 770.90 | 766.22 | 768.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:30:00 | 771.45 | 766.22 | 768.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 773.40 | 767.66 | 768.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 773.40 | 767.66 | 768.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 775.85 | 769.30 | 769.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 775.85 | 769.30 | 769.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 776.30 | 770.70 | 769.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 771.80 | 776.70 | 773.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 12:15:00 | 771.80 | 776.70 | 773.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 771.80 | 776.70 | 773.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:45:00 | 771.20 | 776.70 | 773.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 781.70 | 777.70 | 774.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:00:00 | 783.75 | 778.91 | 775.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 862.13 | 818.32 | 800.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 956.15 | 961.35 | 961.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 946.55 | 958.39 | 960.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 930.55 | 916.79 | 927.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 930.55 | 916.79 | 927.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 930.55 | 916.79 | 927.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 928.25 | 916.79 | 927.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 913.25 | 916.08 | 925.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 909.70 | 915.60 | 922.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 908.00 | 915.23 | 917.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 908.65 | 913.74 | 916.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 912.80 | 907.30 | 906.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 912.80 | 907.30 | 906.89 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 896.00 | 905.10 | 905.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 11:15:00 | 894.40 | 902.96 | 904.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 912.65 | 897.20 | 900.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 912.65 | 897.20 | 900.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 912.65 | 897.20 | 900.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 915.00 | 897.20 | 900.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 916.30 | 901.02 | 901.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 916.30 | 901.02 | 901.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 913.90 | 903.60 | 902.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 932.00 | 913.09 | 908.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 930.65 | 930.98 | 921.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:30:00 | 934.75 | 930.98 | 921.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 941.55 | 937.75 | 933.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 936.00 | 937.75 | 933.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 934.55 | 937.11 | 933.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 934.55 | 937.11 | 933.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 943.10 | 938.30 | 934.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 945.35 | 938.52 | 934.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 924.15 | 935.65 | 933.67 | SL hit (close<static) qty=1.00 sl=933.50 alert=retest2 |

### Cycle 160 — SELL (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 15:15:00 | 927.00 | 932.28 | 932.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 915.35 | 927.94 | 930.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 937.35 | 922.46 | 925.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 937.35 | 922.46 | 925.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 937.35 | 922.46 | 925.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 936.80 | 922.46 | 925.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 947.35 | 927.44 | 927.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 942.00 | 927.44 | 927.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 939.10 | 929.77 | 928.79 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 922.50 | 930.56 | 930.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 919.50 | 925.06 | 927.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 933.20 | 925.53 | 927.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 933.20 | 925.53 | 927.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 933.20 | 925.53 | 927.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 933.20 | 925.53 | 927.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 927.55 | 925.93 | 927.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 923.25 | 925.93 | 927.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 923.95 | 922.18 | 923.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 937.80 | 922.66 | 921.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 937.80 | 922.66 | 921.95 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 922.20 | 928.12 | 928.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 919.50 | 926.39 | 927.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 921.85 | 917.92 | 921.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 13:15:00 | 921.85 | 917.92 | 921.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 921.85 | 917.92 | 921.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:30:00 | 924.70 | 917.92 | 921.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 920.45 | 918.43 | 921.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 912.50 | 918.74 | 921.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 931.00 | 912.17 | 912.50 | SL hit (close>static) qty=1.00 sl=923.05 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 932.20 | 916.17 | 914.29 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 919.40 | 924.13 | 924.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 915.60 | 922.42 | 923.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 918.35 | 915.20 | 918.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 918.35 | 915.20 | 918.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 918.35 | 915.20 | 918.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 918.35 | 915.20 | 918.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 917.00 | 915.56 | 918.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 916.45 | 915.56 | 918.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 906.70 | 913.78 | 917.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 903.00 | 913.78 | 917.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 904.60 | 908.98 | 913.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:30:00 | 904.20 | 905.49 | 910.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 857.85 | 873.31 | 886.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 859.37 | 873.31 | 886.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 858.99 | 873.31 | 886.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 880.00 | 874.65 | 885.78 | SL hit (close>ema200) qty=0.50 sl=874.65 alert=retest2 |

### Cycle 167 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 926.50 | 891.79 | 890.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 932.00 | 904.99 | 896.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 929.30 | 929.69 | 918.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 929.30 | 929.69 | 918.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 919.90 | 927.00 | 921.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 919.90 | 927.00 | 921.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 920.65 | 925.73 | 921.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 921.50 | 925.73 | 921.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 919.20 | 924.42 | 921.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 919.20 | 924.42 | 921.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 917.55 | 923.05 | 920.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 917.55 | 923.05 | 920.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 904.20 | 917.79 | 918.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 897.05 | 913.64 | 916.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 873.55 | 873.04 | 880.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 873.55 | 873.04 | 880.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 874.15 | 867.39 | 873.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 874.15 | 867.39 | 873.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 871.00 | 868.12 | 873.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 877.50 | 868.12 | 873.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 870.30 | 868.55 | 872.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 864.75 | 867.24 | 870.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 863.00 | 865.14 | 868.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:45:00 | 862.70 | 864.67 | 867.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 863.50 | 866.17 | 867.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 873.25 | 867.59 | 868.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 874.85 | 867.59 | 868.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 877.55 | 869.58 | 869.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 877.55 | 869.58 | 869.14 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 15:15:00 | 877.55 | 878.38 | 878.42 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 894.20 | 881.55 | 879.86 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 873.90 | 879.78 | 880.04 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 888.25 | 879.06 | 878.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 889.70 | 881.19 | 879.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 889.05 | 891.38 | 887.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:45:00 | 889.45 | 891.38 | 887.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 885.30 | 890.47 | 888.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 885.30 | 890.47 | 888.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 882.45 | 888.87 | 888.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 882.50 | 888.87 | 888.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 889.40 | 889.29 | 888.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 889.40 | 889.29 | 888.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 887.10 | 888.85 | 888.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 868.00 | 888.85 | 888.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 876.10 | 886.30 | 887.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 860.00 | 871.37 | 876.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 850.50 | 849.65 | 857.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 851.20 | 849.65 | 857.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 831.75 | 846.07 | 854.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:15:00 | 829.40 | 846.07 | 854.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 829.70 | 838.14 | 848.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 852.00 | 846.82 | 846.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 852.00 | 846.82 | 846.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 858.80 | 849.21 | 847.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 844.85 | 849.49 | 848.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 844.85 | 849.49 | 848.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 844.85 | 849.49 | 848.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 844.85 | 849.49 | 848.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 844.45 | 848.49 | 847.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:30:00 | 843.85 | 848.49 | 847.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 846.40 | 847.62 | 847.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 847.15 | 847.62 | 847.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 842.45 | 846.59 | 847.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 836.05 | 843.59 | 845.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 846.30 | 843.13 | 844.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 846.30 | 843.13 | 844.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 846.30 | 843.13 | 844.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 846.30 | 843.13 | 844.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 849.90 | 844.48 | 845.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 849.90 | 844.48 | 845.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 848.75 | 845.34 | 845.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 852.40 | 845.34 | 845.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 854.20 | 847.11 | 846.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 873.95 | 858.39 | 853.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 872.05 | 880.90 | 874.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 872.05 | 880.90 | 874.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 872.05 | 880.90 | 874.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 872.05 | 880.90 | 874.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 871.10 | 878.94 | 874.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 871.10 | 878.94 | 874.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 870.10 | 872.53 | 872.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 867.90 | 872.53 | 872.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 870.50 | 872.12 | 872.34 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 11:15:00 | 875.00 | 872.73 | 872.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 878.50 | 873.88 | 873.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 908.90 | 910.35 | 897.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 895.20 | 905.41 | 901.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 895.20 | 905.41 | 901.16 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 890.10 | 897.88 | 898.54 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 902.00 | 896.02 | 895.66 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 892.95 | 895.41 | 895.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 889.50 | 893.36 | 894.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 885.30 | 884.99 | 888.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 885.30 | 884.99 | 888.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 870.20 | 876.49 | 880.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 875.70 | 876.49 | 880.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 873.20 | 875.11 | 879.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 876.50 | 875.11 | 879.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 857.40 | 867.81 | 873.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:45:00 | 863.35 | 867.81 | 873.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 862.55 | 866.11 | 871.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 857.00 | 864.08 | 870.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 865.45 | 858.31 | 857.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 865.45 | 858.31 | 857.35 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 852.25 | 857.78 | 858.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 847.65 | 855.75 | 857.09 | Break + close below crossover candle low |

### Cycle 185 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 876.80 | 857.94 | 857.40 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 855.40 | 861.88 | 862.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 852.55 | 860.02 | 861.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 852.70 | 852.59 | 856.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 850.00 | 852.59 | 856.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 852.00 | 852.47 | 855.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 836.90 | 849.94 | 852.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:30:00 | 842.20 | 841.65 | 845.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 837.50 | 833.39 | 833.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 837.50 | 833.39 | 833.03 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 829.95 | 832.71 | 832.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 827.10 | 831.59 | 832.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 832.40 | 828.20 | 829.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 832.40 | 828.20 | 829.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 832.40 | 828.20 | 829.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 830.50 | 828.20 | 829.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 830.10 | 828.58 | 829.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 833.00 | 828.58 | 829.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 833.00 | 829.46 | 829.96 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 850.00 | 833.57 | 831.78 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 830.40 | 835.81 | 836.44 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 872.70 | 843.10 | 839.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 907.50 | 873.06 | 858.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 882.10 | 884.16 | 869.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 14:00:00 | 882.10 | 884.16 | 869.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 930.70 | 950.71 | 943.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 929.15 | 950.71 | 943.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 924.90 | 945.55 | 942.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 926.70 | 945.55 | 942.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 928.10 | 939.21 | 939.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 924.95 | 934.09 | 937.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 904.50 | 902.48 | 913.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 904.50 | 902.48 | 913.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 899.45 | 904.11 | 910.83 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 934.20 | 916.19 | 913.92 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 912.00 | 915.48 | 915.59 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 931.85 | 918.47 | 916.90 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 15:15:00 | 917.10 | 920.96 | 921.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 908.30 | 918.43 | 919.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 10:15:00 | 911.30 | 908.78 | 913.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 10:15:00 | 911.30 | 908.78 | 913.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 911.30 | 908.78 | 913.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 911.30 | 908.78 | 913.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 912.00 | 909.42 | 913.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:15:00 | 911.10 | 910.22 | 913.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:15:00 | 908.95 | 910.76 | 913.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 865.54 | 870.83 | 880.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 863.50 | 870.83 | 880.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 10:15:00 | 858.55 | 858.46 | 867.63 | SL hit (close>ema200) qty=0.50 sl=858.46 alert=retest2 |

### Cycle 197 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 805.65 | 796.77 | 796.40 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 790.40 | 797.11 | 797.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 783.20 | 792.48 | 795.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 793.80 | 783.16 | 785.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 793.80 | 783.16 | 785.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 793.80 | 783.16 | 785.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 795.00 | 783.16 | 785.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 795.45 | 785.62 | 786.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 795.50 | 785.62 | 786.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 809.00 | 790.30 | 788.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 815.80 | 802.52 | 795.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 809.90 | 810.55 | 804.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 809.90 | 810.55 | 804.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 809.65 | 810.52 | 807.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 822.45 | 810.54 | 808.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 816.00 | 813.59 | 810.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 820.00 | 816.39 | 812.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 805.00 | 817.04 | 813.79 | SL hit (close<static) qty=1.00 sl=807.10 alert=retest2 |

### Cycle 200 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 800.65 | 810.64 | 811.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 789.80 | 802.13 | 806.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 801.25 | 798.84 | 803.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 12:15:00 | 801.25 | 798.84 | 803.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 801.25 | 798.84 | 803.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 801.25 | 798.84 | 803.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 803.00 | 799.67 | 803.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 802.85 | 799.67 | 803.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 815.10 | 802.75 | 804.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 815.10 | 802.75 | 804.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 814.05 | 805.01 | 805.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 804.05 | 805.01 | 805.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 806.25 | 805.26 | 805.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 806.25 | 805.26 | 805.15 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 802.40 | 804.65 | 804.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 797.55 | 803.23 | 804.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 804.70 | 801.97 | 803.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 804.70 | 801.97 | 803.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 804.70 | 801.97 | 803.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 805.30 | 801.97 | 803.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 805.30 | 802.63 | 803.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:00:00 | 804.25 | 802.96 | 803.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 806.40 | 804.11 | 803.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 806.40 | 804.11 | 803.86 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 799.70 | 803.58 | 803.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 798.00 | 801.57 | 802.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 793.70 | 792.29 | 796.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 793.70 | 792.29 | 796.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 793.70 | 792.29 | 796.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 785.50 | 792.33 | 795.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 786.50 | 789.08 | 792.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 746.22 | 756.27 | 767.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 747.17 | 756.27 | 767.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 741.90 | 740.59 | 748.39 | SL hit (close>ema200) qty=0.50 sl=740.59 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 772.60 | 744.51 | 743.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 777.70 | 761.32 | 752.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 11:15:00 | 768.50 | 772.72 | 762.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 12:00:00 | 768.50 | 772.72 | 762.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 759.45 | 769.34 | 762.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 759.45 | 769.34 | 762.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 758.00 | 767.07 | 762.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 758.00 | 767.07 | 762.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 756.00 | 764.85 | 761.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 756.80 | 764.85 | 761.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 742.65 | 758.21 | 759.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 731.40 | 741.65 | 747.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 728.45 | 725.88 | 733.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 10:15:00 | 735.35 | 727.77 | 733.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 735.35 | 727.77 | 733.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 735.35 | 727.77 | 733.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 741.00 | 730.42 | 734.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 741.00 | 730.42 | 734.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 746.85 | 736.48 | 736.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 756.50 | 743.51 | 740.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 742.90 | 744.51 | 741.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 742.90 | 744.51 | 741.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 729.80 | 741.57 | 740.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 729.80 | 741.57 | 740.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 729.00 | 739.05 | 739.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 724.30 | 733.83 | 736.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 729.90 | 729.73 | 733.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:00:00 | 729.90 | 729.73 | 733.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 736.10 | 731.00 | 733.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 736.10 | 731.00 | 733.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 741.55 | 733.11 | 734.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 741.55 | 733.11 | 734.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 756.80 | 737.85 | 736.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 799.30 | 753.30 | 744.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 812.05 | 812.33 | 797.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 812.05 | 812.33 | 797.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 834.20 | 833.14 | 828.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 822.25 | 830.81 | 827.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 836.75 | 832.00 | 828.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:30:00 | 837.30 | 833.69 | 830.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:00:00 | 840.50 | 836.57 | 832.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:30:00 | 838.00 | 836.34 | 833.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 841.10 | 836.34 | 833.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 838.65 | 837.15 | 834.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 835.00 | 837.15 | 834.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 830.60 | 835.84 | 833.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 826.70 | 835.84 | 833.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 831.35 | 834.94 | 833.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 806.15 | 834.94 | 833.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 805.40 | 829.03 | 831.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 805.40 | 829.03 | 831.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 798.70 | 804.27 | 807.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 15:15:00 | 786.00 | 785.47 | 791.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 792.40 | 785.47 | 791.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 787.50 | 785.88 | 791.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 778.65 | 785.78 | 790.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 793.15 | 785.35 | 784.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 793.15 | 785.35 | 784.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 13:15:00 | 795.95 | 787.47 | 785.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 812.95 | 821.00 | 814.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 812.95 | 821.00 | 814.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 812.95 | 821.00 | 814.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:00:00 | 817.35 | 820.27 | 814.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 12:45:00 | 816.80 | 818.73 | 814.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:00:00 | 816.25 | 818.24 | 814.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:30:00 | 816.85 | 818.05 | 815.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 794.70 | 813.37 | 813.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 794.70 | 813.37 | 813.56 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 823.25 | 810.96 | 809.93 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 810.00 | 812.93 | 812.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 777.45 | 805.83 | 809.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 796.40 | 788.41 | 796.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 796.40 | 788.41 | 796.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 796.40 | 788.41 | 796.50 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 810.65 | 799.94 | 799.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 848.55 | 811.57 | 805.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 822.20 | 827.73 | 818.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 833.25 | 827.73 | 818.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 844.00 | 830.98 | 820.44 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 802.35 | 822.11 | 824.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 796.25 | 816.94 | 821.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 798.10 | 797.32 | 807.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 798.00 | 797.32 | 807.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 806.70 | 798.36 | 805.82 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 814.00 | 808.94 | 808.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 820.90 | 811.33 | 809.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 813.05 | 817.53 | 814.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 813.05 | 817.53 | 814.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 813.05 | 817.53 | 814.49 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 794.50 | 810.64 | 812.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 771.20 | 796.63 | 802.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 784.15 | 777.10 | 786.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 784.15 | 777.10 | 786.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 796.00 | 780.88 | 787.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 796.00 | 780.88 | 787.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 801.60 | 785.03 | 788.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 803.00 | 785.03 | 788.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 802.55 | 791.75 | 791.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 834.40 | 802.23 | 796.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 820.10 | 825.39 | 813.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 814.95 | 819.87 | 815.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 814.95 | 819.87 | 815.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 814.95 | 819.87 | 815.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 811.80 | 818.26 | 814.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 828.05 | 818.26 | 814.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 10:30:00 | 822.05 | 820.72 | 816.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 15:15:00 | 809.95 | 818.53 | 817.34 | SL hit (close<static) qty=1.00 sl=810.00 alert=retest2 |

### Cycle 220 — SELL (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 15:15:00 | 1242.20 | 1246.94 | 1246.98 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1251.60 | 1247.45 | 1247.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1259.30 | 1249.82 | 1248.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 15:15:00 | 1252.30 | 1252.82 | 1250.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 15:15:00 | 1252.30 | 1252.82 | 1250.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1252.30 | 1252.82 | 1250.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1245.40 | 1252.82 | 1250.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1235.60 | 1249.37 | 1249.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 1236.10 | 1249.37 | 1249.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 1245.90 | 1248.68 | 1248.85 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1252.00 | 1249.34 | 1249.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 1267.20 | 1253.88 | 1251.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1292.30 | 1295.53 | 1285.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 1292.30 | 1295.53 | 1285.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1282.30 | 1292.89 | 1285.02 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 09:15:00 | 543.05 | 2024-04-18 14:15:00 | 534.04 | PARTIAL | 0.50 | 1.66% |
| SELL | retest2 | 2024-04-15 09:15:00 | 543.05 | 2024-04-22 09:15:00 | 541.90 | STOP_HIT | 0.50 | 0.21% |
| SELL | retest2 | 2024-04-16 10:15:00 | 562.15 | 2024-04-22 13:15:00 | 552.80 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2024-04-29 09:15:00 | 574.90 | 2024-04-29 14:15:00 | 569.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-04-29 11:00:00 | 573.65 | 2024-04-29 14:15:00 | 569.90 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-04-29 11:30:00 | 574.80 | 2024-04-29 14:15:00 | 569.90 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-18 09:15:00 | 618.35 | 2024-05-22 09:15:00 | 597.40 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-05-21 09:30:00 | 622.80 | 2024-05-22 09:15:00 | 597.40 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2024-05-21 14:45:00 | 615.55 | 2024-05-22 09:15:00 | 597.40 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-05-29 10:00:00 | 599.40 | 2024-05-30 12:15:00 | 606.05 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-29 14:30:00 | 598.55 | 2024-05-30 12:15:00 | 606.05 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-05-29 15:15:00 | 598.90 | 2024-05-30 12:15:00 | 606.05 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-30 09:45:00 | 599.90 | 2024-05-30 12:15:00 | 606.05 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-06-06 12:30:00 | 507.20 | 2024-06-06 15:15:00 | 518.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-06-12 15:00:00 | 536.20 | 2024-06-19 12:15:00 | 534.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-06-13 10:45:00 | 536.00 | 2024-06-19 12:15:00 | 534.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-06-13 11:15:00 | 536.00 | 2024-06-19 12:15:00 | 534.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-06-19 11:15:00 | 536.25 | 2024-06-19 12:15:00 | 534.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-07-01 11:30:00 | 555.00 | 2024-07-04 12:15:00 | 610.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-01 13:45:00 | 554.10 | 2024-07-04 12:15:00 | 609.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 640.40 | 2024-07-18 14:15:00 | 653.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-07-24 09:15:00 | 652.00 | 2024-07-25 13:15:00 | 642.75 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-07-25 10:30:00 | 648.00 | 2024-07-25 13:15:00 | 642.75 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-07-25 12:00:00 | 648.75 | 2024-07-25 13:15:00 | 642.75 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-07-25 12:45:00 | 647.20 | 2024-07-25 13:15:00 | 642.75 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-07-26 13:00:00 | 647.20 | 2024-07-31 09:15:00 | 655.65 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-07-29 09:45:00 | 649.10 | 2024-07-31 09:15:00 | 655.65 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-09 09:15:00 | 649.85 | 2024-08-09 10:15:00 | 644.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-08-09 12:30:00 | 649.95 | 2024-08-14 11:15:00 | 658.70 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-08-12 10:00:00 | 657.00 | 2024-08-14 11:15:00 | 658.70 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-08-21 09:15:00 | 717.70 | 2024-08-27 14:15:00 | 717.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-08-21 09:45:00 | 713.50 | 2024-08-27 14:15:00 | 717.00 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2024-08-21 11:00:00 | 713.35 | 2024-08-27 14:15:00 | 717.00 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2024-08-21 12:00:00 | 713.35 | 2024-08-27 14:15:00 | 717.00 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-09-02 09:15:00 | 704.15 | 2024-09-09 09:15:00 | 668.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:15:00 | 704.15 | 2024-09-09 10:15:00 | 683.30 | STOP_HIT | 0.50 | 2.96% |
| BUY | retest2 | 2024-09-16 09:15:00 | 691.05 | 2024-09-16 09:15:00 | 683.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-19 10:15:00 | 670.70 | 2024-09-23 09:15:00 | 685.10 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-09 11:30:00 | 701.35 | 2024-10-10 09:15:00 | 709.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-10-09 13:00:00 | 701.05 | 2024-10-10 09:15:00 | 709.60 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-10-09 13:45:00 | 697.55 | 2024-10-10 09:15:00 | 709.60 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-10-10 11:00:00 | 701.45 | 2024-10-14 11:15:00 | 707.75 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-10-10 14:15:00 | 697.85 | 2024-10-14 11:15:00 | 707.75 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-11 10:15:00 | 697.05 | 2024-10-14 11:15:00 | 707.75 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-11 14:15:00 | 700.00 | 2024-10-14 11:15:00 | 707.75 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-11 15:00:00 | 696.10 | 2024-10-14 12:15:00 | 708.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-14 10:15:00 | 696.90 | 2024-10-14 12:15:00 | 708.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-10-22 10:15:00 | 695.30 | 2024-10-24 09:15:00 | 716.95 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-10-22 11:00:00 | 699.85 | 2024-10-24 09:15:00 | 716.95 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-10-22 12:45:00 | 702.05 | 2024-10-24 09:15:00 | 716.95 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-10-31 09:45:00 | 725.10 | 2024-11-07 12:15:00 | 797.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-04 11:00:00 | 723.90 | 2024-11-07 12:15:00 | 796.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-04 11:30:00 | 729.10 | 2024-11-07 13:15:00 | 802.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-19 15:15:00 | 681.50 | 2024-11-22 09:15:00 | 700.70 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-11-21 13:45:00 | 685.05 | 2024-11-22 09:15:00 | 700.70 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-12-03 15:15:00 | 784.35 | 2024-12-12 10:15:00 | 783.95 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-12-04 14:30:00 | 788.70 | 2024-12-12 10:15:00 | 783.95 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-12-05 12:00:00 | 784.55 | 2024-12-12 10:15:00 | 783.95 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-12-05 13:00:00 | 784.05 | 2024-12-12 10:15:00 | 783.95 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-12-06 14:00:00 | 789.95 | 2024-12-12 10:15:00 | 783.95 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-13 09:15:00 | 789.00 | 2024-12-16 13:15:00 | 794.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-16 10:45:00 | 792.80 | 2024-12-16 13:15:00 | 794.15 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-12-16 11:45:00 | 789.00 | 2024-12-16 13:15:00 | 794.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-23 15:15:00 | 767.50 | 2024-12-24 10:15:00 | 780.15 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-12-27 15:00:00 | 813.55 | 2025-01-06 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-12-30 15:00:00 | 803.85 | 2025-01-06 09:15:00 | 799.95 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-12-31 10:45:00 | 802.95 | 2025-01-06 09:15:00 | 799.95 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-01-08 12:15:00 | 792.80 | 2025-01-10 09:15:00 | 753.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:15:00 | 792.80 | 2025-01-13 13:15:00 | 713.52 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-17 10:15:00 | 780.00 | 2025-01-22 10:15:00 | 759.50 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-01-17 11:00:00 | 778.25 | 2025-01-22 10:15:00 | 759.50 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-01-20 09:15:00 | 784.85 | 2025-01-22 10:15:00 | 759.50 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-02-11 10:15:00 | 775.05 | 2025-02-14 10:15:00 | 763.20 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-11 14:00:00 | 774.25 | 2025-02-14 11:15:00 | 747.95 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-02-12 11:30:00 | 772.85 | 2025-02-14 11:15:00 | 747.95 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-02-12 12:45:00 | 775.90 | 2025-02-14 11:15:00 | 747.95 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-02-13 09:15:00 | 779.45 | 2025-02-14 11:15:00 | 747.95 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2025-02-18 11:15:00 | 713.90 | 2025-02-20 10:15:00 | 755.05 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest2 | 2025-02-28 09:15:00 | 742.95 | 2025-03-03 09:15:00 | 705.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 742.95 | 2025-03-03 11:15:00 | 668.66 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-11 10:30:00 | 800.65 | 2025-03-18 11:15:00 | 806.10 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-04-02 15:00:00 | 850.15 | 2025-04-03 14:15:00 | 862.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-04-03 09:30:00 | 847.50 | 2025-04-03 14:15:00 | 862.90 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest1 | 2025-04-07 09:15:00 | 778.40 | 2025-04-07 09:15:00 | 700.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:45:00 | 756.10 | 2025-04-15 11:15:00 | 780.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-04-09 09:15:00 | 755.55 | 2025-04-15 11:15:00 | 780.00 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-04-11 10:00:00 | 759.30 | 2025-04-15 11:15:00 | 780.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-04-11 10:45:00 | 759.50 | 2025-04-15 11:15:00 | 780.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-04-21 09:45:00 | 786.20 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-04-21 10:30:00 | 789.70 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-04-21 11:15:00 | 785.35 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-04-21 12:00:00 | 787.90 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-07 10:00:00 | 751.50 | 2025-05-08 11:15:00 | 763.40 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-19 09:15:00 | 787.85 | 2025-05-20 13:15:00 | 772.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-05-19 09:45:00 | 786.10 | 2025-05-20 13:15:00 | 772.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-05-19 10:45:00 | 787.20 | 2025-05-20 13:15:00 | 772.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-05-19 14:15:00 | 788.20 | 2025-05-20 13:15:00 | 772.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-05-22 11:30:00 | 764.40 | 2025-05-26 12:15:00 | 775.85 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-22 12:00:00 | 764.60 | 2025-05-26 12:15:00 | 775.85 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-05-22 12:30:00 | 764.65 | 2025-05-26 12:15:00 | 775.85 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-05-23 13:30:00 | 763.05 | 2025-05-26 12:15:00 | 775.85 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-27 15:00:00 | 783.75 | 2025-05-29 09:15:00 | 862.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 909.70 | 2025-06-20 15:15:00 | 912.80 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-19 09:15:00 | 908.00 | 2025-06-20 15:15:00 | 912.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-19 09:45:00 | 908.65 | 2025-06-20 15:15:00 | 912.80 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-30 12:30:00 | 945.35 | 2025-06-30 13:15:00 | 924.15 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-07-07 09:15:00 | 923.25 | 2025-07-09 10:15:00 | 937.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-08 09:45:00 | 923.95 | 2025-07-09 10:15:00 | 937.80 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-15 09:15:00 | 912.50 | 2025-07-17 09:15:00 | 931.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-07-24 10:15:00 | 903.00 | 2025-07-28 15:15:00 | 857.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:30:00 | 904.60 | 2025-07-28 15:15:00 | 859.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 10:30:00 | 904.20 | 2025-07-28 15:15:00 | 858.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:15:00 | 903.00 | 2025-07-29 09:15:00 | 880.00 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-07-24 13:30:00 | 904.60 | 2025-07-29 09:15:00 | 880.00 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-07-25 10:30:00 | 904.20 | 2025-07-29 09:15:00 | 880.00 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2025-08-11 09:30:00 | 864.75 | 2025-08-12 11:15:00 | 877.55 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-08-11 11:30:00 | 863.00 | 2025-08-12 11:15:00 | 877.55 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-11 13:45:00 | 862.70 | 2025-08-12 11:15:00 | 877.55 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-08-12 09:30:00 | 863.50 | 2025-08-12 11:15:00 | 877.55 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-02 10:15:00 | 829.40 | 2025-09-03 15:15:00 | 852.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-09-02 13:15:00 | 829.70 | 2025-09-03 15:15:00 | 852.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-09-29 11:30:00 | 857.00 | 2025-10-03 10:15:00 | 865.45 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-13 09:15:00 | 836.90 | 2025-10-16 15:15:00 | 837.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-14 09:30:00 | 842.20 | 2025-10-16 15:15:00 | 837.50 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2025-11-18 13:15:00 | 911.10 | 2025-11-24 09:15:00 | 865.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 14:15:00 | 908.95 | 2025-11-24 09:15:00 | 863.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 13:15:00 | 911.10 | 2025-11-25 10:15:00 | 858.55 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-11-18 14:15:00 | 908.95 | 2025-11-25 10:15:00 | 858.55 | STOP_HIT | 0.50 | 5.54% |
| BUY | retest2 | 2025-12-29 10:00:00 | 822.45 | 2025-12-30 09:15:00 | 805.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-29 11:30:00 | 816.00 | 2025-12-30 09:15:00 | 805.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-29 13:45:00 | 820.00 | 2025-12-30 09:15:00 | 805.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-01 09:15:00 | 804.05 | 2026-01-01 09:15:00 | 806.25 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-01-02 12:00:00 | 804.25 | 2026-01-02 14:15:00 | 806.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-01-07 12:30:00 | 785.50 | 2026-01-12 09:15:00 | 746.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:30:00 | 786.50 | 2026-01-12 09:15:00 | 747.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:30:00 | 785.50 | 2026-01-13 14:15:00 | 741.90 | STOP_HIT | 0.50 | 5.55% |
| SELL | retest2 | 2026-01-08 09:30:00 | 786.50 | 2026-01-13 14:15:00 | 741.90 | STOP_HIT | 0.50 | 5.67% |
| BUY | retest2 | 2026-02-11 12:30:00 | 837.30 | 2026-02-13 09:15:00 | 805.40 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-02-12 10:00:00 | 840.50 | 2026-02-13 09:15:00 | 805.40 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-02-12 11:30:00 | 838.00 | 2026-02-13 09:15:00 | 805.40 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2026-02-12 12:15:00 | 841.10 | 2026-02-13 09:15:00 | 805.40 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2026-02-23 10:30:00 | 778.65 | 2026-02-25 12:15:00 | 793.15 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-02 11:00:00 | 817.35 | 2026-03-04 09:15:00 | 794.70 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-03-02 12:45:00 | 816.80 | 2026-03-04 09:15:00 | 794.70 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-03-02 14:00:00 | 816.25 | 2026-03-04 09:15:00 | 794.70 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-03-02 14:30:00 | 816.85 | 2026-03-04 09:15:00 | 794.70 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-03-30 09:15:00 | 828.05 | 2026-03-30 15:15:00 | 809.95 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-03-30 10:30:00 | 822.05 | 2026-03-30 15:15:00 | 809.95 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-04-01 09:15:00 | 841.30 | 2026-04-08 09:15:00 | 909.04 | TARGET_HIT | 1.00 | 8.05% |
| BUY | retest2 | 2026-04-02 10:30:00 | 826.40 | 2026-04-08 11:15:00 | 925.43 | TARGET_HIT | 1.00 | 11.98% |
| BUY | retest2 | 2026-04-13 10:30:00 | 960.45 | 2026-04-15 12:15:00 | 1056.50 | TARGET_HIT | 1.00 | 10.00% |

# Poonawalla Fincorp Ltd. (POONAWALLA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 461.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 240 |
| ALERT1 | 167 |
| ALERT2 | 164 |
| ALERT2_SKIP | 112 |
| ALERT3 | 380 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 148 |
| PARTIAL | 17 |
| TARGET_HIT | 7 |
| STOP_HIT | 149 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 173 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 75 / 98
- **Target hits / Stop hits / Partials:** 7 / 149 / 17
- **Avg / median % per leg:** 0.53% / -0.47%
- **Sum % (uncompounded):** 92.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 28 | 40.6% | 2 | 66 | 1 | -0.32% | -22.3% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 5 | 1 | 1.73% | 10.4% |
| BUY @ 3rd Alert (retest2) | 63 | 22 | 34.9% | 2 | 61 | 0 | -0.52% | -32.7% |
| SELL (all) | 104 | 47 | 45.2% | 5 | 83 | 16 | 1.10% | 114.8% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.23% | -3.7% |
| SELL @ 3rd Alert (retest2) | 101 | 47 | 46.5% | 5 | 80 | 16 | 1.17% | 118.5% |
| retest1 (combined) | 9 | 6 | 66.7% | 0 | 8 | 1 | 0.74% | 6.7% |
| retest2 (combined) | 164 | 69 | 42.1% | 7 | 141 | 16 | 0.52% | 85.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 325.70 | 327.84 | 328.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 09:15:00 | 324.00 | 327.07 | 327.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 13:15:00 | 329.05 | 326.74 | 327.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 13:15:00 | 329.05 | 326.74 | 327.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 329.05 | 326.74 | 327.24 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 341.00 | 330.10 | 328.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 10:15:00 | 342.75 | 332.63 | 329.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 331.40 | 334.64 | 332.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 331.40 | 334.64 | 332.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 331.40 | 334.64 | 332.48 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 12:15:00 | 344.00 | 345.92 | 346.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 12:15:00 | 337.50 | 343.53 | 344.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 09:15:00 | 342.25 | 340.98 | 342.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 342.25 | 340.98 | 342.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 342.25 | 340.98 | 342.17 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 346.70 | 343.45 | 343.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 14:15:00 | 347.10 | 345.73 | 344.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 09:15:00 | 345.50 | 345.89 | 345.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 10:15:00 | 345.40 | 345.79 | 345.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 345.40 | 345.79 | 345.05 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 15:15:00 | 343.95 | 344.77 | 344.77 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 347.05 | 345.23 | 344.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 14:15:00 | 347.80 | 346.53 | 345.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 12:15:00 | 347.25 | 347.36 | 346.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 12:15:00 | 347.25 | 347.36 | 346.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 347.25 | 347.36 | 346.55 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 346.20 | 348.85 | 349.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 09:15:00 | 344.90 | 347.07 | 348.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 348.00 | 345.02 | 346.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 348.00 | 345.02 | 346.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 348.00 | 345.02 | 346.23 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 347.00 | 344.78 | 344.65 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 343.55 | 344.56 | 344.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 14:15:00 | 341.20 | 343.88 | 344.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 10:15:00 | 344.15 | 343.23 | 343.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 10:15:00 | 344.15 | 343.23 | 343.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 344.15 | 343.23 | 343.82 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 361.10 | 346.18 | 344.87 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 342.90 | 347.45 | 347.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 12:15:00 | 338.85 | 341.11 | 343.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 340.10 | 339.20 | 341.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 340.10 | 339.20 | 341.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 340.10 | 339.20 | 341.63 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 344.65 | 342.39 | 342.13 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 12:15:00 | 340.00 | 342.57 | 342.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 10:15:00 | 339.75 | 341.05 | 341.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 12:15:00 | 341.35 | 340.95 | 341.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 12:15:00 | 341.35 | 340.95 | 341.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 341.35 | 340.95 | 341.57 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 09:15:00 | 346.70 | 342.84 | 342.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 13:15:00 | 352.45 | 346.43 | 344.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 12:15:00 | 365.40 | 366.80 | 361.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 362.45 | 365.57 | 361.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 362.45 | 365.57 | 361.43 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 14:15:00 | 367.05 | 368.34 | 368.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 09:15:00 | 365.60 | 367.61 | 368.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 12:15:00 | 369.70 | 367.36 | 367.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 12:15:00 | 369.70 | 367.36 | 367.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 369.70 | 367.36 | 367.72 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 13:15:00 | 371.00 | 368.09 | 368.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 14:15:00 | 371.40 | 368.75 | 368.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 10:15:00 | 368.25 | 369.22 | 368.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 10:15:00 | 368.25 | 369.22 | 368.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 368.25 | 369.22 | 368.69 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 12:15:00 | 366.80 | 368.44 | 368.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 13:15:00 | 365.75 | 367.90 | 368.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 14:15:00 | 369.85 | 368.29 | 368.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 14:15:00 | 369.85 | 368.29 | 368.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 369.85 | 368.29 | 368.51 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 372.55 | 369.14 | 368.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 11:15:00 | 376.90 | 372.25 | 370.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 13:15:00 | 373.80 | 376.54 | 374.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 13:15:00 | 373.80 | 376.54 | 374.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 373.80 | 376.54 | 374.30 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 370.65 | 375.67 | 376.01 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 380.15 | 376.00 | 375.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 389.40 | 381.63 | 379.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 393.75 | 394.35 | 390.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 391.45 | 393.77 | 390.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 391.45 | 393.77 | 390.39 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 434.80 | 436.30 | 436.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 12:15:00 | 434.00 | 435.84 | 436.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 436.45 | 435.39 | 435.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 436.45 | 435.39 | 435.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 436.45 | 435.39 | 435.80 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 10:15:00 | 439.80 | 436.27 | 436.17 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 12:15:00 | 435.35 | 436.09 | 436.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 13:15:00 | 431.75 | 435.22 | 435.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 433.35 | 432.86 | 434.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 433.35 | 432.86 | 434.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 433.35 | 432.86 | 434.33 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 433.90 | 425.91 | 425.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 14:15:00 | 438.00 | 430.94 | 428.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 433.50 | 436.06 | 432.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 14:15:00 | 433.50 | 436.06 | 432.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 433.50 | 436.06 | 432.98 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 433.60 | 436.44 | 436.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 429.90 | 435.13 | 435.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 413.60 | 409.63 | 414.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 10:15:00 | 415.00 | 410.70 | 414.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 415.00 | 410.70 | 414.54 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 14:15:00 | 410.90 | 409.17 | 409.13 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 09:15:00 | 406.95 | 408.70 | 408.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 10:15:00 | 406.00 | 408.16 | 408.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 393.05 | 392.93 | 397.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 398.20 | 392.42 | 394.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 398.20 | 392.42 | 394.94 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 391.10 | 383.37 | 382.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 395.10 | 385.72 | 383.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 15:15:00 | 386.85 | 388.40 | 386.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 15:15:00 | 386.85 | 388.40 | 386.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 386.85 | 388.40 | 386.28 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 379.95 | 384.80 | 385.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 379.00 | 383.64 | 384.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 09:15:00 | 381.50 | 379.01 | 380.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 381.50 | 379.01 | 380.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 381.50 | 379.01 | 380.80 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 380.70 | 377.17 | 376.81 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 375.60 | 377.74 | 377.91 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 382.20 | 378.44 | 378.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 384.35 | 381.51 | 380.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 15:15:00 | 381.75 | 381.91 | 380.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 378.65 | 381.26 | 380.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 378.65 | 381.26 | 380.65 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 376.55 | 380.01 | 380.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 374.50 | 378.91 | 379.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 381.20 | 378.68 | 379.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 381.20 | 378.68 | 379.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 381.20 | 378.68 | 379.21 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 383.15 | 379.43 | 379.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 11:15:00 | 388.80 | 382.08 | 380.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 380.45 | 384.97 | 382.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 380.45 | 384.97 | 382.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 380.45 | 384.97 | 382.99 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 375.75 | 381.13 | 381.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 374.20 | 379.22 | 380.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 379.35 | 378.55 | 380.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 379.35 | 378.55 | 380.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 379.35 | 378.55 | 380.00 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 382.85 | 379.68 | 379.37 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 374.90 | 378.38 | 378.81 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 12:15:00 | 378.20 | 377.47 | 377.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 13:15:00 | 380.15 | 378.01 | 377.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 15:15:00 | 376.60 | 377.81 | 377.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 15:15:00 | 376.60 | 377.81 | 377.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 376.60 | 377.81 | 377.67 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 14:15:00 | 376.85 | 377.88 | 377.88 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 15:15:00 | 379.60 | 378.22 | 378.04 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 375.75 | 377.67 | 377.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 374.85 | 377.10 | 377.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 09:15:00 | 377.90 | 376.31 | 376.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 09:15:00 | 377.90 | 376.31 | 376.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 377.90 | 376.31 | 376.87 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 380.00 | 377.25 | 377.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 10:15:00 | 381.00 | 378.00 | 377.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 12:15:00 | 375.55 | 377.57 | 377.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 12:15:00 | 375.55 | 377.57 | 377.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 375.55 | 377.57 | 377.35 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 15:15:00 | 375.00 | 376.86 | 377.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 368.60 | 375.21 | 376.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 365.50 | 361.24 | 367.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 10:15:00 | 358.80 | 360.75 | 366.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 358.80 | 360.75 | 366.44 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 360.60 | 357.92 | 357.82 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 09:15:00 | 352.90 | 356.91 | 357.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 10:15:00 | 351.75 | 355.88 | 356.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-30 15:15:00 | 353.75 | 353.68 | 355.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 09:15:00 | 357.00 | 354.35 | 355.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 357.00 | 354.35 | 355.37 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 361.60 | 351.37 | 350.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 15:15:00 | 364.40 | 359.13 | 355.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 12:15:00 | 393.00 | 394.13 | 390.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 387.10 | 392.34 | 390.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 387.10 | 392.34 | 390.79 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 13:15:00 | 386.25 | 389.58 | 389.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 09:15:00 | 368.95 | 383.71 | 386.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 367.35 | 366.49 | 372.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 13:15:00 | 365.70 | 363.52 | 364.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 365.70 | 363.52 | 364.75 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 366.25 | 363.56 | 363.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 10:15:00 | 367.40 | 364.33 | 363.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 10:15:00 | 414.00 | 414.49 | 407.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 411.00 | 412.14 | 409.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 411.00 | 412.14 | 409.32 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 415.00 | 418.29 | 418.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 414.20 | 417.47 | 418.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 13:15:00 | 416.00 | 415.77 | 416.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 13:15:00 | 416.00 | 415.77 | 416.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 416.00 | 415.77 | 416.93 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 15:15:00 | 422.00 | 417.93 | 417.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 432.00 | 421.83 | 419.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 438.55 | 439.47 | 436.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 438.55 | 439.47 | 436.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 438.55 | 439.47 | 436.08 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 434.85 | 440.00 | 440.33 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 445.80 | 440.56 | 439.92 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 10:15:00 | 439.30 | 441.42 | 441.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 11:15:00 | 434.20 | 439.98 | 440.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 09:15:00 | 439.00 | 434.12 | 435.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 439.00 | 434.12 | 435.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 439.00 | 434.12 | 435.89 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 438.65 | 436.78 | 436.60 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 431.20 | 436.20 | 436.60 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 10:15:00 | 438.90 | 434.75 | 434.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 13:15:00 | 439.80 | 436.69 | 435.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 458.20 | 458.92 | 452.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 10:15:00 | 494.15 | 497.66 | 493.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 494.15 | 497.66 | 493.63 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 483.55 | 491.25 | 491.42 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 14:15:00 | 496.30 | 492.26 | 491.86 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 487.70 | 491.47 | 491.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 471.45 | 483.61 | 487.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 13:15:00 | 486.05 | 482.80 | 485.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 13:15:00 | 486.05 | 482.80 | 485.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 486.05 | 482.80 | 485.48 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 15:15:00 | 496.00 | 487.16 | 487.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 505.75 | 490.88 | 488.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 496.00 | 497.50 | 493.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 10:15:00 | 493.65 | 496.73 | 493.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 493.65 | 496.73 | 493.77 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 488.95 | 492.50 | 492.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 484.35 | 490.87 | 491.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 12:15:00 | 488.55 | 488.42 | 490.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 13:15:00 | 495.00 | 489.73 | 490.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 495.00 | 489.73 | 490.71 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 11:15:00 | 485.90 | 477.31 | 476.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 12:15:00 | 486.50 | 479.15 | 477.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 15:15:00 | 478.20 | 480.42 | 478.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 15:15:00 | 478.20 | 480.42 | 478.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 478.20 | 480.42 | 478.36 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 476.65 | 486.33 | 487.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 470.50 | 481.04 | 483.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 479.85 | 474.89 | 478.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 479.85 | 474.89 | 478.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 479.85 | 474.89 | 478.46 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 483.00 | 480.22 | 479.92 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 09:15:00 | 474.20 | 479.02 | 479.40 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 489.05 | 480.57 | 479.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 496.50 | 485.10 | 482.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 10:15:00 | 489.15 | 491.77 | 488.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 10:15:00 | 489.15 | 491.77 | 488.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 489.15 | 491.77 | 488.36 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 15:15:00 | 457.00 | 482.67 | 485.30 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 10:15:00 | 472.40 | 466.35 | 466.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 10:15:00 | 477.45 | 472.46 | 469.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 471.60 | 472.84 | 470.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 13:15:00 | 471.60 | 472.84 | 470.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 471.60 | 472.84 | 470.78 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 462.70 | 469.04 | 469.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 14:15:00 | 455.00 | 465.56 | 468.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 460.60 | 460.09 | 463.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 467.80 | 461.74 | 463.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 467.80 | 461.74 | 463.98 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 465.50 | 463.89 | 463.84 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 463.10 | 463.73 | 463.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 14:15:00 | 460.20 | 462.28 | 462.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 10:15:00 | 463.20 | 462.33 | 462.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 10:15:00 | 463.20 | 462.33 | 462.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 463.20 | 462.33 | 462.82 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 14:15:00 | 452.45 | 448.14 | 447.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 09:15:00 | 457.95 | 450.75 | 449.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 10:15:00 | 450.00 | 450.60 | 449.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 10:15:00 | 450.00 | 450.60 | 449.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 450.00 | 450.60 | 449.19 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 10:15:00 | 454.30 | 460.68 | 460.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 11:15:00 | 452.45 | 459.04 | 460.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 12:15:00 | 463.00 | 459.83 | 460.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 12:15:00 | 463.00 | 459.83 | 460.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 463.00 | 459.83 | 460.37 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 13:15:00 | 465.55 | 460.97 | 460.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 15:15:00 | 480.05 | 466.31 | 463.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 12:15:00 | 468.70 | 470.12 | 466.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 13:15:00 | 465.00 | 469.09 | 466.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 465.00 | 469.09 | 466.55 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 15:15:00 | 463.50 | 465.46 | 465.61 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 473.05 | 466.98 | 466.28 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 13:15:00 | 465.85 | 467.19 | 467.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 14:15:00 | 461.30 | 466.01 | 466.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 470.40 | 466.69 | 466.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 470.40 | 466.69 | 466.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 470.40 | 466.69 | 466.93 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 10:15:00 | 476.00 | 468.55 | 467.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 11:15:00 | 479.00 | 470.64 | 468.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 475.90 | 475.93 | 472.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 14:15:00 | 476.00 | 476.75 | 474.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 476.00 | 476.75 | 474.39 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 12:15:00 | 471.30 | 473.45 | 473.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 467.70 | 471.67 | 472.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 476.90 | 471.81 | 472.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 476.90 | 471.81 | 472.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 476.90 | 471.81 | 472.46 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 477.40 | 472.92 | 472.91 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 12:15:00 | 472.05 | 473.24 | 473.37 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 14:15:00 | 474.90 | 473.43 | 473.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 09:15:00 | 484.65 | 475.84 | 474.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 14:15:00 | 493.40 | 494.12 | 488.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 10:15:00 | 490.95 | 495.04 | 492.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 490.95 | 495.04 | 492.72 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 14:15:00 | 488.80 | 491.11 | 491.35 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 14:15:00 | 492.00 | 490.97 | 490.96 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 09:15:00 | 487.15 | 490.31 | 490.67 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 13:15:00 | 494.15 | 491.24 | 490.96 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 489.90 | 491.03 | 491.06 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 14:15:00 | 493.60 | 491.20 | 491.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 15:15:00 | 494.30 | 491.82 | 491.39 | Break + close above crossover candle high |

### Cycle 89 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 480.65 | 489.59 | 490.42 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 15:15:00 | 489.00 | 487.39 | 487.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 496.60 | 489.23 | 488.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 490.95 | 496.27 | 493.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 490.95 | 496.27 | 493.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 490.95 | 496.27 | 493.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 12:30:00 | 494.85 | 496.28 | 493.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 14:30:00 | 494.90 | 496.71 | 495.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 495.25 | 496.19 | 495.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 11:00:00 | 498.10 | 496.43 | 495.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 494.70 | 496.08 | 495.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:30:00 | 495.80 | 496.08 | 495.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 495.00 | 495.87 | 495.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-23 13:15:00 | 494.65 | 495.62 | 495.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 13:15:00 | 494.65 | 495.62 | 495.68 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 498.85 | 496.01 | 495.79 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 12:15:00 | 494.05 | 495.67 | 495.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 492.60 | 494.66 | 495.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 09:15:00 | 494.50 | 494.20 | 494.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 494.50 | 494.20 | 494.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 494.50 | 494.20 | 494.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 10:30:00 | 491.75 | 493.46 | 494.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 12:15:00 | 496.00 | 494.08 | 494.58 | SL hit (close>static) qty=1.00 sl=495.65 alert=retest2 |

### Cycle 94 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 494.40 | 491.61 | 491.24 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 486.70 | 490.60 | 490.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 15:15:00 | 482.90 | 485.92 | 487.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 462.00 | 460.67 | 466.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 10:00:00 | 462.00 | 460.67 | 466.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 462.00 | 461.48 | 465.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:45:00 | 461.25 | 461.24 | 465.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 458.25 | 461.97 | 464.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:45:00 | 461.60 | 458.47 | 460.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 13:15:00 | 468.75 | 462.53 | 462.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 468.75 | 462.53 | 462.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 469.05 | 463.83 | 462.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 13:15:00 | 466.40 | 467.27 | 465.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:45:00 | 467.25 | 467.27 | 465.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 466.00 | 467.02 | 465.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:45:00 | 465.40 | 467.02 | 465.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 467.90 | 467.20 | 465.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 471.70 | 467.20 | 465.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:30:00 | 468.45 | 469.44 | 467.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 469.00 | 468.75 | 467.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 468.30 | 468.49 | 467.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 466.40 | 468.07 | 467.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:00:00 | 466.40 | 468.07 | 467.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 466.75 | 467.81 | 467.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:30:00 | 466.45 | 467.81 | 467.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 467.15 | 467.68 | 467.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:45:00 | 466.85 | 467.68 | 467.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-17 14:15:00 | 463.90 | 466.92 | 467.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 14:15:00 | 463.90 | 466.92 | 467.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 460.80 | 465.04 | 466.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 462.75 | 460.82 | 462.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 462.75 | 460.82 | 462.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 462.75 | 460.82 | 462.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 462.75 | 460.82 | 462.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 460.10 | 460.67 | 462.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 11:30:00 | 458.35 | 460.39 | 462.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 09:45:00 | 459.35 | 458.32 | 460.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 12:15:00 | 459.55 | 458.91 | 460.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:00:00 | 459.50 | 459.03 | 460.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 460.50 | 459.40 | 460.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 460.50 | 459.40 | 460.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 460.50 | 459.62 | 460.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 461.85 | 459.69 | 460.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 460.30 | 459.82 | 460.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:15:00 | 458.00 | 459.67 | 460.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 457.65 | 457.72 | 458.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 452.20 | 447.92 | 447.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 452.20 | 447.92 | 447.64 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 427.05 | 445.26 | 447.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 418.40 | 439.89 | 444.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 432.00 | 429.43 | 435.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 432.00 | 429.43 | 435.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 435.70 | 430.69 | 435.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 435.70 | 430.69 | 435.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 433.95 | 431.34 | 435.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 437.10 | 431.34 | 435.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 433.35 | 431.74 | 434.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 444.75 | 431.74 | 434.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 449.65 | 435.32 | 436.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 449.65 | 435.32 | 436.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 450.60 | 438.38 | 437.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 453.00 | 444.68 | 440.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 454.00 | 459.58 | 453.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 454.00 | 459.58 | 453.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 447.95 | 457.25 | 452.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 447.00 | 457.25 | 452.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 442.30 | 454.26 | 451.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:30:00 | 444.00 | 454.26 | 451.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 13:15:00 | 435.90 | 447.99 | 449.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-10 14:15:00 | 424.65 | 443.32 | 447.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 09:15:00 | 445.75 | 441.27 | 445.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 445.75 | 441.27 | 445.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 445.75 | 441.27 | 445.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:45:00 | 446.55 | 441.27 | 445.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 442.00 | 441.42 | 445.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:30:00 | 445.40 | 441.42 | 445.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 443.70 | 442.26 | 444.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:45:00 | 447.00 | 442.26 | 444.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 443.40 | 442.49 | 444.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 13:30:00 | 444.20 | 442.49 | 444.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 443.95 | 442.08 | 443.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 445.40 | 442.08 | 443.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 446.60 | 442.98 | 444.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 446.60 | 442.98 | 444.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 447.30 | 443.84 | 444.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 12:30:00 | 444.65 | 444.19 | 444.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 13:15:00 | 444.50 | 444.19 | 444.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:15:00 | 422.42 | 427.41 | 431.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:15:00 | 422.27 | 427.41 | 431.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-21 14:15:00 | 421.80 | 421.59 | 424.90 | SL hit (close>ema200) qty=0.50 sl=421.59 alert=retest2 |

### Cycle 102 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 427.80 | 423.15 | 422.89 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 420.80 | 423.37 | 423.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 417.10 | 421.83 | 422.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 412.20 | 411.68 | 415.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:15:00 | 415.50 | 411.68 | 415.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 416.00 | 412.54 | 415.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 415.15 | 412.54 | 415.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 413.15 | 412.66 | 415.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 410.80 | 412.06 | 414.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:00:00 | 408.15 | 412.06 | 414.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 419.70 | 415.06 | 415.08 | SL hit (close>static) qty=1.00 sl=417.15 alert=retest2 |

### Cycle 104 — BUY (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 14:15:00 | 422.00 | 416.44 | 415.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 10:15:00 | 423.50 | 419.50 | 417.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 420.15 | 420.50 | 418.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 14:45:00 | 420.55 | 420.50 | 418.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 419.00 | 420.20 | 418.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 422.70 | 420.20 | 418.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 416.80 | 420.66 | 419.68 | SL hit (close<static) qty=1.00 sl=418.30 alert=retest2 |

### Cycle 105 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 416.65 | 418.78 | 419.01 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 419.65 | 419.12 | 419.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 420.95 | 419.58 | 419.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 14:15:00 | 423.45 | 424.59 | 422.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 14:30:00 | 424.35 | 424.59 | 422.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 423.75 | 424.42 | 422.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 420.80 | 424.42 | 422.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 420.50 | 423.64 | 422.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 420.50 | 423.64 | 422.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 420.50 | 423.01 | 422.45 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 418.70 | 421.42 | 421.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 417.75 | 420.68 | 421.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 411.70 | 404.77 | 406.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 411.70 | 404.77 | 406.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 411.70 | 404.77 | 406.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 411.70 | 404.77 | 406.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 410.55 | 405.93 | 406.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:30:00 | 414.95 | 405.93 | 406.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 411.60 | 407.06 | 406.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 415.50 | 408.75 | 407.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 408.95 | 411.50 | 409.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 408.95 | 411.50 | 409.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 408.95 | 411.50 | 409.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 408.95 | 411.50 | 409.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 411.80 | 411.56 | 409.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 410.50 | 411.56 | 409.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 414.80 | 412.21 | 410.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 411.30 | 412.21 | 410.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 409.40 | 411.96 | 410.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 409.40 | 411.96 | 410.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 413.90 | 412.35 | 410.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 409.00 | 411.08 | 410.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 403.30 | 409.52 | 409.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 11:15:00 | 396.40 | 402.76 | 405.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 13:15:00 | 373.05 | 371.00 | 377.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 14:00:00 | 373.05 | 371.00 | 377.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 375.35 | 371.87 | 377.19 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 15:15:00 | 381.30 | 379.04 | 378.90 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 375.75 | 378.43 | 378.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 14:15:00 | 372.05 | 376.69 | 377.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 11:15:00 | 362.85 | 362.65 | 366.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 09:15:00 | 350.35 | 362.69 | 365.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 348.80 | 347.38 | 351.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 352.40 | 347.38 | 351.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 349.65 | 347.83 | 350.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 352.00 | 347.83 | 350.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 350.55 | 348.36 | 350.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 350.55 | 348.36 | 350.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 354.30 | 349.55 | 350.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 354.30 | 349.55 | 350.95 | SL hit (close>ema400) qty=1.00 sl=350.95 alert=retest1 |

### Cycle 112 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 361.35 | 351.91 | 351.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 371.85 | 355.90 | 353.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 11:15:00 | 366.80 | 367.06 | 362.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:00:00 | 366.80 | 367.06 | 362.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 369.00 | 366.73 | 362.89 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 360.55 | 362.17 | 362.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 358.55 | 361.44 | 361.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 15:15:00 | 355.00 | 354.77 | 357.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 09:15:00 | 362.00 | 354.77 | 357.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 362.15 | 356.25 | 357.55 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 365.65 | 359.60 | 358.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 370.20 | 363.99 | 361.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 393.55 | 397.05 | 389.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 10:00:00 | 393.55 | 397.05 | 389.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 398.75 | 400.03 | 397.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 396.50 | 400.03 | 397.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 402.95 | 403.44 | 401.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 401.25 | 403.44 | 401.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 403.30 | 403.08 | 401.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:30:00 | 402.25 | 403.08 | 401.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 401.00 | 402.60 | 401.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 405.00 | 402.60 | 401.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 399.85 | 403.66 | 402.88 | SL hit (close<static) qty=1.00 sl=401.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 398.00 | 401.90 | 402.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 394.60 | 400.44 | 401.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 392.55 | 390.20 | 393.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 392.55 | 390.20 | 393.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 392.55 | 390.20 | 393.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 393.10 | 390.20 | 393.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 392.50 | 390.66 | 393.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 389.80 | 390.65 | 392.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:15:00 | 389.95 | 390.65 | 392.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 388.70 | 389.67 | 391.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 12:15:00 | 391.10 | 388.20 | 387.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 391.10 | 388.20 | 387.94 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 383.25 | 387.21 | 387.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 374.50 | 383.47 | 385.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 389.10 | 382.79 | 383.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 389.10 | 382.79 | 383.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 389.10 | 382.79 | 383.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 389.60 | 382.79 | 383.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 386.35 | 383.50 | 384.02 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 388.00 | 384.82 | 384.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 390.80 | 386.28 | 385.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 393.45 | 393.95 | 390.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 393.45 | 393.95 | 390.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 396.95 | 399.12 | 397.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 396.95 | 399.12 | 397.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 395.70 | 398.44 | 397.33 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 395.45 | 396.70 | 396.78 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 400.60 | 397.48 | 397.12 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 396.50 | 398.66 | 398.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 392.85 | 396.99 | 397.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 397.35 | 394.95 | 396.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 397.35 | 394.95 | 396.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 397.35 | 394.95 | 396.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 397.35 | 394.95 | 396.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 397.30 | 395.42 | 396.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 394.90 | 395.42 | 396.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 401.00 | 396.53 | 396.98 | SL hit (close>static) qty=1.00 sl=398.85 alert=retest2 |

### Cycle 122 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 403.20 | 397.87 | 397.54 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 397.00 | 400.39 | 400.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 394.05 | 398.95 | 400.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 12:15:00 | 399.50 | 398.11 | 399.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 12:15:00 | 399.50 | 398.11 | 399.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 399.50 | 398.11 | 399.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 398.85 | 398.11 | 399.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 402.00 | 398.89 | 399.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 402.00 | 398.89 | 399.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 403.40 | 399.79 | 399.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:45:00 | 404.85 | 399.79 | 399.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 404.00 | 400.63 | 400.26 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 397.95 | 399.89 | 400.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 395.45 | 398.86 | 399.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 399.00 | 398.88 | 399.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 399.00 | 398.88 | 399.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 399.00 | 398.88 | 399.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 399.70 | 398.88 | 399.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 399.30 | 398.97 | 399.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 399.85 | 398.97 | 399.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 399.25 | 399.02 | 399.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:30:00 | 400.15 | 399.02 | 399.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 398.95 | 399.01 | 399.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:15:00 | 398.25 | 399.01 | 399.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 405.60 | 397.52 | 397.56 | SL hit (close>static) qty=1.00 sl=400.45 alert=retest2 |

### Cycle 126 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 403.45 | 398.70 | 398.09 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 395.05 | 397.93 | 398.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 389.85 | 395.84 | 397.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 394.90 | 394.15 | 395.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 394.90 | 394.15 | 395.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 394.85 | 394.29 | 395.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:45:00 | 395.90 | 394.29 | 395.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 395.00 | 393.58 | 395.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 395.00 | 393.58 | 395.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 394.80 | 393.83 | 395.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 393.15 | 393.83 | 395.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 390.95 | 393.25 | 394.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 388.40 | 393.25 | 394.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:45:00 | 388.45 | 383.94 | 384.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 389.75 | 385.10 | 385.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 389.75 | 385.10 | 385.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 11:15:00 | 391.25 | 387.95 | 386.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 387.00 | 388.54 | 387.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 15:15:00 | 387.00 | 388.54 | 387.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 387.00 | 388.54 | 387.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 385.10 | 388.54 | 387.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 385.30 | 387.89 | 387.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 384.75 | 387.89 | 387.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 383.35 | 386.98 | 386.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 383.35 | 386.98 | 386.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 381.55 | 385.90 | 386.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 13:15:00 | 380.00 | 381.80 | 383.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 15:15:00 | 380.00 | 379.84 | 381.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:15:00 | 382.35 | 379.84 | 381.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 383.20 | 380.51 | 381.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 385.70 | 380.51 | 381.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 382.40 | 380.89 | 381.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 380.70 | 380.89 | 381.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 381.95 | 381.17 | 381.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 361.66 | 369.90 | 373.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 362.85 | 369.90 | 373.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 363.00 | 362.82 | 366.52 | SL hit (close>ema200) qty=0.50 sl=362.82 alert=retest2 |

### Cycle 130 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 360.40 | 336.84 | 334.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 374.40 | 359.78 | 348.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 361.85 | 363.42 | 354.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 12:45:00 | 361.15 | 363.42 | 354.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 364.85 | 370.29 | 364.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:00:00 | 364.85 | 370.29 | 364.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 366.65 | 369.56 | 364.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:30:00 | 365.70 | 369.56 | 364.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 375.05 | 370.17 | 365.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 379.00 | 370.17 | 365.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 365.10 | 369.25 | 368.06 | SL hit (close<static) qty=1.00 sl=365.50 alert=retest2 |

### Cycle 131 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 366.00 | 371.63 | 371.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 362.50 | 366.21 | 368.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 366.20 | 363.69 | 365.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 10:15:00 | 366.20 | 363.69 | 365.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 366.20 | 363.69 | 365.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:45:00 | 367.35 | 363.69 | 365.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 368.50 | 364.66 | 366.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 370.00 | 364.66 | 366.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 367.15 | 365.15 | 366.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 367.00 | 365.15 | 366.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 366.25 | 365.37 | 366.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 361.45 | 365.37 | 366.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 15:00:00 | 362.70 | 364.84 | 365.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 344.56 | 352.73 | 357.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 353.80 | 352.73 | 357.40 | SL hit (close>static) qty=0.50 sl=352.73 alert=retest2 |

### Cycle 132 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 362.10 | 357.25 | 357.15 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 353.90 | 357.29 | 357.71 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 360.80 | 357.99 | 357.99 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 13:15:00 | 357.20 | 358.59 | 358.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 14:15:00 | 355.75 | 358.02 | 358.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 358.35 | 357.59 | 358.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 358.35 | 357.59 | 358.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 358.35 | 357.59 | 358.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 12:30:00 | 357.20 | 357.62 | 357.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 360.20 | 358.54 | 358.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 360.20 | 358.54 | 358.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 364.90 | 359.88 | 359.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 363.00 | 363.13 | 361.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 14:00:00 | 363.00 | 363.13 | 361.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 361.40 | 362.80 | 362.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 372.45 | 362.80 | 362.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 09:15:00 | 357.50 | 368.21 | 366.53 | SL hit (close<static) qty=1.00 sl=361.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 353.30 | 363.54 | 364.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 10:15:00 | 347.20 | 355.04 | 359.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 11:15:00 | 351.60 | 348.18 | 352.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 12:00:00 | 351.60 | 348.18 | 352.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 355.15 | 349.81 | 351.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 353.60 | 349.81 | 351.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 355.30 | 350.91 | 351.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 355.85 | 350.91 | 351.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 360.25 | 353.69 | 353.06 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 354.15 | 355.68 | 355.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 344.50 | 353.20 | 354.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 13:15:00 | 349.55 | 348.61 | 351.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 14:00:00 | 349.55 | 348.61 | 351.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 360.70 | 351.03 | 352.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 360.70 | 351.03 | 352.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 360.10 | 352.84 | 353.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 364.35 | 352.84 | 353.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 362.05 | 354.68 | 353.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 369.95 | 359.72 | 357.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 362.40 | 363.28 | 360.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 362.40 | 363.28 | 360.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 362.40 | 363.28 | 360.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 363.00 | 363.28 | 360.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 360.65 | 362.69 | 360.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 360.65 | 362.69 | 360.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 360.45 | 362.24 | 360.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:15:00 | 359.55 | 362.24 | 360.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 359.60 | 361.71 | 360.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 359.60 | 361.71 | 360.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 349.35 | 358.18 | 359.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 347.15 | 355.97 | 358.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 326.30 | 320.73 | 323.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 326.30 | 320.73 | 323.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 326.30 | 320.73 | 323.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 326.30 | 320.73 | 323.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 322.85 | 321.16 | 323.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 321.80 | 322.07 | 322.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 319.90 | 321.38 | 322.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 321.15 | 320.74 | 321.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 320.80 | 320.57 | 320.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 319.10 | 320.28 | 320.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 318.40 | 320.28 | 320.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 13:00:00 | 318.20 | 315.69 | 316.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 319.70 | 316.49 | 316.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 319.70 | 316.49 | 316.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 321.60 | 317.51 | 316.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 321.50 | 321.50 | 319.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 321.50 | 321.50 | 319.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 319.75 | 321.15 | 319.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:15:00 | 318.95 | 321.15 | 319.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 318.95 | 320.71 | 319.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 326.80 | 320.71 | 319.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 13:15:00 | 315.65 | 319.37 | 319.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 315.65 | 319.37 | 319.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 313.55 | 318.21 | 318.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 317.40 | 316.22 | 317.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 12:15:00 | 317.40 | 316.22 | 317.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 317.40 | 316.22 | 317.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 317.40 | 316.22 | 317.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 318.65 | 316.71 | 317.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 318.65 | 316.71 | 317.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 316.85 | 316.74 | 317.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:45:00 | 318.65 | 316.74 | 317.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 317.25 | 316.84 | 317.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 321.65 | 316.84 | 317.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 319.60 | 317.39 | 317.65 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 319.60 | 317.83 | 317.83 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 317.35 | 317.74 | 317.79 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 12:15:00 | 319.00 | 317.99 | 317.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 320.90 | 318.57 | 318.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 319.30 | 319.64 | 318.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 319.30 | 319.64 | 318.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 319.30 | 319.64 | 318.85 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 315.55 | 317.89 | 318.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 13:15:00 | 314.85 | 317.29 | 317.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 310.25 | 306.26 | 308.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 310.25 | 306.26 | 308.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 310.25 | 306.26 | 308.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 310.25 | 306.26 | 308.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 308.15 | 306.63 | 308.62 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 312.35 | 309.70 | 309.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 314.15 | 311.17 | 310.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 311.90 | 312.18 | 310.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 12:00:00 | 311.90 | 312.18 | 310.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 312.65 | 312.41 | 311.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:30:00 | 311.65 | 312.41 | 311.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 311.00 | 312.13 | 311.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 311.00 | 312.13 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 311.25 | 311.95 | 311.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 311.45 | 311.95 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 312.15 | 311.99 | 311.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 313.35 | 311.83 | 311.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:00:00 | 313.40 | 312.15 | 311.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:15:00 | 314.35 | 313.08 | 312.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:00:00 | 314.40 | 313.34 | 312.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 320.25 | 322.88 | 319.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 319.60 | 322.88 | 319.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 321.00 | 322.50 | 319.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:45:00 | 321.05 | 322.50 | 319.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 321.80 | 322.13 | 320.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:00:00 | 323.00 | 322.30 | 320.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 316.30 | 324.40 | 325.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 316.30 | 324.40 | 325.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 308.35 | 320.26 | 323.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 316.00 | 311.72 | 315.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 316.00 | 311.72 | 315.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 316.00 | 311.72 | 315.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 316.00 | 311.72 | 315.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 313.70 | 312.12 | 315.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:45:00 | 315.95 | 312.12 | 315.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 323.85 | 313.51 | 314.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 323.85 | 313.51 | 314.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 325.40 | 315.89 | 315.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 326.75 | 315.89 | 315.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 322.95 | 317.30 | 316.55 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 315.00 | 316.14 | 316.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 10:15:00 | 309.00 | 314.56 | 315.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 13:15:00 | 310.20 | 310.14 | 311.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 13:15:00 | 310.20 | 310.14 | 311.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 310.20 | 310.14 | 311.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:30:00 | 312.35 | 310.14 | 311.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 309.20 | 310.13 | 311.38 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 312.30 | 310.45 | 310.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 316.95 | 312.67 | 311.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 315.60 | 318.53 | 317.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 13:15:00 | 315.60 | 318.53 | 317.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 315.60 | 318.53 | 317.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 315.60 | 318.53 | 317.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 317.25 | 318.28 | 317.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:45:00 | 316.05 | 318.28 | 317.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 315.00 | 317.62 | 316.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 313.15 | 317.62 | 316.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 310.80 | 316.26 | 316.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 308.20 | 314.65 | 315.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 300.35 | 300.03 | 304.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 300.35 | 300.03 | 304.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 303.60 | 297.21 | 300.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 303.60 | 297.21 | 300.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 303.80 | 298.52 | 301.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 303.70 | 298.52 | 301.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 298.75 | 300.52 | 301.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 293.95 | 300.16 | 301.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 12:15:00 | 290.20 | 288.45 | 288.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 290.20 | 288.45 | 288.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 09:15:00 | 296.35 | 290.89 | 289.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 291.75 | 292.47 | 290.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 13:15:00 | 291.75 | 292.47 | 290.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 291.75 | 292.47 | 290.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 291.75 | 292.47 | 290.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 291.40 | 292.26 | 290.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 291.00 | 292.26 | 290.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 290.45 | 291.89 | 290.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 283.75 | 291.89 | 290.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 283.30 | 290.18 | 290.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 277.85 | 283.05 | 285.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 287.20 | 280.90 | 283.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 287.20 | 280.90 | 283.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 287.20 | 280.90 | 283.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 287.20 | 280.90 | 283.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 285.50 | 281.82 | 283.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:30:00 | 290.25 | 281.82 | 283.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 283.90 | 282.91 | 283.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 283.90 | 282.91 | 283.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 282.00 | 282.72 | 283.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 276.55 | 282.72 | 283.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 273.90 | 278.43 | 280.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 278.90 | 276.78 | 276.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 278.90 | 276.78 | 276.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 281.20 | 277.66 | 277.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 285.00 | 286.23 | 283.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:15:00 | 287.90 | 286.23 | 283.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:00:00 | 287.70 | 286.53 | 283.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 12:30:00 | 286.95 | 286.50 | 284.43 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 13:15:00 | 287.25 | 286.50 | 284.43 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 288.10 | 290.32 | 288.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 288.10 | 290.32 | 288.39 | SL hit (close<ema400) qty=1.00 sl=288.39 alert=retest1 |

### Cycle 157 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 282.20 | 287.11 | 287.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 280.10 | 284.14 | 285.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 283.90 | 283.29 | 284.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 283.90 | 283.29 | 284.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 283.90 | 283.29 | 284.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:45:00 | 284.75 | 283.29 | 284.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 282.60 | 283.15 | 284.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 280.90 | 282.97 | 284.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 281.75 | 282.32 | 283.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 280.75 | 282.04 | 283.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:45:00 | 281.15 | 281.98 | 283.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 283.20 | 281.93 | 282.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 285.25 | 282.59 | 283.18 | SL hit (close>static) qty=1.00 sl=284.85 alert=retest2 |

### Cycle 158 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 289.50 | 283.97 | 283.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 295.90 | 288.69 | 286.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 318.50 | 318.90 | 312.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:45:00 | 318.65 | 318.90 | 312.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 340.70 | 348.22 | 344.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 340.70 | 348.22 | 344.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 341.00 | 346.77 | 344.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 345.00 | 346.77 | 344.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 340.05 | 344.91 | 343.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 339.00 | 344.91 | 343.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 336.45 | 341.77 | 342.43 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 347.30 | 342.90 | 342.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 350.70 | 344.88 | 343.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 343.10 | 345.43 | 344.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 343.10 | 345.43 | 344.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 343.10 | 345.43 | 344.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 343.10 | 345.43 | 344.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 347.65 | 345.88 | 344.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 341.50 | 345.88 | 344.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 346.30 | 347.43 | 345.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 350.20 | 348.18 | 346.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:30:00 | 350.60 | 347.57 | 346.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 354.35 | 347.92 | 346.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 353.35 | 356.76 | 353.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 354.35 | 356.28 | 353.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 354.35 | 356.28 | 353.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 351.60 | 354.96 | 353.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 352.10 | 354.96 | 353.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 353.30 | 354.63 | 353.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 350.05 | 354.63 | 353.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 354.00 | 354.50 | 353.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 342.50 | 354.50 | 353.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 338.90 | 351.38 | 352.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 338.90 | 351.38 | 352.17 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 353.95 | 350.43 | 350.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 359.90 | 353.45 | 352.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 384.10 | 384.41 | 380.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 386.00 | 384.41 | 380.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 11:15:00 | 405.30 | 397.66 | 390.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 403.25 | 403.37 | 396.59 | SL hit (close<ema200) qty=0.50 sl=403.37 alert=retest1 |

### Cycle 163 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 387.95 | 397.38 | 398.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 383.45 | 394.59 | 397.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 12:15:00 | 383.20 | 381.04 | 386.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 12:45:00 | 382.20 | 381.04 | 386.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 384.90 | 382.87 | 386.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 383.50 | 382.87 | 386.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 387.15 | 383.72 | 386.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 388.85 | 383.72 | 386.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 387.60 | 384.50 | 386.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:45:00 | 390.20 | 384.50 | 386.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 385.00 | 384.60 | 386.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 382.75 | 385.04 | 386.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 15:00:00 | 383.20 | 384.67 | 386.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:15:00 | 383.40 | 384.91 | 385.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 384.90 | 382.01 | 381.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 384.90 | 382.01 | 381.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 388.00 | 384.31 | 383.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 378.60 | 383.70 | 383.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 378.60 | 383.70 | 383.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 378.60 | 383.70 | 383.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 378.60 | 383.70 | 383.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 377.80 | 382.52 | 382.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 374.85 | 380.98 | 381.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 375.85 | 375.12 | 378.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 375.85 | 375.12 | 378.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 375.85 | 375.12 | 378.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 376.85 | 375.12 | 378.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 383.80 | 376.85 | 378.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 383.80 | 376.85 | 378.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 386.05 | 378.69 | 379.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 386.05 | 378.69 | 379.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 382.60 | 380.03 | 379.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 384.90 | 381.50 | 380.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 381.75 | 383.17 | 381.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 12:15:00 | 381.75 | 383.17 | 381.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 381.75 | 383.17 | 381.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 381.75 | 383.17 | 381.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 375.55 | 381.65 | 381.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 375.55 | 381.65 | 381.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 373.30 | 379.98 | 380.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 370.30 | 378.04 | 379.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 372.25 | 372.08 | 375.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 378.80 | 372.08 | 375.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 385.05 | 374.68 | 375.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 385.05 | 374.68 | 375.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 388.90 | 377.52 | 377.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 392.50 | 385.85 | 383.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 390.60 | 390.71 | 388.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 14:00:00 | 390.60 | 390.71 | 388.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 390.25 | 390.42 | 388.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 390.80 | 390.42 | 388.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 390.20 | 394.04 | 392.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 389.50 | 394.04 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 394.50 | 394.13 | 392.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 394.80 | 394.13 | 392.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:00:00 | 394.70 | 394.33 | 393.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:30:00 | 394.55 | 393.70 | 393.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:00:00 | 394.80 | 393.70 | 393.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 393.80 | 393.72 | 393.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:30:00 | 394.35 | 393.72 | 393.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 396.20 | 394.21 | 393.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 390.40 | 393.47 | 393.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 390.40 | 393.47 | 393.67 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 395.30 | 394.03 | 393.88 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 10:15:00 | 393.40 | 393.71 | 393.75 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 396.65 | 394.30 | 394.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 397.10 | 394.86 | 394.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 396.85 | 397.81 | 396.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 396.85 | 397.81 | 396.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 396.85 | 397.81 | 396.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 397.75 | 397.81 | 396.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 395.10 | 397.27 | 396.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 395.30 | 397.27 | 396.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 394.15 | 396.64 | 396.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 394.15 | 396.64 | 396.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 397.80 | 396.87 | 396.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 399.60 | 396.87 | 396.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 399.60 | 397.42 | 396.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:00:00 | 398.45 | 398.81 | 397.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:45:00 | 398.15 | 398.20 | 397.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 399.60 | 398.48 | 397.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:30:00 | 401.00 | 398.98 | 398.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 400.15 | 402.21 | 401.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:30:00 | 400.30 | 401.17 | 400.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 403.25 | 404.53 | 404.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 403.25 | 404.53 | 404.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 398.80 | 403.39 | 404.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 403.35 | 403.19 | 403.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 403.35 | 403.19 | 403.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 403.35 | 403.19 | 403.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 403.35 | 403.19 | 403.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 400.00 | 402.55 | 403.47 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 404.80 | 403.12 | 403.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 406.40 | 403.77 | 403.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 427.75 | 429.91 | 425.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 427.75 | 429.91 | 425.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 427.75 | 429.91 | 425.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 427.75 | 429.91 | 425.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 428.45 | 429.08 | 425.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 426.00 | 429.08 | 425.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 423.25 | 427.70 | 426.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 423.40 | 427.70 | 426.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 426.35 | 427.43 | 426.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 426.95 | 427.43 | 426.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 428.00 | 427.06 | 426.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 428.25 | 427.30 | 426.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 421.25 | 426.16 | 426.03 | SL hit (close<static) qty=1.00 sl=422.95 alert=retest2 |

### Cycle 175 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 422.65 | 425.46 | 425.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 416.65 | 423.07 | 424.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 417.70 | 417.52 | 420.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 417.70 | 417.52 | 420.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 419.75 | 418.16 | 419.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 419.10 | 418.16 | 419.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 421.75 | 418.88 | 419.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 420.25 | 418.88 | 419.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 421.50 | 419.40 | 420.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 421.50 | 419.40 | 420.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 419.90 | 419.50 | 420.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 419.00 | 420.15 | 420.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 422.50 | 420.44 | 420.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 422.50 | 420.44 | 420.31 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 418.85 | 420.06 | 420.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 418.55 | 419.70 | 420.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 421.30 | 420.02 | 420.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 421.30 | 420.02 | 420.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 421.30 | 420.02 | 420.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 421.90 | 420.02 | 420.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 419.50 | 419.92 | 420.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 417.40 | 419.92 | 420.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:45:00 | 416.90 | 416.40 | 417.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 417.60 | 416.13 | 416.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 424.15 | 417.74 | 417.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 424.15 | 417.74 | 417.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 430.00 | 420.19 | 418.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 443.50 | 443.50 | 436.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:00:00 | 443.50 | 443.50 | 436.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 435.70 | 441.50 | 436.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 435.70 | 441.50 | 436.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 434.00 | 440.00 | 436.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 434.75 | 440.00 | 436.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 442.80 | 439.51 | 437.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 444.70 | 440.54 | 438.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 467.00 | 468.29 | 468.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 467.00 | 468.29 | 468.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 466.85 | 467.89 | 468.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 469.60 | 468.23 | 468.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 469.60 | 468.23 | 468.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 469.60 | 468.23 | 468.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 469.60 | 468.23 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 469.20 | 468.43 | 468.33 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 466.70 | 468.53 | 468.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 461.00 | 466.39 | 467.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 454.70 | 453.20 | 457.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 454.70 | 453.20 | 457.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 453.50 | 453.37 | 456.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 450.45 | 452.89 | 454.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 449.65 | 451.82 | 453.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 450.55 | 450.08 | 451.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 455.40 | 452.57 | 452.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 455.40 | 452.57 | 452.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 463.60 | 454.57 | 453.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 455.80 | 458.06 | 456.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 455.80 | 458.06 | 456.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 455.80 | 458.06 | 456.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 455.80 | 458.06 | 456.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 456.95 | 457.84 | 456.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 457.15 | 457.84 | 456.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 456.25 | 457.52 | 456.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 456.25 | 457.52 | 456.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 455.40 | 457.09 | 456.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 455.90 | 457.09 | 456.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 452.95 | 456.27 | 455.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 452.95 | 456.27 | 455.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 453.45 | 455.70 | 455.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 449.00 | 454.36 | 455.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 439.20 | 422.38 | 429.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 439.20 | 422.38 | 429.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 439.20 | 422.38 | 429.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 439.20 | 422.38 | 429.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 444.50 | 426.81 | 430.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 444.50 | 426.81 | 430.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 419.65 | 416.27 | 420.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 423.95 | 416.27 | 420.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 420.10 | 417.04 | 420.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 420.55 | 417.04 | 420.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 420.00 | 417.63 | 420.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 420.90 | 417.63 | 420.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 420.00 | 417.20 | 419.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 420.80 | 417.20 | 419.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 422.20 | 418.20 | 419.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 422.20 | 418.20 | 419.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 425.55 | 420.82 | 420.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 426.70 | 422.00 | 421.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 421.00 | 421.80 | 421.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 421.00 | 421.80 | 421.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 421.00 | 421.80 | 421.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 421.00 | 421.80 | 421.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 426.45 | 422.73 | 421.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 419.20 | 422.73 | 421.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 418.30 | 421.84 | 421.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 416.60 | 421.84 | 421.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 419.10 | 421.29 | 421.04 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 415.75 | 420.19 | 420.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 414.95 | 419.14 | 420.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 15:15:00 | 418.50 | 417.93 | 419.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 15:15:00 | 418.50 | 417.93 | 419.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 418.50 | 417.93 | 419.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:30:00 | 416.00 | 418.42 | 419.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 424.25 | 419.58 | 419.71 | SL hit (close>static) qty=1.00 sl=420.50 alert=retest2 |

### Cycle 186 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 426.00 | 420.87 | 420.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 428.65 | 422.42 | 421.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 12:15:00 | 440.80 | 441.33 | 437.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:00:00 | 440.80 | 441.33 | 437.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 441.50 | 441.37 | 437.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 438.10 | 441.37 | 437.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 439.65 | 441.99 | 439.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 439.65 | 441.99 | 439.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 437.40 | 441.07 | 439.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 437.40 | 441.07 | 439.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 437.20 | 440.30 | 439.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:45:00 | 437.70 | 440.30 | 439.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 434.00 | 438.06 | 438.19 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 440.30 | 438.29 | 438.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 442.75 | 439.60 | 438.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 453.85 | 453.89 | 450.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:00:00 | 453.85 | 453.89 | 450.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 466.70 | 467.64 | 464.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 465.65 | 467.64 | 464.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 465.65 | 467.07 | 464.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 464.75 | 467.07 | 464.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 464.45 | 466.54 | 464.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 464.40 | 466.54 | 464.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 463.80 | 466.00 | 464.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 465.00 | 464.68 | 464.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 459.10 | 463.56 | 463.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 459.10 | 463.56 | 463.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 455.00 | 461.85 | 462.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 457.40 | 455.97 | 459.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 457.40 | 455.97 | 459.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 457.40 | 455.97 | 459.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 458.75 | 455.97 | 459.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 459.75 | 456.73 | 459.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 459.75 | 456.73 | 459.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 463.15 | 458.01 | 459.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 463.15 | 458.01 | 459.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 465.20 | 459.45 | 460.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 465.20 | 459.45 | 460.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 465.40 | 461.41 | 460.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 468.05 | 463.19 | 461.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 464.90 | 466.32 | 464.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 464.90 | 466.32 | 464.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 464.90 | 466.32 | 464.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 465.30 | 466.32 | 464.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 463.50 | 465.75 | 464.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 464.20 | 465.75 | 464.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 463.40 | 465.28 | 464.28 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 458.55 | 463.44 | 463.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 453.50 | 461.45 | 462.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 431.00 | 430.79 | 436.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:15:00 | 430.80 | 430.79 | 436.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 434.00 | 429.02 | 431.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 434.00 | 429.02 | 431.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 436.05 | 430.43 | 432.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 435.35 | 430.43 | 432.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 439.75 | 433.34 | 433.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 441.90 | 435.05 | 434.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 437.85 | 438.19 | 436.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 437.85 | 438.19 | 436.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 439.40 | 438.30 | 436.58 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 434.35 | 435.70 | 435.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 11:15:00 | 430.60 | 432.36 | 433.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 432.70 | 430.67 | 432.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 432.70 | 430.67 | 432.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 432.70 | 430.67 | 432.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 432.70 | 430.67 | 432.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 433.10 | 431.16 | 432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 434.40 | 431.16 | 432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 432.15 | 431.36 | 432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 432.70 | 431.36 | 432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 431.45 | 431.37 | 432.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 429.30 | 431.37 | 432.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 430.00 | 431.54 | 432.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 442.85 | 433.56 | 432.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 442.85 | 433.56 | 432.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 446.10 | 436.06 | 434.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 440.80 | 442.95 | 440.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 440.80 | 442.95 | 440.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 440.80 | 442.95 | 440.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 440.80 | 442.95 | 440.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 439.75 | 442.31 | 439.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 439.75 | 442.31 | 439.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 438.40 | 441.53 | 439.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 438.40 | 441.53 | 439.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 439.10 | 441.04 | 439.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 441.00 | 441.04 | 439.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 439.20 | 440.50 | 439.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 439.40 | 440.32 | 439.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 437.35 | 439.53 | 439.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 437.35 | 439.53 | 439.63 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 444.85 | 440.05 | 439.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 451.20 | 443.68 | 441.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 489.10 | 496.72 | 486.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 10:00:00 | 489.10 | 496.72 | 486.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 493.60 | 496.10 | 486.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 488.70 | 496.10 | 486.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 489.25 | 493.58 | 487.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 488.80 | 493.58 | 487.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 487.35 | 492.34 | 487.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:30:00 | 488.55 | 492.34 | 487.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 485.35 | 490.94 | 487.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 485.35 | 490.94 | 487.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 485.90 | 489.93 | 487.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 483.45 | 489.93 | 487.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 485.10 | 488.12 | 486.64 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 483.55 | 485.98 | 485.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 482.20 | 485.23 | 485.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 487.85 | 485.75 | 485.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 487.85 | 485.75 | 485.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 487.85 | 485.75 | 485.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 484.10 | 485.75 | 485.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 487.65 | 486.13 | 486.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 491.05 | 488.85 | 487.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 487.20 | 489.36 | 488.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 487.20 | 489.36 | 488.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 487.20 | 489.36 | 488.27 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 484.40 | 487.59 | 487.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 482.15 | 486.50 | 487.18 | Break + close below crossover candle low |

### Cycle 200 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 495.40 | 487.62 | 487.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 498.65 | 492.25 | 489.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 11:15:00 | 489.15 | 495.49 | 493.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 11:15:00 | 489.15 | 495.49 | 493.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 489.15 | 495.49 | 493.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 489.15 | 495.49 | 493.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 491.80 | 494.75 | 493.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 488.65 | 494.75 | 493.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 490.20 | 493.84 | 492.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 488.05 | 493.84 | 492.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 496.50 | 494.38 | 493.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 502.00 | 495.93 | 494.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-06 09:15:00 | 552.20 | 524.54 | 511.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 525.75 | 527.84 | 528.00 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 530.65 | 528.17 | 528.08 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 523.85 | 527.31 | 527.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 520.75 | 526.00 | 527.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 525.50 | 521.37 | 523.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 525.50 | 521.37 | 523.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 525.50 | 521.37 | 523.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 525.50 | 521.37 | 523.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 529.10 | 522.92 | 524.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 529.10 | 522.92 | 524.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 530.50 | 524.43 | 524.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 531.75 | 524.43 | 524.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 538.40 | 527.23 | 526.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 540.75 | 529.93 | 527.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 532.70 | 536.89 | 534.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 532.70 | 536.89 | 534.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 532.70 | 536.89 | 534.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 532.70 | 536.89 | 534.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 533.75 | 536.26 | 534.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:00:00 | 535.15 | 536.04 | 534.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 529.45 | 534.72 | 533.81 | SL hit (close<static) qty=1.00 sl=532.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 526.80 | 532.30 | 532.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 525.90 | 531.02 | 532.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 515.55 | 514.82 | 521.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 517.40 | 514.82 | 521.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 502.55 | 491.25 | 497.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 499.00 | 491.25 | 497.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 502.85 | 493.57 | 498.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 502.25 | 493.57 | 498.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 497.30 | 496.96 | 498.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 498.35 | 496.96 | 498.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 498.65 | 495.69 | 497.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 496.60 | 495.69 | 497.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 499.45 | 496.44 | 497.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 499.45 | 496.44 | 497.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 500.35 | 497.23 | 498.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:30:00 | 501.80 | 497.23 | 498.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 487.80 | 488.25 | 491.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 488.15 | 488.25 | 491.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 487.00 | 485.02 | 487.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 482.15 | 484.44 | 487.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 481.65 | 482.70 | 485.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 482.70 | 482.67 | 484.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:45:00 | 482.70 | 480.55 | 481.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 458.04 | 464.73 | 470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 458.56 | 464.73 | 470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 458.56 | 464.73 | 470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 464.55 | 464.30 | 468.73 | SL hit (close>ema200) qty=0.50 sl=464.30 alert=retest2 |

### Cycle 206 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 472.95 | 464.87 | 463.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 479.45 | 467.79 | 465.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 482.00 | 482.74 | 477.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 482.00 | 482.74 | 477.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 469.50 | 479.09 | 476.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 469.50 | 479.09 | 476.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 467.00 | 473.87 | 474.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 465.10 | 468.62 | 470.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 468.50 | 468.35 | 469.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 14:45:00 | 465.50 | 467.80 | 469.37 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 13:15:00 | 466.50 | 466.12 | 467.77 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 472.00 | 467.30 | 468.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 472.00 | 467.30 | 468.15 | SL hit (close>ema400) qty=1.00 sl=468.15 alert=retest1 |

### Cycle 208 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 471.30 | 468.28 | 468.01 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 463.25 | 467.13 | 467.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 459.15 | 465.53 | 466.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 463.00 | 461.57 | 464.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 463.00 | 461.57 | 464.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 460.20 | 461.29 | 463.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 458.50 | 460.85 | 463.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 459.30 | 460.55 | 462.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 459.90 | 460.44 | 462.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 459.50 | 460.35 | 462.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 468.90 | 461.92 | 462.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 468.90 | 461.92 | 462.76 | SL hit (close>static) qty=1.00 sl=464.00 alert=retest2 |

### Cycle 210 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 472.40 | 465.13 | 464.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 487.00 | 470.87 | 468.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 475.35 | 475.70 | 471.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 475.35 | 475.70 | 471.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 480.50 | 482.98 | 479.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 477.75 | 482.98 | 479.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 478.75 | 482.13 | 479.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 478.75 | 482.13 | 479.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 476.25 | 480.95 | 479.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 476.25 | 480.95 | 479.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 474.00 | 479.56 | 478.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 473.75 | 479.56 | 478.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 475.25 | 477.90 | 478.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 467.10 | 475.34 | 476.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 465.95 | 465.50 | 469.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:30:00 | 467.15 | 465.50 | 469.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 465.40 | 465.36 | 467.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 465.40 | 465.36 | 467.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 467.55 | 465.79 | 467.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 467.55 | 465.79 | 467.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 467.95 | 466.23 | 467.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 469.70 | 466.23 | 467.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 466.50 | 466.28 | 467.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 468.90 | 466.28 | 467.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 470.15 | 467.05 | 467.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 470.15 | 467.05 | 467.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 470.85 | 467.81 | 468.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 470.85 | 467.81 | 468.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 472.90 | 468.83 | 468.66 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 461.90 | 467.44 | 468.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 460.20 | 466.00 | 467.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 452.35 | 452.31 | 458.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 452.35 | 452.31 | 458.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 456.70 | 453.78 | 457.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 451.50 | 454.40 | 456.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:30:00 | 452.00 | 451.06 | 452.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:00:00 | 451.15 | 451.47 | 452.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 443.95 | 442.61 | 442.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 443.95 | 442.61 | 442.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 448.05 | 444.10 | 443.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 471.60 | 472.30 | 466.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 471.60 | 472.30 | 466.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 472.45 | 471.05 | 467.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 475.80 | 471.83 | 468.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:45:00 | 474.20 | 472.60 | 469.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:45:00 | 475.00 | 473.72 | 470.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:45:00 | 474.30 | 473.64 | 471.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 473.40 | 473.59 | 471.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 475.60 | 473.74 | 471.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 475.70 | 474.17 | 472.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 474.90 | 474.17 | 472.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 477.45 | 474.21 | 472.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 479.35 | 483.22 | 481.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 479.35 | 483.22 | 481.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 482.50 | 483.08 | 481.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 479.00 | 481.05 | 481.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 479.00 | 481.05 | 481.30 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 483.30 | 481.50 | 481.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 485.55 | 482.68 | 482.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 483.55 | 484.19 | 483.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:00:00 | 483.55 | 484.19 | 483.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 480.35 | 483.42 | 482.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 480.75 | 483.42 | 482.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 478.45 | 482.43 | 482.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 475.40 | 480.10 | 481.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 453.25 | 452.93 | 457.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:15:00 | 456.40 | 452.93 | 457.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 460.15 | 455.48 | 457.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 460.15 | 455.48 | 457.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 460.05 | 456.39 | 457.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 462.60 | 457.69 | 458.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 463.25 | 458.80 | 458.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 465.20 | 460.91 | 459.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 465.80 | 466.56 | 463.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:45:00 | 464.30 | 466.56 | 463.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 463.95 | 466.04 | 463.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 463.55 | 466.04 | 463.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 465.75 | 465.98 | 463.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 483.50 | 465.28 | 463.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:30:00 | 467.15 | 467.95 | 465.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 461.45 | 468.52 | 467.04 | SL hit (close<static) qty=1.00 sl=461.55 alert=retest2 |

### Cycle 219 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 457.35 | 464.57 | 465.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 454.50 | 462.56 | 464.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 443.15 | 441.78 | 448.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 14:15:00 | 448.15 | 444.08 | 447.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 448.15 | 444.08 | 447.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 448.15 | 444.08 | 447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 446.00 | 444.47 | 447.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 439.35 | 444.47 | 447.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 417.38 | 427.26 | 436.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 13:15:00 | 395.42 | 409.13 | 422.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 220 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 406.75 | 395.00 | 394.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 411.00 | 398.20 | 395.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 407.80 | 412.41 | 407.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 407.80 | 412.41 | 407.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 407.80 | 412.41 | 407.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 407.80 | 412.41 | 407.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 406.70 | 411.27 | 407.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 405.25 | 411.27 | 407.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 398.20 | 408.65 | 406.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 398.20 | 408.65 | 406.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 397.70 | 406.46 | 405.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 401.55 | 405.52 | 405.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-06 15:15:00 | 441.71 | 418.79 | 412.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 471.80 | 472.60 | 472.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 466.60 | 471.15 | 471.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 477.05 | 470.65 | 471.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 10:15:00 | 477.05 | 470.65 | 471.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 477.05 | 470.65 | 471.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 477.75 | 470.65 | 471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 473.45 | 471.21 | 471.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:30:00 | 468.00 | 470.29 | 471.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 474.40 | 470.06 | 469.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 474.40 | 470.06 | 469.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 477.00 | 471.93 | 470.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 482.20 | 483.73 | 478.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:45:00 | 483.65 | 483.73 | 478.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 476.35 | 481.28 | 478.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 476.35 | 481.28 | 478.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 473.80 | 479.79 | 478.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 473.80 | 479.79 | 478.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 473.45 | 478.52 | 477.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 476.95 | 478.52 | 477.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 473.35 | 476.99 | 477.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 473.35 | 476.99 | 477.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 472.45 | 476.08 | 476.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 476.60 | 473.81 | 475.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 476.60 | 473.81 | 475.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 476.60 | 473.81 | 475.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 478.10 | 473.81 | 475.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 469.10 | 472.87 | 474.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 468.25 | 472.10 | 474.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 467.90 | 471.26 | 473.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 468.20 | 471.80 | 473.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 468.35 | 471.11 | 472.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 462.85 | 468.27 | 470.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 462.25 | 468.27 | 470.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 462.10 | 464.05 | 465.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.84 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.50 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.79 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.93 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 439.14 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 439.00 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 15:15:00 | 421.43 | 430.63 | 439.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 13:15:00 | 419.00 | 412.43 | 412.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 14:15:00 | 422.70 | 414.48 | 413.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 09:15:00 | 415.90 | 416.61 | 414.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 415.90 | 416.61 | 414.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 415.90 | 416.61 | 414.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 415.90 | 416.61 | 414.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 410.40 | 415.37 | 414.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 410.40 | 415.37 | 414.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 407.80 | 413.85 | 413.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 407.80 | 413.85 | 413.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 411.10 | 413.30 | 413.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 403.15 | 409.65 | 410.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 406.75 | 403.92 | 406.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 406.75 | 403.92 | 406.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 406.75 | 403.92 | 406.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 406.15 | 403.92 | 406.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 399.70 | 403.08 | 406.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 402.15 | 403.08 | 406.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 373.70 | 378.94 | 387.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 372.55 | 377.82 | 386.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 394.45 | 383.09 | 385.40 | SL hit (close>static) qty=1.00 sl=388.30 alert=retest2 |

### Cycle 226 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 391.50 | 387.56 | 387.13 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 378.80 | 387.30 | 387.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 378.40 | 385.52 | 386.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 387.10 | 383.91 | 385.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 387.10 | 383.91 | 385.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 387.10 | 383.91 | 385.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 387.10 | 383.91 | 385.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 386.50 | 384.43 | 385.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 380.15 | 384.43 | 385.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 383.00 | 380.47 | 381.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 384.80 | 381.60 | 381.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 384.80 | 381.60 | 381.49 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 370.60 | 379.40 | 380.50 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 393.90 | 382.56 | 381.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 406.00 | 391.99 | 387.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 10:15:00 | 410.35 | 414.24 | 407.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:00:00 | 410.35 | 414.24 | 407.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 401.80 | 411.75 | 407.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:00:00 | 401.80 | 411.75 | 407.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 402.70 | 409.94 | 406.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 13:30:00 | 405.00 | 409.03 | 406.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:45:00 | 410.75 | 406.87 | 406.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 401.35 | 405.52 | 405.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 11:15:00 | 401.35 | 405.52 | 405.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 14:15:00 | 398.95 | 402.50 | 404.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 13:15:00 | 405.10 | 399.15 | 401.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 13:15:00 | 405.10 | 399.15 | 401.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 405.10 | 399.15 | 401.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:00:00 | 405.10 | 399.15 | 401.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 408.35 | 400.99 | 401.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 408.35 | 400.99 | 401.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 410.00 | 402.79 | 402.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 415.05 | 406.40 | 404.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 407.80 | 411.08 | 408.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 407.80 | 411.08 | 408.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 407.80 | 411.08 | 408.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 406.60 | 411.08 | 408.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 403.00 | 409.46 | 407.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 403.00 | 409.46 | 407.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 401.05 | 407.78 | 407.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 400.50 | 407.78 | 407.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 402.15 | 406.65 | 406.69 | EMA200 below EMA400 |

### Cycle 234 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 413.90 | 407.66 | 406.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 419.55 | 412.25 | 409.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 11:15:00 | 421.80 | 422.13 | 418.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:45:00 | 422.00 | 422.13 | 418.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 424.55 | 423.58 | 420.91 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 416.00 | 420.54 | 420.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 413.45 | 418.64 | 419.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 418.00 | 417.64 | 418.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 421.30 | 417.64 | 418.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 420.00 | 418.11 | 419.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 419.55 | 418.11 | 419.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 420.20 | 418.53 | 419.11 | EMA400 retest candle locked (from downside) |

### Cycle 236 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 420.00 | 419.48 | 419.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 421.70 | 420.01 | 419.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 418.00 | 419.61 | 419.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 10:15:00 | 418.00 | 419.61 | 419.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 418.00 | 419.61 | 419.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 418.00 | 419.61 | 419.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 414.65 | 418.61 | 419.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 414.00 | 417.69 | 418.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 414.65 | 414.19 | 416.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 414.65 | 414.19 | 416.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 416.55 | 414.67 | 416.17 | EMA400 retest candle locked (from downside) |

### Cycle 238 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 421.80 | 417.55 | 417.31 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 413.70 | 416.78 | 416.98 | EMA200 below EMA400 |

### Cycle 240 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 418.90 | 417.25 | 417.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 429.00 | 419.60 | 418.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-19 12:30:00 | 494.85 | 2024-04-23 13:15:00 | 494.65 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-04-22 14:30:00 | 494.90 | 2024-04-23 13:15:00 | 494.65 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-04-23 09:15:00 | 495.25 | 2024-04-23 13:15:00 | 494.65 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-04-23 11:00:00 | 498.10 | 2024-04-23 13:15:00 | 494.65 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-04-25 10:30:00 | 491.75 | 2024-04-25 12:15:00 | 496.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-04-26 12:15:00 | 491.40 | 2024-04-30 11:15:00 | 494.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-04-29 13:45:00 | 490.25 | 2024-04-30 11:15:00 | 494.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-04-29 14:30:00 | 490.00 | 2024-04-30 11:15:00 | 494.40 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-05-10 12:45:00 | 461.25 | 2024-05-14 13:15:00 | 468.75 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-05-13 09:15:00 | 458.25 | 2024-05-14 13:15:00 | 468.75 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-05-14 09:45:00 | 461.60 | 2024-05-14 13:15:00 | 468.75 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-05-16 09:15:00 | 471.70 | 2024-05-17 14:15:00 | 463.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-05-16 14:30:00 | 468.45 | 2024-05-17 14:15:00 | 463.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-05-17 09:15:00 | 469.00 | 2024-05-17 14:15:00 | 463.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-05-17 11:00:00 | 468.30 | 2024-05-17 14:15:00 | 463.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-05-22 11:30:00 | 458.35 | 2024-06-03 09:15:00 | 452.20 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2024-05-23 09:45:00 | 459.35 | 2024-06-03 09:15:00 | 452.20 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-05-23 12:15:00 | 459.55 | 2024-06-03 09:15:00 | 452.20 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2024-05-23 13:00:00 | 459.50 | 2024-06-03 09:15:00 | 452.20 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2024-05-24 12:15:00 | 458.00 | 2024-06-03 09:15:00 | 452.20 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2024-05-27 09:30:00 | 457.65 | 2024-06-03 09:15:00 | 452.20 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2024-06-12 12:30:00 | 444.65 | 2024-06-20 09:15:00 | 422.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-12 13:15:00 | 444.50 | 2024-06-20 09:15:00 | 422.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-12 12:30:00 | 444.65 | 2024-06-21 14:15:00 | 421.80 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2024-06-12 13:15:00 | 444.50 | 2024-06-21 14:15:00 | 421.80 | STOP_HIT | 0.50 | 5.11% |
| SELL | retest2 | 2024-06-28 14:30:00 | 410.80 | 2024-07-01 13:15:00 | 419.70 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-06-28 15:00:00 | 408.15 | 2024-07-01 13:15:00 | 419.70 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-07-03 09:15:00 | 422.70 | 2024-07-03 13:15:00 | 416.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-07-04 11:30:00 | 421.45 | 2024-07-04 14:15:00 | 416.65 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-07-04 12:30:00 | 420.70 | 2024-07-04 14:15:00 | 416.65 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest1 | 2024-08-05 09:15:00 | 350.35 | 2024-08-07 15:15:00 | 354.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-08-28 09:15:00 | 405.00 | 2024-08-28 14:15:00 | 399.85 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-09-02 11:45:00 | 389.80 | 2024-09-05 12:15:00 | 391.10 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-09-02 12:15:00 | 389.95 | 2024-09-05 12:15:00 | 391.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-09-03 09:30:00 | 388.70 | 2024-09-05 12:15:00 | 391.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-09-20 09:15:00 | 394.90 | 2024-09-20 09:15:00 | 401.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-27 14:15:00 | 398.25 | 2024-10-01 10:15:00 | 405.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-10-07 10:15:00 | 388.40 | 2024-10-09 13:15:00 | 389.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-10-09 12:45:00 | 388.45 | 2024-10-09 13:15:00 | 389.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-10-16 11:15:00 | 380.70 | 2024-10-22 13:15:00 | 361.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 381.95 | 2024-10-22 13:15:00 | 362.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 11:15:00 | 380.70 | 2024-10-24 09:15:00 | 363.00 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2024-10-17 09:15:00 | 381.95 | 2024-10-24 09:15:00 | 363.00 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2024-11-04 15:15:00 | 379.00 | 2024-11-06 09:15:00 | 365.10 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2024-11-06 14:15:00 | 376.25 | 2024-11-08 10:15:00 | 365.40 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-11-07 09:30:00 | 377.00 | 2024-11-08 10:15:00 | 365.40 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-11-07 12:00:00 | 376.35 | 2024-11-08 10:15:00 | 365.40 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-11-12 14:15:00 | 361.45 | 2024-11-14 09:15:00 | 344.56 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2024-11-12 14:15:00 | 361.45 | 2024-11-14 09:15:00 | 353.80 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2024-11-12 15:00:00 | 362.70 | 2024-11-19 09:15:00 | 362.10 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-11-25 12:30:00 | 357.20 | 2024-11-25 14:15:00 | 360.20 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-11-28 09:15:00 | 372.45 | 2024-11-29 09:15:00 | 357.50 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-12-24 15:00:00 | 321.80 | 2025-01-02 13:15:00 | 319.70 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-12-26 09:30:00 | 319.90 | 2025-01-02 13:15:00 | 319.70 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-12-27 12:45:00 | 321.15 | 2025-01-02 13:15:00 | 319.70 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-12-30 12:15:00 | 320.80 | 2025-01-02 13:15:00 | 319.70 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-12-30 13:15:00 | 318.40 | 2025-01-02 13:15:00 | 319.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-01-02 13:00:00 | 318.20 | 2025-01-02 13:15:00 | 319.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-01-06 09:15:00 | 326.80 | 2025-01-06 13:15:00 | 315.65 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-01-17 11:15:00 | 313.35 | 2025-01-24 14:15:00 | 316.30 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-01-17 12:00:00 | 313.40 | 2025-01-24 14:15:00 | 316.30 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-01-20 10:15:00 | 314.35 | 2025-01-24 14:15:00 | 316.30 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-01-20 11:00:00 | 314.40 | 2025-01-24 14:15:00 | 316.30 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-01-22 14:00:00 | 323.00 | 2025-01-24 14:15:00 | 316.30 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-02-14 09:15:00 | 293.95 | 2025-02-20 12:15:00 | 290.20 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-02-28 09:15:00 | 276.55 | 2025-03-05 11:15:00 | 278.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-03-03 09:15:00 | 273.90 | 2025-03-05 11:15:00 | 278.90 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest1 | 2025-03-07 09:15:00 | 287.90 | 2025-03-10 15:15:00 | 288.10 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest1 | 2025-03-07 10:00:00 | 287.70 | 2025-03-10 15:15:00 | 288.10 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2025-03-07 12:30:00 | 286.95 | 2025-03-10 15:15:00 | 288.10 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest1 | 2025-03-07 13:15:00 | 287.25 | 2025-03-10 15:15:00 | 288.10 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-03-11 10:45:00 | 292.30 | 2025-03-11 12:15:00 | 282.80 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-03-13 09:15:00 | 280.90 | 2025-03-17 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-03-13 10:30:00 | 281.75 | 2025-03-17 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-03-13 14:15:00 | 280.75 | 2025-03-17 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-03-13 14:45:00 | 281.15 | 2025-03-17 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-04-02 12:00:00 | 350.20 | 2025-04-07 09:15:00 | 338.90 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-04-02 14:30:00 | 350.60 | 2025-04-07 09:15:00 | 338.90 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-04-03 09:15:00 | 354.35 | 2025-04-07 09:15:00 | 338.90 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2025-04-04 10:30:00 | 353.35 | 2025-04-07 09:15:00 | 338.90 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest1 | 2025-04-21 09:15:00 | 386.00 | 2025-04-22 11:15:00 | 405.30 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-21 09:15:00 | 386.00 | 2025-04-23 09:15:00 | 403.25 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-04-29 14:15:00 | 382.75 | 2025-05-05 11:15:00 | 384.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-04-29 15:00:00 | 383.20 | 2025-05-05 11:15:00 | 384.90 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-04-30 11:15:00 | 383.40 | 2025-05-05 11:15:00 | 384.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-20 11:15:00 | 394.80 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-05-20 13:00:00 | 394.70 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-05-21 10:30:00 | 394.55 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-21 11:00:00 | 394.80 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-05-27 11:15:00 | 399.60 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-05-27 12:00:00 | 399.60 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-05-28 11:00:00 | 398.45 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-05-28 12:45:00 | 398.15 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-05-28 14:30:00 | 401.00 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-05-30 09:45:00 | 400.15 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-05-30 11:30:00 | 400.30 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-06-11 15:15:00 | 426.95 | 2025-06-12 13:15:00 | 421.25 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-12 11:00:00 | 428.00 | 2025-06-12 13:15:00 | 421.25 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-12 12:00:00 | 428.25 | 2025-06-12 13:15:00 | 421.25 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-17 15:15:00 | 419.00 | 2025-06-18 09:15:00 | 422.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-19 11:15:00 | 417.40 | 2025-06-23 10:15:00 | 424.15 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-06-20 11:45:00 | 416.90 | 2025-06-23 10:15:00 | 424.15 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-23 09:30:00 | 417.60 | 2025-06-23 10:15:00 | 424.15 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-26 13:00:00 | 444.70 | 2025-07-07 15:15:00 | 467.00 | STOP_HIT | 1.00 | 5.01% |
| SELL | retest2 | 2025-07-16 11:30:00 | 450.45 | 2025-07-18 13:15:00 | 455.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-17 09:30:00 | 449.65 | 2025-07-18 13:15:00 | 455.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-18 10:15:00 | 450.55 | 2025-07-18 13:15:00 | 455.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-04 09:30:00 | 416.00 | 2025-08-04 10:15:00 | 424.25 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-21 09:30:00 | 465.00 | 2025-08-21 10:15:00 | 459.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-09 14:15:00 | 429.30 | 2025-09-10 09:15:00 | 442.85 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-09 15:15:00 | 430.00 | 2025-09-10 09:15:00 | 442.85 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-09-12 09:15:00 | 441.00 | 2025-09-15 09:15:00 | 437.35 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-12 13:15:00 | 439.20 | 2025-09-15 09:15:00 | 437.35 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-09-12 13:45:00 | 439.40 | 2025-09-15 09:15:00 | 437.35 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-10-01 13:30:00 | 502.00 | 2025-10-06 09:15:00 | 552.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-17 12:00:00 | 535.15 | 2025-10-17 12:15:00 | 529.45 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-31 11:00:00 | 482.15 | 2025-11-07 09:15:00 | 458.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 15:00:00 | 481.65 | 2025-11-07 09:15:00 | 458.56 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-11-03 11:00:00 | 482.70 | 2025-11-07 09:15:00 | 458.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 11:00:00 | 482.15 | 2025-11-07 12:15:00 | 464.55 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-10-31 15:00:00 | 481.65 | 2025-11-07 12:15:00 | 464.55 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2025-11-03 11:00:00 | 482.70 | 2025-11-07 12:15:00 | 464.55 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-11-04 11:45:00 | 482.70 | 2025-11-10 09:15:00 | 457.57 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-11-04 11:45:00 | 482.70 | 2025-11-11 11:15:00 | 459.90 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-11-11 09:30:00 | 456.15 | 2025-11-11 14:15:00 | 469.20 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-11-11 11:00:00 | 456.55 | 2025-11-11 14:15:00 | 469.20 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest1 | 2025-11-18 14:45:00 | 465.50 | 2025-11-19 13:15:00 | 472.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest1 | 2025-11-19 13:15:00 | 466.50 | 2025-11-19 13:15:00 | 472.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-11-20 09:15:00 | 465.70 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-20 12:15:00 | 466.65 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-20 14:15:00 | 466.50 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-20 14:45:00 | 465.95 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-21 09:15:00 | 464.15 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-11-21 10:15:00 | 463.90 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-25 11:45:00 | 458.50 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-25 12:45:00 | 459.30 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-11-25 14:15:00 | 459.90 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-11-25 15:15:00 | 459.50 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-12-10 10:45:00 | 451.50 | 2025-12-19 12:15:00 | 443.95 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2025-12-11 12:30:00 | 452.00 | 2025-12-19 12:15:00 | 443.95 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2025-12-12 12:00:00 | 451.15 | 2025-12-19 12:15:00 | 443.95 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2025-12-26 12:15:00 | 475.80 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-12-26 12:45:00 | 474.20 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-12-29 09:45:00 | 475.00 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-12-29 10:45:00 | 474.30 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-12-29 15:00:00 | 475.60 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-12-30 09:30:00 | 475.70 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-12-30 10:00:00 | 474.90 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-12-30 11:15:00 | 477.45 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-01-19 09:15:00 | 483.50 | 2026-01-20 09:15:00 | 461.45 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-01-19 12:30:00 | 467.15 | 2026-01-20 09:15:00 | 461.45 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-23 09:15:00 | 439.35 | 2026-01-23 14:15:00 | 417.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 439.35 | 2026-01-27 13:15:00 | 395.42 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-05 13:30:00 | 401.55 | 2026-02-06 15:15:00 | 441.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-16 12:30:00 | 468.00 | 2026-02-17 13:15:00 | 474.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-20 09:15:00 | 476.95 | 2026-02-20 10:15:00 | 473.35 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-02-23 11:45:00 | 468.25 | 2026-03-02 09:15:00 | 444.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 467.90 | 2026-03-02 09:15:00 | 444.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 468.20 | 2026-03-02 09:15:00 | 444.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:00:00 | 468.35 | 2026-03-02 09:15:00 | 444.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 462.25 | 2026-03-02 09:15:00 | 439.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 462.10 | 2026-03-02 09:15:00 | 439.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:45:00 | 468.25 | 2026-03-04 15:15:00 | 421.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 467.90 | 2026-03-04 15:15:00 | 421.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 468.20 | 2026-03-04 15:15:00 | 421.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 10:00:00 | 468.35 | 2026-03-04 15:15:00 | 421.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 462.25 | 2026-03-05 09:15:00 | 432.05 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-02-26 15:15:00 | 462.10 | 2026-03-05 09:15:00 | 432.05 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-03-24 10:30:00 | 372.55 | 2026-03-25 09:15:00 | 394.45 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2026-03-30 09:15:00 | 380.15 | 2026-04-01 15:15:00 | 384.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-04-01 13:30:00 | 383.00 | 2026-04-01 15:15:00 | 384.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-04-09 13:30:00 | 405.00 | 2026-04-10 11:15:00 | 401.35 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-04-10 09:45:00 | 410.75 | 2026-04-10 11:15:00 | 401.35 | STOP_HIT | 1.00 | -2.29% |

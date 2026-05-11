# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 905.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 204 |
| ALERT1 | 150 |
| ALERT2 | 150 |
| ALERT2_SKIP | 104 |
| ALERT3 | 341 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 176 |
| PARTIAL | 11 |
| TARGET_HIT | 8 |
| STOP_HIT | 173 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 192 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 137
- **Target hits / Stop hits / Partials:** 8 / 173 / 11
- **Avg / median % per leg:** -0.10% / -1.03%
- **Sum % (uncompounded):** -20.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 23 | 31.1% | 8 | 66 | 0 | 0.44% | 32.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.74% | -2.2% |
| BUY @ 3rd Alert (retest2) | 71 | 23 | 32.4% | 8 | 63 | 0 | 0.49% | 35.0% |
| SELL (all) | 118 | 32 | 27.1% | 0 | 107 | 11 | -0.45% | -52.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.46% | -4.9% |
| SELL @ 3rd Alert (retest2) | 116 | 32 | 27.6% | 0 | 105 | 11 | -0.41% | -47.9% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.43% | -7.1% |
| retest2 (combined) | 187 | 55 | 29.4% | 8 | 168 | 11 | -0.07% | -13.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 300.45 | 303.23 | 303.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 10:15:00 | 297.60 | 301.19 | 302.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 301.55 | 299.04 | 300.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 301.55 | 299.04 | 300.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 301.55 | 299.04 | 300.33 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 15:15:00 | 299.95 | 297.86 | 297.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 15:15:00 | 302.00 | 298.75 | 298.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 15:15:00 | 303.95 | 304.39 | 302.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 302.00 | 303.92 | 302.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 302.00 | 303.92 | 302.14 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 09:15:00 | 300.65 | 302.58 | 302.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 11:15:00 | 298.80 | 301.52 | 302.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 297.80 | 297.58 | 299.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 10:15:00 | 296.15 | 297.30 | 298.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 296.15 | 297.30 | 298.79 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 315.80 | 298.63 | 297.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 10:15:00 | 321.10 | 315.28 | 310.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 15:15:00 | 317.00 | 317.29 | 313.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 09:15:00 | 313.95 | 316.62 | 313.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 313.95 | 316.62 | 313.60 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 341.90 | 346.52 | 346.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 336.95 | 342.82 | 344.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 346.45 | 339.61 | 341.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 346.45 | 339.61 | 341.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 346.45 | 339.61 | 341.63 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 346.25 | 343.03 | 342.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 348.75 | 344.65 | 343.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 363.60 | 364.42 | 359.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 358.55 | 363.16 | 359.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 358.55 | 363.16 | 359.82 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 12:15:00 | 359.25 | 361.22 | 361.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 13:15:00 | 357.60 | 360.50 | 360.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 15:15:00 | 360.50 | 360.38 | 360.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 15:15:00 | 360.50 | 360.38 | 360.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 360.50 | 360.38 | 360.80 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 365.00 | 361.79 | 361.40 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 14:15:00 | 359.25 | 361.26 | 361.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 14:15:00 | 356.60 | 359.40 | 360.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 357.85 | 354.54 | 356.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 357.85 | 354.54 | 356.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 357.85 | 354.54 | 356.48 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 10:15:00 | 360.65 | 353.67 | 353.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 371.45 | 361.32 | 357.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 373.60 | 374.93 | 368.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 13:15:00 | 381.50 | 381.03 | 378.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 381.50 | 381.03 | 378.49 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 11:15:00 | 392.25 | 397.92 | 398.11 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 11:15:00 | 399.50 | 394.79 | 394.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 404.25 | 399.06 | 396.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 14:15:00 | 399.90 | 401.87 | 399.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 14:15:00 | 399.90 | 401.87 | 399.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 14:15:00 | 399.90 | 401.87 | 399.36 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 15:15:00 | 462.50 | 465.70 | 465.98 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 09:15:00 | 469.70 | 466.50 | 466.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 10:15:00 | 471.85 | 467.57 | 466.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 13:15:00 | 465.55 | 467.46 | 466.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 13:15:00 | 465.55 | 467.46 | 466.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 465.55 | 467.46 | 466.98 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 14:15:00 | 465.00 | 466.71 | 466.89 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 469.35 | 467.16 | 467.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 474.40 | 470.10 | 468.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 14:15:00 | 471.90 | 472.98 | 470.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 15:15:00 | 469.70 | 472.33 | 470.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 469.70 | 472.33 | 470.79 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 09:15:00 | 467.75 | 476.81 | 477.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 466.10 | 470.04 | 472.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 478.00 | 470.84 | 472.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 10:15:00 | 478.00 | 470.84 | 472.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 478.00 | 470.84 | 472.13 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 11:15:00 | 487.05 | 474.09 | 473.49 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 10:15:00 | 473.85 | 481.70 | 482.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 11:15:00 | 472.05 | 479.77 | 481.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 444.00 | 440.02 | 447.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 12:15:00 | 444.00 | 440.02 | 447.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 444.00 | 440.02 | 447.34 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 452.90 | 447.45 | 447.34 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 444.65 | 447.50 | 447.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 442.25 | 445.28 | 446.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 439.00 | 434.09 | 436.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 439.00 | 434.09 | 436.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 439.00 | 434.09 | 436.20 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 15:15:00 | 447.00 | 436.93 | 436.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 10:15:00 | 447.20 | 440.59 | 437.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 12:15:00 | 441.10 | 441.33 | 438.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 10:15:00 | 439.65 | 441.15 | 439.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 439.65 | 441.15 | 439.71 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 439.50 | 442.53 | 442.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 437.20 | 441.26 | 442.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 12:15:00 | 439.80 | 439.22 | 440.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 13:15:00 | 443.95 | 440.16 | 440.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 443.95 | 440.16 | 440.70 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 443.00 | 441.23 | 441.12 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 436.85 | 440.36 | 440.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 432.70 | 435.54 | 437.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 435.70 | 435.57 | 437.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 435.70 | 435.57 | 437.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 435.70 | 435.57 | 437.61 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 12:15:00 | 439.40 | 437.42 | 437.29 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 13:15:00 | 436.00 | 437.14 | 437.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 14:15:00 | 433.70 | 436.45 | 436.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 10:15:00 | 436.85 | 435.96 | 436.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 10:15:00 | 436.85 | 435.96 | 436.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 436.85 | 435.96 | 436.47 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 11:15:00 | 450.15 | 438.80 | 437.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 12:15:00 | 458.10 | 442.66 | 439.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 09:15:00 | 455.75 | 456.66 | 451.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 463.65 | 468.60 | 464.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 463.65 | 468.60 | 464.49 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 461.70 | 465.73 | 465.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 14:15:00 | 458.55 | 463.57 | 464.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 13:15:00 | 435.10 | 434.84 | 443.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 433.60 | 431.80 | 436.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 433.60 | 431.80 | 436.74 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 443.50 | 438.76 | 438.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 12:15:00 | 446.00 | 441.49 | 439.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 14:15:00 | 439.55 | 441.38 | 440.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 14:15:00 | 439.55 | 441.38 | 440.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 439.55 | 441.38 | 440.20 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 436.50 | 440.55 | 440.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 435.75 | 439.01 | 440.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 434.95 | 432.97 | 435.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 09:15:00 | 434.95 | 432.97 | 435.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 434.95 | 432.97 | 435.11 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 13:15:00 | 440.00 | 436.53 | 436.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 443.30 | 438.96 | 437.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 448.75 | 449.95 | 447.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 10:15:00 | 447.05 | 449.42 | 447.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 447.05 | 449.42 | 447.56 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 09:15:00 | 464.75 | 467.87 | 468.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 10:15:00 | 461.05 | 466.51 | 467.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 461.80 | 460.31 | 463.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 461.80 | 460.31 | 463.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 461.80 | 460.31 | 463.33 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 459.05 | 448.67 | 447.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 463.30 | 456.94 | 453.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 12:15:00 | 465.95 | 466.70 | 461.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 13:15:00 | 463.75 | 466.11 | 462.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 13:15:00 | 463.75 | 466.11 | 462.01 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 510.55 | 514.87 | 515.16 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 521.30 | 515.40 | 515.13 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 12:15:00 | 513.00 | 515.54 | 515.88 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 09:15:00 | 521.85 | 515.88 | 515.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 10:15:00 | 534.95 | 519.69 | 517.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 11:15:00 | 536.30 | 537.84 | 533.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 11:15:00 | 536.30 | 537.84 | 533.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 536.30 | 537.84 | 533.40 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 520.50 | 529.32 | 530.44 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 09:15:00 | 543.05 | 532.06 | 531.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 10:15:00 | 559.85 | 537.62 | 534.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 613.50 | 613.56 | 601.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 618.90 | 622.97 | 615.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 618.90 | 622.97 | 615.91 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 644.70 | 652.44 | 652.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 642.05 | 650.36 | 651.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 14:15:00 | 638.00 | 634.40 | 639.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 15:15:00 | 637.00 | 634.92 | 639.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 637.00 | 634.92 | 639.49 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 663.60 | 644.83 | 642.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 15:15:00 | 670.60 | 649.99 | 645.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 673.35 | 677.89 | 669.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 673.35 | 677.89 | 669.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 673.35 | 677.89 | 669.97 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 657.05 | 671.80 | 671.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 653.75 | 664.53 | 668.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 657.20 | 656.48 | 661.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 12:15:00 | 654.75 | 656.86 | 661.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 654.75 | 656.86 | 661.05 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 672.75 | 661.26 | 660.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 10:15:00 | 691.70 | 671.82 | 666.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 10:15:00 | 689.35 | 690.27 | 680.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 11:15:00 | 678.50 | 687.91 | 680.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 678.50 | 687.91 | 680.20 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 664.00 | 675.92 | 676.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 663.65 | 672.57 | 674.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 675.60 | 673.17 | 674.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 12:15:00 | 675.60 | 673.17 | 674.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 675.60 | 673.17 | 674.67 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 12:15:00 | 675.70 | 675.22 | 675.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 15:15:00 | 678.15 | 676.18 | 675.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 706.65 | 718.52 | 707.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 706.65 | 718.52 | 707.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 706.65 | 718.52 | 707.50 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 705.00 | 705.70 | 705.79 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 714.65 | 706.23 | 705.89 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 698.35 | 705.13 | 705.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 686.15 | 698.95 | 702.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 698.05 | 696.58 | 700.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 698.05 | 696.58 | 700.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 698.05 | 696.58 | 700.77 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 675.00 | 670.39 | 670.20 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 09:15:00 | 664.30 | 670.71 | 671.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 13:15:00 | 660.30 | 665.14 | 667.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 667.65 | 665.25 | 666.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 667.65 | 665.25 | 666.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 667.65 | 665.25 | 666.92 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 14:15:00 | 666.70 | 655.98 | 654.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 667.55 | 659.42 | 656.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 657.00 | 660.75 | 658.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 14:15:00 | 657.00 | 660.75 | 658.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 657.00 | 660.75 | 658.55 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 652.45 | 657.09 | 657.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 14:15:00 | 648.65 | 655.41 | 656.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 641.70 | 635.47 | 642.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 12:15:00 | 640.40 | 636.46 | 642.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 640.40 | 636.46 | 642.06 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 678.15 | 649.61 | 646.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 688.35 | 665.73 | 655.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 10:15:00 | 667.65 | 669.34 | 660.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 15:15:00 | 664.00 | 666.81 | 662.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 664.00 | 666.81 | 662.40 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 653.85 | 659.45 | 659.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 631.30 | 651.59 | 655.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 643.55 | 636.49 | 643.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 643.55 | 636.49 | 643.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 643.55 | 636.49 | 643.89 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 13:15:00 | 568.90 | 565.81 | 565.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 14:15:00 | 580.00 | 568.65 | 566.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 11:15:00 | 572.00 | 572.04 | 569.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 11:15:00 | 572.00 | 572.04 | 569.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 572.00 | 572.04 | 569.30 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 612.05 | 616.50 | 617.00 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 619.75 | 617.28 | 616.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 625.75 | 619.38 | 618.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-08 13:15:00 | 620.60 | 622.09 | 620.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 13:15:00 | 620.60 | 622.09 | 620.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 620.60 | 622.09 | 620.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 625.80 | 629.42 | 626.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:45:00 | 626.20 | 628.16 | 626.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 10:45:00 | 624.95 | 628.11 | 626.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 13:15:00 | 619.85 | 624.44 | 624.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 619.85 | 624.44 | 624.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 602.10 | 618.89 | 622.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 603.75 | 602.02 | 607.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 603.75 | 602.02 | 607.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 603.75 | 602.02 | 607.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 598.00 | 606.20 | 607.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 12:45:00 | 598.15 | 600.24 | 604.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 607.90 | 603.84 | 603.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 607.90 | 603.84 | 603.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 10:15:00 | 610.85 | 605.71 | 604.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 642.05 | 642.46 | 633.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 10:00:00 | 642.05 | 642.46 | 633.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 639.85 | 643.31 | 638.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 636.00 | 641.85 | 637.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 634.00 | 640.28 | 637.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:00:00 | 637.45 | 639.71 | 637.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 627.50 | 635.89 | 636.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 627.50 | 635.89 | 636.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 11:15:00 | 626.00 | 633.91 | 635.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 591.15 | 589.13 | 597.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 591.15 | 589.13 | 597.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 591.80 | 589.28 | 595.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 595.55 | 589.28 | 595.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 582.30 | 588.83 | 593.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 580.10 | 588.83 | 593.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 14:00:00 | 581.30 | 575.95 | 580.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 15:15:00 | 582.00 | 577.40 | 581.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 15:00:00 | 580.90 | 579.84 | 580.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 579.80 | 579.83 | 580.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 591.35 | 579.83 | 580.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 588.20 | 581.51 | 581.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 588.20 | 581.51 | 581.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 595.75 | 586.39 | 583.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 600.00 | 602.36 | 596.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 596.50 | 601.18 | 596.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 596.50 | 601.18 | 596.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:00:00 | 596.50 | 601.18 | 596.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 597.05 | 600.36 | 596.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:15:00 | 596.05 | 600.36 | 596.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 594.95 | 599.28 | 596.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:30:00 | 594.60 | 599.28 | 596.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 594.95 | 598.41 | 596.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:15:00 | 594.55 | 598.41 | 596.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 591.00 | 596.93 | 595.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 591.00 | 596.93 | 595.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 591.00 | 594.46 | 594.81 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 597.80 | 595.13 | 595.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 623.00 | 600.70 | 597.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 636.50 | 638.33 | 626.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 636.50 | 638.33 | 626.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 632.00 | 635.85 | 627.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:00:00 | 640.00 | 636.68 | 628.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 11:00:00 | 638.90 | 637.13 | 629.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 629.80 | 641.96 | 642.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 629.80 | 641.96 | 642.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 622.65 | 631.68 | 636.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 626.80 | 623.05 | 627.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 14:00:00 | 626.80 | 623.05 | 627.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 623.20 | 623.08 | 626.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:45:00 | 620.20 | 621.74 | 625.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:00:00 | 621.30 | 621.65 | 625.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:45:00 | 621.85 | 621.98 | 624.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 621.00 | 622.12 | 624.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 611.05 | 619.90 | 622.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 609.45 | 619.90 | 622.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 608.90 | 615.55 | 619.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:15:00 | 608.00 | 614.84 | 618.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 13:15:00 | 625.20 | 619.59 | 619.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 625.20 | 619.59 | 619.52 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 608.00 | 617.89 | 618.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 541.65 | 596.66 | 608.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 567.95 | 567.88 | 583.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 567.85 | 567.88 | 583.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 596.55 | 573.84 | 581.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 596.55 | 573.84 | 581.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 607.60 | 580.59 | 583.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 606.85 | 580.59 | 583.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 611.00 | 586.67 | 586.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 631.10 | 612.35 | 602.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 11:15:00 | 673.00 | 674.05 | 667.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 11:45:00 | 674.05 | 674.05 | 667.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 665.85 | 672.61 | 669.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 665.85 | 672.61 | 669.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 667.15 | 671.52 | 668.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 665.45 | 671.52 | 668.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 662.35 | 669.68 | 668.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 662.35 | 669.68 | 668.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 13:15:00 | 665.95 | 667.32 | 667.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 651.40 | 662.11 | 664.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 15:15:00 | 656.60 | 655.85 | 659.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 09:15:00 | 662.30 | 655.85 | 659.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 661.25 | 656.93 | 659.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 661.25 | 656.93 | 659.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 659.90 | 657.52 | 659.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 655.45 | 657.14 | 659.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 15:00:00 | 658.10 | 657.75 | 659.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 15:15:00 | 664.00 | 659.00 | 659.68 | SL hit (close>static) qty=1.00 sl=661.40 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 664.50 | 658.48 | 658.39 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 655.35 | 657.85 | 658.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 651.25 | 656.69 | 657.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 657.05 | 656.76 | 657.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 657.05 | 656.76 | 657.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 657.05 | 656.76 | 657.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 657.05 | 656.76 | 657.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 652.50 | 655.91 | 657.01 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 674.55 | 659.22 | 658.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 681.50 | 669.44 | 664.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 703.95 | 705.78 | 691.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 13:00:00 | 703.95 | 705.78 | 691.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 723.90 | 723.38 | 717.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 738.50 | 726.26 | 719.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 714.05 | 723.09 | 720.71 | SL hit (close<static) qty=1.00 sl=716.05 alert=retest2 |

### Cycle 73 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 709.00 | 719.30 | 719.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 707.05 | 712.41 | 715.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 10:15:00 | 693.45 | 687.33 | 695.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 10:15:00 | 693.45 | 687.33 | 695.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 693.45 | 687.33 | 695.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 694.20 | 687.33 | 695.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 695.25 | 688.91 | 695.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 695.25 | 688.91 | 695.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 696.15 | 690.36 | 695.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 694.40 | 690.36 | 695.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 695.90 | 691.47 | 695.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 695.90 | 691.47 | 695.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 692.70 | 691.71 | 695.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 15:15:00 | 687.30 | 691.71 | 695.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:00:00 | 689.60 | 690.59 | 693.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:15:00 | 690.05 | 690.69 | 693.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:00:00 | 689.75 | 690.16 | 692.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 692.35 | 687.71 | 690.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 692.80 | 687.71 | 690.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 692.45 | 688.66 | 690.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:15:00 | 690.15 | 688.66 | 690.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 13:15:00 | 693.85 | 690.02 | 690.85 | SL hit (close>static) qty=1.00 sl=693.40 alert=retest2 |

### Cycle 74 — BUY (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 14:15:00 | 694.15 | 691.46 | 691.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 15:15:00 | 695.60 | 692.29 | 691.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 687.90 | 691.41 | 691.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 687.90 | 691.41 | 691.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 687.90 | 691.41 | 691.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 686.40 | 691.41 | 691.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 678.25 | 688.78 | 690.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 655.70 | 680.07 | 685.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 15:15:00 | 668.00 | 667.37 | 675.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 09:15:00 | 656.00 | 667.37 | 675.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 12:30:00 | 661.55 | 667.83 | 672.98 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 675.00 | 669.26 | 673.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-22 13:15:00 | 675.00 | 669.26 | 673.17 | SL hit (close>ema400) qty=1.00 sl=673.17 alert=retest1 |

### Cycle 76 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 680.85 | 673.07 | 672.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 15:15:00 | 685.00 | 675.45 | 673.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 681.50 | 686.47 | 682.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 681.50 | 686.47 | 682.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 681.50 | 686.47 | 682.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:45:00 | 680.40 | 686.47 | 682.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 682.90 | 685.76 | 682.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:30:00 | 681.00 | 685.76 | 682.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 683.00 | 685.21 | 682.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:30:00 | 685.75 | 684.96 | 682.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:45:00 | 685.05 | 685.16 | 682.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 694.35 | 684.33 | 682.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:00:00 | 685.25 | 685.58 | 683.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 713.20 | 717.06 | 709.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 710.00 | 717.06 | 709.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2024-08-01 09:15:00 | 754.33 | 733.31 | 721.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 717.75 | 729.17 | 729.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 683.50 | 720.04 | 725.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 705.70 | 701.48 | 708.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 13:00:00 | 705.70 | 701.48 | 708.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 701.30 | 698.97 | 705.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:15:00 | 718.10 | 698.97 | 705.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 721.80 | 703.54 | 706.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 722.70 | 703.54 | 706.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 724.30 | 707.69 | 708.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 732.85 | 707.69 | 708.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 719.60 | 710.07 | 709.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 736.55 | 717.30 | 712.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 719.80 | 721.28 | 717.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:15:00 | 742.50 | 721.22 | 718.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 732.50 | 735.54 | 728.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 14:30:00 | 745.80 | 738.81 | 732.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 15:00:00 | 745.30 | 738.81 | 732.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 750.65 | 739.45 | 733.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:00:00 | 751.00 | 742.29 | 736.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 747.30 | 743.86 | 737.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 747.30 | 743.86 | 737.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 738.00 | 744.45 | 739.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-13 15:15:00 | 738.00 | 744.45 | 739.80 | SL hit (close<ema400) qty=1.00 sl=739.80 alert=retest1 |

### Cycle 79 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 732.70 | 737.07 | 737.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 09:15:00 | 720.50 | 732.98 | 735.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 12:15:00 | 720.80 | 717.84 | 723.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 13:00:00 | 720.80 | 717.84 | 723.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 719.90 | 717.42 | 721.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 734.75 | 717.42 | 721.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 739.30 | 721.79 | 723.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 741.90 | 721.79 | 723.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 749.35 | 727.31 | 725.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 756.30 | 733.10 | 728.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 748.65 | 749.49 | 739.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:00:00 | 748.65 | 749.49 | 739.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 814.65 | 817.80 | 812.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 812.00 | 817.80 | 812.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 813.50 | 816.94 | 812.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 809.05 | 816.94 | 812.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 800.85 | 813.72 | 811.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 800.85 | 813.72 | 811.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 800.00 | 810.98 | 810.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:30:00 | 800.25 | 810.98 | 810.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 799.90 | 808.76 | 809.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 796.00 | 804.62 | 807.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 12:15:00 | 792.80 | 784.65 | 790.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 12:15:00 | 792.80 | 784.65 | 790.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 792.80 | 784.65 | 790.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:00:00 | 792.80 | 784.65 | 790.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 796.00 | 786.92 | 790.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 796.00 | 786.92 | 790.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 797.50 | 789.03 | 791.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 797.50 | 789.03 | 791.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 795.00 | 790.23 | 791.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 798.00 | 790.23 | 791.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 794.05 | 791.80 | 792.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 791.55 | 791.80 | 792.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 786.20 | 790.68 | 791.62 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 801.00 | 792.97 | 792.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 815.70 | 801.54 | 797.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 816.40 | 827.21 | 820.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 816.40 | 827.21 | 820.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 816.40 | 827.21 | 820.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:45:00 | 816.95 | 827.21 | 820.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 808.70 | 823.51 | 819.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 809.80 | 823.51 | 819.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 803.65 | 814.88 | 816.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 796.95 | 811.29 | 814.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 810.00 | 808.49 | 812.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 10:45:00 | 809.00 | 808.49 | 812.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 811.50 | 809.09 | 811.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 810.25 | 809.09 | 811.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 813.85 | 810.05 | 812.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 813.85 | 810.05 | 812.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 812.55 | 810.55 | 812.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 810.35 | 810.55 | 812.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 812.00 | 810.84 | 812.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 812.00 | 810.84 | 812.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 815.00 | 811.67 | 812.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 827.35 | 811.67 | 812.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 818.75 | 813.09 | 813.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 832.00 | 822.87 | 819.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 878.75 | 883.76 | 868.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 878.75 | 883.76 | 868.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 880.00 | 886.85 | 882.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 880.00 | 886.85 | 882.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 863.60 | 882.20 | 880.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 863.60 | 882.20 | 880.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 861.75 | 878.11 | 879.19 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 907.00 | 881.55 | 879.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 931.05 | 898.81 | 888.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 12:15:00 | 934.30 | 936.31 | 923.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:45:00 | 934.80 | 936.31 | 923.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 921.85 | 932.29 | 925.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 921.85 | 932.29 | 925.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 928.10 | 931.45 | 925.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:15:00 | 933.95 | 931.45 | 925.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 12:00:00 | 929.85 | 931.13 | 925.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:00:00 | 930.25 | 927.92 | 925.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 13:15:00 | 917.75 | 923.95 | 924.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 917.75 | 923.95 | 924.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 10:15:00 | 910.15 | 918.48 | 921.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 915.10 | 910.89 | 915.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 915.10 | 910.89 | 915.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 915.10 | 910.89 | 915.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:30:00 | 895.70 | 907.19 | 911.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 895.80 | 907.50 | 908.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 914.50 | 908.90 | 908.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 914.50 | 908.90 | 908.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 923.45 | 911.81 | 910.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 911.60 | 913.91 | 911.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 911.60 | 913.91 | 911.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 911.60 | 913.91 | 911.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:45:00 | 913.70 | 913.91 | 911.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 908.60 | 912.85 | 911.43 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 886.05 | 907.03 | 909.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 876.40 | 900.91 | 906.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 898.00 | 896.68 | 902.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 898.00 | 896.68 | 902.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 899.00 | 897.14 | 901.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 891.95 | 897.14 | 901.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 905.30 | 899.58 | 901.81 | SL hit (close>static) qty=1.00 sl=904.15 alert=retest2 |

### Cycle 90 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 922.25 | 902.48 | 902.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 929.00 | 907.79 | 904.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 924.00 | 929.22 | 922.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 15:15:00 | 924.00 | 929.22 | 922.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 924.00 | 929.22 | 922.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 919.15 | 929.22 | 922.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 919.95 | 927.37 | 922.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 918.25 | 927.37 | 922.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 916.25 | 925.14 | 922.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 916.25 | 925.14 | 922.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 912.45 | 919.07 | 919.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 910.10 | 916.18 | 918.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 913.00 | 911.93 | 914.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 913.00 | 911.93 | 914.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 913.00 | 911.93 | 914.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 913.00 | 911.93 | 914.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 915.00 | 912.54 | 914.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:30:00 | 915.65 | 912.54 | 914.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 916.00 | 913.23 | 914.54 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 919.95 | 915.62 | 915.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 15:15:00 | 930.00 | 918.49 | 916.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 915.00 | 917.79 | 916.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 915.00 | 917.79 | 916.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 915.00 | 917.79 | 916.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 915.00 | 917.79 | 916.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 913.65 | 916.97 | 916.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 911.50 | 916.97 | 916.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 913.05 | 916.18 | 915.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 912.75 | 916.18 | 915.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 915.70 | 915.90 | 915.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:45:00 | 914.55 | 915.90 | 915.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 928.65 | 918.45 | 916.95 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 886.90 | 911.89 | 914.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 876.95 | 897.65 | 905.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 11:15:00 | 829.95 | 828.20 | 840.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 11:30:00 | 828.25 | 828.20 | 840.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 805.40 | 804.17 | 809.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 792.15 | 803.93 | 809.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 818.00 | 806.75 | 809.82 | SL hit (close>static) qty=1.00 sl=815.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 834.55 | 815.44 | 813.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 838.60 | 823.20 | 817.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 839.75 | 842.74 | 836.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 839.75 | 842.74 | 836.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 839.75 | 842.74 | 836.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 839.75 | 842.74 | 836.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 822.50 | 838.69 | 835.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 822.50 | 838.69 | 835.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 831.05 | 837.16 | 834.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 819.15 | 837.16 | 834.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 837.40 | 835.18 | 834.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 840.00 | 835.18 | 834.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 841.60 | 862.73 | 864.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 841.60 | 862.73 | 864.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 830.70 | 856.33 | 861.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 811.45 | 807.40 | 823.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 811.45 | 807.40 | 823.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 811.45 | 807.40 | 823.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 15:15:00 | 799.95 | 813.46 | 820.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 10:00:00 | 803.00 | 798.32 | 806.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 820.15 | 811.42 | 810.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 820.15 | 811.42 | 810.38 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 800.00 | 812.89 | 814.32 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 818.50 | 813.24 | 813.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 839.70 | 829.25 | 823.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 838.55 | 840.55 | 835.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 13:15:00 | 838.25 | 839.46 | 835.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 838.25 | 839.46 | 835.81 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 09:15:00 | 827.20 | 834.10 | 834.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 13:15:00 | 820.85 | 827.92 | 831.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 11:15:00 | 830.80 | 824.74 | 828.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 11:15:00 | 830.80 | 824.74 | 828.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 830.80 | 824.74 | 828.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 830.80 | 824.74 | 828.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 825.00 | 824.79 | 827.74 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 837.05 | 830.52 | 829.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 843.55 | 834.46 | 831.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 14:15:00 | 846.25 | 847.19 | 841.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 846.25 | 847.19 | 841.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 846.25 | 847.19 | 841.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 843.50 | 847.19 | 841.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 839.95 | 849.15 | 846.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 839.95 | 849.15 | 846.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 839.45 | 847.21 | 845.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 839.45 | 847.21 | 845.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 838.75 | 845.52 | 845.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 838.75 | 845.52 | 845.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 847.65 | 845.72 | 845.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:45:00 | 844.45 | 845.72 | 845.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 846.10 | 845.79 | 845.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 852.05 | 846.30 | 845.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 842.50 | 852.92 | 853.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 842.50 | 852.92 | 853.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 840.00 | 850.34 | 852.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 798.60 | 798.27 | 807.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:00:00 | 798.60 | 798.27 | 807.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 790.85 | 797.23 | 803.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:15:00 | 787.85 | 797.23 | 803.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 787.90 | 794.71 | 800.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 748.46 | 777.97 | 790.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 748.50 | 777.97 | 790.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 13:15:00 | 753.85 | 751.85 | 763.40 | SL hit (close>ema200) qty=0.50 sl=751.85 alert=retest2 |

### Cycle 102 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 749.00 | 731.72 | 731.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 755.20 | 743.10 | 738.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 768.95 | 769.51 | 757.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 768.95 | 769.51 | 757.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 768.95 | 769.51 | 757.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 766.70 | 769.51 | 757.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 750.50 | 765.71 | 756.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 750.50 | 765.71 | 756.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 765.35 | 765.63 | 757.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 770.15 | 764.71 | 759.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 11:15:00 | 769.95 | 764.79 | 760.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 09:15:00 | 754.35 | 777.69 | 778.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 754.35 | 777.69 | 778.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 744.50 | 762.81 | 769.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 748.95 | 748.91 | 756.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:45:00 | 748.35 | 748.91 | 756.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 755.80 | 749.60 | 753.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 764.50 | 749.60 | 753.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 753.80 | 750.44 | 753.40 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 765.05 | 755.77 | 755.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 768.90 | 758.39 | 756.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 09:15:00 | 792.30 | 805.67 | 795.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 792.30 | 805.67 | 795.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 792.30 | 805.67 | 795.51 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 777.15 | 794.32 | 796.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 737.95 | 780.58 | 789.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 733.80 | 730.80 | 750.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 733.80 | 730.80 | 750.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 742.40 | 731.30 | 741.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 742.60 | 731.30 | 741.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 734.70 | 731.98 | 740.88 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 772.55 | 748.00 | 745.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 776.25 | 756.44 | 750.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 755.80 | 764.63 | 756.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 755.80 | 764.63 | 756.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 755.80 | 764.63 | 756.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:00:00 | 776.95 | 767.01 | 759.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 13:00:00 | 778.00 | 769.21 | 760.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:00:00 | 776.40 | 763.99 | 762.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 779.35 | 767.01 | 763.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 753.70 | 766.27 | 763.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 751.70 | 766.27 | 763.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 756.55 | 764.32 | 763.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 758.00 | 764.32 | 763.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:00:00 | 758.10 | 763.08 | 762.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 12:15:00 | 760.45 | 762.55 | 762.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 760.45 | 762.55 | 762.58 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 765.70 | 763.06 | 762.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 768.65 | 765.12 | 763.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 777.30 | 780.08 | 775.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 777.30 | 780.08 | 775.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 774.40 | 778.94 | 775.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 774.70 | 778.94 | 775.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 776.25 | 778.40 | 775.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 773.20 | 778.40 | 775.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 770.40 | 777.87 | 775.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 769.05 | 777.87 | 775.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 781.95 | 778.69 | 776.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 784.15 | 779.36 | 776.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:30:00 | 782.10 | 781.24 | 778.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 759.45 | 777.47 | 777.36 | SL hit (close<static) qty=1.00 sl=762.20 alert=retest2 |

### Cycle 109 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 763.55 | 774.68 | 776.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 737.65 | 759.75 | 767.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 731.00 | 730.52 | 744.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 731.55 | 730.52 | 744.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 736.50 | 731.69 | 739.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:00:00 | 718.35 | 733.32 | 737.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 719.30 | 729.71 | 733.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:45:00 | 720.75 | 726.31 | 731.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 15:00:00 | 722.40 | 721.49 | 726.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 703.80 | 718.04 | 723.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 701.25 | 718.04 | 723.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:30:00 | 701.55 | 711.05 | 718.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:00:00 | 699.40 | 712.92 | 716.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 14:15:00 | 682.43 | 707.68 | 713.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 14:15:00 | 683.33 | 707.68 | 713.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 14:15:00 | 684.71 | 707.68 | 713.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 14:15:00 | 686.28 | 707.68 | 713.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 704.80 | 704.09 | 710.17 | SL hit (close>ema200) qty=0.50 sl=704.09 alert=retest2 |

### Cycle 110 — BUY (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 12:15:00 | 720.10 | 712.65 | 711.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 14:15:00 | 728.95 | 717.60 | 714.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 717.50 | 719.25 | 715.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 717.50 | 719.25 | 715.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 717.50 | 719.25 | 715.63 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 701.70 | 714.10 | 715.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 696.30 | 705.88 | 708.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 708.00 | 704.90 | 707.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 708.00 | 704.90 | 707.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 708.00 | 704.90 | 707.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:30:00 | 707.55 | 704.90 | 707.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 705.30 | 704.98 | 707.29 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 723.65 | 711.02 | 709.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 729.00 | 721.60 | 716.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 789.00 | 807.35 | 792.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 789.00 | 807.35 | 792.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 789.00 | 807.35 | 792.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 792.25 | 807.35 | 792.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 800.20 | 805.92 | 793.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:15:00 | 808.00 | 802.18 | 795.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 12:00:00 | 812.00 | 808.04 | 800.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:15:00 | 807.25 | 814.63 | 813.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:00:00 | 811.20 | 813.94 | 813.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 819.60 | 816.22 | 814.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 818.35 | 816.22 | 814.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 822.25 | 817.43 | 815.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 11:30:00 | 830.00 | 820.98 | 817.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 09:15:00 | 888.80 | 866.29 | 856.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 853.70 | 859.67 | 859.74 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 860.65 | 856.45 | 856.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 13:15:00 | 868.10 | 859.70 | 857.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 851.40 | 858.04 | 857.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 851.40 | 858.04 | 857.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 851.40 | 858.04 | 857.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 851.40 | 858.04 | 857.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 857.00 | 857.83 | 857.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 854.35 | 857.83 | 857.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 858.20 | 857.91 | 857.31 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 853.55 | 856.25 | 856.61 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 862.90 | 857.37 | 857.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 893.90 | 865.30 | 860.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 15:15:00 | 901.00 | 906.09 | 894.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 09:15:00 | 889.45 | 906.09 | 894.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 879.05 | 900.68 | 893.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 879.05 | 900.68 | 893.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 881.60 | 896.86 | 891.98 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 858.70 | 885.89 | 887.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 854.95 | 879.70 | 884.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 828.70 | 821.76 | 841.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 828.70 | 821.76 | 841.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 828.70 | 821.76 | 841.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 813.85 | 821.01 | 838.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 809.60 | 827.18 | 835.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 845.05 | 830.15 | 832.19 | SL hit (close>static) qty=1.00 sl=842.65 alert=retest2 |

### Cycle 118 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 842.10 | 833.88 | 833.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 848.50 | 836.81 | 834.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 888.60 | 889.27 | 875.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 15:15:00 | 903.00 | 893.91 | 883.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 13:00:00 | 900.15 | 895.92 | 888.40 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 894.30 | 905.84 | 901.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 894.30 | 905.84 | 901.13 | SL hit (close<ema400) qty=1.00 sl=901.13 alert=retest1 |

### Cycle 119 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 875.95 | 896.37 | 898.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 864.70 | 871.81 | 876.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 12:15:00 | 874.30 | 872.31 | 876.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 12:15:00 | 874.30 | 872.31 | 876.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 874.30 | 872.31 | 876.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 874.30 | 872.31 | 876.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 872.25 | 872.30 | 875.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:45:00 | 870.70 | 870.14 | 874.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 883.65 | 868.61 | 868.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 883.65 | 868.61 | 868.26 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 859.30 | 872.83 | 873.51 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 897.20 | 876.40 | 874.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 898.15 | 884.05 | 878.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 880.85 | 884.87 | 879.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 880.85 | 884.87 | 879.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 880.85 | 884.87 | 879.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 880.85 | 884.87 | 879.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 879.95 | 883.89 | 879.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 879.45 | 883.89 | 879.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 880.00 | 883.11 | 879.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 861.30 | 883.11 | 879.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 867.45 | 879.98 | 878.79 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 863.35 | 876.65 | 877.39 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 886.65 | 875.66 | 875.40 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 12:15:00 | 870.85 | 874.82 | 875.07 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 880.00 | 875.86 | 875.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 894.85 | 879.66 | 877.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 902.00 | 902.40 | 894.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 902.00 | 902.40 | 894.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 895.75 | 901.07 | 894.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:30:00 | 893.10 | 901.07 | 894.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 898.55 | 900.56 | 894.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 893.25 | 900.56 | 894.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 897.00 | 928.43 | 924.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 903.80 | 928.43 | 924.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 903.05 | 923.35 | 922.19 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 907.85 | 920.25 | 920.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 894.90 | 907.33 | 911.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 905.95 | 905.15 | 909.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:30:00 | 906.40 | 905.15 | 909.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 910.00 | 906.12 | 909.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 899.35 | 906.12 | 909.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 905.15 | 905.92 | 908.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 897.20 | 903.90 | 906.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 887.45 | 904.34 | 905.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:15:00 | 852.34 | 866.54 | 870.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 11:15:00 | 843.08 | 858.05 | 865.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 850.65 | 848.46 | 857.89 | SL hit (close>ema200) qty=0.50 sl=848.46 alert=retest2 |

### Cycle 128 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 870.35 | 862.91 | 862.44 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 858.70 | 863.30 | 863.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 856.75 | 861.99 | 862.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 857.70 | 856.19 | 859.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 857.70 | 856.19 | 859.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 857.70 | 856.19 | 859.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 848.95 | 855.88 | 858.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 861.00 | 855.79 | 858.14 | SL hit (close>static) qty=1.00 sl=859.75 alert=retest2 |

### Cycle 130 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 868.00 | 859.65 | 859.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 872.40 | 864.03 | 861.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 866.20 | 868.69 | 865.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 866.20 | 868.69 | 865.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 872.00 | 869.35 | 865.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:45:00 | 874.60 | 870.75 | 866.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:00:00 | 875.45 | 872.69 | 868.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:45:00 | 876.90 | 874.34 | 870.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 865.05 | 873.79 | 873.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 865.05 | 873.79 | 873.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 861.35 | 871.31 | 872.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 841.50 | 838.55 | 846.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 841.50 | 838.55 | 846.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 841.50 | 838.55 | 846.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 844.70 | 838.55 | 846.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 845.85 | 839.54 | 843.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 846.35 | 839.54 | 843.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 843.70 | 840.37 | 843.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:15:00 | 848.80 | 840.37 | 843.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 841.00 | 840.50 | 843.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 839.60 | 840.50 | 843.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 840.10 | 839.79 | 842.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 831.95 | 826.56 | 826.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 831.95 | 826.56 | 826.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 839.15 | 830.36 | 828.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 833.35 | 834.58 | 831.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 834.55 | 834.58 | 831.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 832.80 | 834.22 | 831.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 832.80 | 834.22 | 831.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 841.45 | 835.39 | 832.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:30:00 | 850.10 | 838.35 | 834.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 845.50 | 840.50 | 836.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 846.55 | 841.80 | 838.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 869.95 | 877.53 | 877.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 869.95 | 877.53 | 877.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 866.00 | 875.22 | 876.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 872.95 | 870.96 | 873.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 872.95 | 870.96 | 873.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 872.95 | 870.96 | 873.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 872.95 | 870.96 | 873.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 873.00 | 871.37 | 873.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 866.40 | 870.59 | 873.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 867.10 | 867.37 | 870.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 866.50 | 866.90 | 869.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 864.05 | 854.23 | 853.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 864.05 | 854.23 | 853.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 867.90 | 856.96 | 855.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 14:15:00 | 909.45 | 913.14 | 903.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 909.45 | 913.14 | 903.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 905.00 | 911.51 | 903.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 921.55 | 911.51 | 903.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 975.90 | 980.00 | 980.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 975.90 | 980.00 | 980.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 971.25 | 978.25 | 979.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 980.00 | 978.60 | 979.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 10:15:00 | 980.00 | 978.60 | 979.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 980.00 | 978.60 | 979.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 980.50 | 978.60 | 979.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 983.60 | 979.60 | 980.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:45:00 | 984.70 | 979.60 | 980.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 981.00 | 979.88 | 980.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 985.00 | 979.88 | 980.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 980.10 | 979.92 | 980.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 980.40 | 979.92 | 980.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 982.10 | 980.36 | 980.30 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 968.65 | 978.48 | 979.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 960.35 | 974.85 | 977.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 959.00 | 958.66 | 967.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 953.60 | 956.49 | 963.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 953.60 | 956.49 | 963.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 956.20 | 956.49 | 963.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 965.75 | 958.08 | 962.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 965.75 | 958.08 | 962.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 963.85 | 959.23 | 963.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 953.65 | 959.23 | 963.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 960.15 | 960.71 | 962.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 958.70 | 955.16 | 958.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 952.20 | 955.19 | 958.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 951.80 | 952.08 | 955.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 947.25 | 951.00 | 954.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:15:00 | 948.20 | 950.06 | 953.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 948.00 | 949.65 | 952.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 948.05 | 949.33 | 952.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 966.55 | 952.59 | 953.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 966.55 | 952.59 | 953.36 | SL hit (close>static) qty=1.00 sl=956.80 alert=retest2 |

### Cycle 138 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 970.25 | 956.12 | 954.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 975.60 | 961.34 | 959.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 966.55 | 966.56 | 962.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 14:15:00 | 966.55 | 966.56 | 962.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 966.55 | 966.56 | 962.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 965.80 | 966.56 | 962.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 962.00 | 965.22 | 962.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 961.45 | 965.22 | 962.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 957.00 | 963.57 | 962.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 957.00 | 963.57 | 962.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 955.35 | 961.93 | 961.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 953.10 | 961.93 | 961.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 952.10 | 959.96 | 960.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 15:15:00 | 947.95 | 955.17 | 958.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 954.00 | 953.57 | 956.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 954.45 | 953.57 | 956.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 956.15 | 951.59 | 954.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:45:00 | 940.25 | 949.56 | 952.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 941.45 | 947.78 | 950.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 942.15 | 946.33 | 949.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 935.55 | 945.93 | 946.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 938.60 | 944.46 | 945.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:45:00 | 933.85 | 943.97 | 945.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 921.75 | 941.74 | 943.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 931.00 | 912.88 | 912.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 931.00 | 912.88 | 912.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 931.90 | 916.69 | 914.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 926.10 | 927.77 | 922.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 926.10 | 927.77 | 922.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 920.30 | 925.81 | 922.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 920.30 | 925.81 | 922.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 923.00 | 925.25 | 922.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 920.70 | 925.25 | 922.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 920.00 | 924.20 | 922.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:00:00 | 929.50 | 925.26 | 923.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 928.25 | 925.78 | 924.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 944.90 | 930.36 | 926.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 930.10 | 934.09 | 933.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 927.95 | 932.86 | 932.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 927.95 | 932.86 | 932.92 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 936.40 | 933.57 | 933.23 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 930.40 | 932.56 | 932.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 926.10 | 931.27 | 932.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 918.00 | 915.93 | 920.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 918.00 | 915.93 | 920.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 916.10 | 915.51 | 919.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 916.10 | 915.51 | 919.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 927.35 | 917.88 | 919.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 927.35 | 917.88 | 919.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 925.00 | 919.30 | 920.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 926.50 | 919.30 | 920.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 928.25 | 921.09 | 921.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 928.75 | 923.55 | 922.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 922.30 | 923.30 | 922.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 922.30 | 923.30 | 922.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 922.30 | 923.30 | 922.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 922.30 | 923.30 | 922.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 926.80 | 924.00 | 922.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 928.45 | 923.90 | 922.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 927.80 | 924.73 | 923.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 915.75 | 923.43 | 923.06 | SL hit (close<static) qty=1.00 sl=922.20 alert=retest2 |

### Cycle 145 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 916.05 | 921.95 | 922.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 913.95 | 918.59 | 920.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 915.40 | 914.44 | 916.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:30:00 | 916.60 | 914.44 | 916.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 913.70 | 914.38 | 916.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 901.70 | 910.52 | 913.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 906.45 | 909.95 | 913.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 909.60 | 910.36 | 913.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 922.90 | 912.87 | 914.09 | SL hit (close>static) qty=1.00 sl=916.95 alert=retest2 |

### Cycle 146 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 922.55 | 915.22 | 914.97 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 909.80 | 915.17 | 915.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 903.00 | 909.70 | 912.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 916.05 | 908.69 | 911.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 916.05 | 908.69 | 911.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 916.05 | 908.69 | 911.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 916.05 | 908.69 | 911.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 925.85 | 912.12 | 912.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 925.85 | 912.12 | 912.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 925.55 | 914.81 | 913.77 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 905.70 | 915.60 | 916.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 904.55 | 909.62 | 912.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 918.45 | 911.38 | 913.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 918.45 | 911.38 | 913.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 918.45 | 911.38 | 913.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 928.25 | 911.38 | 913.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 920.00 | 913.11 | 914.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 922.45 | 913.11 | 914.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 915.65 | 913.52 | 914.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 915.65 | 913.52 | 914.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 911.20 | 913.05 | 913.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 908.00 | 911.98 | 913.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 907.40 | 911.06 | 912.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 909.55 | 910.39 | 912.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 909.00 | 910.39 | 912.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 907.35 | 909.78 | 911.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 906.15 | 909.78 | 911.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 912.40 | 910.34 | 911.54 | SL hit (close>static) qty=1.00 sl=911.70 alert=retest2 |

### Cycle 150 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 919.60 | 912.44 | 912.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 924.45 | 914.85 | 913.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 962.20 | 962.57 | 948.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:45:00 | 963.20 | 962.57 | 948.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 950.00 | 959.15 | 951.25 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 928.70 | 944.79 | 946.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 925.80 | 940.99 | 944.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 937.50 | 934.08 | 937.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 937.50 | 934.08 | 937.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 937.50 | 934.08 | 937.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 939.10 | 934.08 | 937.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 948.20 | 936.91 | 938.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 948.20 | 936.91 | 938.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 948.50 | 939.22 | 939.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 949.05 | 939.22 | 939.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 944.45 | 940.27 | 939.92 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 929.45 | 938.50 | 939.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 923.35 | 931.85 | 935.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 930.50 | 927.39 | 930.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 930.50 | 927.39 | 930.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 930.50 | 927.39 | 930.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 921.55 | 925.40 | 929.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 914.90 | 923.30 | 928.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:45:00 | 920.00 | 923.23 | 927.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 922.05 | 923.23 | 927.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 925.05 | 923.02 | 926.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 925.05 | 923.02 | 926.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 925.00 | 923.41 | 926.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 922.65 | 923.41 | 926.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 921.20 | 917.16 | 920.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 922.45 | 919.74 | 920.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 925.20 | 919.66 | 919.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 925.20 | 919.66 | 919.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 927.00 | 922.87 | 920.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 913.30 | 920.96 | 920.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 913.30 | 920.96 | 920.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 913.30 | 920.96 | 920.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 913.30 | 920.96 | 920.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 909.25 | 918.62 | 919.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 908.40 | 914.36 | 916.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 905.30 | 900.03 | 903.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 905.30 | 900.03 | 903.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 905.30 | 900.03 | 903.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 907.35 | 900.03 | 903.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 907.65 | 901.56 | 903.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 908.70 | 901.56 | 903.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 909.10 | 903.06 | 904.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 911.40 | 903.06 | 904.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 903.95 | 903.62 | 904.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 903.95 | 903.62 | 904.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 903.00 | 903.50 | 904.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 903.80 | 903.50 | 904.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 924.40 | 907.68 | 906.08 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 905.00 | 908.49 | 908.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 899.45 | 905.71 | 907.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 896.00 | 889.23 | 895.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 896.00 | 889.23 | 895.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 896.00 | 889.23 | 895.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 896.00 | 889.23 | 895.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 895.00 | 890.38 | 895.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:45:00 | 890.00 | 890.48 | 894.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 890.00 | 892.10 | 895.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 906.00 | 894.88 | 896.00 | SL hit (close>static) qty=1.00 sl=899.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 901.00 | 897.09 | 896.87 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 889.70 | 895.61 | 896.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 888.25 | 894.14 | 895.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 870.70 | 856.37 | 868.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 870.70 | 856.37 | 868.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 870.70 | 856.37 | 868.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 870.70 | 856.37 | 868.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 864.05 | 857.90 | 868.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:00:00 | 863.05 | 861.51 | 868.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 862.55 | 861.72 | 867.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 854.10 | 862.37 | 867.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 859.15 | 853.47 | 855.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 864.40 | 855.66 | 856.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 863.45 | 855.66 | 856.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 865.05 | 857.54 | 857.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 865.05 | 857.54 | 857.09 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 852.75 | 858.53 | 858.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 849.20 | 855.87 | 857.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 856.60 | 855.64 | 856.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 11:15:00 | 856.60 | 855.64 | 856.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 856.60 | 855.64 | 856.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 856.05 | 855.64 | 856.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 853.35 | 855.18 | 856.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 855.35 | 855.18 | 856.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 854.45 | 855.04 | 856.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 854.50 | 855.04 | 856.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 854.85 | 855.00 | 856.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 854.85 | 855.00 | 856.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 856.00 | 855.20 | 856.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 851.05 | 855.20 | 856.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 859.70 | 856.10 | 856.54 | SL hit (close>static) qty=1.00 sl=858.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 828.55 | 824.23 | 823.89 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 818.55 | 823.43 | 823.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 815.50 | 820.84 | 822.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 14:15:00 | 821.35 | 820.30 | 821.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 15:00:00 | 821.35 | 820.30 | 821.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 825.65 | 817.45 | 818.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 825.45 | 817.45 | 818.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 825.45 | 819.05 | 819.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:15:00 | 821.95 | 819.05 | 819.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 820.95 | 819.74 | 819.61 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 818.35 | 819.50 | 819.53 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 820.00 | 819.60 | 819.57 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 814.20 | 818.52 | 819.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 809.75 | 814.44 | 816.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 801.90 | 801.76 | 805.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 09:15:00 | 801.80 | 801.76 | 805.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 792.20 | 799.85 | 804.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:45:00 | 789.50 | 796.70 | 801.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 784.00 | 800.36 | 800.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:30:00 | 788.10 | 795.48 | 797.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 819.65 | 802.23 | 800.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 819.65 | 802.23 | 800.50 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 796.70 | 801.21 | 801.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 792.70 | 799.51 | 800.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 797.85 | 797.17 | 799.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 797.85 | 797.17 | 799.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 797.85 | 797.17 | 799.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 795.95 | 797.17 | 799.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 790.90 | 795.92 | 798.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 803.80 | 795.92 | 798.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 796.60 | 794.96 | 796.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 797.80 | 794.96 | 796.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 799.50 | 795.87 | 797.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 801.00 | 795.87 | 797.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 799.65 | 796.63 | 797.32 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 801.25 | 797.97 | 797.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 809.85 | 801.56 | 799.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 802.90 | 806.17 | 803.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 802.90 | 806.17 | 803.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 802.90 | 806.17 | 803.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 800.35 | 806.17 | 803.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 805.00 | 805.94 | 803.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 801.75 | 805.94 | 803.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 803.50 | 805.45 | 803.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 805.55 | 805.45 | 803.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 808.45 | 806.05 | 804.11 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 802.15 | 804.25 | 804.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 800.00 | 803.38 | 804.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 801.60 | 801.56 | 802.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 801.60 | 801.56 | 802.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 801.60 | 801.56 | 802.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 801.60 | 801.56 | 802.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 820.05 | 805.25 | 804.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 821.70 | 808.54 | 805.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 825.00 | 825.38 | 819.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 833.25 | 825.38 | 819.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 829.85 | 826.43 | 822.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 828.00 | 826.43 | 822.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 825.15 | 825.30 | 822.79 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 818.75 | 822.32 | 822.33 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 825.60 | 822.98 | 822.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 10:15:00 | 829.85 | 824.35 | 823.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 821.40 | 824.42 | 823.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 12:15:00 | 821.40 | 824.42 | 823.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 821.40 | 824.42 | 823.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 821.40 | 824.42 | 823.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 823.70 | 824.27 | 823.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:30:00 | 821.30 | 824.27 | 823.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 820.90 | 823.60 | 823.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 820.90 | 823.60 | 823.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 819.10 | 822.70 | 822.93 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 823.80 | 823.10 | 823.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 827.25 | 824.39 | 823.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 838.50 | 842.27 | 837.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 838.50 | 842.27 | 837.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 838.50 | 842.27 | 837.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 838.50 | 842.27 | 837.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 838.40 | 841.50 | 837.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 841.20 | 841.31 | 837.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 844.00 | 841.15 | 838.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 838.85 | 844.07 | 844.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 838.85 | 844.07 | 844.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 832.85 | 840.22 | 842.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 806.50 | 803.40 | 809.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 806.50 | 803.40 | 809.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 813.95 | 806.16 | 808.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 813.95 | 806.16 | 808.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 814.00 | 807.73 | 809.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 812.95 | 807.73 | 809.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 814.05 | 810.62 | 810.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 823.95 | 813.29 | 811.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 815.65 | 825.80 | 820.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 11:15:00 | 815.65 | 825.80 | 820.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 815.65 | 825.80 | 820.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 815.65 | 825.80 | 820.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 816.50 | 823.94 | 820.53 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 812.90 | 818.35 | 818.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 807.55 | 816.19 | 817.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 13:15:00 | 816.00 | 812.73 | 815.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 816.00 | 812.73 | 815.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 816.00 | 812.73 | 815.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 816.00 | 812.73 | 815.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 820.15 | 814.21 | 815.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 820.15 | 814.21 | 815.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 813.90 | 814.15 | 815.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 807.05 | 814.15 | 815.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 766.70 | 789.63 | 795.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 12:15:00 | 791.25 | 788.80 | 793.42 | SL hit (close>ema200) qty=0.50 sl=788.80 alert=retest2 |

### Cycle 180 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 812.20 | 796.89 | 795.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 821.50 | 801.81 | 798.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 843.00 | 848.72 | 833.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:15:00 | 841.45 | 848.72 | 833.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 846.05 | 850.42 | 841.62 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 830.00 | 841.00 | 841.04 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 847.00 | 839.40 | 839.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 852.60 | 842.04 | 840.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 846.10 | 848.32 | 845.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 846.10 | 848.32 | 845.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 847.95 | 848.24 | 845.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 847.50 | 848.24 | 845.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 846.55 | 851.20 | 848.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 843.00 | 851.20 | 848.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 853.45 | 851.65 | 848.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:30:00 | 856.20 | 852.86 | 849.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 854.70 | 855.13 | 852.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:15:00 | 855.05 | 854.78 | 852.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 881.75 | 893.91 | 894.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 881.75 | 893.91 | 894.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 877.00 | 886.36 | 890.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 883.70 | 880.45 | 885.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 13:15:00 | 883.70 | 880.45 | 885.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 883.70 | 880.45 | 885.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 883.30 | 880.45 | 885.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 881.20 | 880.60 | 884.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 884.25 | 880.60 | 884.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 880.10 | 880.50 | 884.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 887.05 | 880.50 | 884.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 891.95 | 882.79 | 884.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 891.95 | 882.79 | 884.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 891.00 | 884.43 | 885.49 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 893.00 | 887.52 | 886.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 897.20 | 890.05 | 888.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 12:15:00 | 890.85 | 894.02 | 891.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 890.85 | 894.02 | 891.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 890.85 | 894.02 | 891.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 890.85 | 894.02 | 891.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 891.75 | 893.57 | 891.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 894.00 | 893.58 | 891.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 884.75 | 891.56 | 890.94 | SL hit (close<static) qty=1.00 sl=890.10 alert=retest2 |

### Cycle 185 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 883.00 | 889.85 | 890.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 873.75 | 885.38 | 888.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 848.00 | 836.94 | 843.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 848.00 | 836.94 | 843.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 848.00 | 836.94 | 843.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 848.00 | 836.94 | 843.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 854.70 | 840.49 | 844.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 856.45 | 840.49 | 844.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 855.50 | 848.10 | 847.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 862.25 | 850.93 | 848.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 844.60 | 850.75 | 849.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 844.60 | 850.75 | 849.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 844.60 | 850.75 | 849.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 844.60 | 850.75 | 849.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 851.60 | 850.92 | 849.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:15:00 | 853.80 | 850.89 | 849.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 844.15 | 851.46 | 850.74 | SL hit (close<static) qty=1.00 sl=844.50 alert=retest2 |

### Cycle 187 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 841.90 | 851.05 | 851.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 838.05 | 848.45 | 850.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 781.80 | 775.43 | 787.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 10:15:00 | 786.10 | 777.56 | 786.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 786.10 | 777.56 | 786.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 787.30 | 777.56 | 786.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 786.25 | 779.30 | 786.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 786.25 | 779.30 | 786.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 787.80 | 781.00 | 786.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:45:00 | 788.00 | 781.00 | 786.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 793.75 | 783.55 | 787.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:45:00 | 792.80 | 783.55 | 787.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 790.75 | 784.99 | 787.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 794.50 | 784.99 | 787.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 785.30 | 786.87 | 788.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 12:30:00 | 783.50 | 787.41 | 788.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 794.85 | 788.89 | 788.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 794.85 | 788.89 | 788.81 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 785.85 | 788.29 | 788.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 776.55 | 786.06 | 787.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 786.10 | 786.07 | 787.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 786.10 | 786.07 | 787.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 788.10 | 786.47 | 787.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 788.10 | 786.47 | 787.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 787.40 | 786.66 | 787.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 789.85 | 786.66 | 787.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 788.45 | 787.02 | 787.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 789.70 | 787.02 | 787.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 780.00 | 785.61 | 786.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:15:00 | 778.00 | 785.61 | 786.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 798.05 | 767.99 | 770.66 | SL hit (close>static) qty=1.00 sl=790.60 alert=retest2 |

### Cycle 190 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 780.00 | 773.53 | 772.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 788.10 | 778.78 | 775.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 783.30 | 785.51 | 782.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 783.30 | 785.51 | 782.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 784.90 | 785.39 | 782.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 784.90 | 785.39 | 782.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 787.80 | 785.52 | 783.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 787.80 | 785.52 | 783.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 776.60 | 788.65 | 786.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:15:00 | 775.00 | 788.65 | 786.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 776.35 | 786.19 | 785.74 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 776.35 | 784.22 | 784.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 771.95 | 778.92 | 781.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 789.50 | 780.15 | 781.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 789.50 | 780.15 | 781.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 789.50 | 780.15 | 781.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 789.50 | 780.15 | 781.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 789.00 | 781.92 | 782.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 791.75 | 781.92 | 782.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 787.45 | 783.02 | 782.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 809.60 | 788.65 | 785.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 794.20 | 798.60 | 792.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 794.20 | 798.60 | 792.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 794.20 | 798.60 | 792.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 794.20 | 798.60 | 792.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 791.20 | 797.12 | 792.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 778.90 | 797.12 | 792.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 773.50 | 792.40 | 790.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 773.50 | 792.40 | 790.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 772.45 | 788.41 | 789.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 770.20 | 784.77 | 787.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 776.35 | 775.33 | 781.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 09:45:00 | 776.20 | 775.33 | 781.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 787.85 | 778.08 | 781.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:30:00 | 790.60 | 778.08 | 781.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 785.60 | 779.59 | 781.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:15:00 | 778.60 | 779.59 | 781.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 793.25 | 783.69 | 782.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 793.25 | 783.69 | 782.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 798.55 | 786.66 | 783.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 770.00 | 787.23 | 785.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 770.00 | 787.23 | 785.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 770.00 | 787.23 | 785.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 770.00 | 787.23 | 785.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 773.00 | 784.39 | 784.41 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 789.00 | 785.22 | 784.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 793.90 | 786.96 | 785.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 778.00 | 785.48 | 785.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 778.00 | 785.48 | 785.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 778.00 | 785.48 | 785.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 778.00 | 785.48 | 785.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 781.35 | 784.65 | 784.82 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 789.30 | 785.66 | 785.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 792.70 | 787.07 | 785.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 831.85 | 832.68 | 821.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 831.85 | 832.68 | 821.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 828.00 | 833.28 | 827.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 839.10 | 833.28 | 827.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 823.80 | 834.49 | 831.79 | SL hit (close<static) qty=1.00 sl=826.70 alert=retest2 |

### Cycle 199 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 851.65 | 856.94 | 857.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 850.00 | 855.55 | 856.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 11:15:00 | 848.15 | 822.05 | 828.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 11:15:00 | 848.15 | 822.05 | 828.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 848.15 | 822.05 | 828.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:00:00 | 848.15 | 822.05 | 828.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 831.00 | 823.84 | 828.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:15:00 | 829.00 | 823.84 | 828.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:00:00 | 823.30 | 821.95 | 825.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 882.70 | 833.51 | 829.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 882.70 | 833.51 | 829.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 900.95 | 885.62 | 872.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 884.00 | 887.83 | 877.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:30:00 | 879.80 | 887.83 | 877.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 873.45 | 883.87 | 877.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 884.50 | 883.38 | 878.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 865.00 | 878.26 | 876.75 | SL hit (close<static) qty=1.00 sl=866.40 alert=retest2 |

### Cycle 201 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 870.00 | 875.16 | 875.58 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 888.10 | 877.75 | 876.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 896.00 | 881.40 | 878.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 889.40 | 894.75 | 888.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 889.40 | 894.75 | 888.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 889.40 | 894.75 | 888.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 889.40 | 894.75 | 888.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 890.25 | 893.85 | 888.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 890.25 | 893.85 | 888.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 885.40 | 892.16 | 888.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 885.40 | 892.16 | 888.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 890.40 | 891.81 | 888.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 895.65 | 892.61 | 889.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:45:00 | 893.75 | 893.72 | 890.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:15:00 | 894.50 | 893.68 | 890.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 894.10 | 893.78 | 891.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 887.70 | 892.46 | 891.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 895.10 | 892.46 | 891.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 894.15 | 892.80 | 891.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 883.00 | 890.84 | 890.61 | SL hit (close<static) qty=1.00 sl=885.40 alert=retest2 |

### Cycle 203 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 882.15 | 889.10 | 889.84 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 923.65 | 892.35 | 890.30 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 625.80 | 2024-04-12 13:15:00 | 619.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-04-12 09:45:00 | 626.20 | 2024-04-12 13:15:00 | 619.85 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-04-12 10:45:00 | 624.95 | 2024-04-12 13:15:00 | 619.85 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-04-19 09:15:00 | 598.00 | 2024-04-22 14:15:00 | 607.90 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-04-19 12:45:00 | 598.15 | 2024-04-22 14:15:00 | 607.90 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-04-29 12:00:00 | 637.45 | 2024-04-30 10:15:00 | 627.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-05-09 10:15:00 | 580.10 | 2024-05-14 09:15:00 | 588.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-05-10 14:00:00 | 581.30 | 2024-05-14 09:15:00 | 588.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-10 15:15:00 | 582.00 | 2024-05-14 09:15:00 | 588.20 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-05-13 15:00:00 | 580.90 | 2024-05-14 09:15:00 | 588.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-05-22 10:00:00 | 640.00 | 2024-05-27 09:15:00 | 629.80 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-05-22 11:00:00 | 638.90 | 2024-05-27 09:15:00 | 629.80 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-05-30 09:45:00 | 620.20 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-05-30 11:00:00 | 621.30 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-05-30 14:45:00 | 621.85 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-05-31 09:15:00 | 621.00 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-05-31 10:15:00 | 609.45 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-05-31 14:15:00 | 608.90 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-05-31 15:15:00 | 608.00 | 2024-06-03 13:15:00 | 625.20 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-06-25 11:30:00 | 655.45 | 2024-06-25 15:15:00 | 664.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-06-25 15:00:00 | 658.10 | 2024-06-25 15:15:00 | 664.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-06-26 09:30:00 | 650.95 | 2024-06-26 14:15:00 | 665.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-07-05 12:00:00 | 738.50 | 2024-07-08 09:15:00 | 714.05 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2024-07-11 15:15:00 | 687.30 | 2024-07-15 13:15:00 | 693.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-07-12 10:00:00 | 689.60 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-07-12 12:15:00 | 690.05 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-07-12 14:00:00 | 689.75 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-07-15 12:15:00 | 690.15 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-07-15 14:30:00 | 689.85 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-07-16 10:15:00 | 688.45 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-07-16 11:00:00 | 689.75 | 2024-07-16 14:15:00 | 694.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-07-22 09:15:00 | 656.00 | 2024-07-22 13:15:00 | 675.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest1 | 2024-07-22 12:30:00 | 661.55 | 2024-07-22 13:15:00 | 675.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-07-23 09:15:00 | 665.10 | 2024-07-23 13:15:00 | 681.05 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-07-23 11:00:00 | 667.00 | 2024-07-23 13:15:00 | 681.05 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-07-23 12:15:00 | 665.00 | 2024-07-23 13:15:00 | 681.05 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-07-23 13:00:00 | 666.00 | 2024-07-23 13:15:00 | 681.05 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-07-25 12:30:00 | 685.75 | 2024-08-01 09:15:00 | 754.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 13:45:00 | 685.05 | 2024-08-01 09:15:00 | 753.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 09:15:00 | 694.35 | 2024-08-01 09:15:00 | 753.78 | TARGET_HIT | 1.00 | 8.56% |
| BUY | retest2 | 2024-07-26 12:00:00 | 685.25 | 2024-08-01 10:15:00 | 763.79 | TARGET_HIT | 1.00 | 11.46% |
| BUY | retest1 | 2024-08-09 09:15:00 | 742.50 | 2024-08-13 15:15:00 | 738.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-12 14:30:00 | 745.80 | 2024-08-14 10:15:00 | 725.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-08-12 15:00:00 | 745.30 | 2024-08-14 10:15:00 | 725.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-08-13 09:15:00 | 750.65 | 2024-08-14 10:15:00 | 725.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2024-08-13 11:00:00 | 751.00 | 2024-08-14 10:15:00 | 725.00 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-09-25 11:15:00 | 933.95 | 2024-09-26 13:15:00 | 917.75 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-09-25 12:00:00 | 929.85 | 2024-09-26 13:15:00 | 917.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-09-26 10:00:00 | 930.25 | 2024-09-26 13:15:00 | 917.75 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-10-01 10:30:00 | 895.70 | 2024-10-04 09:15:00 | 914.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-04 09:15:00 | 895.80 | 2024-10-04 09:15:00 | 914.50 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-10-08 09:15:00 | 891.95 | 2024-10-08 11:15:00 | 905.30 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-10-08 12:45:00 | 896.00 | 2024-10-09 09:15:00 | 922.25 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-10-08 13:45:00 | 896.05 | 2024-10-09 09:15:00 | 922.25 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-10-29 10:30:00 | 792.15 | 2024-10-29 11:15:00 | 818.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-11-04 15:15:00 | 840.00 | 2024-11-08 10:15:00 | 841.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-11-12 15:15:00 | 799.95 | 2024-11-18 09:15:00 | 820.15 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-11-14 10:00:00 | 803.00 | 2024-11-18 09:15:00 | 820.15 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-12-10 09:15:00 | 852.05 | 2024-12-12 10:15:00 | 842.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-20 10:15:00 | 787.85 | 2024-12-23 09:15:00 | 748.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:45:00 | 787.90 | 2024-12-23 09:15:00 | 748.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 10:15:00 | 787.85 | 2024-12-24 13:15:00 | 753.85 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2024-12-20 12:45:00 | 787.90 | 2024-12-24 13:15:00 | 753.85 | STOP_HIT | 0.50 | 4.32% |
| BUY | retest2 | 2025-01-07 09:15:00 | 770.15 | 2025-01-13 09:15:00 | 754.35 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-01-07 11:15:00 | 769.95 | 2025-01-13 09:15:00 | 754.35 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-01-31 12:00:00 | 776.95 | 2025-02-03 12:15:00 | 760.45 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-01-31 13:00:00 | 778.00 | 2025-02-03 12:15:00 | 760.45 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-02-01 14:00:00 | 776.40 | 2025-02-03 12:15:00 | 760.45 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-02-01 14:45:00 | 779.35 | 2025-02-03 12:15:00 | 760.45 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-02-03 11:15:00 | 758.00 | 2025-02-03 12:15:00 | 760.45 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-02-03 12:00:00 | 758.10 | 2025-02-03 12:15:00 | 760.45 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-02-07 11:30:00 | 784.15 | 2025-02-10 09:15:00 | 759.45 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-02-07 14:30:00 | 782.10 | 2025-02-10 09:15:00 | 759.45 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-02-14 10:00:00 | 718.35 | 2025-02-19 14:15:00 | 682.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 719.30 | 2025-02-19 14:15:00 | 683.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 09:45:00 | 720.75 | 2025-02-19 14:15:00 | 684.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 15:00:00 | 722.40 | 2025-02-19 14:15:00 | 686.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:00:00 | 718.35 | 2025-02-20 10:15:00 | 704.80 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-02-17 09:15:00 | 719.30 | 2025-02-20 10:15:00 | 704.80 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2025-02-17 09:45:00 | 720.75 | 2025-02-20 10:15:00 | 704.80 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2025-02-17 15:00:00 | 722.40 | 2025-02-20 10:15:00 | 704.80 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-02-18 10:15:00 | 701.25 | 2025-02-21 12:15:00 | 720.10 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-02-18 12:30:00 | 701.55 | 2025-02-21 12:15:00 | 720.10 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-02-19 13:00:00 | 699.40 | 2025-02-21 12:15:00 | 720.10 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-03-11 15:15:00 | 808.00 | 2025-03-24 09:15:00 | 888.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-12 12:00:00 | 812.00 | 2025-03-24 09:15:00 | 893.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:15:00 | 807.25 | 2025-03-24 09:15:00 | 887.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 14:00:00 | 811.20 | 2025-03-24 09:15:00 | 892.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 11:30:00 | 830.00 | 2025-03-25 12:15:00 | 853.70 | STOP_HIT | 1.00 | 2.86% |
| SELL | retest2 | 2025-04-08 11:30:00 | 813.85 | 2025-04-11 09:15:00 | 845.05 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-04-09 10:00:00 | 809.60 | 2025-04-11 09:15:00 | 845.05 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest1 | 2025-04-17 15:15:00 | 903.00 | 2025-04-23 09:15:00 | 894.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2025-04-21 13:00:00 | 900.15 | 2025-04-23 09:15:00 | 894.30 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-04-24 11:00:00 | 908.95 | 2025-04-25 09:15:00 | 875.95 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-04-30 14:45:00 | 870.70 | 2025-05-05 11:15:00 | 883.65 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-05-22 14:15:00 | 897.20 | 2025-05-30 09:15:00 | 852.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 09:15:00 | 887.45 | 2025-05-30 11:15:00 | 843.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 14:15:00 | 897.20 | 2025-05-30 15:15:00 | 850.65 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2025-05-26 09:15:00 | 887.45 | 2025-05-30 15:15:00 | 850.65 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2025-06-04 15:15:00 | 848.95 | 2025-06-05 09:15:00 | 861.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-06-09 09:45:00 | 874.60 | 2025-06-11 11:15:00 | 865.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-09 12:00:00 | 875.45 | 2025-06-11 11:15:00 | 865.05 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-09 14:45:00 | 876.90 | 2025-06-11 11:15:00 | 865.05 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-06-17 12:15:00 | 839.60 | 2025-06-23 14:15:00 | 831.95 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-06-17 14:30:00 | 840.10 | 2025-06-23 14:15:00 | 831.95 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-06-25 11:30:00 | 850.10 | 2025-07-07 09:15:00 | 869.95 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-06-26 09:15:00 | 845.50 | 2025-07-07 09:15:00 | 869.95 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-06-26 12:00:00 | 846.55 | 2025-07-07 09:15:00 | 869.95 | STOP_HIT | 1.00 | 2.76% |
| SELL | retest2 | 2025-07-08 09:30:00 | 866.40 | 2025-07-15 10:15:00 | 864.05 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-07-08 15:15:00 | 867.10 | 2025-07-15 10:15:00 | 864.05 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-07-09 09:45:00 | 866.50 | 2025-07-15 10:15:00 | 864.05 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-07-21 09:15:00 | 921.55 | 2025-08-04 15:15:00 | 975.90 | STOP_HIT | 1.00 | 5.90% |
| SELL | retest2 | 2025-08-08 09:15:00 | 953.65 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-08 12:15:00 | 960.15 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-08-11 09:30:00 | 958.70 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-11 11:30:00 | 952.20 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-12 09:30:00 | 947.25 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-08-12 13:15:00 | 948.20 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-12 14:00:00 | 948.00 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-08-12 15:00:00 | 948.05 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-08-21 12:45:00 | 940.25 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-08-22 09:30:00 | 941.45 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-08-22 10:30:00 | 942.15 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-08-26 09:15:00 | 935.55 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-08-26 10:45:00 | 933.85 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-28 09:15:00 | 921.75 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-05 11:00:00 | 929.50 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-09-05 15:15:00 | 928.25 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-09-08 09:30:00 | 944.90 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-09-09 14:45:00 | 930.10 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-16 13:15:00 | 928.45 | 2025-09-17 09:15:00 | 915.75 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-09-16 15:15:00 | 927.80 | 2025-09-17 09:15:00 | 915.75 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-19 14:00:00 | 901.70 | 2025-09-22 09:15:00 | 922.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-09-19 14:30:00 | 906.45 | 2025-09-22 09:15:00 | 922.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-22 09:15:00 | 909.60 | 2025-09-22 09:15:00 | 922.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-30 09:15:00 | 908.00 | 2025-09-30 14:15:00 | 912.40 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-30 10:00:00 | 907.40 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-30 11:30:00 | 909.55 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-09-30 12:15:00 | 909.00 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-30 13:15:00 | 906.15 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-14 10:30:00 | 921.55 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-14 12:00:00 | 914.90 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-14 12:45:00 | 920.00 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-14 13:15:00 | 922.05 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-15 09:15:00 | 922.65 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-16 09:45:00 | 921.20 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-16 13:45:00 | 922.45 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-11-03 13:45:00 | 890.00 | 2025-11-03 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-03 14:45:00 | 890.00 | 2025-11-03 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-07 14:00:00 | 863.05 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-11-07 15:00:00 | 862.55 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-10 09:15:00 | 854.10 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-12 12:00:00 | 859.15 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-11-17 09:15:00 | 851.05 | 2025-11-17 09:15:00 | 859.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-11-18 09:15:00 | 851.20 | 2025-11-24 14:15:00 | 808.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 09:30:00 | 847.15 | 2025-11-24 15:15:00 | 804.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 851.20 | 2025-11-25 14:15:00 | 817.95 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-11-24 09:30:00 | 847.15 | 2025-11-25 14:15:00 | 817.95 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-12-05 11:45:00 | 789.50 | 2025-12-09 14:15:00 | 819.65 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-12-09 09:15:00 | 784.00 | 2025-12-09 14:15:00 | 819.65 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2025-12-09 12:30:00 | 788.10 | 2025-12-09 14:15:00 | 819.65 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-01-01 14:15:00 | 841.20 | 2026-01-06 11:15:00 | 838.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-01-01 15:15:00 | 844.00 | 2026-01-06 11:15:00 | 838.85 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-20 09:15:00 | 807.05 | 2026-01-27 09:15:00 | 766.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 807.05 | 2026-01-27 12:15:00 | 791.25 | STOP_HIT | 0.50 | 1.96% |
| BUY | retest2 | 2026-02-05 11:30:00 | 856.20 | 2026-02-13 10:15:00 | 881.75 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest2 | 2026-02-06 11:30:00 | 854.70 | 2026-02-13 10:15:00 | 881.75 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2026-02-06 13:15:00 | 855.05 | 2026-02-13 10:15:00 | 881.75 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2026-02-18 14:45:00 | 894.00 | 2026-02-19 09:15:00 | 884.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-26 13:15:00 | 853.80 | 2026-02-27 10:15:00 | 844.15 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-02-27 12:15:00 | 857.10 | 2026-03-02 10:15:00 | 843.25 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-03-02 09:30:00 | 853.90 | 2026-03-02 10:15:00 | 843.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-03-11 12:30:00 | 783.50 | 2026-03-11 13:15:00 | 794.85 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-03-12 15:15:00 | 778.00 | 2026-03-17 09:15:00 | 798.05 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-03-17 11:00:00 | 779.60 | 2026-03-17 12:15:00 | 780.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-03-17 11:30:00 | 778.45 | 2026-03-17 12:15:00 | 780.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-03-30 13:15:00 | 778.60 | 2026-04-01 11:15:00 | 793.25 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-04-10 09:15:00 | 839.10 | 2026-04-13 09:15:00 | 823.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-13 10:45:00 | 834.00 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2026-04-13 11:15:00 | 833.45 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2026-04-13 11:45:00 | 833.25 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2026-04-15 09:15:00 | 848.50 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-04-23 13:15:00 | 829.00 | 2026-04-27 09:15:00 | 882.70 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest2 | 2026-04-24 13:00:00 | 823.30 | 2026-04-27 09:15:00 | 882.70 | STOP_HIT | 1.00 | -7.21% |
| BUY | retest2 | 2026-04-30 11:15:00 | 884.50 | 2026-04-30 12:15:00 | 865.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-05-06 09:30:00 | 895.65 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-05-06 11:45:00 | 893.75 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-06 13:15:00 | 894.50 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-05-06 13:45:00 | 894.10 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-05-07 09:15:00 | 895.10 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-05-07 10:00:00 | 894.15 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-05-07 10:30:00 | 894.70 | 2026-05-07 11:15:00 | 882.15 | STOP_HIT | 1.00 | -1.40% |

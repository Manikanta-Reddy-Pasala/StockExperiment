# Ramkrishna Forgings Ltd. (RKFORGE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 607.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 143 |
| ALERT2 | 139 |
| ALERT2_SKIP | 93 |
| ALERT3 | 277 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 147 |
| PARTIAL | 16 |
| TARGET_HIT | 11 |
| STOP_HIT | 137 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 164 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 108
- **Target hits / Stop hits / Partials:** 11 / 137 / 16
- **Avg / median % per leg:** 0.13% / -1.01%
- **Sum % (uncompounded):** 20.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 23 | 31.5% | 5 | 67 | 1 | -0.18% | -13.1% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 71 | 21 | 29.6% | 4 | 67 | 0 | -0.40% | -28.1% |
| SELL (all) | 91 | 33 | 36.3% | 6 | 70 | 15 | 0.37% | 34.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 91 | 33 | 36.3% | 6 | 70 | 15 | 0.37% | 34.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 162 | 54 | 33.3% | 10 | 137 | 15 | 0.04% | 5.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 345.55 | 343.06 | 342.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 14:15:00 | 347.15 | 344.67 | 343.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 15:15:00 | 345.50 | 347.15 | 345.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 15:15:00 | 345.50 | 347.15 | 345.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 345.50 | 347.15 | 345.86 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 13:15:00 | 343.85 | 345.48 | 345.52 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 15:15:00 | 347.35 | 345.80 | 345.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 09:15:00 | 351.70 | 346.98 | 346.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 357.00 | 358.15 | 353.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 13:15:00 | 359.25 | 363.27 | 360.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 13:15:00 | 359.25 | 363.27 | 360.67 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 11:15:00 | 356.05 | 359.48 | 359.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 12:15:00 | 354.00 | 358.38 | 359.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 09:15:00 | 361.35 | 357.60 | 358.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 361.35 | 357.60 | 358.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 361.35 | 357.60 | 358.33 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 12:15:00 | 360.40 | 359.00 | 358.86 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 14:15:00 | 356.95 | 358.54 | 358.67 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 368.20 | 360.41 | 359.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 10:15:00 | 372.25 | 362.78 | 360.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 14:15:00 | 365.00 | 365.30 | 362.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 12:15:00 | 384.15 | 389.82 | 386.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 384.15 | 389.82 | 386.62 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 09:15:00 | 378.35 | 383.97 | 384.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 10:15:00 | 374.95 | 379.97 | 381.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 10:15:00 | 378.50 | 375.93 | 378.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 10:15:00 | 378.50 | 375.93 | 378.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 378.50 | 375.93 | 378.48 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 11:15:00 | 382.70 | 379.66 | 379.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 09:15:00 | 390.90 | 383.53 | 381.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 13:15:00 | 387.85 | 387.85 | 384.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 13:15:00 | 386.50 | 388.62 | 386.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 386.50 | 388.62 | 386.87 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 436.85 | 455.10 | 455.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 14:15:00 | 425.80 | 441.62 | 447.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 441.00 | 439.55 | 445.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 451.65 | 443.56 | 445.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 451.65 | 443.56 | 445.49 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 463.70 | 448.76 | 447.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 10:15:00 | 467.95 | 452.60 | 449.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 11:15:00 | 457.60 | 459.77 | 455.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 13:15:00 | 454.25 | 458.47 | 455.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 454.25 | 458.47 | 455.80 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 12:15:00 | 446.85 | 453.33 | 454.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 13:15:00 | 438.70 | 450.41 | 452.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 09:15:00 | 447.85 | 442.64 | 445.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 447.85 | 442.64 | 445.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 447.85 | 442.64 | 445.82 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 12:15:00 | 446.80 | 446.63 | 446.62 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 13:15:00 | 445.00 | 446.30 | 446.47 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 448.95 | 446.79 | 446.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 11:15:00 | 449.90 | 447.41 | 446.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 15:15:00 | 446.60 | 447.64 | 447.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 15:15:00 | 446.60 | 447.64 | 447.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 446.60 | 447.64 | 447.20 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 443.65 | 446.90 | 446.94 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 11:15:00 | 448.70 | 447.26 | 447.10 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 444.50 | 446.75 | 446.90 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 14:15:00 | 448.05 | 447.01 | 447.00 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 446.00 | 446.81 | 446.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 444.90 | 446.42 | 446.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 445.35 | 442.72 | 444.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 445.35 | 442.72 | 444.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 445.35 | 442.72 | 444.27 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 10:15:00 | 447.55 | 445.08 | 444.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 11:15:00 | 448.45 | 445.75 | 445.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 445.00 | 456.14 | 452.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 445.00 | 456.14 | 452.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 445.00 | 456.14 | 452.89 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 14:15:00 | 541.25 | 549.14 | 549.27 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 557.75 | 550.57 | 549.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 11:15:00 | 562.00 | 552.85 | 550.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 13:15:00 | 557.30 | 560.61 | 557.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 13:15:00 | 557.30 | 560.61 | 557.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 557.30 | 560.61 | 557.41 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 548.00 | 557.13 | 557.29 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 562.85 | 556.77 | 556.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 10:15:00 | 567.65 | 558.95 | 557.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 13:15:00 | 560.95 | 561.17 | 558.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 595.70 | 588.80 | 582.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 595.70 | 588.80 | 582.24 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 14:15:00 | 569.90 | 579.31 | 579.86 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 09:15:00 | 589.00 | 579.43 | 578.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 10:15:00 | 593.60 | 586.86 | 583.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 10:15:00 | 605.75 | 608.83 | 601.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 10:15:00 | 605.75 | 608.83 | 601.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 605.75 | 608.83 | 601.67 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 653.80 | 658.89 | 659.13 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 13:15:00 | 659.75 | 658.40 | 658.39 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 14:15:00 | 656.95 | 658.11 | 658.26 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 15:15:00 | 661.00 | 658.69 | 658.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 10:15:00 | 668.80 | 661.06 | 659.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 14:15:00 | 713.45 | 713.55 | 696.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 714.55 | 716.24 | 710.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 714.55 | 716.24 | 710.71 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 09:15:00 | 711.00 | 712.49 | 712.57 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 11:15:00 | 714.10 | 712.79 | 712.69 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 12:15:00 | 710.00 | 712.24 | 712.45 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 714.60 | 712.71 | 712.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 720.95 | 714.51 | 713.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 15:15:00 | 717.60 | 718.27 | 716.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 697.20 | 720.34 | 719.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 697.20 | 720.34 | 719.98 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 689.05 | 714.08 | 717.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 660.30 | 698.16 | 709.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 680.10 | 671.56 | 686.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 12:15:00 | 680.10 | 671.56 | 686.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 680.10 | 671.56 | 686.69 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 712.00 | 691.41 | 691.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 11:15:00 | 713.50 | 695.82 | 693.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 09:15:00 | 695.40 | 705.37 | 700.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 695.40 | 705.37 | 700.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 695.40 | 705.37 | 700.16 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 14:15:00 | 681.55 | 697.52 | 698.01 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 09:15:00 | 704.50 | 699.31 | 698.76 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 693.20 | 697.82 | 698.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 678.95 | 692.64 | 695.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 11:15:00 | 651.35 | 645.47 | 657.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 642.55 | 639.11 | 644.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 642.55 | 639.11 | 644.60 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 647.95 | 642.50 | 642.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 657.45 | 645.49 | 643.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 646.25 | 647.19 | 644.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 643.20 | 646.39 | 644.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 643.20 | 646.39 | 644.75 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 634.20 | 643.57 | 644.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 632.05 | 638.26 | 640.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 14:15:00 | 631.60 | 629.95 | 633.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 645.00 | 632.81 | 634.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 645.00 | 632.81 | 634.13 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 645.05 | 635.26 | 635.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 12:15:00 | 651.60 | 640.25 | 637.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 641.20 | 644.91 | 641.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 641.20 | 644.91 | 641.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 641.20 | 644.91 | 641.01 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 631.00 | 639.39 | 639.49 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 644.15 | 638.62 | 638.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 14:15:00 | 645.00 | 639.90 | 639.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 15:15:00 | 645.75 | 647.64 | 644.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 652.90 | 648.69 | 645.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 652.90 | 648.69 | 645.43 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 14:15:00 | 637.15 | 670.94 | 672.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 623.00 | 649.04 | 658.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 11:15:00 | 630.00 | 627.38 | 638.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 14:15:00 | 635.25 | 626.77 | 635.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 635.25 | 626.77 | 635.14 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 634.25 | 622.26 | 621.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 11:15:00 | 635.95 | 625.00 | 623.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 689.00 | 689.74 | 684.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 699.25 | 691.64 | 686.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 699.25 | 691.64 | 686.09 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 758.15 | 760.30 | 760.34 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 773.00 | 762.04 | 761.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 10:15:00 | 778.05 | 765.24 | 762.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 792.35 | 800.95 | 793.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 14:15:00 | 792.35 | 800.95 | 793.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 792.35 | 800.95 | 793.83 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 784.00 | 791.04 | 791.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 781.15 | 785.11 | 787.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 753.70 | 739.33 | 747.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 753.70 | 739.33 | 747.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 753.70 | 739.33 | 747.55 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 756.80 | 746.50 | 745.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 765.95 | 758.32 | 756.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 11:15:00 | 757.85 | 759.18 | 757.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 11:15:00 | 757.85 | 759.18 | 757.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 757.85 | 759.18 | 757.26 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 729.60 | 752.16 | 754.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 713.45 | 744.41 | 750.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 735.50 | 727.90 | 735.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 735.50 | 727.90 | 735.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 735.50 | 727.90 | 735.34 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 13:15:00 | 727.75 | 720.26 | 720.08 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 11:15:00 | 715.30 | 719.80 | 720.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 12:15:00 | 712.80 | 718.40 | 719.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 10:15:00 | 710.80 | 707.49 | 710.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 10:15:00 | 710.80 | 707.49 | 710.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 710.80 | 707.49 | 710.94 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 15:15:00 | 717.60 | 712.07 | 711.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 731.15 | 715.88 | 713.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 734.55 | 736.43 | 729.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 14:15:00 | 738.55 | 736.85 | 730.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 738.55 | 736.85 | 730.04 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 776.65 | 783.20 | 783.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 12:15:00 | 775.20 | 780.42 | 781.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 10:15:00 | 781.20 | 779.47 | 780.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 10:15:00 | 781.20 | 779.47 | 780.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 781.20 | 779.47 | 780.73 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 768.80 | 765.17 | 765.10 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 10:15:00 | 763.00 | 764.85 | 764.98 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 11:15:00 | 766.00 | 765.08 | 765.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 12:15:00 | 769.15 | 765.90 | 765.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 13:15:00 | 764.80 | 765.68 | 765.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 13:15:00 | 764.80 | 765.68 | 765.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 764.80 | 765.68 | 765.38 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 761.45 | 764.60 | 764.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 758.70 | 763.42 | 764.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 741.45 | 737.21 | 743.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 741.45 | 737.21 | 743.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 741.45 | 737.21 | 743.92 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 14:15:00 | 747.30 | 743.52 | 743.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 10:15:00 | 758.70 | 747.59 | 745.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 760.90 | 761.95 | 757.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 13:15:00 | 767.50 | 762.88 | 758.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 767.50 | 762.88 | 758.49 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 15:15:00 | 755.95 | 764.22 | 764.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 15:15:00 | 742.00 | 751.91 | 757.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 15:15:00 | 742.85 | 740.47 | 747.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 703.65 | 710.14 | 721.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 703.65 | 710.14 | 721.88 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 729.95 | 716.93 | 715.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 15:15:00 | 734.80 | 720.50 | 717.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 14:15:00 | 726.25 | 728.69 | 724.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 14:15:00 | 726.25 | 728.69 | 724.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 726.25 | 728.69 | 724.02 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 12:15:00 | 722.85 | 729.15 | 729.84 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 741.30 | 729.44 | 729.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 11:15:00 | 748.30 | 733.21 | 731.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 757.55 | 761.03 | 754.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 14:15:00 | 758.65 | 760.55 | 754.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 758.65 | 760.55 | 754.50 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 740.20 | 749.99 | 751.01 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 764.00 | 751.20 | 750.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 768.05 | 756.12 | 753.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 763.00 | 763.72 | 758.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 767.30 | 764.55 | 760.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 767.30 | 764.55 | 760.06 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 749.65 | 764.55 | 766.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 743.90 | 760.42 | 764.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 783.40 | 764.17 | 764.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 11:15:00 | 783.40 | 764.17 | 764.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 783.40 | 764.17 | 764.95 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 770.00 | 766.27 | 765.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 780.75 | 769.16 | 767.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 11:15:00 | 771.80 | 774.65 | 770.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 11:15:00 | 771.80 | 774.65 | 770.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 771.80 | 774.65 | 770.98 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 756.80 | 769.74 | 771.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 744.65 | 759.89 | 765.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 728.75 | 726.92 | 738.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 728.75 | 726.92 | 738.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 728.75 | 726.92 | 738.07 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 12:15:00 | 640.80 | 633.87 | 633.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 13:15:00 | 645.75 | 636.24 | 634.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 15:15:00 | 663.75 | 665.90 | 657.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 670.25 | 675.88 | 668.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 670.25 | 675.88 | 668.61 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 15:15:00 | 656.00 | 665.71 | 666.18 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 675.45 | 667.39 | 666.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 13:15:00 | 684.05 | 671.62 | 668.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 11:15:00 | 709.30 | 709.36 | 700.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 715.00 | 712.98 | 705.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 715.00 | 712.98 | 705.83 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 09:15:00 | 701.90 | 707.24 | 707.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 11:15:00 | 696.55 | 703.79 | 705.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 09:15:00 | 701.00 | 698.42 | 701.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 09:15:00 | 701.00 | 698.42 | 701.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 701.00 | 698.42 | 701.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 694.30 | 694.90 | 696.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 14:30:00 | 694.90 | 694.59 | 695.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:00:00 | 688.90 | 694.59 | 695.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 659.58 | 694.81 | 695.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 702.00 | 694.81 | 695.58 | SL hit (close>static) qty=0.50 sl=694.81 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 10:15:00 | 705.95 | 697.04 | 696.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 12:15:00 | 715.60 | 702.33 | 699.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 733.05 | 734.27 | 723.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 730.50 | 732.75 | 725.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 730.50 | 732.75 | 725.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 13:15:00 | 742.80 | 733.22 | 727.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 10:15:00 | 741.00 | 748.54 | 743.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 10:45:00 | 739.50 | 746.66 | 743.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 749.55 | 761.62 | 762.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 749.55 | 761.62 | 762.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 746.45 | 754.66 | 757.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 14:15:00 | 724.65 | 713.79 | 723.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 14:15:00 | 724.65 | 713.79 | 723.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 724.65 | 713.79 | 723.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 15:00:00 | 724.65 | 713.79 | 723.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 719.65 | 714.96 | 723.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 710.00 | 714.96 | 723.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 674.50 | 689.65 | 700.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 13:15:00 | 687.95 | 685.88 | 694.91 | SL hit (close>ema200) qty=0.50 sl=685.88 alert=retest2 |

### Cycle 77 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 697.45 | 692.57 | 692.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 703.60 | 695.86 | 693.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 14:15:00 | 707.55 | 711.03 | 706.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 707.55 | 711.03 | 706.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 707.55 | 711.03 | 706.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 707.55 | 711.03 | 706.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 712.70 | 711.37 | 707.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 718.55 | 711.37 | 707.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 717.50 | 725.72 | 726.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 717.50 | 725.72 | 726.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 15:15:00 | 716.00 | 720.71 | 723.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 14:15:00 | 698.50 | 697.95 | 705.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 698.50 | 697.95 | 705.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 702.15 | 698.87 | 703.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:30:00 | 702.20 | 698.87 | 703.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 705.80 | 700.25 | 704.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 705.80 | 700.25 | 704.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 699.15 | 700.03 | 703.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 14:15:00 | 692.85 | 699.05 | 702.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 707.30 | 678.03 | 679.02 | SL hit (close>static) qty=1.00 sl=705.80 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 706.80 | 683.78 | 681.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 719.00 | 690.83 | 684.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 693.85 | 700.88 | 693.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 693.85 | 700.88 | 693.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 693.85 | 700.88 | 693.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 686.40 | 700.88 | 693.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 658.60 | 692.42 | 690.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 658.60 | 692.42 | 690.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 641.00 | 682.14 | 685.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 12:15:00 | 634.95 | 672.70 | 681.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 652.00 | 651.12 | 663.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 652.00 | 651.12 | 663.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 658.25 | 652.54 | 662.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 658.25 | 652.54 | 662.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 666.50 | 655.33 | 663.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 666.50 | 655.33 | 663.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 671.00 | 658.47 | 663.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 689.10 | 658.47 | 663.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 706.55 | 673.26 | 670.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 710.15 | 694.00 | 683.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 706.05 | 707.19 | 697.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:30:00 | 708.30 | 707.19 | 697.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 701.50 | 704.27 | 699.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:45:00 | 699.90 | 704.27 | 699.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 701.95 | 703.80 | 699.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 706.05 | 703.80 | 699.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 12:15:00 | 776.65 | 753.19 | 747.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 14:15:00 | 883.05 | 898.45 | 900.19 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 905.05 | 898.46 | 898.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 908.00 | 901.05 | 899.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 955.00 | 956.08 | 943.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 09:30:00 | 950.65 | 956.08 | 943.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 945.20 | 965.71 | 958.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 945.20 | 965.71 | 958.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 955.45 | 963.66 | 957.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:15:00 | 961.95 | 963.66 | 957.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 964.10 | 961.96 | 958.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 932.50 | 954.37 | 955.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 10:15:00 | 932.50 | 954.37 | 955.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 926.55 | 948.81 | 952.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 918.00 | 913.81 | 926.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 922.10 | 913.81 | 926.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 899.10 | 910.87 | 924.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 897.00 | 908.10 | 921.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 891.05 | 904.58 | 918.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 15:00:00 | 893.65 | 902.10 | 914.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 896.05 | 898.08 | 910.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 942.00 | 906.86 | 913.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 942.00 | 906.86 | 913.01 | SL hit (close>static) qty=1.00 sl=926.10 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 918.85 | 915.06 | 914.98 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 11:15:00 | 910.30 | 914.18 | 914.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 905.75 | 912.50 | 913.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 880.50 | 871.21 | 882.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 880.50 | 871.21 | 882.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 877.40 | 872.45 | 882.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 886.00 | 872.45 | 882.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 877.70 | 873.50 | 881.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 875.90 | 875.02 | 881.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 13:15:00 | 885.55 | 877.12 | 882.07 | SL hit (close>static) qty=1.00 sl=882.80 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 885.85 | 883.84 | 883.62 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 874.35 | 884.81 | 885.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 09:15:00 | 861.00 | 873.95 | 878.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 855.95 | 855.29 | 865.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 855.95 | 855.29 | 865.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 855.95 | 855.29 | 865.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 11:30:00 | 848.00 | 853.54 | 862.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 13:45:00 | 849.00 | 852.20 | 860.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 848.35 | 853.65 | 855.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 13:30:00 | 848.10 | 850.69 | 853.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 857.00 | 851.96 | 853.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 857.00 | 851.96 | 853.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 859.10 | 853.38 | 854.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 856.55 | 853.38 | 854.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 853.15 | 853.34 | 854.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:00:00 | 847.95 | 853.01 | 853.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 902.60 | 861.90 | 857.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 902.60 | 861.90 | 857.06 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 843.70 | 862.02 | 863.40 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 13:15:00 | 864.00 | 861.23 | 860.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 09:15:00 | 900.85 | 869.17 | 864.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 10:15:00 | 931.10 | 931.17 | 915.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 918.95 | 929.05 | 921.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 918.95 | 929.05 | 921.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 921.60 | 929.05 | 921.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 928.30 | 928.90 | 922.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:30:00 | 929.00 | 927.77 | 922.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 15:15:00 | 928.70 | 928.47 | 924.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:45:00 | 933.45 | 931.03 | 926.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 12:30:00 | 931.00 | 931.04 | 927.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 929.00 | 931.14 | 928.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 929.00 | 931.14 | 928.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 924.05 | 929.72 | 927.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 929.40 | 929.72 | 927.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 924.20 | 928.62 | 927.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 911.05 | 928.62 | 927.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 930.00 | 928.90 | 927.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:00:00 | 937.65 | 930.65 | 928.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:45:00 | 938.25 | 931.33 | 929.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:00:00 | 938.20 | 932.70 | 929.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 944.55 | 931.96 | 930.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 929.15 | 932.19 | 930.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 929.15 | 932.19 | 930.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 936.00 | 932.95 | 931.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 943.30 | 933.79 | 932.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 12:15:00 | 942.35 | 936.94 | 934.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 13:00:00 | 942.80 | 938.11 | 934.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 13:45:00 | 941.75 | 938.77 | 935.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 931.95 | 944.26 | 939.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 931.95 | 944.26 | 939.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 925.00 | 940.41 | 938.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 925.00 | 940.41 | 938.34 | SL hit (close<static) qty=1.00 sl=929.15 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 927.50 | 936.32 | 936.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 14:15:00 | 927.10 | 934.48 | 935.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 934.20 | 932.25 | 934.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 934.20 | 932.25 | 934.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 934.20 | 932.25 | 934.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 934.85 | 932.25 | 934.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 934.10 | 932.62 | 934.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 14:45:00 | 929.20 | 932.74 | 934.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 10:00:00 | 928.40 | 931.28 | 933.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 11:45:00 | 930.50 | 930.00 | 932.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 13:15:00 | 948.00 | 932.15 | 932.70 | SL hit (close>static) qty=1.00 sl=943.65 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 14:15:00 | 958.60 | 937.44 | 935.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 11:15:00 | 962.35 | 950.63 | 942.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 13:15:00 | 951.50 | 951.76 | 944.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 14:00:00 | 951.50 | 951.76 | 944.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 947.50 | 954.60 | 950.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 947.50 | 954.60 | 950.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 945.95 | 952.87 | 949.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 943.00 | 952.87 | 949.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 950.00 | 952.10 | 949.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 954.20 | 952.10 | 949.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:00:00 | 954.65 | 952.61 | 950.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 11:15:00 | 939.05 | 949.38 | 949.26 | SL hit (close<static) qty=1.00 sl=945.60 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 942.75 | 948.06 | 948.67 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 964.25 | 950.03 | 948.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 11:15:00 | 981.40 | 964.15 | 960.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 15:15:00 | 973.00 | 974.24 | 967.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 09:15:00 | 983.70 | 974.24 | 967.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 983.10 | 976.01 | 968.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:15:00 | 998.00 | 979.41 | 972.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:15:00 | 996.95 | 984.35 | 975.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 11:00:00 | 996.80 | 990.39 | 981.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 13:45:00 | 1005.30 | 993.77 | 985.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 990.00 | 992.25 | 985.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 992.90 | 992.25 | 985.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 975.10 | 988.82 | 984.90 | SL hit (close<static) qty=1.00 sl=982.50 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 977.15 | 981.68 | 982.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 969.55 | 979.18 | 980.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 11:15:00 | 974.90 | 971.85 | 976.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 11:15:00 | 974.90 | 971.85 | 976.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 974.90 | 971.85 | 976.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:15:00 | 967.00 | 972.90 | 974.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 981.00 | 974.70 | 974.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 14:15:00 | 981.00 | 974.70 | 974.33 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 966.85 | 973.18 | 973.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 11:15:00 | 963.10 | 970.17 | 972.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 980.00 | 972.13 | 972.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 12:15:00 | 980.00 | 972.13 | 972.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 980.00 | 972.13 | 972.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 980.00 | 972.13 | 972.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 980.00 | 973.71 | 973.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1000.00 | 981.21 | 977.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 994.60 | 1003.20 | 993.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 10:15:00 | 994.60 | 1003.20 | 993.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 994.60 | 1003.20 | 993.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 994.60 | 1003.20 | 993.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 990.95 | 1000.75 | 993.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 990.95 | 1000.75 | 993.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 994.15 | 999.43 | 993.36 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 977.95 | 990.82 | 990.90 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 1018.25 | 991.54 | 989.15 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 995.15 | 1004.37 | 1004.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 15:15:00 | 994.00 | 1002.30 | 1003.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 12:15:00 | 999.90 | 999.54 | 1001.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 13:00:00 | 999.90 | 999.54 | 1001.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 999.60 | 999.55 | 1001.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 999.60 | 999.55 | 1001.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1002.05 | 1000.05 | 1001.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 1002.05 | 1000.05 | 1001.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1005.55 | 1001.15 | 1002.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 1006.60 | 1002.02 | 1002.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 1013.70 | 1004.36 | 1003.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 1016.75 | 1010.56 | 1007.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 997.65 | 1010.57 | 1009.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 997.65 | 1010.57 | 1009.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 997.65 | 1010.57 | 1009.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 995.05 | 1010.57 | 1009.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1003.40 | 1009.13 | 1008.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:15:00 | 1006.00 | 1009.13 | 1008.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:45:00 | 1005.45 | 1008.56 | 1008.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 1006.05 | 1008.06 | 1008.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 1006.05 | 1008.06 | 1008.27 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 1015.80 | 1009.85 | 1009.05 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1008.10 | 1009.57 | 1009.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 1002.85 | 1008.23 | 1008.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 928.05 | 918.51 | 933.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 10:00:00 | 928.05 | 918.51 | 933.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 931.00 | 922.28 | 931.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:45:00 | 934.00 | 922.28 | 931.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 927.05 | 923.23 | 931.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:45:00 | 929.05 | 923.23 | 931.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 938.00 | 927.07 | 931.13 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 11:15:00 | 946.75 | 934.00 | 933.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 13:15:00 | 949.25 | 938.95 | 936.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 1037.50 | 1039.93 | 1021.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:45:00 | 1037.05 | 1039.93 | 1021.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1023.45 | 1034.02 | 1022.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 1023.60 | 1034.02 | 1022.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 1016.90 | 1030.60 | 1021.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 1016.90 | 1030.60 | 1021.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1023.00 | 1029.08 | 1021.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 1013.85 | 1029.08 | 1021.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 1041.95 | 1031.65 | 1023.64 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1020.90 | 1024.95 | 1025.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 1015.05 | 1020.56 | 1022.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 12:15:00 | 1021.05 | 1020.57 | 1022.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 13:00:00 | 1021.05 | 1020.57 | 1022.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1008.00 | 1018.05 | 1020.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 998.05 | 1013.41 | 1018.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 948.15 | 973.40 | 992.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 970.10 | 968.62 | 985.21 | SL hit (close>ema200) qty=0.50 sl=968.62 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 909.80 | 898.92 | 898.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 930.05 | 905.80 | 901.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 911.00 | 919.45 | 912.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 911.00 | 919.45 | 912.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 911.00 | 919.45 | 912.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 911.00 | 919.45 | 912.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 915.10 | 918.58 | 913.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 922.15 | 918.58 | 913.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 921.20 | 916.18 | 913.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 921.75 | 916.91 | 914.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:30:00 | 920.50 | 918.64 | 915.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 919.05 | 919.11 | 916.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:45:00 | 917.40 | 919.11 | 916.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 915.00 | 918.28 | 916.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 930.00 | 918.28 | 916.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 14:15:00 | 946.85 | 957.66 | 959.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 946.85 | 957.66 | 959.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 935.10 | 953.15 | 956.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 937.35 | 930.01 | 939.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 937.35 | 930.01 | 939.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 937.35 | 930.01 | 939.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 937.35 | 930.01 | 939.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 929.25 | 929.86 | 938.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 935.45 | 929.86 | 938.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 934.15 | 931.40 | 937.08 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 949.95 | 940.76 | 939.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 957.00 | 945.65 | 942.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 13:15:00 | 949.80 | 956.30 | 949.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 13:15:00 | 949.80 | 956.30 | 949.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 949.80 | 956.30 | 949.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 949.80 | 956.30 | 949.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 944.80 | 954.00 | 949.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 944.80 | 954.00 | 949.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 947.00 | 952.60 | 949.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 960.15 | 952.60 | 949.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 965.25 | 955.13 | 950.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 978.40 | 961.11 | 956.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 970.55 | 964.25 | 960.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:45:00 | 971.25 | 965.80 | 961.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 12:15:00 | 970.15 | 966.52 | 962.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 971.90 | 970.37 | 966.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 971.90 | 970.37 | 966.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 963.35 | 968.97 | 966.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:00:00 | 963.35 | 968.97 | 966.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 956.70 | 966.52 | 965.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:00:00 | 956.70 | 966.52 | 965.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 956.95 | 963.08 | 963.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 956.95 | 963.08 | 963.77 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 966.25 | 964.48 | 964.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 13:15:00 | 975.15 | 968.30 | 966.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 974.00 | 974.32 | 970.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:00:00 | 974.00 | 974.32 | 970.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 971.95 | 973.84 | 970.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 969.80 | 973.84 | 970.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 976.50 | 974.37 | 971.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:30:00 | 980.20 | 974.62 | 971.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 969.55 | 973.10 | 971.63 | SL hit (close<static) qty=1.00 sl=969.60 alert=retest2 |

### Cycle 114 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 965.00 | 969.75 | 970.38 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 975.60 | 970.42 | 970.21 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 964.00 | 969.13 | 969.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 11:15:00 | 958.45 | 967.00 | 968.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 15:15:00 | 964.00 | 963.38 | 966.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:15:00 | 969.50 | 963.38 | 966.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 967.50 | 964.20 | 966.20 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 975.75 | 968.00 | 967.57 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 965.70 | 967.56 | 967.73 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 970.40 | 968.13 | 967.97 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 967.50 | 968.23 | 968.24 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 969.80 | 968.54 | 968.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 13:15:00 | 976.85 | 970.15 | 969.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 11:15:00 | 968.55 | 971.06 | 970.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 11:15:00 | 968.55 | 971.06 | 970.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 968.55 | 971.06 | 970.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:00:00 | 968.55 | 971.06 | 970.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 965.60 | 969.97 | 969.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 965.40 | 969.97 | 969.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 967.00 | 969.09 | 969.36 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 981.85 | 971.46 | 970.38 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 970.05 | 972.16 | 972.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 960.15 | 969.76 | 971.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 907.00 | 906.64 | 915.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 09:15:00 | 905.85 | 906.64 | 915.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 899.65 | 898.58 | 906.34 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 914.95 | 907.77 | 907.34 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 899.60 | 907.94 | 908.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 10:15:00 | 896.65 | 905.68 | 907.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 897.35 | 896.50 | 901.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 897.35 | 896.50 | 901.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 897.35 | 896.50 | 901.15 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 918.90 | 905.30 | 904.15 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 898.10 | 905.04 | 905.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 892.70 | 902.57 | 904.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 901.25 | 901.01 | 902.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 901.25 | 901.01 | 902.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 901.25 | 901.01 | 902.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:45:00 | 906.00 | 901.01 | 902.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 901.40 | 900.24 | 901.99 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 911.45 | 903.82 | 903.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 922.50 | 915.62 | 911.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 916.00 | 919.62 | 915.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 15:15:00 | 916.00 | 919.62 | 915.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 916.00 | 919.62 | 915.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 916.80 | 919.62 | 915.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 920.85 | 919.87 | 916.24 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 905.00 | 914.35 | 915.18 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 921.40 | 914.98 | 914.96 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 913.75 | 914.73 | 914.85 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 916.25 | 915.03 | 914.98 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 912.05 | 914.44 | 914.71 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 919.05 | 915.36 | 915.11 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 912.00 | 914.58 | 914.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 14:15:00 | 903.50 | 912.20 | 913.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 10:15:00 | 917.30 | 912.11 | 913.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 917.30 | 912.11 | 913.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 917.30 | 912.11 | 913.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 917.30 | 912.11 | 913.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 914.25 | 912.54 | 913.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:30:00 | 904.65 | 912.00 | 912.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 917.85 | 910.30 | 911.65 | SL hit (close>static) qty=1.00 sl=917.75 alert=retest2 |

### Cycle 137 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 981.55 | 924.55 | 918.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 995.10 | 978.49 | 964.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 974.70 | 977.80 | 966.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 974.70 | 977.80 | 966.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1007.30 | 986.99 | 975.91 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 959.00 | 970.91 | 971.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 948.80 | 963.77 | 968.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 778.95 | 757.67 | 781.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 778.95 | 757.67 | 781.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 767.35 | 759.60 | 780.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:45:00 | 766.55 | 760.61 | 778.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 766.35 | 763.39 | 776.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 787.00 | 771.69 | 777.56 | SL hit (close>static) qty=1.00 sl=780.25 alert=retest2 |

### Cycle 139 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 783.40 | 774.56 | 774.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 814.75 | 783.95 | 778.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 788.05 | 789.73 | 782.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 12:30:00 | 796.10 | 789.73 | 782.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 793.00 | 790.38 | 783.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:30:00 | 796.25 | 792.56 | 785.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 753.50 | 784.02 | 782.77 | SL hit (close<static) qty=1.00 sl=782.20 alert=retest2 |

### Cycle 140 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 751.25 | 777.46 | 779.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 11:15:00 | 748.75 | 758.98 | 766.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 766.15 | 756.50 | 762.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 766.15 | 756.50 | 762.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 766.15 | 756.50 | 762.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 766.15 | 756.50 | 762.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 769.15 | 759.03 | 762.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 769.15 | 759.03 | 762.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 758.75 | 759.44 | 761.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:30:00 | 762.40 | 759.44 | 761.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 761.65 | 759.72 | 761.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:30:00 | 754.20 | 759.25 | 760.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 753.60 | 759.25 | 760.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 716.49 | 739.13 | 748.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 715.92 | 739.13 | 748.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 11:15:00 | 678.78 | 701.96 | 720.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 141 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 677.80 | 648.56 | 648.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 694.80 | 657.81 | 652.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 706.95 | 707.50 | 696.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 706.00 | 707.50 | 696.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 708.10 | 707.62 | 697.60 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 693.75 | 699.06 | 699.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 688.20 | 696.89 | 698.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 651.35 | 651.19 | 665.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 632.05 | 651.19 | 665.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 640.30 | 633.87 | 646.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 647.40 | 633.87 | 646.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 660.25 | 639.15 | 647.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 660.25 | 639.15 | 647.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 662.70 | 643.86 | 648.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 664.70 | 643.86 | 648.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 670.20 | 652.66 | 652.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 679.00 | 657.93 | 654.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 690.00 | 694.12 | 684.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 702.30 | 694.12 | 684.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 709.00 | 697.09 | 686.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 713.00 | 697.09 | 686.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:00:00 | 712.25 | 712.13 | 701.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 675.65 | 696.65 | 698.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 675.65 | 696.65 | 698.19 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 700.35 | 689.03 | 687.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 718.70 | 700.47 | 694.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 743.00 | 746.76 | 736.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 751.75 | 746.76 | 736.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:15:00 | 789.34 | 765.49 | 748.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-03-21 14:15:00 | 826.93 | 790.79 | 764.92 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 146 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 795.20 | 803.27 | 804.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 779.80 | 793.41 | 798.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 785.90 | 785.68 | 791.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 785.90 | 785.68 | 791.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 785.90 | 785.68 | 791.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 785.90 | 785.68 | 791.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 775.15 | 770.53 | 777.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 775.15 | 770.53 | 777.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 761.10 | 768.40 | 775.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:15:00 | 749.80 | 761.84 | 767.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 674.82 | 727.86 | 741.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 769.40 | 734.57 | 731.60 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 14:15:00 | 735.80 | 750.85 | 752.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 09:15:00 | 725.15 | 743.65 | 749.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 691.05 | 685.97 | 698.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 09:45:00 | 689.60 | 685.97 | 698.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 607.55 | 603.37 | 607.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:30:00 | 608.45 | 603.37 | 607.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 600.00 | 602.70 | 606.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 597.00 | 601.23 | 605.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:45:00 | 597.85 | 600.28 | 604.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 567.15 | 579.00 | 585.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 567.96 | 579.00 | 585.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 573.60 | 569.44 | 577.02 | SL hit (close>ema200) qty=0.50 sl=569.44 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 580.50 | 576.38 | 576.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 582.60 | 578.90 | 577.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 618.95 | 620.88 | 612.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 616.65 | 620.88 | 612.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 615.00 | 617.36 | 614.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 607.20 | 617.36 | 614.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 617.40 | 617.37 | 614.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 615.10 | 617.37 | 614.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 617.35 | 617.66 | 615.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 615.00 | 617.66 | 615.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 614.30 | 616.99 | 614.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 612.95 | 616.99 | 614.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 611.15 | 615.82 | 614.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 611.15 | 615.82 | 614.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 617.20 | 616.10 | 614.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 619.60 | 616.10 | 614.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 613.75 | 614.75 | 614.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 613.75 | 614.75 | 614.82 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 616.75 | 615.03 | 614.84 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 613.55 | 614.79 | 614.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 14:15:00 | 610.50 | 613.09 | 613.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 610.40 | 610.38 | 611.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:45:00 | 609.50 | 610.38 | 611.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 610.90 | 610.48 | 611.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:45:00 | 609.50 | 610.25 | 611.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 609.00 | 600.83 | 603.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 610.20 | 605.68 | 605.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 610.20 | 605.68 | 605.26 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 601.95 | 605.68 | 605.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 596.90 | 603.10 | 604.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 600.50 | 599.98 | 602.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 600.50 | 599.98 | 602.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 600.50 | 599.98 | 602.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 600.50 | 599.98 | 602.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 599.00 | 599.79 | 602.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 599.60 | 599.79 | 602.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 593.00 | 598.43 | 601.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 10:45:00 | 589.80 | 596.79 | 600.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:00:00 | 590.35 | 594.42 | 598.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 607.35 | 596.23 | 597.30 | SL hit (close>static) qty=1.00 sl=603.80 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 609.20 | 598.82 | 598.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 651.30 | 611.37 | 604.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 666.35 | 668.74 | 658.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:00:00 | 666.35 | 668.74 | 658.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 660.40 | 666.04 | 660.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 661.40 | 666.04 | 660.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 661.10 | 665.06 | 660.71 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 646.00 | 658.22 | 659.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 638.20 | 648.66 | 653.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 647.80 | 647.46 | 650.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 647.80 | 647.46 | 650.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 647.25 | 647.81 | 650.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 644.10 | 647.07 | 649.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:00:00 | 644.20 | 645.59 | 648.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 637.80 | 636.41 | 639.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:45:00 | 643.80 | 637.35 | 639.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 642.80 | 638.44 | 639.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 642.80 | 638.44 | 639.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 642.70 | 639.29 | 640.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:00:00 | 642.70 | 639.29 | 640.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 641.00 | 639.63 | 640.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:45:00 | 644.15 | 639.63 | 640.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 633.55 | 638.42 | 639.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 635.90 | 638.42 | 639.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 632.55 | 637.24 | 638.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 627.85 | 633.86 | 636.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 624.00 | 631.87 | 635.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 646.10 | 638.26 | 637.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 646.10 | 638.26 | 637.43 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 628.85 | 636.42 | 636.74 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 646.95 | 638.48 | 637.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 13:15:00 | 650.25 | 643.11 | 640.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 10:15:00 | 676.35 | 677.72 | 670.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 11:00:00 | 676.35 | 677.72 | 670.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 668.70 | 674.95 | 670.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 668.70 | 674.95 | 670.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 675.60 | 675.08 | 670.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 684.05 | 674.47 | 671.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 683.50 | 676.19 | 673.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 671.00 | 672.89 | 673.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 671.00 | 672.89 | 673.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 662.85 | 670.88 | 672.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 667.45 | 665.29 | 667.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 09:45:00 | 666.65 | 665.29 | 667.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 665.60 | 665.35 | 667.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:45:00 | 663.55 | 665.24 | 667.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 663.25 | 664.40 | 666.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 662.30 | 663.64 | 665.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 663.40 | 661.18 | 663.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 661.30 | 661.20 | 662.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 653.00 | 659.88 | 661.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 656.85 | 657.61 | 660.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:15:00 | 654.20 | 656.57 | 659.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 657.00 | 654.63 | 656.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 660.80 | 655.87 | 657.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:15:00 | 663.95 | 655.87 | 657.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 659.60 | 656.61 | 657.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 660.35 | 656.61 | 657.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 664.40 | 658.17 | 658.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 664.40 | 658.17 | 658.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 12:15:00 | 673.75 | 664.73 | 661.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 671.05 | 672.66 | 667.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:00:00 | 671.05 | 672.66 | 667.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 667.55 | 671.64 | 667.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:30:00 | 667.70 | 671.64 | 667.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 666.00 | 670.51 | 667.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 666.00 | 670.51 | 667.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 664.25 | 669.26 | 667.15 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 09:15:00 | 654.35 | 663.64 | 664.90 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 665.40 | 659.97 | 659.48 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 651.40 | 659.27 | 659.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 647.45 | 653.01 | 655.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 642.40 | 640.75 | 645.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 642.40 | 640.75 | 645.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 642.40 | 640.75 | 645.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 648.00 | 640.75 | 645.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 637.00 | 638.77 | 641.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 633.90 | 638.09 | 641.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 634.65 | 637.10 | 640.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 602.20 | 611.08 | 616.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 602.92 | 611.08 | 616.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-31 10:15:00 | 570.51 | 591.25 | 599.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 165 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 584.00 | 571.23 | 570.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 09:15:00 | 585.00 | 573.98 | 572.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 575.30 | 580.86 | 577.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 575.30 | 580.86 | 577.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 575.30 | 580.86 | 577.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 569.95 | 580.86 | 577.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 583.65 | 581.42 | 578.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 11:15:00 | 586.35 | 581.42 | 578.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 586.30 | 582.30 | 578.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 11:45:00 | 588.30 | 583.04 | 580.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 591.15 | 585.95 | 583.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 588.55 | 586.47 | 583.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 573.00 | 581.23 | 581.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 573.00 | 581.23 | 581.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 568.20 | 576.37 | 579.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 577.00 | 571.55 | 574.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 577.00 | 571.55 | 574.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 577.00 | 571.55 | 574.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:15:00 | 576.40 | 571.55 | 574.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 579.10 | 573.06 | 575.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 573.65 | 573.89 | 575.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:15:00 | 574.40 | 573.89 | 575.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 15:15:00 | 579.00 | 575.71 | 575.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 579.00 | 575.71 | 575.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 581.45 | 576.86 | 576.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 581.10 | 581.76 | 579.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:45:00 | 582.00 | 581.76 | 579.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 578.00 | 581.01 | 579.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:45:00 | 577.95 | 581.01 | 579.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 575.00 | 579.81 | 579.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 575.00 | 579.81 | 579.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 574.10 | 578.67 | 578.84 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 600.00 | 582.93 | 580.76 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 576.00 | 580.29 | 580.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 571.90 | 577.71 | 579.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 565.30 | 563.59 | 567.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 565.30 | 563.59 | 567.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 565.30 | 563.59 | 567.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 565.30 | 563.59 | 567.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 572.30 | 565.72 | 567.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 572.30 | 565.72 | 567.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 574.65 | 567.51 | 568.45 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 12:15:00 | 573.50 | 569.65 | 569.32 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 566.40 | 569.98 | 570.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 15:15:00 | 565.50 | 569.08 | 569.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 571.00 | 569.46 | 569.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 571.00 | 569.46 | 569.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 571.00 | 569.46 | 569.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 574.80 | 569.46 | 569.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 569.80 | 569.53 | 569.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 569.10 | 569.53 | 569.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 573.80 | 570.59 | 570.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 573.80 | 570.59 | 570.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 581.15 | 573.20 | 571.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 572.20 | 574.41 | 572.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 572.20 | 574.41 | 572.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 572.20 | 574.41 | 572.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 572.20 | 574.41 | 572.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 578.75 | 575.28 | 573.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 582.40 | 575.83 | 574.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 569.10 | 573.77 | 574.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 569.10 | 573.77 | 574.00 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 578.05 | 574.42 | 574.25 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 571.00 | 573.68 | 573.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 570.00 | 572.30 | 573.19 | Break + close below crossover candle low |

### Cycle 177 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 595.95 | 576.82 | 575.08 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 578.55 | 582.52 | 582.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 576.80 | 580.55 | 581.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 580.80 | 578.88 | 580.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 580.80 | 578.88 | 580.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 580.80 | 578.88 | 580.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 580.80 | 578.88 | 580.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 579.85 | 579.07 | 580.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 580.75 | 579.07 | 580.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 578.40 | 578.94 | 579.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 579.60 | 578.94 | 579.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 580.75 | 578.64 | 579.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 582.95 | 578.64 | 579.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 578.70 | 578.65 | 579.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 578.15 | 578.50 | 579.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 577.00 | 577.68 | 578.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 576.80 | 577.22 | 578.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 582.65 | 578.58 | 578.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 582.65 | 578.58 | 578.26 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 573.80 | 577.68 | 577.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 569.30 | 575.67 | 576.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 561.75 | 559.62 | 563.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 561.75 | 559.62 | 563.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 563.55 | 560.60 | 563.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 564.85 | 560.60 | 563.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 564.20 | 561.32 | 563.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 559.70 | 561.32 | 563.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 562.00 | 561.45 | 563.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 558.40 | 560.97 | 563.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 558.40 | 560.30 | 561.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:00:00 | 558.30 | 560.30 | 561.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 530.48 | 545.09 | 552.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 530.48 | 545.09 | 552.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 530.38 | 545.09 | 552.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 533.65 | 530.05 | 536.54 | SL hit (close>ema200) qty=0.50 sl=530.05 alert=retest2 |

### Cycle 181 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 539.00 | 537.65 | 537.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 540.80 | 538.58 | 538.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 550.00 | 550.02 | 546.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 550.00 | 550.02 | 546.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 548.15 | 549.64 | 546.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 547.80 | 549.64 | 546.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 555.50 | 550.59 | 547.55 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 544.60 | 547.36 | 547.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 540.95 | 546.08 | 546.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 540.55 | 539.80 | 542.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 13:45:00 | 539.60 | 539.80 | 542.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 550.00 | 541.84 | 543.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 550.00 | 541.84 | 543.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 550.00 | 543.47 | 543.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 547.50 | 543.47 | 543.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 550.90 | 544.96 | 544.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 550.90 | 544.96 | 544.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 557.50 | 550.38 | 548.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 549.05 | 553.95 | 551.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 549.05 | 553.95 | 551.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 549.05 | 553.95 | 551.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 549.05 | 553.95 | 551.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 547.35 | 552.63 | 551.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 547.35 | 552.63 | 551.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 545.85 | 550.08 | 550.26 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 553.30 | 550.72 | 550.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 15:15:00 | 558.90 | 552.36 | 551.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 10:15:00 | 550.75 | 552.09 | 551.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 550.75 | 552.09 | 551.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 550.75 | 552.09 | 551.36 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 549.45 | 550.72 | 550.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 10:15:00 | 548.25 | 550.02 | 550.45 | Break + close below crossover candle low |

### Cycle 187 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 553.75 | 550.77 | 550.75 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 549.40 | 550.49 | 550.62 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 553.50 | 550.94 | 550.79 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 547.70 | 550.43 | 550.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 544.00 | 548.54 | 549.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 555.30 | 542.40 | 544.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 555.30 | 542.40 | 544.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 555.30 | 542.40 | 544.74 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 554.30 | 546.82 | 546.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 561.60 | 550.60 | 548.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 552.80 | 554.58 | 551.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 552.80 | 554.58 | 551.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 550.05 | 553.45 | 551.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 550.05 | 553.45 | 551.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 544.20 | 551.60 | 550.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 544.20 | 551.60 | 550.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 542.50 | 549.78 | 550.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 541.25 | 546.83 | 548.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 538.50 | 536.84 | 540.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 538.50 | 536.84 | 540.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 535.95 | 536.66 | 540.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 535.20 | 536.33 | 539.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 545.20 | 538.75 | 539.45 | SL hit (close>static) qty=1.00 sl=540.50 alert=retest2 |

### Cycle 193 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 549.70 | 540.94 | 540.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 550.50 | 542.85 | 541.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 545.50 | 545.90 | 543.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:00:00 | 545.50 | 545.90 | 543.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 546.45 | 546.01 | 544.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 544.85 | 546.01 | 544.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 543.95 | 545.60 | 544.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 547.60 | 545.60 | 544.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 540.80 | 544.23 | 544.08 | SL hit (close<static) qty=1.00 sl=542.25 alert=retest2 |

### Cycle 194 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 542.05 | 543.79 | 543.89 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 547.50 | 544.53 | 544.22 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 542.30 | 543.76 | 543.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 12:15:00 | 540.75 | 543.16 | 543.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 543.30 | 543.10 | 543.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 543.30 | 543.10 | 543.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 543.30 | 543.10 | 543.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 543.30 | 543.10 | 543.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 543.40 | 543.16 | 543.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 540.70 | 543.16 | 543.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 538.00 | 542.13 | 542.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 536.75 | 541.19 | 542.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 535.95 | 541.19 | 542.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 538.05 | 532.29 | 532.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 538.05 | 532.29 | 532.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 546.35 | 536.23 | 533.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 11:15:00 | 534.00 | 538.40 | 536.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 11:15:00 | 534.00 | 538.40 | 536.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 534.00 | 538.40 | 536.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 534.00 | 538.40 | 536.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 536.25 | 537.97 | 536.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:45:00 | 538.45 | 537.79 | 536.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 545.65 | 550.70 | 550.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 545.65 | 550.70 | 550.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 542.00 | 547.03 | 548.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 541.95 | 540.94 | 543.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:45:00 | 541.35 | 540.94 | 543.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 550.40 | 542.83 | 544.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 553.70 | 542.83 | 544.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 548.25 | 543.91 | 544.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 547.50 | 543.91 | 544.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 542.35 | 543.48 | 544.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:30:00 | 542.85 | 543.48 | 544.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 541.95 | 541.56 | 543.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 541.95 | 541.56 | 543.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 541.70 | 541.59 | 542.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 542.60 | 541.59 | 542.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 532.80 | 524.60 | 528.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 532.80 | 524.60 | 528.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 528.05 | 525.29 | 528.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 540.00 | 525.29 | 528.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 537.85 | 527.80 | 529.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 539.40 | 527.80 | 529.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 537.80 | 529.80 | 529.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 540.20 | 529.80 | 529.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 537.05 | 531.25 | 530.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 542.15 | 535.21 | 532.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 542.10 | 542.12 | 537.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:00:00 | 542.10 | 542.12 | 537.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 542.40 | 542.62 | 539.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:00:00 | 547.50 | 543.60 | 540.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 538.50 | 544.40 | 543.06 | SL hit (close<static) qty=1.00 sl=539.35 alert=retest2 |

### Cycle 200 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 533.70 | 540.84 | 541.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 525.85 | 536.51 | 539.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 528.00 | 526.07 | 531.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 11:00:00 | 528.00 | 526.07 | 531.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 531.45 | 527.25 | 530.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 531.45 | 527.25 | 530.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 531.00 | 528.00 | 530.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 530.75 | 528.00 | 530.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 534.35 | 529.27 | 531.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 534.35 | 529.27 | 531.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 533.80 | 530.18 | 531.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 533.20 | 530.18 | 531.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 531.70 | 530.72 | 531.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 535.15 | 530.72 | 531.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 530.20 | 530.70 | 531.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 530.85 | 530.70 | 531.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 531.35 | 530.83 | 531.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 531.35 | 530.83 | 531.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 532.50 | 531.16 | 531.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 532.50 | 531.16 | 531.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 532.00 | 531.33 | 531.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 527.00 | 531.33 | 531.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 528.30 | 529.60 | 530.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 533.40 | 527.29 | 528.19 | SL hit (close>static) qty=1.00 sl=533.00 alert=retest2 |

### Cycle 201 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 505.40 | 494.74 | 493.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 524.45 | 504.86 | 498.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 519.25 | 519.28 | 511.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:15:00 | 518.60 | 519.28 | 511.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 518.25 | 517.69 | 513.89 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 504.60 | 512.12 | 512.95 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 515.45 | 513.29 | 513.20 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 506.20 | 511.88 | 512.56 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 517.45 | 513.46 | 513.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 527.60 | 517.12 | 514.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 518.70 | 520.09 | 517.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 518.70 | 520.09 | 517.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 519.65 | 520.00 | 517.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 524.20 | 517.82 | 517.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 523.25 | 524.53 | 524.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 523.25 | 524.53 | 524.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 518.45 | 523.05 | 523.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 520.85 | 518.63 | 520.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 520.85 | 518.63 | 520.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 520.85 | 518.63 | 520.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 515.05 | 516.83 | 519.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 489.30 | 501.87 | 508.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 10:15:00 | 502.85 | 502.07 | 507.58 | SL hit (close>ema200) qty=0.50 sl=502.07 alert=retest2 |

### Cycle 207 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 510.40 | 493.51 | 492.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 515.95 | 505.57 | 499.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 511.35 | 512.63 | 506.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 511.35 | 512.63 | 506.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 506.35 | 511.38 | 506.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 514.20 | 511.38 | 506.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:45:00 | 512.50 | 511.31 | 507.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 503.60 | 509.77 | 506.70 | SL hit (close<static) qty=1.00 sl=506.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 498.15 | 504.17 | 504.83 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 508.05 | 504.46 | 504.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 512.05 | 505.98 | 505.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 501.30 | 506.18 | 505.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 501.30 | 506.18 | 505.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 501.30 | 506.18 | 505.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 499.95 | 506.18 | 505.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 504.55 | 505.86 | 505.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 504.15 | 505.86 | 505.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 507.50 | 505.90 | 505.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 511.10 | 508.39 | 506.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 509.40 | 510.02 | 508.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 509.55 | 509.06 | 508.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 494.40 | 506.12 | 507.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 494.40 | 506.12 | 507.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 489.00 | 502.69 | 505.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 507.65 | 501.25 | 503.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 507.65 | 501.25 | 503.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 507.65 | 501.25 | 503.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 507.65 | 501.25 | 503.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 506.00 | 502.20 | 504.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 544.10 | 502.20 | 504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 561.80 | 514.12 | 509.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 13:15:00 | 575.25 | 571.07 | 568.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 12:15:00 | 564.40 | 572.20 | 570.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 12:15:00 | 564.40 | 572.20 | 570.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 564.40 | 572.20 | 570.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:00:00 | 564.40 | 572.20 | 570.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 576.40 | 573.04 | 570.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 578.85 | 574.20 | 571.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 578.25 | 574.70 | 572.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 576.65 | 574.83 | 572.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 563.00 | 571.34 | 571.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 563.00 | 571.34 | 571.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 556.60 | 563.27 | 565.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 567.55 | 557.62 | 560.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 567.55 | 557.62 | 560.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 567.55 | 557.62 | 560.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 567.55 | 557.62 | 560.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 560.20 | 558.14 | 560.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 557.00 | 557.91 | 560.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 572.00 | 560.73 | 561.19 | SL hit (close>static) qty=1.00 sl=570.50 alert=retest2 |

### Cycle 213 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 570.15 | 562.61 | 562.00 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 551.85 | 560.46 | 561.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 549.00 | 557.01 | 559.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 560.15 | 556.47 | 558.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 560.15 | 556.47 | 558.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 560.15 | 556.47 | 558.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 561.25 | 556.47 | 558.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 558.20 | 556.81 | 558.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 556.80 | 556.23 | 558.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 557.00 | 556.10 | 558.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 551.70 | 547.35 | 546.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 11:15:00 | 551.70 | 547.35 | 546.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 14:15:00 | 553.50 | 549.26 | 547.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 13:15:00 | 551.05 | 551.20 | 549.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 551.05 | 551.20 | 549.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 551.05 | 551.20 | 549.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 551.05 | 551.20 | 549.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 557.40 | 552.44 | 550.35 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 545.75 | 549.00 | 549.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 15:15:00 | 542.00 | 547.13 | 548.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 554.30 | 548.57 | 548.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 554.30 | 548.57 | 548.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 554.30 | 548.57 | 548.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 556.05 | 548.57 | 548.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 556.40 | 550.13 | 549.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 564.65 | 554.93 | 552.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 548.25 | 557.76 | 555.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 548.25 | 557.76 | 555.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 548.25 | 557.76 | 555.53 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 548.70 | 553.19 | 553.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 546.20 | 551.79 | 553.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 15:15:00 | 554.00 | 551.81 | 552.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 15:15:00 | 554.00 | 551.81 | 552.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 554.00 | 551.81 | 552.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 557.15 | 551.81 | 552.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 554.80 | 552.41 | 553.00 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 556.60 | 553.67 | 553.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 560.00 | 555.68 | 554.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 554.50 | 558.37 | 556.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 554.50 | 558.37 | 556.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 554.50 | 558.37 | 556.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 554.50 | 558.37 | 556.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 551.95 | 557.09 | 556.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 552.50 | 557.09 | 556.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 549.55 | 555.58 | 555.90 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 561.80 | 556.49 | 556.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 562.15 | 557.62 | 556.75 | Break + close above crossover candle high |

### Cycle 222 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 544.50 | 555.82 | 556.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 14:15:00 | 531.90 | 542.65 | 545.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 480.70 | 477.21 | 491.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 480.15 | 477.21 | 491.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 490.65 | 480.93 | 491.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 489.00 | 480.93 | 491.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 480.60 | 480.87 | 490.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 480.10 | 481.09 | 489.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 509.50 | 486.62 | 490.46 | SL hit (close>static) qty=1.00 sl=491.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 509.20 | 494.07 | 493.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 512.10 | 497.68 | 495.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 502.00 | 502.03 | 498.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 498.25 | 502.03 | 498.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 488.95 | 499.41 | 497.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 488.95 | 499.41 | 497.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 485.80 | 496.69 | 496.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 483.35 | 496.69 | 496.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 484.25 | 494.20 | 495.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 482.10 | 491.78 | 493.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 498.00 | 479.60 | 483.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 498.00 | 479.60 | 483.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 498.00 | 479.60 | 483.76 | EMA400 retest candle locked (from downside) |

### Cycle 225 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 499.65 | 487.58 | 486.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 501.75 | 492.14 | 489.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 480.90 | 491.18 | 489.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 480.90 | 491.18 | 489.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 480.90 | 491.18 | 489.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 480.90 | 491.18 | 489.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 492.05 | 490.10 | 489.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 495.40 | 491.65 | 489.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 496.60 | 494.42 | 491.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 496.35 | 494.74 | 492.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 500.85 | 494.74 | 492.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 505.80 | 502.13 | 497.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 507.85 | 502.13 | 497.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 519.40 | 503.65 | 500.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 14:15:00 | 544.94 | 538.88 | 530.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 525.80 | 533.28 | 534.16 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 535.40 | 534.20 | 534.19 | EMA200 above EMA400 |

### Cycle 228 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 529.70 | 533.72 | 534.01 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 563.95 | 538.34 | 535.78 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 544.15 | 549.20 | 549.86 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 554.25 | 549.71 | 549.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 568.00 | 558.35 | 554.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 599.10 | 599.68 | 586.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:45:00 | 597.60 | 599.68 | 586.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 588.85 | 595.64 | 587.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 588.90 | 595.64 | 587.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 589.80 | 594.47 | 588.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 595.15 | 593.53 | 588.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:15:00 | 593.35 | 593.53 | 588.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:30:00 | 600.90 | 596.61 | 591.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 14:15:00 | 611.35 | 617.67 | 617.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 611.35 | 617.67 | 617.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 607.80 | 615.70 | 616.78 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:15:00 | 694.30 | 2024-04-15 09:15:00 | 659.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 09:15:00 | 694.30 | 2024-04-15 09:15:00 | 702.00 | STOP_HIT | 0.50 | -1.11% |
| SELL | retest2 | 2024-04-12 14:30:00 | 694.90 | 2024-04-15 09:15:00 | 660.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 14:30:00 | 694.90 | 2024-04-15 09:15:00 | 702.00 | STOP_HIT | 0.50 | -1.02% |
| SELL | retest2 | 2024-04-12 15:00:00 | 688.90 | 2024-04-15 10:15:00 | 705.95 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-04-19 13:15:00 | 742.80 | 2024-04-30 10:15:00 | 749.55 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-04-23 10:15:00 | 741.00 | 2024-04-30 10:15:00 | 749.55 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2024-04-23 10:45:00 | 739.50 | 2024-04-30 10:15:00 | 749.55 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2024-05-08 09:15:00 | 710.00 | 2024-05-10 09:15:00 | 674.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 710.00 | 2024-05-10 13:15:00 | 687.95 | STOP_HIT | 0.50 | 3.11% |
| BUY | retest2 | 2024-05-17 09:15:00 | 718.55 | 2024-05-23 11:15:00 | 717.50 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-05-28 14:15:00 | 692.85 | 2024-06-03 09:15:00 | 707.30 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-06-11 09:15:00 | 706.05 | 2024-06-18 12:15:00 | 776.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 13:15:00 | 961.95 | 2024-07-09 10:15:00 | 932.50 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-07-08 15:15:00 | 964.10 | 2024-07-09 10:15:00 | 932.50 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-07-11 11:00:00 | 897.00 | 2024-07-12 10:15:00 | 942.00 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2024-07-11 11:45:00 | 891.05 | 2024-07-12 10:15:00 | 942.00 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2024-07-11 15:00:00 | 893.65 | 2024-07-12 10:15:00 | 942.00 | STOP_HIT | 1.00 | -5.41% |
| SELL | retest2 | 2024-07-12 09:30:00 | 896.05 | 2024-07-12 10:15:00 | 942.00 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2024-07-12 12:15:00 | 925.65 | 2024-07-16 09:15:00 | 918.85 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-07-12 13:15:00 | 926.25 | 2024-07-16 09:15:00 | 918.85 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-07-22 12:30:00 | 875.90 | 2024-07-22 13:15:00 | 885.55 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-07-23 09:30:00 | 874.30 | 2024-07-23 10:15:00 | 884.55 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-07-23 12:15:00 | 856.65 | 2024-07-23 13:15:00 | 895.85 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2024-07-29 11:30:00 | 848.00 | 2024-08-02 10:15:00 | 902.60 | STOP_HIT | 1.00 | -6.44% |
| SELL | retest2 | 2024-07-29 13:45:00 | 849.00 | 2024-08-02 10:15:00 | 902.60 | STOP_HIT | 1.00 | -6.31% |
| SELL | retest2 | 2024-07-31 09:15:00 | 848.35 | 2024-08-02 10:15:00 | 902.60 | STOP_HIT | 1.00 | -6.39% |
| SELL | retest2 | 2024-07-31 13:30:00 | 848.10 | 2024-08-02 10:15:00 | 902.60 | STOP_HIT | 1.00 | -6.43% |
| SELL | retest2 | 2024-08-01 12:00:00 | 847.95 | 2024-08-02 10:15:00 | 902.60 | STOP_HIT | 1.00 | -6.44% |
| BUY | retest2 | 2024-08-12 12:30:00 | 929.00 | 2024-08-20 11:15:00 | 925.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-08-12 15:15:00 | 928.70 | 2024-08-20 11:15:00 | 925.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-08-13 11:45:00 | 933.45 | 2024-08-20 11:15:00 | 925.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-08-13 12:30:00 | 931.00 | 2024-08-20 11:15:00 | 925.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-14 12:00:00 | 937.65 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-08-14 12:45:00 | 938.25 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-08-14 14:00:00 | 938.20 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-08-16 09:15:00 | 944.55 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-08-19 09:15:00 | 943.30 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-08-19 12:15:00 | 942.35 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-08-19 13:00:00 | 942.80 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-08-19 13:45:00 | 941.75 | 2024-08-20 13:15:00 | 927.50 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-08-21 14:45:00 | 929.20 | 2024-08-22 13:15:00 | 948.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-08-22 10:00:00 | 928.40 | 2024-08-22 13:15:00 | 948.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-08-22 11:45:00 | 930.50 | 2024-08-22 13:15:00 | 948.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-08-27 09:15:00 | 954.20 | 2024-08-27 11:15:00 | 939.05 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-08-27 10:00:00 | 954.65 | 2024-08-27 11:15:00 | 939.05 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-09-04 13:15:00 | 998.00 | 2024-09-06 09:15:00 | 975.10 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-09-04 15:15:00 | 996.95 | 2024-09-06 12:15:00 | 977.15 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-09-05 11:00:00 | 996.80 | 2024-09-06 12:15:00 | 977.15 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-09-05 13:45:00 | 1005.30 | 2024-09-06 12:15:00 | 977.15 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-09-06 09:15:00 | 992.90 | 2024-09-06 12:15:00 | 977.15 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-09-10 15:15:00 | 967.00 | 2024-09-11 14:15:00 | 981.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-09-30 11:15:00 | 1006.00 | 2024-09-30 13:15:00 | 1006.05 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-09-30 12:45:00 | 1005.45 | 2024-09-30 13:15:00 | 1006.05 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-10-21 14:30:00 | 998.05 | 2024-10-22 14:15:00 | 948.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:30:00 | 998.05 | 2024-10-23 10:15:00 | 970.10 | STOP_HIT | 0.50 | 2.80% |
| BUY | retest2 | 2024-11-04 11:15:00 | 922.15 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | 2.68% |
| BUY | retest2 | 2024-11-05 09:15:00 | 921.20 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2024-11-05 09:45:00 | 921.75 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2024-11-05 12:30:00 | 920.50 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2024-11-06 09:15:00 | 930.00 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2024-11-22 09:15:00 | 978.40 | 2024-11-26 13:15:00 | 956.95 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-11-25 10:15:00 | 970.55 | 2024-11-26 13:15:00 | 956.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-11-25 10:45:00 | 971.25 | 2024-11-26 13:15:00 | 956.95 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-11-25 12:15:00 | 970.15 | 2024-11-26 13:15:00 | 956.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-11-28 14:30:00 | 980.20 | 2024-11-29 10:15:00 | 969.55 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-01-09 14:30:00 | 904.65 | 2025-01-10 10:15:00 | 917.85 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-01-10 10:45:00 | 911.50 | 2025-01-10 11:15:00 | 981.55 | STOP_HIT | 1.00 | -7.69% |
| SELL | retest2 | 2025-01-29 11:45:00 | 766.55 | 2025-01-30 09:15:00 | 787.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-01-29 13:30:00 | 766.35 | 2025-01-30 09:15:00 | 787.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-01-30 13:30:00 | 765.85 | 2025-01-31 12:15:00 | 782.05 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-01-31 10:00:00 | 764.45 | 2025-01-31 12:15:00 | 782.05 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-02-01 14:30:00 | 796.25 | 2025-02-03 09:15:00 | 753.50 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2025-02-06 14:30:00 | 754.20 | 2025-02-10 09:15:00 | 716.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 15:15:00 | 753.60 | 2025-02-10 09:15:00 | 715.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 14:30:00 | 754.20 | 2025-02-11 11:15:00 | 678.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 15:15:00 | 753.60 | 2025-02-11 11:15:00 | 678.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-07 10:15:00 | 713.00 | 2025-03-11 09:15:00 | 675.65 | STOP_HIT | 1.00 | -5.24% |
| BUY | retest2 | 2025-03-10 10:00:00 | 712.25 | 2025-03-11 09:15:00 | 675.65 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest1 | 2025-03-21 09:15:00 | 751.75 | 2025-03-21 11:15:00 | 789.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-21 09:15:00 | 751.75 | 2025-03-21 14:15:00 | 826.93 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-26 09:15:00 | 826.90 | 2025-03-26 14:15:00 | 789.65 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-04-03 11:15:00 | 749.80 | 2025-04-07 09:15:00 | 674.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 597.00 | 2025-05-09 09:15:00 | 567.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 10:45:00 | 597.85 | 2025-05-09 09:15:00 | 567.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 597.00 | 2025-05-09 15:15:00 | 573.60 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-05-06 10:45:00 | 597.85 | 2025-05-09 15:15:00 | 573.60 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2025-05-20 15:15:00 | 619.60 | 2025-05-21 14:15:00 | 613.75 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-28 12:45:00 | 609.50 | 2025-06-02 09:15:00 | 610.20 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-05-30 10:15:00 | 609.00 | 2025-06-02 09:15:00 | 610.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-06-04 10:45:00 | 589.80 | 2025-06-05 11:15:00 | 607.35 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-06-04 13:00:00 | 590.35 | 2025-06-05 11:15:00 | 607.35 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-06-17 11:00:00 | 644.10 | 2025-06-23 11:15:00 | 646.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-17 14:00:00 | 644.20 | 2025-06-23 11:15:00 | 646.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-06-19 11:15:00 | 637.80 | 2025-06-23 11:15:00 | 646.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-19 11:45:00 | 643.80 | 2025-06-23 11:15:00 | 646.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-06-20 12:45:00 | 627.85 | 2025-06-23 11:15:00 | 646.10 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-06-20 15:15:00 | 624.00 | 2025-06-23 11:15:00 | 646.10 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-06-30 09:30:00 | 684.05 | 2025-07-01 15:15:00 | 671.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-01 09:15:00 | 683.50 | 2025-07-01 15:15:00 | 671.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-07-03 11:45:00 | 663.55 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-03 12:30:00 | 663.25 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-07-04 09:30:00 | 662.30 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-07 09:15:00 | 663.40 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-07-07 15:15:00 | 653.00 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-07-08 09:30:00 | 656.85 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-08 12:15:00 | 654.20 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-09 09:30:00 | 657.00 | 2025-07-09 12:15:00 | 664.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-22 10:45:00 | 633.90 | 2025-07-29 10:15:00 | 602.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 13:15:00 | 634.65 | 2025-07-29 10:15:00 | 602.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 633.90 | 2025-07-31 10:15:00 | 570.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 13:15:00 | 634.65 | 2025-07-31 10:15:00 | 571.18 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-11 11:15:00 | 586.35 | 2025-08-13 13:15:00 | 573.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-11 12:15:00 | 586.30 | 2025-08-13 13:15:00 | 573.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-08-12 11:45:00 | 588.30 | 2025-08-13 13:15:00 | 573.00 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-13 09:15:00 | 591.15 | 2025-08-13 13:15:00 | 573.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-08-18 12:30:00 | 573.65 | 2025-08-18 15:15:00 | 579.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-08-18 13:15:00 | 574.40 | 2025-08-18 15:15:00 | 579.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-01 11:15:00 | 569.10 | 2025-09-01 14:15:00 | 573.80 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-04 09:30:00 | 582.40 | 2025-09-04 14:15:00 | 569.10 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-16 11:30:00 | 578.15 | 2025-09-18 10:15:00 | 582.65 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-09-16 14:45:00 | 577.00 | 2025-09-18 10:15:00 | 582.65 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-17 09:45:00 | 576.80 | 2025-09-18 10:15:00 | 582.65 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-24 11:15:00 | 558.40 | 2025-09-26 09:15:00 | 530.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 558.40 | 2025-09-26 09:15:00 | 530.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:00:00 | 558.30 | 2025-09-26 09:15:00 | 530.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 11:15:00 | 558.40 | 2025-09-30 09:15:00 | 533.65 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-09-25 09:30:00 | 558.40 | 2025-09-30 09:15:00 | 533.65 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-09-25 10:00:00 | 558.30 | 2025-09-30 09:15:00 | 533.65 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-10-10 09:15:00 | 547.50 | 2025-10-10 09:15:00 | 550.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-28 12:45:00 | 535.20 | 2025-10-29 11:15:00 | 545.20 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-31 09:15:00 | 547.60 | 2025-10-31 14:15:00 | 540.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-04 10:30:00 | 536.75 | 2025-11-11 10:15:00 | 538.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-11-04 11:15:00 | 535.95 | 2025-11-11 10:15:00 | 538.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-11-12 14:45:00 | 538.45 | 2025-11-18 10:15:00 | 545.65 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-11-28 11:00:00 | 547.50 | 2025-12-01 11:15:00 | 538.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-12-05 09:15:00 | 527.00 | 2025-12-09 11:15:00 | 533.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-08 09:30:00 | 528.30 | 2025-12-09 11:15:00 | 533.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-09 12:30:00 | 528.80 | 2025-12-17 13:15:00 | 502.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-09 12:30:00 | 528.80 | 2025-12-19 14:15:00 | 475.92 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-02 09:15:00 | 524.20 | 2026-01-06 13:15:00 | 523.25 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-01-08 11:45:00 | 515.05 | 2026-01-12 09:15:00 | 489.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 515.05 | 2026-01-12 10:15:00 | 502.85 | STOP_HIT | 0.50 | 2.37% |
| BUY | retest2 | 2026-01-27 09:15:00 | 514.20 | 2026-01-27 10:15:00 | 503.60 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-27 09:45:00 | 512.50 | 2026-01-27 10:15:00 | 503.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-30 09:30:00 | 511.10 | 2026-02-02 09:15:00 | 494.40 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-02-01 09:15:00 | 509.40 | 2026-02-02 09:15:00 | 494.40 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2026-02-01 14:30:00 | 509.55 | 2026-02-02 09:15:00 | 494.40 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-02-12 15:00:00 | 578.85 | 2026-02-16 09:15:00 | 563.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-02-13 10:30:00 | 578.25 | 2026-02-16 09:15:00 | 563.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-02-13 12:15:00 | 576.65 | 2026-02-16 09:15:00 | 563.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-19 12:00:00 | 557.00 | 2026-02-19 12:15:00 | 572.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-23 11:30:00 | 556.80 | 2026-02-27 11:15:00 | 551.70 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2026-02-23 12:30:00 | 557.00 | 2026-02-27 11:15:00 | 551.70 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2026-03-24 15:15:00 | 480.10 | 2026-03-25 09:15:00 | 509.50 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest2 | 2026-04-02 13:45:00 | 495.40 | 2026-04-10 14:15:00 | 544.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 496.60 | 2026-04-10 14:15:00 | 546.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:30:00 | 496.35 | 2026-04-10 14:15:00 | 545.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:15:00 | 500.85 | 2026-04-16 12:15:00 | 525.80 | STOP_HIT | 1.00 | 4.98% |
| BUY | retest2 | 2026-04-07 10:15:00 | 507.85 | 2026-04-16 12:15:00 | 525.80 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2026-04-08 09:15:00 | 519.40 | 2026-04-16 12:15:00 | 525.80 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2026-04-30 12:45:00 | 595.15 | 2026-05-08 14:15:00 | 611.35 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2026-04-30 13:15:00 | 593.35 | 2026-05-08 14:15:00 | 611.35 | STOP_HIT | 1.00 | 3.03% |
| BUY | retest2 | 2026-05-04 09:30:00 | 600.90 | 2026-05-08 14:15:00 | 611.35 | STOP_HIT | 1.00 | 1.74% |

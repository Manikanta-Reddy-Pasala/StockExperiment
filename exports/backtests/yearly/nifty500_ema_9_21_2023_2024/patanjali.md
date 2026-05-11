# Patanjali Foods Ltd. (PATANJALI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 459.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 242 |
| ALERT1 | 141 |
| ALERT2 | 139 |
| ALERT2_SKIP | 90 |
| ALERT3 | 318 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 142 |
| PARTIAL | 15 |
| TARGET_HIT | 5 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 159 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 102
- **Target hits / Stop hits / Partials:** 5 / 139 / 15
- **Avg / median % per leg:** 0.58% / -0.71%
- **Sum % (uncompounded):** 91.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 15 | 23.8% | 5 | 58 | 0 | 0.24% | 15.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 63 | 15 | 23.8% | 5 | 58 | 0 | 0.24% | 15.4% |
| SELL (all) | 96 | 42 | 43.8% | 0 | 81 | 15 | 0.79% | 76.1% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.46% | 10.4% |
| SELL @ 3rd Alert (retest2) | 93 | 39 | 41.9% | 0 | 79 | 14 | 0.71% | 65.7% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.46% | 10.4% |
| retest2 (combined) | 156 | 54 | 34.6% | 5 | 137 | 14 | 0.52% | 81.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 09:15:00 | 307.72 | 309.81 | 310.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 12:15:00 | 307.02 | 308.68 | 309.44 | Break + close below crossover candle low |

### Cycle 2 — BUY (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 14:15:00 | 317.68 | 310.18 | 309.97 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 12:15:00 | 311.67 | 313.32 | 313.37 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 14:15:00 | 321.00 | 314.64 | 313.95 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 13:15:00 | 313.42 | 313.91 | 313.92 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 315.67 | 314.26 | 314.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 15:15:00 | 318.33 | 315.08 | 314.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 09:15:00 | 313.48 | 314.76 | 314.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 313.48 | 314.76 | 314.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 313.48 | 314.76 | 314.38 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 14:15:00 | 313.33 | 314.24 | 314.27 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 15:15:00 | 316.00 | 314.59 | 314.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 11:15:00 | 318.38 | 315.30 | 314.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 11:15:00 | 338.33 | 338.83 | 334.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 10:15:00 | 338.37 | 339.51 | 336.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 338.37 | 339.51 | 336.64 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 341.67 | 343.84 | 343.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 15:15:00 | 340.33 | 343.14 | 343.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 09:15:00 | 343.33 | 343.18 | 343.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 09:15:00 | 343.33 | 343.18 | 343.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 343.33 | 343.18 | 343.51 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 14:15:00 | 349.00 | 341.11 | 340.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 10:15:00 | 353.33 | 345.35 | 343.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 12:15:00 | 390.02 | 390.29 | 385.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 13:15:00 | 390.00 | 390.23 | 385.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 13:15:00 | 390.00 | 390.23 | 385.91 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 13:15:00 | 386.30 | 389.01 | 389.02 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 397.83 | 390.78 | 389.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 14:15:00 | 403.00 | 396.69 | 395.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-05 15:15:00 | 396.67 | 396.69 | 395.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 398.17 | 397.28 | 396.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 398.17 | 397.28 | 396.43 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 395.00 | 395.82 | 395.93 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 14:15:00 | 397.38 | 395.96 | 395.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 09:15:00 | 401.67 | 397.22 | 396.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 15:15:00 | 401.33 | 402.01 | 399.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 15:15:00 | 403.33 | 404.29 | 402.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 403.33 | 404.29 | 402.44 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 10:15:00 | 388.37 | 400.61 | 402.27 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 11:15:00 | 407.78 | 399.90 | 399.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 416.67 | 406.97 | 403.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 15:15:00 | 443.67 | 445.25 | 437.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 440.00 | 444.20 | 437.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 440.00 | 444.20 | 437.39 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 430.58 | 437.19 | 437.27 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 440.00 | 434.92 | 434.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 443.53 | 439.07 | 437.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 445.00 | 445.52 | 443.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 445.00 | 445.52 | 443.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 445.00 | 445.52 | 443.32 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 438.32 | 442.35 | 442.37 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 14:15:00 | 443.12 | 442.50 | 442.44 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 440.00 | 442.08 | 442.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 436.00 | 440.27 | 441.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 442.00 | 438.90 | 440.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 442.00 | 438.90 | 440.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 442.00 | 438.90 | 440.13 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 453.53 | 441.83 | 441.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 458.40 | 445.14 | 442.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 11:15:00 | 453.00 | 454.08 | 449.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 451.68 | 454.73 | 451.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 451.68 | 454.73 | 451.84 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 15:15:00 | 448.17 | 450.18 | 450.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 13:15:00 | 446.68 | 448.36 | 449.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 447.67 | 443.73 | 445.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 447.67 | 443.73 | 445.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 447.67 | 443.73 | 445.55 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 435.62 | 431.97 | 431.59 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 09:15:00 | 427.43 | 431.55 | 431.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 09:15:00 | 426.00 | 428.42 | 429.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 14:15:00 | 418.33 | 415.96 | 420.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 14:15:00 | 418.33 | 415.96 | 420.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 418.33 | 415.96 | 420.64 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 14:15:00 | 415.00 | 408.29 | 408.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 14:15:00 | 418.33 | 413.46 | 411.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 11:15:00 | 425.62 | 428.11 | 425.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 426.00 | 427.04 | 425.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 426.00 | 427.04 | 425.60 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 425.02 | 429.42 | 429.54 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 432.50 | 429.44 | 429.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 437.98 | 431.77 | 430.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 10:15:00 | 431.08 | 432.64 | 431.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 10:15:00 | 431.08 | 432.64 | 431.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 431.08 | 432.64 | 431.41 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 428.33 | 431.20 | 431.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 11:15:00 | 426.67 | 430.29 | 430.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 12:15:00 | 427.30 | 427.27 | 428.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 13:15:00 | 426.67 | 427.15 | 428.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 426.67 | 427.15 | 428.52 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 11:15:00 | 431.28 | 429.51 | 429.37 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 426.33 | 428.88 | 429.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 423.33 | 427.47 | 428.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 428.95 | 427.76 | 428.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 10:15:00 | 428.95 | 427.76 | 428.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 428.95 | 427.76 | 428.40 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 13:15:00 | 415.02 | 413.02 | 412.76 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 13:15:00 | 412.00 | 413.02 | 413.08 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 416.67 | 413.75 | 413.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 15:15:00 | 420.00 | 415.00 | 414.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 415.00 | 415.00 | 414.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 415.00 | 415.00 | 414.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 415.00 | 415.00 | 414.10 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 412.27 | 413.50 | 413.61 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 14:15:00 | 415.67 | 413.93 | 413.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 09:15:00 | 424.33 | 415.86 | 414.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 09:15:00 | 443.67 | 446.47 | 442.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 443.67 | 446.47 | 442.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 443.67 | 446.47 | 442.42 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 439.00 | 441.64 | 442.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 437.67 | 440.85 | 441.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 13:15:00 | 441.27 | 440.53 | 441.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 443.13 | 441.05 | 441.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 443.13 | 441.05 | 441.47 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 10:15:00 | 445.83 | 441.46 | 440.94 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 436.67 | 440.27 | 440.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 434.98 | 438.39 | 439.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 413.33 | 410.50 | 416.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 15:15:00 | 417.35 | 411.87 | 416.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 417.35 | 411.87 | 416.76 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 431.62 | 419.32 | 419.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 14:15:00 | 435.33 | 429.22 | 425.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 14:15:00 | 449.42 | 453.04 | 446.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 11:15:00 | 442.70 | 450.10 | 446.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 442.70 | 450.10 | 446.77 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 11:15:00 | 480.83 | 482.69 | 482.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 14:15:00 | 477.28 | 481.14 | 481.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 454.50 | 452.55 | 457.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 454.50 | 452.55 | 457.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 454.50 | 452.55 | 457.66 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 461.78 | 449.91 | 449.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 11:15:00 | 464.50 | 452.83 | 450.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 09:15:00 | 465.50 | 466.93 | 462.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 10:15:00 | 463.27 | 466.19 | 462.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 463.27 | 466.19 | 462.56 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 517.62 | 525.45 | 526.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 513.32 | 523.03 | 525.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 12:15:00 | 514.33 | 512.75 | 517.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 13:15:00 | 515.68 | 513.34 | 517.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 515.68 | 513.34 | 517.58 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 14:15:00 | 522.30 | 519.43 | 519.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 15:15:00 | 524.67 | 520.48 | 519.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 517.73 | 520.21 | 519.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 10:15:00 | 517.73 | 520.21 | 519.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 517.73 | 520.21 | 519.67 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 511.67 | 526.13 | 527.13 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 537.67 | 522.68 | 521.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 541.17 | 531.83 | 526.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 10:15:00 | 534.13 | 536.30 | 533.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 13:15:00 | 531.45 | 534.90 | 533.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 531.45 | 534.90 | 533.30 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 09:15:00 | 525.92 | 531.91 | 532.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 11:15:00 | 523.80 | 529.55 | 531.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 11:15:00 | 523.35 | 522.01 | 525.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 11:15:00 | 523.35 | 522.01 | 525.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 523.35 | 522.01 | 525.49 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 10:15:00 | 533.33 | 524.77 | 523.79 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 526.92 | 530.04 | 530.28 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 13:15:00 | 536.97 | 531.10 | 530.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 548.00 | 534.79 | 532.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 10:15:00 | 555.52 | 555.72 | 547.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 10:15:00 | 553.33 | 554.00 | 550.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 553.33 | 554.00 | 550.42 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 545.93 | 549.69 | 549.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 543.00 | 547.87 | 549.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 546.00 | 545.46 | 546.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 546.00 | 545.46 | 546.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 546.00 | 545.46 | 546.92 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 530.12 | 523.07 | 522.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 533.97 | 525.25 | 523.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 14:15:00 | 525.77 | 527.42 | 525.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 14:15:00 | 525.77 | 527.42 | 525.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 525.77 | 527.42 | 525.76 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 14:15:00 | 523.87 | 524.98 | 525.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 15:15:00 | 521.33 | 524.25 | 524.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 14:15:00 | 524.95 | 522.37 | 523.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 524.95 | 522.37 | 523.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 524.95 | 522.37 | 523.25 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 10:15:00 | 527.25 | 523.80 | 523.73 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 09:15:00 | 515.53 | 523.00 | 523.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 513.20 | 518.66 | 520.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 13:15:00 | 516.63 | 516.59 | 519.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 516.88 | 514.31 | 517.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 516.88 | 514.31 | 517.20 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 13:15:00 | 527.32 | 519.41 | 518.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 535.00 | 522.53 | 520.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 527.70 | 546.27 | 541.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 527.70 | 546.27 | 541.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 527.70 | 546.27 | 541.43 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 532.33 | 538.15 | 538.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 526.00 | 533.88 | 536.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 516.00 | 514.52 | 520.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 518.20 | 515.25 | 520.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 518.20 | 515.25 | 520.30 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 528.37 | 521.99 | 521.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 532.25 | 526.21 | 524.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 10:15:00 | 528.27 | 529.94 | 527.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 10:15:00 | 528.27 | 529.94 | 527.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 528.27 | 529.94 | 527.10 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 15:15:00 | 546.00 | 547.60 | 547.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 541.00 | 546.28 | 547.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 520.68 | 518.44 | 524.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 14:15:00 | 521.33 | 518.64 | 522.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 521.33 | 518.64 | 522.41 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 534.33 | 524.60 | 524.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 539.55 | 527.59 | 525.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 13:15:00 | 531.33 | 532.23 | 528.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 533.65 | 532.51 | 529.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 533.65 | 532.51 | 529.31 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 14:15:00 | 523.28 | 528.10 | 528.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 515.70 | 525.18 | 527.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 513.63 | 513.45 | 518.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 12:15:00 | 461.70 | 453.59 | 462.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 461.70 | 453.59 | 462.78 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 468.00 | 465.08 | 464.73 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 10:15:00 | 458.37 | 463.40 | 464.01 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 472.67 | 464.53 | 464.25 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 454.88 | 462.67 | 463.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 11:15:00 | 451.80 | 460.50 | 462.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 453.05 | 452.74 | 456.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 455.90 | 453.70 | 455.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 455.90 | 453.70 | 455.78 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 458.43 | 456.50 | 456.44 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 10:15:00 | 455.67 | 456.33 | 456.37 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 11:15:00 | 463.77 | 457.82 | 457.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 465.27 | 459.31 | 457.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 460.32 | 460.43 | 458.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 15:15:00 | 460.00 | 460.34 | 458.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 460.00 | 460.34 | 458.74 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 13:15:00 | 453.90 | 457.44 | 457.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 14:15:00 | 447.87 | 455.53 | 456.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 457.38 | 455.01 | 456.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 457.38 | 455.01 | 456.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 457.38 | 455.01 | 456.40 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 13:15:00 | 458.68 | 454.67 | 454.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 463.90 | 456.52 | 455.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 09:15:00 | 461.52 | 462.13 | 459.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 461.52 | 462.13 | 459.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 461.52 | 462.13 | 459.72 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 10:15:00 | 460.00 | 461.93 | 461.93 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 13:15:00 | 463.63 | 462.15 | 462.01 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 457.48 | 461.31 | 461.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 442.18 | 447.82 | 451.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 14:15:00 | 442.65 | 442.46 | 446.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 14:15:00 | 442.65 | 442.46 | 446.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 442.65 | 442.46 | 446.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 442.65 | 442.46 | 446.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 441.67 | 441.09 | 443.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 441.67 | 441.09 | 443.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 447.52 | 442.41 | 444.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 447.52 | 442.41 | 444.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 452.32 | 444.39 | 444.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 453.33 | 444.39 | 444.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 11:15:00 | 450.92 | 445.70 | 445.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 14:15:00 | 468.03 | 451.53 | 448.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 458.48 | 461.37 | 457.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 458.48 | 461.37 | 457.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 458.48 | 461.37 | 457.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 12:45:00 | 462.78 | 461.61 | 458.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 13:45:00 | 463.42 | 462.22 | 458.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-24 11:15:00 | 509.06 | 498.35 | 488.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 12:15:00 | 501.63 | 510.22 | 511.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 15:15:00 | 499.67 | 506.36 | 509.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 15:15:00 | 483.23 | 482.36 | 489.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 12:15:00 | 482.98 | 483.23 | 487.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 482.98 | 483.23 | 487.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:15:00 | 481.97 | 483.23 | 487.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 479.02 | 478.77 | 481.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 12:15:00 | 457.87 | 466.04 | 472.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 13:15:00 | 455.07 | 464.14 | 470.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 15:15:00 | 454.63 | 454.20 | 460.18 | SL hit (close>ema200) qty=0.50 sl=454.20 alert=retest2 |

### Cycle 76 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 457.28 | 450.21 | 450.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 15:15:00 | 461.00 | 453.60 | 451.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 11:15:00 | 469.67 | 470.83 | 466.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 12:00:00 | 469.67 | 470.83 | 466.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 475.00 | 470.75 | 468.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:15:00 | 478.33 | 471.57 | 470.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 480.38 | 473.75 | 472.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:15:00 | 478.35 | 484.43 | 483.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 478.03 | 482.16 | 482.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 478.03 | 482.16 | 482.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 15:15:00 | 476.50 | 480.21 | 481.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 477.97 | 477.51 | 479.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 477.97 | 477.51 | 479.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 477.97 | 477.51 | 479.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 477.97 | 477.51 | 479.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 495.00 | 480.44 | 480.22 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 442.88 | 476.43 | 479.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 389.98 | 459.14 | 471.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 440.58 | 435.88 | 452.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:45:00 | 441.40 | 435.88 | 452.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 449.67 | 441.11 | 448.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 452.45 | 441.11 | 448.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 450.90 | 443.06 | 448.82 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 455.00 | 451.38 | 451.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 458.27 | 452.76 | 451.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 492.83 | 492.83 | 483.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 492.83 | 492.83 | 483.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 488.97 | 489.97 | 486.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:30:00 | 481.67 | 489.97 | 486.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 482.12 | 488.13 | 486.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 482.12 | 488.13 | 486.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 483.50 | 487.20 | 485.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:15:00 | 481.80 | 487.20 | 485.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 485.83 | 486.06 | 485.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:30:00 | 486.20 | 486.06 | 485.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 492.00 | 487.25 | 486.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:45:00 | 494.97 | 489.80 | 487.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:30:00 | 495.33 | 490.57 | 488.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 496.00 | 492.01 | 489.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 484.05 | 489.19 | 489.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 484.05 | 489.19 | 489.49 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 490.70 | 488.85 | 488.70 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 486.38 | 488.67 | 488.73 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 489.35 | 488.80 | 488.79 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 487.55 | 488.84 | 488.85 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 489.00 | 488.87 | 488.86 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 484.80 | 488.06 | 488.49 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 515.00 | 492.40 | 490.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 519.70 | 512.66 | 505.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 507.93 | 513.51 | 509.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 507.93 | 513.51 | 509.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 507.93 | 513.51 | 509.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 507.93 | 513.51 | 509.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 505.03 | 511.82 | 509.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:45:00 | 504.33 | 511.82 | 509.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 543.33 | 556.31 | 544.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 542.17 | 556.31 | 544.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 546.63 | 554.37 | 545.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:45:00 | 548.93 | 553.64 | 545.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:45:00 | 548.72 | 553.12 | 547.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 550.88 | 547.99 | 546.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:30:00 | 549.43 | 548.29 | 547.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 546.87 | 548.00 | 547.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 546.87 | 548.00 | 547.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 546.25 | 547.65 | 547.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 546.25 | 547.65 | 547.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 544.00 | 546.92 | 546.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 544.87 | 546.92 | 546.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 549.67 | 547.50 | 547.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 545.15 | 547.39 | 547.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 545.15 | 547.39 | 547.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 542.65 | 546.44 | 546.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 551.77 | 546.07 | 546.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 551.77 | 546.07 | 546.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 551.77 | 546.07 | 546.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 551.77 | 546.07 | 546.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 554.67 | 547.79 | 547.18 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 14:15:00 | 546.20 | 547.79 | 547.95 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 15:15:00 | 549.47 | 548.13 | 548.09 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 545.73 | 547.65 | 547.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 10:15:00 | 543.60 | 546.84 | 547.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 09:15:00 | 527.92 | 521.31 | 525.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 527.92 | 521.31 | 525.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 527.92 | 521.31 | 525.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:00:00 | 527.92 | 521.31 | 525.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 528.82 | 522.81 | 525.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 528.82 | 522.81 | 525.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 530.33 | 524.31 | 526.19 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 537.65 | 528.21 | 527.71 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 517.35 | 525.85 | 526.88 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 14:15:00 | 530.37 | 527.22 | 527.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 10:15:00 | 537.03 | 529.85 | 528.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 15:15:00 | 533.33 | 535.72 | 532.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 534.42 | 535.72 | 532.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 534.95 | 535.56 | 532.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 531.58 | 535.56 | 532.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 536.33 | 535.72 | 533.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 531.83 | 535.72 | 533.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 534.08 | 535.39 | 533.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:15:00 | 521.75 | 535.39 | 533.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 533.35 | 534.98 | 533.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 14:00:00 | 535.83 | 535.15 | 533.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 537.67 | 535.15 | 533.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 565.33 | 568.91 | 568.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 565.33 | 568.91 | 568.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 560.67 | 566.91 | 568.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 15:15:00 | 563.70 | 560.76 | 563.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 15:15:00 | 563.70 | 560.76 | 563.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 563.70 | 560.76 | 563.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 589.57 | 560.76 | 563.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 589.67 | 566.54 | 565.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 11:15:00 | 599.67 | 587.18 | 579.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 600.02 | 600.30 | 592.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 14:00:00 | 600.02 | 600.30 | 592.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 595.00 | 599.24 | 593.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 593.05 | 599.24 | 593.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 592.38 | 597.19 | 593.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 602.98 | 596.29 | 594.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 589.92 | 593.78 | 593.31 | SL hit (close<static) qty=1.00 sl=590.45 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 588.48 | 592.72 | 592.87 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 600.03 | 593.88 | 593.21 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 588.35 | 592.76 | 592.82 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 15:15:00 | 599.67 | 592.59 | 592.59 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 590.43 | 592.16 | 592.39 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 597.52 | 592.14 | 591.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 598.33 | 594.32 | 592.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 598.87 | 603.81 | 600.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 598.87 | 603.81 | 600.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 598.87 | 603.81 | 600.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 598.87 | 603.81 | 600.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 605.17 | 604.08 | 600.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 598.83 | 604.08 | 600.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 598.30 | 602.93 | 600.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 598.32 | 602.93 | 600.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 597.65 | 601.87 | 600.36 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 593.57 | 599.39 | 599.45 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 602.72 | 599.46 | 599.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 627.23 | 605.47 | 602.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 13:15:00 | 632.70 | 632.91 | 626.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 14:00:00 | 632.70 | 632.91 | 626.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 631.87 | 631.95 | 628.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:30:00 | 630.37 | 631.95 | 628.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 630.47 | 631.65 | 629.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 630.47 | 631.65 | 629.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 630.95 | 631.51 | 629.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 628.33 | 631.51 | 629.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 630.12 | 631.23 | 629.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 629.02 | 631.23 | 629.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 627.98 | 630.58 | 629.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 627.98 | 630.58 | 629.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 628.33 | 630.13 | 629.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 628.33 | 630.13 | 629.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 627.95 | 629.70 | 629.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:30:00 | 632.92 | 629.32 | 628.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:00:00 | 638.73 | 631.20 | 629.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 10:15:00 | 626.00 | 630.54 | 629.96 | SL hit (close<static) qty=1.00 sl=627.33 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 624.15 | 629.26 | 629.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 622.82 | 627.97 | 628.83 | Break + close below crossover candle low |

### Cycle 108 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 641.00 | 630.33 | 629.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 644.50 | 639.43 | 635.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 652.72 | 654.07 | 649.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 14:00:00 | 652.72 | 654.07 | 649.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 647.67 | 652.58 | 649.94 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 643.33 | 647.88 | 648.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 13:15:00 | 633.33 | 644.15 | 646.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 14:15:00 | 643.77 | 641.13 | 643.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 643.77 | 641.13 | 643.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 643.77 | 641.13 | 643.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:45:00 | 646.33 | 641.13 | 643.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 640.68 | 641.04 | 643.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 639.37 | 640.01 | 642.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 643.85 | 638.43 | 638.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 643.85 | 638.43 | 638.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 648.33 | 641.55 | 639.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 638.33 | 641.84 | 640.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 638.33 | 641.84 | 640.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 638.33 | 641.84 | 640.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 638.33 | 641.84 | 640.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 641.00 | 641.67 | 640.71 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 638.70 | 640.00 | 640.15 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 643.30 | 640.59 | 640.30 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 09:15:00 | 619.50 | 636.55 | 638.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 615.88 | 623.43 | 629.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 616.00 | 615.52 | 620.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:00:00 | 616.00 | 615.52 | 620.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 578.13 | 574.02 | 576.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 578.13 | 574.02 | 576.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 578.33 | 574.88 | 576.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 575.38 | 574.88 | 576.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:45:00 | 575.02 | 574.98 | 576.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 574.48 | 574.74 | 575.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 574.77 | 573.77 | 575.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 568.63 | 563.87 | 566.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 568.63 | 563.87 | 566.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 567.67 | 564.63 | 566.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 568.62 | 564.63 | 566.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 566.83 | 565.54 | 566.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 559.05 | 565.54 | 566.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 546.61 | 556.12 | 560.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 546.27 | 556.12 | 560.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 545.76 | 556.12 | 560.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 546.03 | 556.12 | 560.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 544.98 | 542.19 | 547.86 | SL hit (close>ema200) qty=0.50 sl=542.19 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 562.37 | 550.54 | 549.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 564.00 | 554.96 | 551.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 571.78 | 572.35 | 566.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:00:00 | 571.78 | 572.35 | 566.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 570.80 | 572.36 | 568.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 570.80 | 572.36 | 568.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 579.67 | 573.82 | 569.32 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 568.33 | 570.41 | 570.63 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 578.00 | 571.69 | 571.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 12:15:00 | 582.38 | 576.60 | 573.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 580.47 | 581.00 | 577.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:30:00 | 581.95 | 581.00 | 577.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 578.50 | 580.50 | 577.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 578.50 | 580.50 | 577.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 575.62 | 579.14 | 577.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 575.62 | 579.14 | 577.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 580.33 | 579.38 | 577.74 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 574.28 | 576.62 | 576.92 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 579.63 | 577.22 | 577.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 582.97 | 578.37 | 577.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 13:15:00 | 582.60 | 584.38 | 581.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 14:00:00 | 582.60 | 584.38 | 581.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 592.73 | 586.05 | 582.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:30:00 | 586.28 | 586.05 | 582.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 579.53 | 585.85 | 583.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 579.53 | 585.85 | 583.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 580.33 | 584.75 | 583.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 577.45 | 584.75 | 583.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 589.00 | 583.92 | 583.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:30:00 | 585.07 | 583.92 | 583.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 579.28 | 583.81 | 583.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 578.52 | 583.81 | 583.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 578.33 | 582.71 | 582.73 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 14:15:00 | 589.75 | 582.61 | 582.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 15:15:00 | 592.67 | 584.62 | 583.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 14:15:00 | 591.52 | 591.57 | 588.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 591.52 | 591.57 | 588.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 584.97 | 590.25 | 587.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 576.00 | 590.25 | 587.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 571.55 | 586.51 | 586.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 571.55 | 586.51 | 586.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 567.45 | 582.70 | 584.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 561.20 | 574.30 | 579.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 568.02 | 566.18 | 574.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:00:00 | 568.02 | 566.18 | 574.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 572.62 | 567.47 | 574.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:30:00 | 569.82 | 567.47 | 574.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 572.05 | 568.38 | 573.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 572.58 | 568.38 | 573.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 577.13 | 570.13 | 574.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 577.13 | 570.13 | 574.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 578.80 | 571.87 | 574.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 578.80 | 571.87 | 574.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 585.67 | 576.79 | 576.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 596.60 | 584.82 | 580.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 590.62 | 592.06 | 586.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 590.62 | 592.06 | 586.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 588.75 | 591.06 | 587.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 592.92 | 590.66 | 587.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 12:00:00 | 591.13 | 591.08 | 588.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 593.67 | 591.89 | 589.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 592.92 | 594.30 | 592.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 596.00 | 594.64 | 592.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:15:00 | 597.40 | 594.64 | 592.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 613.30 | 594.91 | 593.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 13:15:00 | 613.53 | 620.25 | 620.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 613.53 | 620.25 | 620.41 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 622.45 | 620.84 | 620.64 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 615.67 | 620.02 | 620.44 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 618.18 | 616.05 | 615.86 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 12:15:00 | 610.92 | 615.48 | 615.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 591.17 | 610.29 | 613.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 597.07 | 591.14 | 596.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 14:15:00 | 597.07 | 591.14 | 596.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 597.07 | 591.14 | 596.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 598.00 | 591.14 | 596.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 596.63 | 592.24 | 596.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 601.40 | 592.24 | 596.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 593.55 | 592.50 | 596.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 10:45:00 | 591.03 | 591.83 | 595.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 595.67 | 588.84 | 588.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 595.67 | 588.84 | 588.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 599.47 | 592.75 | 590.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 620.00 | 621.02 | 615.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:00:00 | 622.97 | 621.41 | 615.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 618.90 | 621.16 | 616.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 618.90 | 621.16 | 616.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 615.97 | 619.35 | 616.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 615.97 | 619.35 | 616.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 619.33 | 619.34 | 617.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 617.68 | 619.01 | 617.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 618.68 | 618.95 | 617.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 616.80 | 618.95 | 617.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 618.95 | 618.95 | 617.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 616.53 | 618.95 | 617.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 618.58 | 618.87 | 617.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:30:00 | 617.82 | 618.87 | 617.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 625.28 | 620.99 | 619.07 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 12:15:00 | 612.50 | 617.83 | 617.98 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 624.07 | 619.06 | 618.51 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 615.30 | 617.98 | 618.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 10:15:00 | 612.33 | 615.69 | 616.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 614.47 | 613.50 | 615.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 614.47 | 613.50 | 615.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 614.47 | 613.50 | 615.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:45:00 | 616.33 | 613.50 | 615.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 614.50 | 613.70 | 615.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 618.02 | 613.70 | 615.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 613.03 | 613.56 | 614.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 10:45:00 | 611.25 | 613.17 | 614.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:00:00 | 611.78 | 612.89 | 614.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:00:00 | 611.78 | 612.85 | 614.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:30:00 | 611.65 | 612.29 | 613.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 610.45 | 611.92 | 613.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:45:00 | 608.78 | 610.77 | 612.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:00:00 | 607.85 | 608.05 | 610.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 615.47 | 606.49 | 607.31 | SL hit (close>static) qty=1.00 sl=613.63 alert=retest2 |

### Cycle 132 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 613.93 | 608.94 | 608.35 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 598.95 | 607.58 | 608.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 593.02 | 604.67 | 606.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 14:15:00 | 598.67 | 595.91 | 599.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 14:15:00 | 598.67 | 595.91 | 599.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 598.67 | 595.91 | 599.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 598.67 | 595.91 | 599.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 599.67 | 596.66 | 599.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 589.00 | 596.66 | 599.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 602.30 | 595.55 | 596.47 | SL hit (close>static) qty=1.00 sl=600.67 alert=retest2 |

### Cycle 134 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 601.80 | 597.94 | 597.47 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 586.68 | 595.68 | 596.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 15:15:00 | 576.67 | 591.88 | 594.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 580.50 | 580.43 | 586.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 15:00:00 | 580.50 | 580.43 | 586.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 575.85 | 574.05 | 579.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 575.85 | 574.05 | 579.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 571.83 | 573.76 | 578.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:45:00 | 568.17 | 571.40 | 575.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 14:15:00 | 568.17 | 571.40 | 575.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 13:15:00 | 579.63 | 574.85 | 575.42 | SL hit (close>static) qty=1.00 sl=578.83 alert=retest2 |

### Cycle 136 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 582.73 | 576.43 | 576.08 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 572.33 | 575.92 | 576.16 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 591.20 | 578.98 | 577.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 594.98 | 588.39 | 584.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 11:15:00 | 616.50 | 617.56 | 610.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:30:00 | 619.92 | 617.56 | 610.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 611.98 | 616.07 | 611.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 611.72 | 616.07 | 611.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 618.40 | 616.53 | 611.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 608.25 | 616.53 | 611.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 614.82 | 619.22 | 616.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 611.97 | 619.22 | 616.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 613.50 | 618.08 | 616.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 13:00:00 | 615.70 | 616.83 | 615.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:45:00 | 616.67 | 616.26 | 615.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 616.73 | 615.81 | 615.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:45:00 | 618.82 | 616.71 | 616.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 618.97 | 618.12 | 617.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 618.97 | 618.12 | 617.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 618.32 | 618.23 | 617.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 614.13 | 618.23 | 617.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 611.87 | 616.96 | 616.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 611.87 | 616.96 | 616.76 | SL hit (close<static) qty=1.00 sl=612.27 alert=retest2 |

### Cycle 139 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 612.47 | 616.06 | 616.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 607.80 | 613.37 | 615.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 599.18 | 595.56 | 601.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 599.18 | 595.56 | 601.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 599.18 | 595.56 | 601.57 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 612.80 | 604.38 | 603.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 618.88 | 615.48 | 611.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 616.05 | 616.36 | 613.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:00:00 | 616.05 | 616.36 | 613.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 612.33 | 615.51 | 613.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 613.32 | 615.51 | 613.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 609.88 | 614.38 | 613.33 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 608.33 | 612.20 | 612.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 12:15:00 | 608.30 | 611.42 | 612.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 612.38 | 609.74 | 610.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 612.38 | 609.74 | 610.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 612.38 | 609.74 | 610.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 11:00:00 | 605.52 | 608.90 | 610.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 605.95 | 608.31 | 610.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:00:00 | 605.72 | 606.89 | 608.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:30:00 | 606.73 | 607.14 | 608.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 612.62 | 608.24 | 608.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 612.62 | 608.24 | 608.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 606.10 | 607.81 | 608.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 12:15:00 | 604.98 | 607.81 | 608.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:15:00 | 603.17 | 607.75 | 608.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 622.92 | 610.22 | 609.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 622.92 | 610.22 | 609.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 625.32 | 614.92 | 611.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 11:15:00 | 615.07 | 616.08 | 613.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:00:00 | 615.07 | 616.08 | 613.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 613.15 | 615.50 | 613.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 621.55 | 616.35 | 613.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 606.70 | 613.08 | 612.99 | SL hit (close<static) qty=1.00 sl=611.60 alert=retest2 |

### Cycle 143 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 607.00 | 611.87 | 612.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 600.93 | 609.68 | 611.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 604.88 | 595.31 | 601.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 604.88 | 595.31 | 601.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 604.88 | 595.31 | 601.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 604.88 | 595.31 | 601.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 601.00 | 596.45 | 601.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 597.33 | 596.45 | 601.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 587.70 | 594.70 | 600.31 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 610.67 | 601.91 | 600.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 612.35 | 605.74 | 603.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 610.65 | 611.42 | 607.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:45:00 | 610.88 | 611.42 | 607.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 606.68 | 610.47 | 607.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 606.68 | 610.47 | 607.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 612.67 | 610.91 | 607.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 612.18 | 610.91 | 607.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 608.33 | 610.40 | 607.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 614.00 | 610.40 | 607.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:45:00 | 613.33 | 610.90 | 608.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 613.97 | 610.90 | 608.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 604.25 | 609.22 | 608.52 | SL hit (close<static) qty=1.00 sl=607.33 alert=retest2 |

### Cycle 145 — SELL (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 15:15:00 | 602.65 | 607.91 | 607.98 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 609.32 | 608.19 | 608.11 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 606.00 | 608.02 | 608.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 15:15:00 | 602.32 | 606.88 | 607.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 598.67 | 598.51 | 602.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:15:00 | 603.87 | 598.51 | 602.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 603.15 | 599.44 | 602.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 606.62 | 599.44 | 602.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 598.28 | 599.21 | 601.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 13:45:00 | 595.47 | 598.01 | 600.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:45:00 | 594.97 | 599.15 | 600.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 608.70 | 600.91 | 600.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 608.70 | 600.91 | 600.59 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 595.97 | 600.46 | 600.58 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 14:15:00 | 607.47 | 601.73 | 601.13 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 595.83 | 600.10 | 600.59 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 616.33 | 602.10 | 601.27 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 595.57 | 604.63 | 605.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 14:15:00 | 593.45 | 600.79 | 603.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 608.23 | 600.39 | 602.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 608.23 | 600.39 | 602.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 608.23 | 600.39 | 602.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 608.23 | 600.39 | 602.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 609.87 | 602.29 | 603.12 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 609.70 | 603.77 | 603.72 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 600.75 | 603.59 | 603.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 586.53 | 598.88 | 601.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 594.80 | 588.77 | 594.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 14:15:00 | 594.80 | 588.77 | 594.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 594.80 | 588.77 | 594.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 594.80 | 588.77 | 594.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 590.87 | 589.19 | 594.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 602.67 | 589.19 | 594.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 600.22 | 591.40 | 594.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:00:00 | 593.82 | 593.52 | 595.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 607.17 | 597.62 | 596.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 607.17 | 597.62 | 596.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 608.23 | 602.65 | 600.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 14:15:00 | 617.63 | 617.78 | 613.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 15:00:00 | 617.63 | 617.78 | 613.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 609.93 | 615.72 | 612.97 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 608.83 | 611.35 | 611.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 606.62 | 609.40 | 610.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 578.90 | 569.14 | 584.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 578.90 | 569.14 | 584.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 578.90 | 569.14 | 584.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 578.90 | 569.14 | 584.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 593.33 | 573.98 | 585.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 572.40 | 573.98 | 585.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:15:00 | 578.07 | 575.39 | 580.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 575.98 | 576.23 | 578.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 583.52 | 579.71 | 579.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 583.52 | 579.71 | 579.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 586.12 | 581.79 | 580.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 584.53 | 584.81 | 582.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 13:45:00 | 585.17 | 584.81 | 582.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 584.97 | 585.70 | 583.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 584.83 | 585.70 | 583.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 582.97 | 585.16 | 583.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:00:00 | 582.97 | 585.16 | 583.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 581.02 | 584.33 | 583.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 581.02 | 584.33 | 583.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 581.33 | 583.73 | 583.38 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 581.67 | 582.91 | 583.06 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 10:15:00 | 585.00 | 583.25 | 583.18 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 582.83 | 583.30 | 583.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 15:15:00 | 580.00 | 582.42 | 582.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 573.92 | 573.57 | 576.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 573.92 | 573.57 | 576.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 573.92 | 573.57 | 576.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 577.58 | 573.57 | 576.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 572.80 | 573.42 | 576.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 573.17 | 573.42 | 576.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 575.33 | 573.94 | 575.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:30:00 | 576.50 | 573.94 | 575.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 574.70 | 574.09 | 575.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 575.33 | 574.09 | 575.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 576.00 | 574.47 | 575.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 576.38 | 574.47 | 575.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 579.47 | 575.47 | 575.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 579.47 | 575.47 | 575.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 580.30 | 576.44 | 576.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 585.00 | 580.48 | 578.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 583.00 | 583.01 | 580.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 583.00 | 583.01 | 580.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 582.07 | 582.75 | 581.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 582.07 | 582.75 | 581.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 585.30 | 583.26 | 581.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 584.75 | 583.26 | 581.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 582.73 | 584.05 | 582.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:15:00 | 584.28 | 584.05 | 582.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 585.08 | 584.25 | 582.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 585.37 | 584.25 | 582.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 13:00:00 | 585.60 | 585.32 | 583.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:15:00 | 587.00 | 585.27 | 583.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 15:00:00 | 587.15 | 585.64 | 584.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 587.60 | 593.73 | 590.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 587.60 | 593.73 | 590.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 589.23 | 592.83 | 590.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 15:00:00 | 591.18 | 591.15 | 590.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:00:00 | 594.02 | 592.12 | 590.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:45:00 | 593.67 | 594.10 | 592.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 586.97 | 612.66 | 613.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 586.97 | 612.66 | 613.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 12:15:00 | 585.15 | 600.46 | 606.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 602.22 | 597.10 | 602.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 602.22 | 597.10 | 602.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 602.22 | 597.10 | 602.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:15:00 | 603.57 | 597.10 | 602.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 599.82 | 597.65 | 602.52 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 604.67 | 603.81 | 603.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 15:15:00 | 605.88 | 604.22 | 603.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 657.97 | 660.59 | 651.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 657.97 | 660.59 | 651.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 657.93 | 658.88 | 654.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:30:00 | 656.07 | 658.88 | 654.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 655.67 | 658.37 | 656.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:00:00 | 655.67 | 658.37 | 656.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 658.00 | 658.29 | 656.46 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 15:15:00 | 653.00 | 655.62 | 655.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 649.40 | 654.38 | 655.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 13:15:00 | 657.60 | 650.66 | 652.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 13:15:00 | 657.60 | 650.66 | 652.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 657.60 | 650.66 | 652.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 657.57 | 650.66 | 652.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 655.93 | 651.72 | 653.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:30:00 | 659.47 | 651.72 | 653.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 654.77 | 650.36 | 651.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 652.87 | 650.36 | 651.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 653.50 | 650.99 | 652.00 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 14:15:00 | 656.57 | 653.10 | 652.84 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 11:15:00 | 650.67 | 652.57 | 652.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 12:15:00 | 646.93 | 651.44 | 652.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 10:15:00 | 642.33 | 641.72 | 644.68 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-30 11:15:00 | 638.83 | 641.72 | 644.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 606.89 | 614.09 | 620.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 610.80 | 607.56 | 614.42 | SL hit (close>ema200) qty=0.50 sl=607.56 alert=retest1 |

### Cycle 168 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 606.67 | 601.83 | 601.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 609.20 | 603.97 | 602.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 603.60 | 605.12 | 604.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 603.60 | 605.12 | 604.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 603.60 | 605.12 | 604.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:00:00 | 603.60 | 605.12 | 604.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 602.97 | 604.69 | 604.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:30:00 | 604.53 | 604.69 | 604.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 600.17 | 603.79 | 603.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:00:00 | 600.17 | 603.79 | 603.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 598.17 | 602.66 | 603.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 13:15:00 | 596.37 | 601.40 | 602.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 597.77 | 595.65 | 598.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 14:00:00 | 597.77 | 595.65 | 598.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 602.33 | 596.99 | 598.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 602.93 | 596.99 | 598.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 602.00 | 597.99 | 599.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 590.70 | 597.99 | 599.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 594.70 | 593.42 | 595.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 594.70 | 593.42 | 595.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 586.03 | 591.93 | 594.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 14:00:00 | 576.33 | 584.70 | 590.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 15:00:00 | 577.53 | 583.26 | 588.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 575.83 | 571.72 | 571.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 575.83 | 571.72 | 571.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 579.80 | 574.53 | 573.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 576.70 | 577.58 | 575.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:00:00 | 576.70 | 577.58 | 575.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 575.83 | 577.23 | 575.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 575.83 | 577.23 | 575.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 579.17 | 577.62 | 575.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 581.90 | 578.66 | 576.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 581.80 | 579.40 | 577.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:45:00 | 582.57 | 579.80 | 577.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 14:45:00 | 585.30 | 578.34 | 577.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 581.00 | 578.87 | 578.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 578.67 | 578.87 | 578.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 574.80 | 578.06 | 577.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 574.80 | 578.06 | 577.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 571.17 | 576.68 | 577.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 571.17 | 576.68 | 577.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 569.07 | 575.16 | 576.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 572.47 | 568.35 | 572.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 572.47 | 568.35 | 572.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 572.47 | 568.35 | 572.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 570.23 | 568.35 | 572.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 571.20 | 568.92 | 572.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 571.73 | 568.92 | 572.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 570.50 | 569.24 | 571.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:15:00 | 566.63 | 569.52 | 571.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 567.87 | 567.89 | 570.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 569.40 | 562.11 | 561.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 569.40 | 562.11 | 561.26 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 562.43 | 563.52 | 563.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 561.50 | 563.12 | 563.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 558.83 | 558.15 | 559.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 558.83 | 558.15 | 559.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 558.83 | 558.15 | 559.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 558.83 | 558.15 | 559.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 557.97 | 557.57 | 558.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 559.63 | 557.57 | 558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 559.23 | 557.90 | 558.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 558.33 | 557.90 | 558.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 560.00 | 558.32 | 558.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 560.00 | 558.32 | 558.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 559.90 | 558.69 | 558.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 559.90 | 558.69 | 558.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 560.00 | 558.95 | 558.91 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 558.17 | 558.94 | 558.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 13:15:00 | 554.97 | 558.05 | 558.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 15:15:00 | 553.67 | 553.43 | 555.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 15:15:00 | 553.67 | 553.43 | 555.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 553.67 | 553.43 | 555.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 555.97 | 553.43 | 555.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 552.83 | 553.31 | 554.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 550.00 | 553.12 | 554.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 14:15:00 | 544.93 | 541.91 | 541.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 544.93 | 541.91 | 541.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 549.63 | 543.42 | 542.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 13:15:00 | 546.60 | 546.85 | 544.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:45:00 | 547.07 | 546.85 | 544.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 550.60 | 547.60 | 545.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 554.33 | 548.05 | 545.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 544.43 | 547.19 | 546.28 | SL hit (close<static) qty=1.00 sl=545.20 alert=retest2 |

### Cycle 177 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 543.93 | 545.65 | 545.86 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 14:15:00 | 548.57 | 546.23 | 546.11 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 543.73 | 545.94 | 546.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 542.93 | 544.86 | 545.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 545.23 | 544.57 | 545.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 545.23 | 544.57 | 545.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 545.23 | 544.57 | 545.23 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 551.87 | 546.70 | 546.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 553.30 | 548.02 | 546.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 550.57 | 550.67 | 548.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:30:00 | 549.67 | 550.67 | 548.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 550.13 | 551.15 | 549.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 550.83 | 551.15 | 549.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 549.63 | 550.85 | 549.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 549.63 | 550.85 | 549.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 549.70 | 550.62 | 549.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 549.70 | 550.62 | 549.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 545.67 | 549.63 | 549.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 542.67 | 549.63 | 549.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 545.97 | 548.90 | 549.10 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 551.30 | 549.38 | 549.30 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 544.27 | 548.96 | 549.34 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 553.30 | 549.05 | 548.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 11:15:00 | 554.67 | 551.90 | 550.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 553.93 | 554.13 | 552.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 553.93 | 554.13 | 552.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 553.93 | 554.13 | 552.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 552.63 | 554.13 | 552.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 552.73 | 553.85 | 552.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 555.30 | 553.95 | 552.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 555.47 | 554.19 | 552.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 555.67 | 554.90 | 553.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 14:15:00 | 610.83 | 597.97 | 582.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 632.70 | 637.69 | 638.09 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 646.67 | 639.71 | 638.90 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 633.70 | 637.84 | 638.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 630.03 | 634.86 | 636.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 621.37 | 619.14 | 624.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 621.37 | 619.14 | 624.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 621.37 | 619.14 | 624.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 621.37 | 619.14 | 624.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 624.60 | 620.58 | 623.86 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 633.20 | 626.55 | 625.86 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 621.33 | 627.14 | 627.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 618.70 | 624.55 | 626.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 606.73 | 605.55 | 609.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 606.50 | 605.55 | 609.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 603.00 | 605.04 | 607.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 601.67 | 604.10 | 606.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 597.40 | 602.02 | 605.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:45:00 | 601.77 | 598.20 | 600.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:45:00 | 600.83 | 599.62 | 601.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 604.83 | 600.66 | 601.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 604.83 | 600.66 | 601.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 605.83 | 601.70 | 601.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 605.83 | 601.70 | 601.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 606.33 | 602.62 | 602.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 606.33 | 602.62 | 602.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 607.63 | 603.62 | 602.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 600.97 | 603.09 | 602.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 10:15:00 | 600.97 | 603.09 | 602.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 600.97 | 603.09 | 602.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 600.97 | 603.09 | 602.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 599.67 | 602.41 | 602.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 599.67 | 602.41 | 602.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 596.83 | 601.29 | 601.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 15:15:00 | 596.03 | 599.47 | 600.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 599.13 | 593.30 | 596.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 599.13 | 593.30 | 596.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 599.13 | 593.30 | 596.06 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 597.83 | 596.15 | 596.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 600.00 | 597.03 | 596.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 10:15:00 | 602.60 | 602.98 | 600.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:00:00 | 602.60 | 602.98 | 600.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 602.10 | 602.80 | 600.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 602.10 | 602.80 | 600.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 600.87 | 602.42 | 600.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 603.20 | 601.93 | 600.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 602.97 | 601.77 | 600.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 602.63 | 601.44 | 600.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 598.30 | 600.81 | 600.61 | SL hit (close<static) qty=1.00 sl=598.67 alert=retest2 |

### Cycle 193 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 598.10 | 601.07 | 601.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 596.47 | 598.89 | 600.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 595.67 | 594.78 | 597.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:15:00 | 597.67 | 594.78 | 597.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 598.00 | 595.42 | 597.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 597.13 | 595.42 | 597.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 595.87 | 595.51 | 597.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 595.70 | 595.43 | 596.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 595.27 | 594.80 | 596.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:45:00 | 594.77 | 594.76 | 595.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 598.30 | 595.97 | 596.17 | SL hit (close>static) qty=1.00 sl=598.13 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 601.77 | 597.35 | 596.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 604.33 | 598.74 | 597.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 600.20 | 600.42 | 599.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 600.20 | 600.42 | 599.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 600.20 | 600.42 | 599.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 600.27 | 600.42 | 599.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 599.40 | 600.22 | 599.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 598.97 | 600.22 | 599.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 603.67 | 600.91 | 599.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 606.00 | 601.93 | 600.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:45:00 | 605.57 | 604.34 | 602.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:30:00 | 605.70 | 604.83 | 602.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:30:00 | 605.17 | 604.81 | 602.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 603.07 | 604.43 | 602.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 597.23 | 602.11 | 602.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 597.23 | 602.11 | 602.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 595.67 | 599.87 | 601.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 598.57 | 598.25 | 599.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 598.57 | 598.25 | 599.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 598.57 | 598.25 | 599.81 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 600.70 | 599.71 | 599.68 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 10:15:00 | 598.33 | 599.66 | 599.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 11:15:00 | 598.23 | 599.37 | 599.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 600.10 | 598.97 | 599.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 600.10 | 598.97 | 599.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 600.10 | 598.97 | 599.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 600.10 | 598.97 | 599.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 600.27 | 599.23 | 599.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 593.80 | 599.23 | 599.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 595.60 | 598.50 | 599.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 593.40 | 596.94 | 598.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:30:00 | 592.90 | 595.69 | 597.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 591.60 | 595.40 | 596.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 591.80 | 592.01 | 594.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 594.00 | 592.37 | 594.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 594.00 | 592.37 | 594.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 592.40 | 592.38 | 594.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 593.10 | 592.38 | 594.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 592.50 | 592.40 | 593.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 592.50 | 592.40 | 593.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 596.40 | 593.20 | 594.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 596.60 | 593.20 | 594.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 597.90 | 594.14 | 594.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 597.90 | 594.14 | 594.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 598.80 | 595.07 | 594.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 598.80 | 595.07 | 594.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 14:15:00 | 602.50 | 599.08 | 597.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 602.00 | 607.01 | 604.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 602.00 | 607.01 | 604.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 602.00 | 607.01 | 604.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 601.50 | 607.01 | 604.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 599.10 | 605.43 | 604.27 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 599.10 | 603.37 | 603.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 596.40 | 601.98 | 602.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 603.00 | 600.71 | 601.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 603.00 | 600.71 | 601.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 603.00 | 600.71 | 601.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 603.00 | 600.71 | 601.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 597.80 | 600.13 | 601.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 596.60 | 599.50 | 600.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 603.60 | 600.87 | 600.93 | SL hit (close>static) qty=1.00 sl=603.30 alert=retest2 |

### Cycle 200 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 593.80 | 586.38 | 585.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 597.15 | 593.06 | 589.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 593.80 | 594.83 | 591.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:15:00 | 593.40 | 594.83 | 591.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 591.05 | 594.08 | 591.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 591.05 | 594.08 | 591.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 596.25 | 594.51 | 592.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 598.35 | 595.17 | 592.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:00:00 | 598.55 | 595.98 | 594.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 599.10 | 596.17 | 594.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 592.00 | 594.70 | 594.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 592.00 | 594.70 | 594.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 584.00 | 591.91 | 593.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 591.25 | 590.17 | 591.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 591.25 | 590.17 | 591.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 591.25 | 590.17 | 591.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 590.00 | 590.17 | 591.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 588.00 | 589.74 | 591.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:30:00 | 586.05 | 588.04 | 590.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 586.00 | 588.04 | 590.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:45:00 | 585.60 | 584.93 | 587.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 585.75 | 585.22 | 587.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 585.95 | 585.12 | 586.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 586.80 | 585.12 | 586.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 584.50 | 585.00 | 586.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:30:00 | 587.45 | 585.00 | 586.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 587.15 | 585.37 | 586.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 586.10 | 585.37 | 586.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 586.45 | 585.58 | 586.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 587.00 | 585.58 | 586.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 585.75 | 585.62 | 586.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:15:00 | 585.80 | 585.62 | 586.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 590.95 | 586.68 | 586.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 588.50 | 586.68 | 586.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 591.15 | 587.58 | 587.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 591.15 | 587.58 | 587.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 595.85 | 590.76 | 589.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 593.85 | 593.93 | 592.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:15:00 | 590.80 | 593.93 | 592.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 587.95 | 592.74 | 591.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 587.95 | 592.74 | 591.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 584.25 | 591.04 | 591.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 580.70 | 586.77 | 588.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 585.85 | 584.86 | 587.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 586.15 | 584.86 | 587.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 583.90 | 584.67 | 587.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 584.90 | 584.67 | 587.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 588.90 | 585.52 | 587.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 588.90 | 585.52 | 587.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 590.20 | 586.45 | 587.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 590.20 | 586.45 | 587.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 588.85 | 586.93 | 587.60 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 591.00 | 588.29 | 588.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 591.95 | 589.02 | 588.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 11:15:00 | 607.55 | 608.04 | 603.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:00:00 | 607.55 | 608.04 | 603.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 603.30 | 607.09 | 603.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 603.30 | 607.09 | 603.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 605.05 | 606.68 | 603.96 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 578.00 | 599.74 | 601.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 572.90 | 594.38 | 598.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 14:15:00 | 576.70 | 576.44 | 583.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 15:00:00 | 576.70 | 576.44 | 583.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 575.00 | 575.68 | 581.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 571.60 | 575.14 | 580.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 581.30 | 576.32 | 576.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 581.30 | 576.32 | 576.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 587.45 | 581.35 | 579.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 585.15 | 588.61 | 586.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 585.15 | 588.61 | 586.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 585.15 | 588.61 | 586.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 585.15 | 588.61 | 586.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 584.75 | 587.84 | 586.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 582.30 | 587.84 | 586.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 585.00 | 586.81 | 586.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:00:00 | 585.00 | 586.81 | 586.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 587.65 | 586.98 | 586.29 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 582.55 | 586.10 | 586.42 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 11:15:00 | 593.20 | 587.52 | 587.03 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 584.45 | 587.28 | 587.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 583.45 | 585.83 | 586.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 584.70 | 584.55 | 585.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:00:00 | 584.70 | 584.55 | 585.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 582.00 | 583.15 | 584.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 582.00 | 583.15 | 584.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 579.30 | 579.98 | 581.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 578.55 | 579.98 | 581.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 572.55 | 570.35 | 573.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 572.55 | 570.35 | 573.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 569.90 | 570.42 | 571.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 567.40 | 569.97 | 571.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 566.15 | 568.81 | 570.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 566.75 | 567.77 | 569.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 569.00 | 567.44 | 568.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 569.00 | 567.75 | 568.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 567.75 | 567.75 | 568.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 14:15:00 | 539.03 | 547.88 | 554.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 14:15:00 | 537.84 | 547.88 | 554.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 14:15:00 | 538.41 | 547.88 | 554.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 14:15:00 | 540.55 | 547.88 | 554.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 14:15:00 | 539.36 | 547.88 | 554.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 543.55 | 542.35 | 550.19 | SL hit (close>ema200) qty=0.50 sl=542.35 alert=retest2 |

### Cycle 210 — BUY (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 14:15:00 | 537.65 | 533.53 | 533.36 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 533.20 | 533.48 | 533.49 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 13:15:00 | 537.75 | 534.33 | 533.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 14:15:00 | 541.10 | 535.69 | 534.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 556.95 | 557.49 | 552.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 11:00:00 | 556.95 | 557.49 | 552.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 549.30 | 554.76 | 552.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 549.30 | 554.76 | 552.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 550.95 | 554.00 | 552.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 547.85 | 554.00 | 552.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 547.50 | 552.52 | 552.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 547.50 | 552.52 | 552.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 11:15:00 | 548.05 | 551.63 | 551.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 545.85 | 548.93 | 549.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 548.15 | 548.07 | 549.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:30:00 | 548.05 | 548.07 | 549.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 550.00 | 548.45 | 549.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 550.00 | 548.45 | 549.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 548.65 | 548.49 | 549.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 548.50 | 548.49 | 549.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 548.80 | 548.56 | 549.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 548.80 | 548.56 | 549.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 546.90 | 547.78 | 548.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 545.80 | 547.78 | 548.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 546.10 | 542.90 | 544.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 542.85 | 543.85 | 544.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 545.00 | 541.90 | 542.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 547.90 | 543.63 | 543.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 547.90 | 543.63 | 543.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 552.00 | 547.75 | 545.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 570.80 | 571.47 | 565.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 13:45:00 | 569.85 | 571.47 | 565.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 567.35 | 574.53 | 571.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 567.35 | 574.53 | 571.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 563.35 | 572.30 | 570.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 563.35 | 572.30 | 570.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 564.80 | 569.34 | 569.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 564.05 | 568.28 | 569.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 15:15:00 | 527.90 | 522.96 | 529.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 516.50 | 521.30 | 528.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 502.95 | 501.66 | 509.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 502.95 | 501.66 | 509.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 511.45 | 504.15 | 509.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 511.45 | 504.15 | 509.55 | SL hit (close>ema400) qty=1.00 sl=509.55 alert=retest1 |

### Cycle 216 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 503.40 | 497.03 | 496.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 506.25 | 500.93 | 498.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 505.25 | 506.36 | 503.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 505.25 | 506.36 | 503.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 508.60 | 506.81 | 503.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 508.60 | 506.81 | 503.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 512.30 | 511.14 | 507.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 518.95 | 512.70 | 508.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 504.50 | 514.40 | 512.43 | SL hit (close<static) qty=1.00 sl=507.10 alert=retest2 |

### Cycle 217 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 515.45 | 520.76 | 521.09 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 525.10 | 520.18 | 519.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 528.00 | 522.11 | 520.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 15:15:00 | 532.35 | 532.93 | 529.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 532.60 | 532.94 | 530.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 532.60 | 532.94 | 530.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 532.60 | 532.94 | 530.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 530.55 | 532.47 | 530.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 530.55 | 532.47 | 530.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 530.00 | 531.97 | 530.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 528.15 | 531.97 | 530.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 529.85 | 531.55 | 530.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:00:00 | 535.85 | 531.13 | 530.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:00:00 | 535.60 | 531.77 | 531.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 534.45 | 531.46 | 531.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 525.25 | 529.75 | 530.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 525.25 | 529.75 | 530.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 522.20 | 527.47 | 529.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 526.20 | 525.61 | 527.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 526.20 | 525.61 | 527.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 522.85 | 525.05 | 527.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 519.95 | 523.48 | 526.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 517.50 | 521.63 | 524.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 493.95 | 502.25 | 506.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 503.10 | 502.42 | 506.46 | SL hit (close>ema200) qty=0.50 sl=502.42 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 500.00 | 497.24 | 496.90 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 494.75 | 496.75 | 496.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 490.70 | 495.34 | 496.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 10:15:00 | 491.15 | 491.00 | 493.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:45:00 | 491.60 | 491.00 | 493.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 491.00 | 491.00 | 492.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:45:00 | 492.70 | 491.00 | 492.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 483.85 | 481.82 | 485.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 484.75 | 481.82 | 485.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 484.40 | 482.45 | 485.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 481.95 | 482.76 | 485.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 489.30 | 486.06 | 485.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 489.30 | 486.06 | 485.76 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 476.95 | 486.20 | 486.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 476.25 | 481.13 | 483.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 479.00 | 478.31 | 481.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 479.00 | 478.31 | 481.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 479.00 | 478.31 | 481.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 476.00 | 478.26 | 480.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 473.75 | 478.26 | 480.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 481.20 | 473.51 | 472.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 481.20 | 473.51 | 472.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 484.10 | 475.63 | 473.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 479.75 | 481.46 | 477.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 479.75 | 481.46 | 477.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 477.40 | 480.65 | 477.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 477.40 | 480.65 | 477.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 476.60 | 479.84 | 477.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 476.40 | 479.84 | 477.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 477.85 | 479.44 | 477.77 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 470.40 | 475.84 | 476.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 465.30 | 473.73 | 475.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 468.95 | 464.56 | 469.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 468.95 | 464.56 | 469.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 468.95 | 464.56 | 469.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 465.55 | 464.90 | 468.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 460.05 | 468.24 | 469.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 463.00 | 465.06 | 466.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 471.80 | 467.02 | 466.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 471.80 | 467.02 | 466.68 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 464.95 | 466.54 | 466.68 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 468.40 | 466.88 | 466.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 473.00 | 468.10 | 467.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 469.95 | 470.01 | 468.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:00:00 | 469.95 | 470.01 | 468.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 468.95 | 469.80 | 468.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 468.95 | 469.80 | 468.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 467.00 | 469.24 | 468.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 467.25 | 469.24 | 468.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — SELL (started 2026-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 09:15:00 | 459.95 | 467.38 | 467.75 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 467.35 | 465.34 | 465.09 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 454.50 | 463.17 | 464.12 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 465.45 | 463.14 | 463.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 471.40 | 465.63 | 464.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 463.00 | 466.76 | 465.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 463.00 | 466.76 | 465.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 463.00 | 466.76 | 465.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 461.65 | 466.76 | 465.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — SELL (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 10:15:00 | 460.00 | 465.41 | 465.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 458.30 | 462.65 | 464.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 461.85 | 461.43 | 462.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 461.85 | 461.43 | 462.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 461.85 | 461.43 | 462.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 462.45 | 461.43 | 462.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 464.15 | 461.98 | 462.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:45:00 | 464.25 | 461.98 | 462.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 464.90 | 462.57 | 463.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:15:00 | 467.20 | 462.57 | 463.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 234 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 467.80 | 463.61 | 463.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 471.85 | 466.05 | 464.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 468.95 | 470.42 | 468.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 468.95 | 470.42 | 468.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 468.95 | 470.42 | 468.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:45:00 | 472.35 | 469.96 | 468.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 461.85 | 467.64 | 467.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 461.85 | 467.64 | 467.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 458.40 | 465.79 | 466.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 462.95 | 462.40 | 464.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 462.95 | 462.40 | 464.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 462.95 | 462.40 | 464.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 462.95 | 462.40 | 464.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 469.60 | 463.94 | 465.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 469.60 | 463.94 | 465.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 469.00 | 466.29 | 465.94 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 464.00 | 465.59 | 465.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 461.80 | 464.21 | 465.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 466.95 | 464.46 | 464.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 466.95 | 464.46 | 464.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 466.95 | 464.46 | 464.91 | EMA400 retest candle locked (from downside) |

### Cycle 238 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 468.00 | 465.47 | 465.31 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 463.35 | 465.13 | 465.19 | EMA200 below EMA400 |

### Cycle 240 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 465.85 | 465.27 | 465.25 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 465.00 | 465.22 | 465.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 455.90 | 463.36 | 464.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 463.20 | 460.13 | 461.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 463.20 | 460.13 | 461.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 463.20 | 460.13 | 461.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 465.05 | 460.13 | 461.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 461.50 | 460.41 | 461.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 460.50 | 460.04 | 461.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 459.75 | 457.28 | 457.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 458.30 | 457.76 | 457.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 242 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 458.30 | 457.76 | 457.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 460.05 | 458.37 | 458.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 460.50 | 460.76 | 459.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 460.50 | 460.76 | 459.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 459.90 | 460.59 | 459.76 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-19 12:45:00 | 462.78 | 2024-04-24 11:15:00 | 509.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 13:45:00 | 463.42 | 2024-04-24 11:15:00 | 509.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-06 13:15:00 | 481.97 | 2024-05-09 12:15:00 | 457.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 479.02 | 2024-05-09 13:15:00 | 455.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 13:15:00 | 481.97 | 2024-05-10 15:15:00 | 454.63 | STOP_HIT | 0.50 | 5.67% |
| SELL | retest2 | 2024-05-08 09:15:00 | 479.02 | 2024-05-10 15:15:00 | 454.63 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2024-05-24 12:15:00 | 478.33 | 2024-05-30 11:15:00 | 478.03 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-05-27 15:15:00 | 480.38 | 2024-05-30 11:15:00 | 478.03 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-05-30 10:15:00 | 478.35 | 2024-05-30 11:15:00 | 478.03 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-06-14 10:45:00 | 494.97 | 2024-06-19 09:15:00 | 484.05 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-06-14 12:30:00 | 495.33 | 2024-06-19 09:15:00 | 484.05 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-06-18 09:15:00 | 496.00 | 2024-06-19 09:15:00 | 484.05 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-07-02 14:45:00 | 548.93 | 2024-07-08 10:15:00 | 545.15 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-07-03 10:45:00 | 548.72 | 2024-07-08 10:15:00 | 545.15 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-07-04 09:15:00 | 550.88 | 2024-07-08 10:15:00 | 545.15 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-04 12:30:00 | 549.43 | 2024-07-08 10:15:00 | 545.15 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-23 14:00:00 | 535.83 | 2024-08-02 14:15:00 | 565.33 | STOP_HIT | 1.00 | 5.51% |
| BUY | retest2 | 2024-07-23 15:15:00 | 537.67 | 2024-08-02 14:15:00 | 565.33 | STOP_HIT | 1.00 | 5.14% |
| BUY | retest2 | 2024-08-12 09:15:00 | 602.98 | 2024-08-12 11:15:00 | 589.92 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-08-27 13:30:00 | 632.92 | 2024-08-28 10:15:00 | 626.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-27 15:00:00 | 638.73 | 2024-08-28 10:15:00 | 626.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-09-06 09:30:00 | 639.37 | 2024-09-10 09:15:00 | 643.85 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-09-26 09:15:00 | 575.38 | 2024-10-04 09:15:00 | 546.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 11:45:00 | 575.02 | 2024-10-04 09:15:00 | 546.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 14:30:00 | 574.48 | 2024-10-04 09:15:00 | 545.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 09:30:00 | 574.77 | 2024-10-04 09:15:00 | 546.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 09:15:00 | 575.38 | 2024-10-07 14:15:00 | 544.98 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest2 | 2024-09-26 11:45:00 | 575.02 | 2024-10-07 14:15:00 | 544.98 | STOP_HIT | 0.50 | 5.22% |
| SELL | retest2 | 2024-09-26 14:30:00 | 574.48 | 2024-10-07 14:15:00 | 544.98 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2024-09-27 09:30:00 | 574.77 | 2024-10-07 14:15:00 | 544.98 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2024-10-03 09:15:00 | 559.05 | 2024-10-08 14:15:00 | 562.37 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-10-31 10:15:00 | 592.92 | 2024-11-11 13:15:00 | 613.53 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2024-10-31 12:00:00 | 591.13 | 2024-11-11 13:15:00 | 613.53 | STOP_HIT | 1.00 | 3.79% |
| BUY | retest2 | 2024-10-31 14:45:00 | 593.67 | 2024-11-11 13:15:00 | 613.53 | STOP_HIT | 1.00 | 3.35% |
| BUY | retest2 | 2024-11-04 11:30:00 | 592.92 | 2024-11-11 13:15:00 | 613.53 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2024-11-04 13:15:00 | 597.40 | 2024-11-11 13:15:00 | 613.53 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest2 | 2024-11-05 09:15:00 | 613.30 | 2024-11-11 13:15:00 | 613.53 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-11-25 10:45:00 | 591.03 | 2024-11-28 10:15:00 | 595.67 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-11 10:45:00 | 611.25 | 2024-12-16 09:15:00 | 615.47 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-12-11 12:00:00 | 611.78 | 2024-12-16 09:15:00 | 615.47 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-12-11 14:00:00 | 611.78 | 2024-12-16 11:15:00 | 613.93 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-12-11 14:30:00 | 611.65 | 2024-12-16 11:15:00 | 613.93 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-12-12 09:45:00 | 608.78 | 2024-12-16 11:15:00 | 613.93 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-12 15:00:00 | 607.85 | 2024-12-16 11:15:00 | 613.93 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 589.00 | 2024-12-20 09:15:00 | 602.30 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-12-26 13:45:00 | 568.17 | 2024-12-27 13:15:00 | 579.63 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-26 14:15:00 | 568.17 | 2024-12-27 13:15:00 | 579.63 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-01-08 13:00:00 | 615.70 | 2025-01-10 09:15:00 | 611.87 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-01-08 14:45:00 | 616.67 | 2025-01-10 09:15:00 | 611.87 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-01-09 09:15:00 | 616.73 | 2025-01-10 09:15:00 | 611.87 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-01-09 09:45:00 | 618.82 | 2025-01-10 09:15:00 | 611.87 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-01-21 11:00:00 | 605.52 | 2025-01-22 14:15:00 | 622.92 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-01-21 12:00:00 | 605.95 | 2025-01-22 14:15:00 | 622.92 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-01-21 15:00:00 | 605.72 | 2025-01-22 14:15:00 | 622.92 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-01-22 09:30:00 | 606.73 | 2025-01-22 14:15:00 | 622.92 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-01-22 12:15:00 | 604.98 | 2025-01-22 14:15:00 | 622.92 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-01-22 13:15:00 | 603.17 | 2025-01-22 14:15:00 | 622.92 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-01-23 15:00:00 | 621.55 | 2025-01-24 11:15:00 | 606.70 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-01-31 09:15:00 | 614.00 | 2025-01-31 14:15:00 | 604.25 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-31 09:45:00 | 613.33 | 2025-01-31 14:15:00 | 604.25 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-01-31 10:15:00 | 613.97 | 2025-01-31 14:15:00 | 604.25 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-02-04 13:45:00 | 595.47 | 2025-02-06 09:15:00 | 608.70 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-02-05 11:45:00 | 594.97 | 2025-02-06 09:15:00 | 608.70 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-02-18 12:00:00 | 593.82 | 2025-02-18 14:15:00 | 607.17 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-03-03 09:15:00 | 572.40 | 2025-03-05 11:15:00 | 583.52 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-03 15:15:00 | 578.07 | 2025-03-05 11:15:00 | 583.52 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-03-04 15:00:00 | 575.98 | 2025-03-05 11:15:00 | 583.52 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-03-21 10:15:00 | 585.37 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-03-21 13:00:00 | 585.60 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-03-21 14:15:00 | 587.00 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-03-21 15:00:00 | 587.15 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-03-25 15:00:00 | 591.18 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-03-26 10:00:00 | 594.02 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-27 10:45:00 | 593.67 | 2025-04-07 09:15:00 | 586.97 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest1 | 2025-04-30 11:15:00 | 638.83 | 2025-05-07 09:15:00 | 606.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-04-30 11:15:00 | 638.83 | 2025-05-07 14:15:00 | 610.80 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2025-05-08 13:00:00 | 609.00 | 2025-05-12 14:15:00 | 606.67 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-05-19 14:00:00 | 576.33 | 2025-05-26 11:15:00 | 575.83 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-05-19 15:00:00 | 577.53 | 2025-05-26 11:15:00 | 575.83 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-05-28 12:00:00 | 581.90 | 2025-05-30 10:15:00 | 571.17 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-05-28 13:30:00 | 581.80 | 2025-05-30 10:15:00 | 571.17 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-28 14:45:00 | 582.57 | 2025-05-30 10:15:00 | 571.17 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-05-29 14:45:00 | 585.30 | 2025-05-30 10:15:00 | 571.17 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-06-02 14:15:00 | 566.63 | 2025-06-06 14:15:00 | 569.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-06-03 09:30:00 | 567.87 | 2025-06-06 14:15:00 | 569.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-06-19 11:15:00 | 550.00 | 2025-06-25 14:15:00 | 544.93 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2025-06-27 09:15:00 | 554.33 | 2025-06-27 13:15:00 | 544.43 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-11 13:45:00 | 555.30 | 2025-07-16 14:15:00 | 610.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-11 14:45:00 | 555.47 | 2025-07-16 14:15:00 | 611.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-14 09:45:00 | 555.67 | 2025-07-16 14:15:00 | 611.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-08 15:00:00 | 601.67 | 2025-08-12 15:15:00 | 606.33 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-08-11 09:30:00 | 597.40 | 2025-08-12 15:15:00 | 606.33 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-08-12 10:45:00 | 601.77 | 2025-08-12 15:15:00 | 606.33 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-12 12:45:00 | 600.83 | 2025-08-12 15:15:00 | 606.33 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-21 14:30:00 | 603.20 | 2025-08-22 12:15:00 | 598.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-22 09:45:00 | 602.97 | 2025-08-22 12:15:00 | 598.30 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-08-22 11:30:00 | 602.63 | 2025-08-22 12:15:00 | 598.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-25 09:30:00 | 603.30 | 2025-08-26 10:15:00 | 598.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-29 13:30:00 | 595.70 | 2025-09-01 14:15:00 | 598.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-01 10:15:00 | 595.27 | 2025-09-01 14:15:00 | 598.30 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-01 12:45:00 | 594.77 | 2025-09-01 14:15:00 | 598.30 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-03 13:00:00 | 606.00 | 2025-09-05 10:15:00 | 597.23 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-04 10:45:00 | 605.57 | 2025-09-05 10:15:00 | 597.23 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-04 11:30:00 | 605.70 | 2025-09-05 10:15:00 | 597.23 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-09-04 12:30:00 | 605.17 | 2025-09-05 10:15:00 | 597.23 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-11 11:30:00 | 593.40 | 2025-09-15 14:15:00 | 598.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-11 12:30:00 | 592.90 | 2025-09-15 14:15:00 | 598.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-12 10:45:00 | 591.60 | 2025-09-15 14:15:00 | 598.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-12 15:15:00 | 591.80 | 2025-09-15 14:15:00 | 598.80 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-24 11:15:00 | 596.60 | 2025-09-24 13:15:00 | 603.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-26 09:15:00 | 596.50 | 2025-10-06 09:15:00 | 593.80 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-09-26 11:00:00 | 596.50 | 2025-10-06 09:15:00 | 593.80 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-09-29 12:00:00 | 595.50 | 2025-10-06 09:15:00 | 593.80 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-09-30 10:00:00 | 594.00 | 2025-10-06 09:15:00 | 593.80 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-10-08 09:15:00 | 598.35 | 2025-10-10 13:15:00 | 592.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-08 15:00:00 | 598.55 | 2025-10-10 13:15:00 | 592.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-09 09:45:00 | 599.10 | 2025-10-10 13:15:00 | 592.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-14 11:30:00 | 586.05 | 2025-10-16 13:15:00 | 591.15 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-14 12:15:00 | 586.00 | 2025-10-16 13:15:00 | 591.15 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-15 09:45:00 | 585.60 | 2025-10-16 13:15:00 | 591.15 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-10-15 11:15:00 | 585.75 | 2025-10-16 13:15:00 | 591.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-06 10:30:00 | 571.60 | 2025-11-10 10:15:00 | 581.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-11-27 10:45:00 | 567.40 | 2025-12-04 14:15:00 | 539.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 09:15:00 | 566.15 | 2025-12-04 14:15:00 | 537.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 15:00:00 | 566.75 | 2025-12-04 14:15:00 | 538.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 569.00 | 2025-12-04 14:15:00 | 540.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:15:00 | 567.75 | 2025-12-04 14:15:00 | 539.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:45:00 | 567.40 | 2025-12-05 10:15:00 | 543.55 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-11-28 09:15:00 | 566.15 | 2025-12-05 10:15:00 | 543.55 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-11-28 15:00:00 | 566.75 | 2025-12-05 10:15:00 | 543.55 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-01 15:15:00 | 569.00 | 2025-12-05 10:15:00 | 543.55 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-12-02 09:15:00 | 567.75 | 2025-12-05 10:15:00 | 543.55 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-12-29 10:15:00 | 545.80 | 2026-01-01 09:15:00 | 547.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-12-30 13:00:00 | 546.10 | 2026-01-01 09:15:00 | 547.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-12-30 15:00:00 | 542.85 | 2026-01-01 09:15:00 | 547.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-31 15:00:00 | 545.00 | 2026-01-01 09:15:00 | 547.90 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-01-20 09:30:00 | 516.50 | 2026-01-22 09:15:00 | 511.45 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2026-01-22 11:30:00 | 506.55 | 2026-01-22 14:15:00 | 513.15 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-23 13:15:00 | 506.80 | 2026-02-02 11:15:00 | 481.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 13:15:00 | 506.80 | 2026-02-02 14:15:00 | 496.20 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2026-01-23 15:15:00 | 505.00 | 2026-02-03 11:15:00 | 503.40 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-02-06 11:00:00 | 518.95 | 2026-02-09 10:15:00 | 504.50 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-09 12:00:00 | 519.50 | 2026-02-13 09:15:00 | 515.45 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-02-20 14:00:00 | 535.85 | 2026-02-24 11:15:00 | 525.25 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-23 12:00:00 | 535.60 | 2026-02-24 11:15:00 | 525.25 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2026-02-23 13:30:00 | 534.45 | 2026-02-24 11:15:00 | 525.25 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-25 12:45:00 | 519.95 | 2026-03-04 09:15:00 | 493.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 519.95 | 2026-03-04 10:15:00 | 503.10 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2026-02-25 15:00:00 | 517.50 | 2026-03-09 09:15:00 | 491.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:00:00 | 517.50 | 2026-03-09 15:15:00 | 492.25 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2026-03-17 11:15:00 | 481.95 | 2026-03-18 10:15:00 | 489.30 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-03-20 14:30:00 | 476.00 | 2026-03-25 10:15:00 | 481.20 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-20 15:00:00 | 473.75 | 2026-03-25 10:15:00 | 481.20 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-04-01 10:45:00 | 465.55 | 2026-04-06 14:15:00 | 471.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-04-02 09:15:00 | 460.05 | 2026-04-06 14:15:00 | 471.80 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-04-06 09:15:00 | 463.00 | 2026-04-06 14:15:00 | 471.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-23 13:45:00 | 472.35 | 2026-04-24 09:15:00 | 461.85 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-05-04 11:45:00 | 460.50 | 2026-05-07 10:15:00 | 458.30 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2026-05-06 15:00:00 | 459.75 | 2026-05-07 10:15:00 | 458.30 | STOP_HIT | 1.00 | 0.32% |

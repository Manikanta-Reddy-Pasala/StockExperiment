# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1009.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 238 |
| ALERT1 | 143 |
| ALERT2 | 139 |
| ALERT2_SKIP | 100 |
| ALERT3 | 337 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 168 |
| PARTIAL | 30 |
| TARGET_HIT | 8 |
| STOP_HIT | 161 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 199 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 80 / 119
- **Target hits / Stop hits / Partials:** 8 / 161 / 30
- **Avg / median % per leg:** 0.80% / -0.79%
- **Sum % (uncompounded):** 158.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 87 | 21 | 24.1% | 5 | 82 | 0 | -0.22% | -19.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 87 | 21 | 24.1% | 5 | 82 | 0 | -0.22% | -19.5% |
| SELL (all) | 112 | 59 | 52.7% | 3 | 79 | 30 | 1.59% | 178.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.94% | -1.9% |
| SELL @ 3rd Alert (retest2) | 111 | 59 | 53.2% | 3 | 78 | 30 | 1.62% | 179.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.94% | -1.9% |
| retest2 (combined) | 198 | 80 | 40.4% | 8 | 160 | 30 | 0.81% | 160.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 336.35 | 342.35 | 342.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 09:15:00 | 332.20 | 339.47 | 341.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 336.05 | 335.39 | 338.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 14:15:00 | 336.05 | 335.39 | 338.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 336.05 | 335.39 | 338.27 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 15:15:00 | 340.00 | 338.58 | 338.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 344.15 | 339.72 | 339.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 09:15:00 | 347.90 | 347.91 | 345.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 15:15:00 | 348.45 | 349.00 | 347.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 348.45 | 349.00 | 347.09 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 09:15:00 | 333.35 | 354.49 | 356.29 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 11:15:00 | 338.00 | 335.35 | 335.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 340.15 | 337.61 | 336.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 15:15:00 | 360.80 | 361.03 | 358.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 412.20 | 413.95 | 407.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 412.20 | 413.95 | 407.41 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 396.90 | 404.39 | 404.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 15:15:00 | 394.50 | 402.41 | 403.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 11:15:00 | 402.00 | 401.63 | 403.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 402.00 | 401.63 | 403.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 402.00 | 401.63 | 403.04 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 10:15:00 | 405.65 | 399.60 | 399.15 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 14:15:00 | 399.40 | 402.61 | 403.02 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 409.25 | 403.19 | 403.17 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 15:15:00 | 402.40 | 403.44 | 403.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 10:15:00 | 397.55 | 401.56 | 402.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 11:15:00 | 392.95 | 391.40 | 393.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 392.85 | 391.60 | 393.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 392.85 | 391.60 | 393.05 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 12:15:00 | 392.20 | 389.64 | 389.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 14:15:00 | 394.70 | 391.00 | 390.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 11:15:00 | 392.00 | 393.40 | 391.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 11:15:00 | 392.00 | 393.40 | 391.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 392.00 | 393.40 | 391.78 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 389.95 | 391.30 | 391.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 385.95 | 390.23 | 390.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 10:15:00 | 391.40 | 389.36 | 390.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 10:15:00 | 391.40 | 389.36 | 390.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 391.40 | 389.36 | 390.19 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 13:15:00 | 392.45 | 390.75 | 390.68 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 14:15:00 | 389.85 | 390.57 | 390.61 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 394.25 | 391.24 | 390.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 397.05 | 392.75 | 391.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 13:15:00 | 389.70 | 392.77 | 391.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 13:15:00 | 389.70 | 392.77 | 391.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 389.70 | 392.77 | 391.90 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 15:15:00 | 388.00 | 391.03 | 391.21 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 09:15:00 | 405.75 | 393.97 | 392.54 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 14:15:00 | 386.10 | 393.51 | 394.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 15:15:00 | 378.70 | 384.34 | 388.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 379.95 | 376.61 | 381.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 379.95 | 376.61 | 381.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 379.95 | 376.61 | 381.02 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 12:15:00 | 378.00 | 370.84 | 370.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 14:15:00 | 381.70 | 373.93 | 371.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 12:15:00 | 375.60 | 376.95 | 374.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 12:15:00 | 375.60 | 376.95 | 374.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 375.60 | 376.95 | 374.51 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 14:15:00 | 373.70 | 374.54 | 374.55 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 09:15:00 | 377.05 | 374.59 | 374.54 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 373.35 | 374.34 | 374.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 11:15:00 | 372.45 | 373.96 | 374.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 373.45 | 373.21 | 373.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 14:15:00 | 373.45 | 373.21 | 373.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 373.45 | 373.21 | 373.78 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 374.85 | 374.08 | 374.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 15:15:00 | 376.00 | 374.47 | 374.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 10:15:00 | 373.90 | 374.41 | 374.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 10:15:00 | 373.90 | 374.41 | 374.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 373.90 | 374.41 | 374.22 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 437.55 | 444.83 | 445.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 437.35 | 443.33 | 444.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 442.15 | 440.36 | 442.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 442.15 | 440.36 | 442.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 442.15 | 440.36 | 442.69 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 446.90 | 443.49 | 443.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 449.85 | 445.32 | 444.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 12:15:00 | 454.10 | 459.03 | 454.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 12:15:00 | 454.10 | 459.03 | 454.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 454.10 | 459.03 | 454.30 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 444.35 | 451.28 | 452.06 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 13:15:00 | 460.00 | 452.17 | 451.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 14:15:00 | 468.20 | 455.37 | 452.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 10:15:00 | 461.40 | 462.64 | 459.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 461.30 | 463.88 | 461.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 461.30 | 463.88 | 461.75 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 12:15:00 | 463.55 | 468.35 | 468.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 12:15:00 | 454.05 | 461.94 | 465.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 10:15:00 | 457.95 | 457.91 | 461.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 11:15:00 | 464.90 | 459.31 | 461.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 464.90 | 459.31 | 461.83 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 427.85 | 427.09 | 427.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 429.55 | 427.58 | 427.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 428.05 | 428.12 | 427.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 428.05 | 428.12 | 427.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 428.05 | 428.12 | 427.69 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 425.00 | 427.08 | 427.26 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 429.00 | 427.46 | 427.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 10:15:00 | 435.10 | 428.99 | 428.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 13:15:00 | 437.35 | 437.53 | 434.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 15:15:00 | 433.00 | 436.39 | 434.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 433.00 | 436.39 | 434.37 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 429.40 | 433.12 | 433.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 425.25 | 429.89 | 431.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 12:15:00 | 427.15 | 426.82 | 428.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 13:15:00 | 427.05 | 426.87 | 428.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 427.05 | 426.87 | 428.65 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 13:15:00 | 419.70 | 414.78 | 414.49 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 13:15:00 | 410.50 | 414.08 | 414.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 14:15:00 | 406.50 | 409.74 | 411.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 10:15:00 | 409.95 | 408.98 | 410.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 354.50 | 341.22 | 350.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 354.50 | 341.22 | 350.91 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 397.00 | 358.80 | 356.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 411.15 | 405.92 | 403.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 15:15:00 | 411.50 | 414.24 | 409.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 15:15:00 | 411.50 | 414.24 | 409.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 411.50 | 414.24 | 409.88 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 424.20 | 426.93 | 427.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 12:15:00 | 423.20 | 426.18 | 426.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 420.75 | 420.67 | 422.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 420.75 | 420.67 | 422.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 420.75 | 420.67 | 422.51 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 11:15:00 | 433.35 | 423.45 | 422.51 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 11:15:00 | 424.80 | 425.80 | 425.91 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 426.60 | 426.10 | 426.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 434.30 | 428.21 | 427.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 12:15:00 | 429.60 | 429.92 | 428.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 13:15:00 | 442.95 | 432.53 | 429.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 13:15:00 | 442.95 | 432.53 | 429.58 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 14:15:00 | 458.00 | 465.62 | 466.14 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 470.70 | 462.69 | 461.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 10:15:00 | 478.95 | 465.94 | 463.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 512.65 | 512.67 | 497.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 504.95 | 511.57 | 505.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 504.95 | 511.57 | 505.23 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 491.95 | 501.45 | 502.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 11:15:00 | 483.00 | 491.73 | 496.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 515.00 | 493.39 | 495.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 515.00 | 493.39 | 495.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 515.00 | 493.39 | 495.95 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-12-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 15:15:00 | 521.00 | 498.91 | 498.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 13:15:00 | 528.00 | 511.40 | 505.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 538.35 | 538.47 | 528.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 537.45 | 544.04 | 538.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 537.45 | 544.04 | 538.00 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 570.80 | 578.33 | 578.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 557.45 | 572.67 | 576.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 13:15:00 | 569.00 | 568.83 | 572.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 570.90 | 568.43 | 571.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 570.90 | 568.43 | 571.09 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 11:15:00 | 575.60 | 571.58 | 571.26 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 09:15:00 | 566.50 | 571.04 | 571.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 11:15:00 | 565.50 | 569.21 | 570.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 566.85 | 565.75 | 567.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 10:15:00 | 563.00 | 565.20 | 567.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 563.00 | 565.20 | 567.48 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 15:15:00 | 572.00 | 563.84 | 563.15 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 550.80 | 561.23 | 562.03 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 570.45 | 561.03 | 560.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 11:15:00 | 575.35 | 563.90 | 562.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 13:15:00 | 570.50 | 570.60 | 567.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 557.65 | 568.15 | 567.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 557.65 | 568.15 | 567.44 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 559.00 | 566.32 | 566.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 548.90 | 562.84 | 565.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 554.15 | 546.25 | 552.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 554.15 | 546.25 | 552.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 554.15 | 546.25 | 552.18 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 567.90 | 556.49 | 555.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 11:15:00 | 579.00 | 562.59 | 558.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 11:15:00 | 566.70 | 566.73 | 563.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 11:15:00 | 566.70 | 566.73 | 563.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 566.70 | 566.73 | 563.22 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 571.00 | 576.08 | 576.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 561.80 | 569.98 | 572.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 13:15:00 | 561.80 | 558.86 | 563.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 14:15:00 | 580.40 | 563.17 | 564.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 580.40 | 563.17 | 564.76 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 09:15:00 | 579.05 | 567.76 | 566.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 617.50 | 589.48 | 582.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 616.45 | 616.63 | 607.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 11:15:00 | 609.65 | 614.37 | 608.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 609.65 | 614.37 | 608.20 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 14:15:00 | 600.05 | 607.84 | 608.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 10:15:00 | 590.25 | 602.55 | 605.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 14:15:00 | 577.85 | 574.71 | 579.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 573.40 | 574.80 | 578.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 573.40 | 574.80 | 578.50 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 567.60 | 564.76 | 564.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 12:15:00 | 570.50 | 565.91 | 565.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 13:15:00 | 571.35 | 571.61 | 569.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 568.20 | 570.93 | 569.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 568.20 | 570.93 | 569.62 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 570.00 | 586.04 | 587.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 11:15:00 | 563.30 | 581.49 | 585.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 12:15:00 | 566.80 | 566.08 | 573.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 13:15:00 | 578.25 | 568.51 | 573.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 578.25 | 568.51 | 573.65 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 567.85 | 564.49 | 564.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 11:15:00 | 569.90 | 567.27 | 566.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 14:15:00 | 567.95 | 568.25 | 566.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 570.65 | 568.69 | 567.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 570.65 | 568.69 | 567.29 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 10:15:00 | 566.65 | 567.51 | 567.54 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 11:15:00 | 574.50 | 568.91 | 568.18 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 13:15:00 | 565.75 | 568.56 | 568.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 14:15:00 | 565.00 | 567.85 | 568.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 10:15:00 | 567.95 | 567.37 | 568.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 10:15:00 | 567.95 | 567.37 | 568.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 567.95 | 567.37 | 568.10 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 09:15:00 | 580.45 | 570.09 | 569.04 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 569.95 | 571.48 | 571.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 15:15:00 | 566.40 | 570.46 | 571.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 574.80 | 571.33 | 571.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 574.80 | 571.33 | 571.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 574.80 | 571.33 | 571.41 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 573.25 | 571.71 | 571.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 11:15:00 | 578.30 | 573.27 | 572.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 13:15:00 | 581.80 | 582.62 | 579.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 14:15:00 | 578.80 | 581.85 | 579.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 578.80 | 581.85 | 579.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 649.15 | 652.37 | 648.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 14:15:00 | 636.15 | 645.81 | 646.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 636.15 | 645.81 | 646.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 629.60 | 641.32 | 644.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 14:15:00 | 651.10 | 636.98 | 640.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 14:15:00 | 651.10 | 636.98 | 640.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 651.10 | 636.98 | 640.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 651.10 | 636.98 | 640.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 641.75 | 637.93 | 640.43 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 10:15:00 | 657.80 | 643.22 | 642.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 11:15:00 | 671.70 | 648.92 | 645.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 689.70 | 694.65 | 679.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 10:15:00 | 679.95 | 691.71 | 679.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 10:15:00 | 679.95 | 691.71 | 679.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 11:00:00 | 679.95 | 691.71 | 679.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 11:15:00 | 671.20 | 687.61 | 678.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:00:00 | 671.20 | 687.61 | 678.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 670.55 | 684.20 | 677.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:45:00 | 670.00 | 684.20 | 677.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 684.00 | 682.24 | 679.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 686.95 | 682.24 | 679.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 11:00:00 | 686.90 | 682.25 | 679.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 12:00:00 | 685.70 | 682.94 | 680.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 13:45:00 | 684.60 | 684.93 | 681.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 683.00 | 684.64 | 682.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:15:00 | 691.55 | 682.04 | 681.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:45:00 | 686.70 | 682.93 | 682.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 14:15:00 | 680.20 | 681.77 | 681.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 14:15:00 | 680.20 | 681.77 | 681.94 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 12:15:00 | 682.80 | 681.83 | 681.77 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 13:15:00 | 681.10 | 681.68 | 681.71 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 09:15:00 | 688.90 | 682.90 | 682.23 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 679.85 | 682.93 | 683.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 677.05 | 680.35 | 681.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 09:15:00 | 680.55 | 679.50 | 680.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 680.55 | 679.50 | 680.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 680.55 | 679.50 | 680.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:00:00 | 680.55 | 679.50 | 680.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 682.20 | 680.04 | 681.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:45:00 | 682.40 | 680.04 | 681.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 682.10 | 680.45 | 681.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 13:15:00 | 680.65 | 680.67 | 681.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 14:15:00 | 681.30 | 680.93 | 681.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 14:45:00 | 680.95 | 680.67 | 681.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 681.40 | 680.87 | 681.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 671.90 | 679.08 | 680.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-03 14:15:00 | 686.25 | 680.86 | 680.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 14:15:00 | 686.25 | 680.86 | 680.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 15:15:00 | 688.70 | 682.43 | 681.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 09:15:00 | 679.00 | 681.74 | 681.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 679.00 | 681.74 | 681.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 679.00 | 681.74 | 681.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 679.00 | 681.74 | 681.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 680.05 | 681.40 | 681.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:30:00 | 683.95 | 681.78 | 681.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 13:30:00 | 683.45 | 683.52 | 682.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 687.45 | 682.41 | 681.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 12:15:00 | 678.00 | 681.40 | 681.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 678.00 | 681.40 | 681.61 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 13:15:00 | 698.70 | 684.86 | 683.17 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 678.85 | 690.60 | 691.38 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 11:15:00 | 696.30 | 691.24 | 691.20 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 684.35 | 691.02 | 691.62 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 699.95 | 691.87 | 691.63 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 11:15:00 | 681.95 | 689.89 | 690.75 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 693.55 | 691.36 | 691.31 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 688.30 | 690.75 | 691.04 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 698.05 | 691.76 | 691.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 709.00 | 695.21 | 693.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 12:15:00 | 725.55 | 725.78 | 717.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 13:00:00 | 725.55 | 725.78 | 717.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 723.30 | 725.28 | 718.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:30:00 | 724.20 | 725.28 | 718.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 726.00 | 724.74 | 718.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 730.60 | 724.74 | 718.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-18 12:15:00 | 714.50 | 720.85 | 718.48 | SL hit (close<static) qty=1.00 sl=715.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 696.15 | 715.91 | 716.45 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 714.90 | 709.24 | 709.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 718.40 | 711.07 | 710.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 15:15:00 | 712.00 | 713.49 | 711.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 15:15:00 | 712.00 | 713.49 | 711.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 712.00 | 713.49 | 711.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 711.80 | 713.49 | 711.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 706.90 | 712.18 | 711.26 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 704.80 | 710.01 | 710.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 699.65 | 707.94 | 709.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 11:15:00 | 701.65 | 698.94 | 702.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 11:15:00 | 701.65 | 698.94 | 702.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 701.65 | 698.94 | 702.53 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 719.40 | 704.71 | 704.02 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 689.65 | 707.39 | 708.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 10:15:00 | 674.00 | 695.64 | 702.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 659.95 | 649.78 | 663.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 659.95 | 649.78 | 663.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 659.95 | 649.78 | 663.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 659.95 | 649.78 | 663.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 679.05 | 655.63 | 665.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 679.05 | 655.63 | 665.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 666.90 | 657.89 | 665.26 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 688.00 | 672.09 | 670.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 710.40 | 679.75 | 674.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 742.60 | 743.52 | 725.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:45:00 | 742.70 | 743.52 | 725.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 739.00 | 743.65 | 737.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 744.10 | 743.65 | 737.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:45:00 | 744.00 | 743.48 | 738.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 12:00:00 | 743.90 | 744.55 | 739.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:15:00 | 742.70 | 743.47 | 740.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 738.85 | 742.55 | 739.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 738.85 | 742.55 | 739.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 735.20 | 741.08 | 739.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 740.85 | 741.08 | 739.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:45:00 | 741.40 | 741.32 | 739.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 14:15:00 | 727.60 | 744.78 | 746.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 727.60 | 744.78 | 746.79 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 737.20 | 732.00 | 731.78 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 729.10 | 731.42 | 731.53 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 739.05 | 732.95 | 732.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 13:15:00 | 745.85 | 737.49 | 734.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 741.05 | 747.48 | 742.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 14:15:00 | 741.05 | 747.48 | 742.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 741.05 | 747.48 | 742.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 741.05 | 747.48 | 742.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 740.00 | 745.99 | 742.71 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 11:15:00 | 733.05 | 739.38 | 740.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 09:15:00 | 730.00 | 735.47 | 736.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 14:15:00 | 735.60 | 732.14 | 734.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 14:15:00 | 735.60 | 732.14 | 734.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 735.60 | 732.14 | 734.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 735.60 | 732.14 | 734.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 735.25 | 732.76 | 734.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 737.25 | 732.76 | 734.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 734.95 | 733.77 | 734.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:15:00 | 740.20 | 733.77 | 734.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 749.00 | 736.82 | 735.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 12:15:00 | 755.50 | 740.56 | 737.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 14:15:00 | 752.60 | 754.15 | 748.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 14:45:00 | 751.60 | 754.15 | 748.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 751.80 | 753.68 | 748.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 752.50 | 753.68 | 748.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 747.30 | 752.40 | 748.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 747.60 | 752.40 | 748.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 744.05 | 750.73 | 747.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 744.05 | 750.73 | 747.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 736.35 | 747.86 | 746.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 736.35 | 747.86 | 746.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 739.25 | 744.99 | 745.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 729.50 | 739.49 | 742.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 727.35 | 726.03 | 729.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 727.35 | 726.03 | 729.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 727.35 | 726.03 | 729.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:30:00 | 722.85 | 725.08 | 728.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 722.50 | 723.59 | 726.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 721.50 | 723.59 | 726.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 12:15:00 | 730.85 | 725.65 | 726.38 | SL hit (close>static) qty=1.00 sl=729.90 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 733.65 | 727.25 | 727.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 743.50 | 730.48 | 728.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 729.25 | 735.96 | 733.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 729.25 | 735.96 | 733.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 729.25 | 735.96 | 733.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 729.25 | 735.96 | 733.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 730.55 | 734.88 | 733.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 727.55 | 734.88 | 733.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 726.55 | 731.97 | 732.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 712.85 | 727.71 | 730.27 | Break + close below crossover candle low |

### Cycle 96 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 770.05 | 730.29 | 729.87 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 11:15:00 | 716.60 | 730.55 | 730.90 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 734.65 | 725.08 | 724.63 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 722.20 | 725.90 | 726.10 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 12:15:00 | 730.20 | 726.76 | 726.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 15:15:00 | 737.00 | 729.62 | 727.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 843.35 | 846.42 | 825.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 10:15:00 | 814.90 | 840.12 | 824.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 814.90 | 840.12 | 824.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 814.90 | 840.12 | 824.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 829.70 | 838.03 | 825.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:15:00 | 837.60 | 838.03 | 825.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 13:00:00 | 836.00 | 840.29 | 833.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 836.90 | 839.00 | 833.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 14:45:00 | 840.55 | 839.43 | 834.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 840.00 | 839.54 | 835.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:30:00 | 861.80 | 849.58 | 846.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:00:00 | 860.00 | 853.49 | 848.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:45:00 | 858.05 | 856.51 | 853.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 15:15:00 | 851.00 | 853.00 | 853.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 851.00 | 853.00 | 853.07 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 857.55 | 853.21 | 853.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 14:15:00 | 864.00 | 855.75 | 854.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 15:15:00 | 852.30 | 855.06 | 854.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 15:15:00 | 852.30 | 855.06 | 854.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 852.30 | 855.06 | 854.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 871.00 | 855.06 | 854.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 09:15:00 | 849.80 | 854.01 | 853.70 | SL hit (close<static) qty=1.00 sl=852.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 851.10 | 853.19 | 853.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 12:15:00 | 845.45 | 851.64 | 852.64 | Break + close below crossover candle low |

### Cycle 104 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 862.50 | 853.81 | 853.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 864.00 | 855.87 | 854.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 888.30 | 890.47 | 876.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 09:45:00 | 891.20 | 890.47 | 876.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 894.85 | 895.63 | 890.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 898.60 | 895.63 | 890.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 894.35 | 895.61 | 891.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:30:00 | 899.95 | 895.53 | 893.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 12:15:00 | 917.85 | 919.08 | 919.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 917.85 | 919.08 | 919.18 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 13:15:00 | 922.10 | 919.68 | 919.44 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 914.50 | 919.74 | 920.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 908.20 | 915.13 | 917.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 912.70 | 912.59 | 915.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 912.70 | 912.59 | 915.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 912.70 | 912.59 | 915.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 913.00 | 912.59 | 915.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 916.00 | 913.33 | 915.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:00:00 | 916.00 | 913.33 | 915.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 914.00 | 913.47 | 915.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:30:00 | 911.05 | 913.74 | 914.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 11:15:00 | 928.55 | 916.70 | 916.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 928.55 | 916.70 | 916.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 10:15:00 | 943.65 | 937.16 | 930.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 1020.55 | 1021.81 | 1004.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 10:45:00 | 1015.10 | 1021.81 | 1004.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1198.00 | 1210.93 | 1195.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 1198.00 | 1210.93 | 1195.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1195.00 | 1207.74 | 1195.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 1161.90 | 1207.74 | 1195.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1144.30 | 1195.05 | 1190.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 1137.35 | 1195.05 | 1190.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1158.95 | 1187.83 | 1187.57 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 1168.00 | 1183.87 | 1185.79 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 1193.05 | 1184.95 | 1184.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1208.00 | 1191.62 | 1187.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 1187.30 | 1198.19 | 1192.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 11:15:00 | 1187.30 | 1198.19 | 1192.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 1187.30 | 1198.19 | 1192.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:00:00 | 1187.30 | 1198.19 | 1192.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 1207.20 | 1199.99 | 1193.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:30:00 | 1211.70 | 1203.91 | 1196.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:30:00 | 1208.10 | 1203.23 | 1201.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 10:15:00 | 1209.05 | 1203.23 | 1201.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:15:00 | 1209.00 | 1204.19 | 1202.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 1214.35 | 1206.22 | 1203.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:30:00 | 1213.00 | 1206.22 | 1203.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1203.85 | 1205.75 | 1203.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 1203.85 | 1205.75 | 1203.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1197.75 | 1204.15 | 1202.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:30:00 | 1197.45 | 1204.15 | 1202.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1204.00 | 1204.12 | 1203.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 1210.00 | 1204.12 | 1203.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:30:00 | 1214.70 | 1206.78 | 1204.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 1192.10 | 1203.42 | 1203.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 1192.10 | 1203.42 | 1203.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 14:15:00 | 1180.60 | 1197.33 | 1200.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 13:15:00 | 1190.10 | 1178.15 | 1187.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 13:15:00 | 1190.10 | 1178.15 | 1187.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1190.10 | 1178.15 | 1187.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 1190.10 | 1178.15 | 1187.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1165.30 | 1175.58 | 1185.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1156.50 | 1174.67 | 1183.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:15:00 | 1157.45 | 1172.97 | 1182.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:00:00 | 1164.85 | 1156.12 | 1166.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 13:45:00 | 1164.00 | 1156.90 | 1163.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1173.05 | 1160.13 | 1164.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 1173.05 | 1160.13 | 1164.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1167.00 | 1161.50 | 1164.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1149.75 | 1161.50 | 1164.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 1106.61 | 1136.19 | 1148.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 1105.80 | 1136.19 | 1148.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1098.67 | 1132.06 | 1145.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1099.58 | 1132.06 | 1145.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1092.26 | 1132.06 | 1145.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 1133.30 | 1131.74 | 1143.25 | SL hit (close>ema200) qty=0.50 sl=1131.74 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1137.40 | 1119.82 | 1117.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 1145.05 | 1124.87 | 1119.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 1173.10 | 1173.68 | 1160.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 15:15:00 | 1173.10 | 1173.68 | 1160.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1173.10 | 1173.68 | 1160.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 1159.75 | 1170.53 | 1159.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1150.00 | 1166.42 | 1159.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 1146.00 | 1166.42 | 1159.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 1146.80 | 1154.93 | 1155.28 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 1161.00 | 1155.84 | 1155.44 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 1147.90 | 1154.91 | 1155.13 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 1198.55 | 1163.05 | 1158.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 12:15:00 | 1206.55 | 1171.75 | 1163.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1211.00 | 1214.06 | 1198.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 1211.00 | 1214.06 | 1198.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1195.75 | 1208.70 | 1198.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 1190.80 | 1208.70 | 1198.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1191.65 | 1205.29 | 1197.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:00:00 | 1191.65 | 1205.29 | 1197.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1193.50 | 1202.93 | 1197.55 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 1178.55 | 1192.75 | 1193.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 1164.25 | 1180.80 | 1187.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 1182.55 | 1181.15 | 1186.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 1182.55 | 1181.15 | 1186.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1182.55 | 1181.15 | 1186.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 1182.55 | 1181.15 | 1186.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1180.05 | 1180.93 | 1186.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 1172.90 | 1180.93 | 1186.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 1173.30 | 1179.55 | 1184.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:15:00 | 1172.80 | 1179.55 | 1184.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 1173.80 | 1178.21 | 1182.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1188.70 | 1180.44 | 1183.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:30:00 | 1190.55 | 1180.44 | 1183.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 1181.15 | 1180.58 | 1182.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:15:00 | 1190.95 | 1180.58 | 1182.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1190.95 | 1182.66 | 1183.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 1169.75 | 1182.66 | 1183.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 1114.26 | 1147.98 | 1164.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 1114.63 | 1147.98 | 1164.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 1114.16 | 1147.98 | 1164.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 1115.11 | 1147.98 | 1164.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1111.26 | 1139.76 | 1155.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 1140.90 | 1139.99 | 1154.62 | SL hit (close>ema200) qty=0.50 sl=1139.99 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 13:15:00 | 1143.00 | 1102.20 | 1096.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 1148.40 | 1120.62 | 1107.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1201.55 | 1209.07 | 1170.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 09:45:00 | 1211.60 | 1209.07 | 1170.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1238.50 | 1237.57 | 1223.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 1221.40 | 1237.57 | 1223.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 1223.15 | 1234.03 | 1224.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 1223.15 | 1234.03 | 1224.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 1239.20 | 1235.07 | 1225.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 15:15:00 | 1276.30 | 1236.01 | 1226.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:45:00 | 1243.90 | 1244.57 | 1244.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:15:00 | 1247.15 | 1244.57 | 1244.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 15:15:00 | 1233.00 | 1242.16 | 1243.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 1233.00 | 1242.16 | 1243.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1229.55 | 1237.88 | 1240.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1243.00 | 1235.01 | 1238.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1243.00 | 1235.01 | 1238.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1243.00 | 1235.01 | 1238.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 1243.00 | 1235.01 | 1238.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1246.45 | 1237.29 | 1239.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:00:00 | 1224.80 | 1232.87 | 1236.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 15:15:00 | 1163.56 | 1210.34 | 1225.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 09:15:00 | 1102.32 | 1188.48 | 1213.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 1150.25 | 1129.54 | 1128.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 12:15:00 | 1182.65 | 1151.35 | 1141.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 11:15:00 | 1207.40 | 1215.04 | 1192.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 12:00:00 | 1207.40 | 1215.04 | 1192.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1204.00 | 1209.00 | 1196.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:30:00 | 1209.20 | 1210.19 | 1198.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 1205.00 | 1219.54 | 1218.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 1205.00 | 1216.63 | 1217.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 1205.00 | 1216.63 | 1217.61 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1223.30 | 1218.01 | 1217.82 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 1215.45 | 1217.68 | 1217.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 1179.35 | 1209.32 | 1213.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 12:15:00 | 1152.00 | 1150.51 | 1167.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 12:45:00 | 1151.85 | 1150.51 | 1167.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1106.10 | 1115.80 | 1128.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 1114.95 | 1115.80 | 1128.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1091.15 | 1087.75 | 1100.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 1097.95 | 1087.75 | 1100.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1094.90 | 1090.60 | 1098.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 1101.45 | 1090.60 | 1098.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1091.15 | 1090.71 | 1097.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:45:00 | 1100.05 | 1090.71 | 1097.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 1093.35 | 1090.08 | 1095.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:30:00 | 1097.50 | 1090.08 | 1095.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1085.30 | 1086.29 | 1091.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 1090.80 | 1086.29 | 1091.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 1089.35 | 1086.90 | 1091.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:30:00 | 1088.10 | 1086.90 | 1091.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1092.80 | 1088.08 | 1091.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 1092.40 | 1088.08 | 1091.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1092.00 | 1088.86 | 1091.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:15:00 | 1098.00 | 1088.86 | 1091.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1098.00 | 1090.69 | 1092.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 1097.80 | 1090.69 | 1092.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1087.15 | 1089.98 | 1091.62 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 1104.05 | 1094.92 | 1093.71 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1082.70 | 1092.48 | 1092.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1072.10 | 1088.40 | 1090.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 15:15:00 | 1087.05 | 1083.32 | 1087.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 15:15:00 | 1087.05 | 1083.32 | 1087.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1087.05 | 1083.32 | 1087.81 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 1099.65 | 1079.04 | 1078.91 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 13:15:00 | 1071.40 | 1080.18 | 1080.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 14:15:00 | 1068.60 | 1077.86 | 1079.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 15:15:00 | 1079.00 | 1078.09 | 1079.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 15:15:00 | 1079.00 | 1078.09 | 1079.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1079.00 | 1078.09 | 1079.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1093.50 | 1078.09 | 1079.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1076.50 | 1077.77 | 1079.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 1109.60 | 1077.77 | 1079.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1073.75 | 1076.97 | 1078.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 1075.75 | 1076.97 | 1078.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1080.00 | 1077.57 | 1078.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 1077.50 | 1077.57 | 1078.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1072.00 | 1076.46 | 1078.16 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 1089.50 | 1079.38 | 1078.95 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 1075.45 | 1078.59 | 1078.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 1062.40 | 1075.35 | 1077.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 1095.70 | 1077.57 | 1077.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 1095.70 | 1077.57 | 1077.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1095.70 | 1077.57 | 1077.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1095.70 | 1077.57 | 1077.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 1085.00 | 1079.05 | 1078.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 1123.90 | 1101.03 | 1091.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 11:15:00 | 1084.70 | 1099.65 | 1092.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 11:15:00 | 1084.70 | 1099.65 | 1092.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1084.70 | 1099.65 | 1092.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:45:00 | 1084.30 | 1099.65 | 1092.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 1070.05 | 1093.73 | 1090.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:00:00 | 1070.05 | 1093.73 | 1090.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 1084.05 | 1087.89 | 1088.19 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1102.75 | 1090.87 | 1089.51 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 1080.00 | 1088.27 | 1088.54 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 15:15:00 | 1092.90 | 1088.28 | 1087.84 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1077.05 | 1086.03 | 1086.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 1019.00 | 1051.21 | 1067.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 15:15:00 | 1034.00 | 1030.62 | 1045.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 09:15:00 | 1006.40 | 1030.62 | 1045.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1041.95 | 1023.48 | 1033.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1041.95 | 1023.48 | 1033.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1025.00 | 1023.78 | 1032.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1038.00 | 1023.78 | 1032.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1034.95 | 1026.01 | 1032.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:30:00 | 1027.65 | 1028.36 | 1032.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:15:00 | 1021.05 | 1028.36 | 1032.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 976.27 | 1009.96 | 1021.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 970.00 | 988.67 | 1005.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 924.89 | 956.46 | 980.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 136 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 946.00 | 941.19 | 940.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 957.50 | 944.45 | 942.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 963.10 | 967.83 | 958.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 963.10 | 967.83 | 958.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 956.15 | 965.43 | 958.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 956.15 | 965.43 | 958.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 950.50 | 962.44 | 957.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:00:00 | 950.50 | 962.44 | 957.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 932.00 | 951.60 | 953.62 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 960.55 | 949.69 | 949.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 15:15:00 | 965.00 | 956.10 | 952.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 938.55 | 952.59 | 951.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 938.55 | 952.59 | 951.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 938.55 | 952.59 | 951.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 938.55 | 952.59 | 951.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 935.80 | 949.23 | 949.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 920.60 | 938.17 | 944.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 909.15 | 906.94 | 921.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 909.15 | 906.94 | 921.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 903.85 | 890.82 | 901.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 903.85 | 890.82 | 901.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 921.35 | 896.93 | 902.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 921.35 | 896.93 | 902.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 924.80 | 902.50 | 904.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 924.80 | 902.50 | 904.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 934.05 | 908.81 | 907.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 951.60 | 926.90 | 917.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 951.60 | 960.86 | 946.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 13:15:00 | 951.60 | 960.86 | 946.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 951.60 | 960.86 | 946.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 951.60 | 960.86 | 946.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 974.60 | 963.60 | 949.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 985.00 | 965.26 | 951.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:00:00 | 983.60 | 968.93 | 954.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 983.60 | 974.60 | 959.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 984.65 | 978.10 | 965.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 942.45 | 972.18 | 964.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 942.45 | 972.18 | 964.84 | SL hit (close<static) qty=1.00 sl=946.35 alert=retest2 |

### Cycle 141 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 924.70 | 957.12 | 958.92 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 967.45 | 956.20 | 955.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 974.00 | 959.38 | 957.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1035.20 | 1037.01 | 1015.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 1035.20 | 1037.01 | 1015.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1010.00 | 1035.28 | 1020.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 1017.35 | 1035.28 | 1020.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1009.00 | 1030.03 | 1019.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:45:00 | 1004.75 | 1030.03 | 1019.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1018.40 | 1026.26 | 1019.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 1018.40 | 1026.26 | 1019.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1000.00 | 1021.01 | 1017.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 1000.00 | 1021.01 | 1017.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1003.85 | 1017.57 | 1016.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:15:00 | 1000.00 | 1017.57 | 1016.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 1000.00 | 1014.06 | 1014.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 993.05 | 1009.86 | 1012.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 993.80 | 977.47 | 989.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 14:15:00 | 993.80 | 977.47 | 989.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 993.80 | 977.47 | 989.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 993.80 | 977.47 | 989.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 997.00 | 981.38 | 990.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 953.50 | 981.38 | 990.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 990.00 | 978.48 | 983.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 990.15 | 983.02 | 984.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 940.50 | 968.62 | 976.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 940.64 | 968.62 | 976.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 943.95 | 941.69 | 958.17 | SL hit (close>ema200) qty=0.50 sl=941.69 alert=retest2 |

### Cycle 144 — BUY (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 12:15:00 | 992.20 | 966.11 | 965.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 1011.60 | 975.21 | 969.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 09:15:00 | 957.55 | 977.86 | 972.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 957.55 | 977.86 | 972.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 957.55 | 977.86 | 972.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:00:00 | 957.55 | 977.86 | 972.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 970.50 | 976.39 | 972.29 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 960.55 | 969.71 | 969.75 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 970.60 | 969.89 | 969.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 1000.15 | 975.94 | 972.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 12:15:00 | 974.15 | 982.83 | 977.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 12:15:00 | 974.15 | 982.83 | 977.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 974.15 | 982.83 | 977.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:00:00 | 974.15 | 982.83 | 977.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 972.45 | 980.75 | 977.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:45:00 | 971.75 | 980.75 | 977.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 980.50 | 979.01 | 977.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 978.90 | 979.01 | 977.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 978.15 | 978.84 | 977.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:45:00 | 978.00 | 978.84 | 977.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 979.00 | 978.87 | 977.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:15:00 | 975.55 | 978.87 | 977.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 997.30 | 982.56 | 979.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 1005.75 | 985.67 | 981.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 968.00 | 978.56 | 979.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 968.00 | 978.56 | 979.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 964.00 | 975.64 | 978.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 952.70 | 946.80 | 956.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 12:15:00 | 952.70 | 946.80 | 956.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 952.70 | 946.80 | 956.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:30:00 | 954.70 | 946.80 | 956.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 944.25 | 946.29 | 955.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:00:00 | 936.10 | 944.25 | 953.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 889.29 | 914.30 | 930.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 900.35 | 899.76 | 915.86 | SL hit (close>ema200) qty=0.50 sl=899.76 alert=retest2 |

### Cycle 148 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 914.95 | 911.54 | 911.26 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 10:15:00 | 903.80 | 911.94 | 911.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 11:15:00 | 900.85 | 909.73 | 910.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 918.25 | 905.41 | 907.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 918.25 | 905.41 | 907.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 918.25 | 905.41 | 907.76 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 921.30 | 910.92 | 910.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 935.90 | 919.67 | 915.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 916.95 | 919.13 | 915.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 10:15:00 | 916.95 | 919.13 | 915.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 916.95 | 919.13 | 915.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:45:00 | 917.45 | 919.13 | 915.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 917.35 | 918.40 | 915.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:30:00 | 919.80 | 918.40 | 915.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 909.20 | 916.56 | 915.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 909.20 | 916.56 | 915.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 910.00 | 915.25 | 914.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 902.25 | 915.25 | 914.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 909.65 | 914.13 | 914.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 883.40 | 901.24 | 907.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 896.95 | 891.86 | 898.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 896.95 | 891.86 | 898.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 896.95 | 891.86 | 898.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 896.95 | 891.86 | 898.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 890.80 | 891.65 | 898.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 885.20 | 892.14 | 897.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 888.80 | 871.12 | 870.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 888.80 | 871.12 | 870.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 894.80 | 879.89 | 875.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 892.50 | 892.83 | 885.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 892.50 | 892.83 | 885.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 908.00 | 904.95 | 897.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:30:00 | 905.80 | 904.95 | 897.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 897.75 | 903.51 | 897.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:30:00 | 899.45 | 903.51 | 897.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 898.35 | 902.48 | 897.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:30:00 | 895.20 | 902.48 | 897.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 903.20 | 902.62 | 898.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:30:00 | 897.80 | 902.62 | 898.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 894.90 | 901.21 | 898.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 893.60 | 901.21 | 898.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 902.65 | 901.50 | 898.70 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 877.00 | 894.11 | 896.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 12:15:00 | 872.75 | 887.47 | 892.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 865.00 | 864.35 | 873.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:15:00 | 870.00 | 864.35 | 873.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 872.85 | 866.05 | 873.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 876.40 | 866.05 | 873.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 880.95 | 869.03 | 874.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 881.85 | 869.03 | 874.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 884.80 | 872.18 | 875.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 884.80 | 872.18 | 875.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 890.90 | 878.37 | 877.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 897.75 | 889.66 | 884.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 11:15:00 | 887.45 | 892.36 | 887.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 11:15:00 | 887.45 | 892.36 | 887.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 887.45 | 892.36 | 887.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:00:00 | 887.45 | 892.36 | 887.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 892.85 | 892.46 | 888.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:30:00 | 888.90 | 892.46 | 888.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 904.60 | 894.88 | 889.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 914.95 | 894.88 | 889.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 09:45:00 | 906.20 | 901.62 | 894.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 859.05 | 921.93 | 926.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 859.05 | 921.93 | 926.65 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 923.60 | 885.61 | 880.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 941.15 | 924.57 | 919.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 916.95 | 942.18 | 934.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 916.95 | 942.18 | 934.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 916.95 | 942.18 | 934.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 916.95 | 942.18 | 934.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 907.65 | 935.27 | 931.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 907.65 | 935.27 | 931.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 915.00 | 928.94 | 929.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 912.10 | 925.57 | 927.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 911.65 | 911.63 | 916.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 911.65 | 911.63 | 916.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 911.65 | 911.63 | 916.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:15:00 | 903.10 | 911.64 | 915.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 893.90 | 896.06 | 898.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:15:00 | 904.95 | 898.73 | 899.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:45:00 | 902.25 | 898.90 | 899.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 896.95 | 898.51 | 899.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:15:00 | 902.45 | 898.51 | 899.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-05 12:15:00 | 914.85 | 901.78 | 900.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 914.85 | 901.78 | 900.74 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 883.15 | 898.28 | 899.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 873.20 | 890.40 | 895.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 877.95 | 877.50 | 885.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:00:00 | 877.95 | 877.50 | 885.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 882.25 | 878.85 | 884.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:45:00 | 878.50 | 879.05 | 884.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:45:00 | 878.05 | 878.86 | 883.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 870.00 | 884.60 | 885.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 890.55 | 879.13 | 877.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 890.55 | 879.13 | 877.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 896.05 | 886.45 | 882.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 899.80 | 900.27 | 894.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:45:00 | 900.45 | 900.27 | 894.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1026.45 | 999.10 | 982.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:45:00 | 1034.90 | 1006.48 | 987.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-27 10:15:00 | 1138.39 | 1116.02 | 1087.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1132.70 | 1153.70 | 1153.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 1128.70 | 1148.70 | 1151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 1154.30 | 1146.96 | 1150.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 1154.30 | 1146.96 | 1150.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1154.30 | 1146.96 | 1150.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 1154.30 | 1146.96 | 1150.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1158.00 | 1149.16 | 1150.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 1161.30 | 1149.16 | 1150.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1157.00 | 1150.73 | 1151.35 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1160.40 | 1152.67 | 1152.17 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1146.30 | 1154.04 | 1154.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1141.50 | 1151.53 | 1153.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1143.30 | 1140.77 | 1145.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1143.30 | 1140.77 | 1145.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1143.30 | 1140.77 | 1145.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 1135.60 | 1140.77 | 1145.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 1141.30 | 1144.11 | 1145.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 1151.00 | 1146.73 | 1146.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 1151.00 | 1146.73 | 1146.27 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 1143.40 | 1146.06 | 1146.15 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 1155.50 | 1147.95 | 1147.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 11:15:00 | 1156.30 | 1149.62 | 1147.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1143.60 | 1149.08 | 1147.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 1143.60 | 1149.08 | 1147.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1143.60 | 1149.08 | 1147.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1145.10 | 1149.08 | 1147.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1146.00 | 1148.47 | 1147.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 1146.50 | 1148.47 | 1147.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1145.00 | 1147.77 | 1147.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1143.20 | 1147.77 | 1147.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1145.40 | 1147.30 | 1147.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1138.50 | 1144.55 | 1146.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 12:15:00 | 1146.20 | 1144.88 | 1146.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1146.20 | 1144.88 | 1146.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1146.20 | 1144.88 | 1146.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 1148.00 | 1144.88 | 1146.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1131.00 | 1142.10 | 1144.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 15:00:00 | 1128.20 | 1139.32 | 1143.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1071.79 | 1135.09 | 1140.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:45:00 | 1118.10 | 1135.09 | 1140.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1130.10 | 1111.22 | 1122.49 | SL hit (close>ema200) qty=0.50 sl=1111.22 alert=retest2 |

### Cycle 168 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 1130.40 | 1124.19 | 1123.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 1135.30 | 1128.54 | 1126.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1124.40 | 1128.69 | 1127.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 14:15:00 | 1124.40 | 1128.69 | 1127.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1124.40 | 1128.69 | 1127.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 1124.40 | 1128.69 | 1127.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1127.00 | 1128.35 | 1127.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 1130.70 | 1128.35 | 1127.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:45:00 | 1141.10 | 1132.68 | 1129.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 13:45:00 | 1130.00 | 1130.79 | 1129.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 1131.00 | 1135.64 | 1131.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1131.70 | 1138.06 | 1134.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 1131.70 | 1138.06 | 1134.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1129.80 | 1136.41 | 1133.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1129.80 | 1136.41 | 1133.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 1154.00 | 1127.45 | 1125.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 1162.00 | 1134.36 | 1129.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 1172.60 | 1181.41 | 1172.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1172.60 | 1181.41 | 1172.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1172.60 | 1181.41 | 1172.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1172.60 | 1181.41 | 1172.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1193.10 | 1183.04 | 1174.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 1215.50 | 1189.07 | 1178.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 1190.90 | 1196.37 | 1196.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1190.90 | 1196.37 | 1196.98 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 1204.00 | 1197.88 | 1197.50 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1170.50 | 1192.90 | 1195.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1155.70 | 1185.46 | 1191.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 1160.70 | 1157.10 | 1166.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:30:00 | 1161.40 | 1157.10 | 1166.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1169.70 | 1159.70 | 1165.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1169.70 | 1159.70 | 1165.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1167.40 | 1161.24 | 1166.10 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1180.00 | 1168.86 | 1168.47 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 1161.90 | 1167.87 | 1168.22 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1177.50 | 1169.79 | 1169.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1182.00 | 1173.15 | 1170.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1194.80 | 1198.21 | 1190.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1194.80 | 1198.21 | 1190.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1194.80 | 1198.21 | 1190.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1187.70 | 1198.21 | 1190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1192.00 | 1196.97 | 1190.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 1192.50 | 1196.97 | 1190.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1195.00 | 1195.81 | 1192.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1200.00 | 1195.81 | 1192.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1185.50 | 1193.75 | 1191.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 1185.50 | 1193.75 | 1191.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1192.20 | 1193.44 | 1191.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1183.60 | 1193.44 | 1191.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1200.40 | 1194.83 | 1192.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 1201.80 | 1196.93 | 1194.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 15:00:00 | 1201.90 | 1196.93 | 1194.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1207.00 | 1200.33 | 1196.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 1205.90 | 1211.80 | 1212.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1205.90 | 1211.80 | 1212.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 10:15:00 | 1198.50 | 1207.94 | 1210.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 1177.30 | 1175.67 | 1186.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 13:00:00 | 1177.30 | 1175.67 | 1186.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1181.50 | 1176.91 | 1184.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1168.60 | 1176.71 | 1184.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:30:00 | 1167.20 | 1172.06 | 1180.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 1189.80 | 1175.60 | 1181.47 | SL hit (close>static) qty=1.00 sl=1185.60 alert=retest2 |

### Cycle 178 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 1193.50 | 1184.66 | 1183.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 1209.70 | 1189.67 | 1185.89 | Break + close above crossover candle high |

### Cycle 179 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 1150.90 | 1181.92 | 1182.71 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 1214.20 | 1185.00 | 1182.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1222.00 | 1192.40 | 1186.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1196.80 | 1212.97 | 1201.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1196.80 | 1212.97 | 1201.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1196.80 | 1212.97 | 1201.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1196.80 | 1212.97 | 1201.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1202.90 | 1210.96 | 1201.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 1204.80 | 1210.96 | 1201.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1210.00 | 1210.77 | 1202.04 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1160.10 | 1195.15 | 1197.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 1160.00 | 1188.12 | 1194.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1151.50 | 1150.50 | 1165.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 1151.50 | 1150.50 | 1165.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1142.70 | 1148.59 | 1161.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 1129.10 | 1140.99 | 1155.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1072.64 | 1096.30 | 1104.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 1095.10 | 1092.65 | 1099.75 | SL hit (close>ema200) qty=0.50 sl=1092.65 alert=retest2 |

### Cycle 182 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 1094.00 | 1068.58 | 1066.34 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 1061.00 | 1073.22 | 1073.45 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 1078.10 | 1074.19 | 1073.87 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1066.70 | 1072.70 | 1073.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1062.50 | 1068.42 | 1070.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1072.50 | 1069.23 | 1070.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 1072.50 | 1069.23 | 1070.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1072.50 | 1069.23 | 1070.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1072.50 | 1069.23 | 1070.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1082.70 | 1071.93 | 1072.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 1082.70 | 1071.93 | 1072.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 1084.40 | 1074.42 | 1073.16 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1058.30 | 1075.74 | 1075.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1054.80 | 1065.72 | 1070.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1025.50 | 1023.73 | 1034.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 1025.50 | 1023.73 | 1034.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1030.50 | 1024.65 | 1033.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1034.10 | 1024.65 | 1033.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1041.30 | 1027.98 | 1034.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1041.30 | 1027.98 | 1034.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1046.40 | 1031.66 | 1035.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1046.40 | 1031.66 | 1035.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1047.50 | 1038.01 | 1037.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 12:15:00 | 1055.40 | 1045.77 | 1042.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1046.60 | 1049.45 | 1045.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1046.60 | 1049.45 | 1045.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1046.60 | 1049.45 | 1045.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 1048.60 | 1049.45 | 1045.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1043.40 | 1048.24 | 1045.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 1045.00 | 1048.24 | 1045.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1059.50 | 1050.49 | 1046.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:00:00 | 1061.50 | 1052.69 | 1048.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1065.70 | 1058.03 | 1052.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:00:00 | 1061.00 | 1058.63 | 1052.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 1115.80 | 1120.00 | 1120.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 10:15:00 | 1115.80 | 1120.00 | 1120.30 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 1125.70 | 1120.44 | 1120.40 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1117.30 | 1120.47 | 1120.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 1113.00 | 1118.97 | 1119.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 13:15:00 | 1113.00 | 1110.60 | 1114.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1113.00 | 1110.60 | 1114.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1113.70 | 1111.22 | 1114.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1113.70 | 1111.22 | 1114.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1111.00 | 1111.18 | 1114.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1114.20 | 1111.18 | 1114.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1112.70 | 1111.48 | 1114.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 1099.30 | 1109.47 | 1112.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 1116.80 | 1109.09 | 1111.91 | SL hit (close>static) qty=1.00 sl=1116.70 alert=retest2 |

### Cycle 192 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 1082.40 | 1063.57 | 1063.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1102.70 | 1083.44 | 1076.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 1116.20 | 1116.94 | 1105.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 1116.20 | 1116.94 | 1105.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1105.00 | 1113.51 | 1105.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 1104.60 | 1113.51 | 1105.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1105.20 | 1111.84 | 1105.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 1105.80 | 1111.84 | 1105.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1098.10 | 1109.10 | 1104.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 1098.10 | 1109.10 | 1104.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1096.60 | 1106.60 | 1104.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1102.60 | 1106.60 | 1104.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1087.80 | 1100.34 | 1101.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 1087.80 | 1100.34 | 1101.47 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1100.90 | 1096.09 | 1095.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1108.50 | 1099.40 | 1097.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1103.30 | 1111.41 | 1106.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1103.30 | 1111.41 | 1106.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1103.30 | 1111.41 | 1106.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 1106.90 | 1111.41 | 1106.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1104.70 | 1110.07 | 1105.97 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 1099.00 | 1103.22 | 1103.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1077.50 | 1098.08 | 1101.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1081.20 | 1080.70 | 1088.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 1086.60 | 1080.70 | 1088.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1064.20 | 1066.92 | 1077.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1064.20 | 1066.92 | 1077.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1082.60 | 1069.12 | 1076.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 1070.80 | 1074.14 | 1077.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 1068.40 | 1073.59 | 1076.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 1068.70 | 1072.52 | 1075.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1080.90 | 1076.67 | 1076.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1080.90 | 1076.67 | 1076.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 1089.60 | 1080.25 | 1077.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 11:15:00 | 1121.80 | 1123.73 | 1113.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:45:00 | 1119.10 | 1123.73 | 1113.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1122.00 | 1121.47 | 1115.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 1125.70 | 1120.96 | 1116.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1141.00 | 1121.93 | 1118.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 1127.10 | 1123.86 | 1120.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1127.40 | 1121.72 | 1120.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1121.10 | 1121.59 | 1120.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1121.10 | 1121.59 | 1120.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1127.00 | 1122.68 | 1120.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 1131.50 | 1124.55 | 1122.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1095.50 | 1117.34 | 1120.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1113.30 | 1108.07 | 1114.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1113.30 | 1108.07 | 1114.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1113.30 | 1108.07 | 1114.49 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1139.50 | 1119.86 | 1119.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1165.50 | 1131.67 | 1124.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1150.30 | 1162.69 | 1150.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1150.30 | 1162.69 | 1150.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1150.30 | 1162.69 | 1150.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 1154.80 | 1162.69 | 1150.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1154.40 | 1161.03 | 1151.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 1151.60 | 1161.03 | 1151.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1151.50 | 1159.13 | 1151.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 1155.80 | 1159.13 | 1151.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1157.00 | 1158.70 | 1151.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 1151.10 | 1158.70 | 1151.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1170.00 | 1161.87 | 1154.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 1141.10 | 1161.87 | 1154.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1132.90 | 1156.08 | 1152.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 1132.90 | 1156.08 | 1152.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1135.50 | 1151.96 | 1151.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 1131.90 | 1151.96 | 1151.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 1131.10 | 1147.79 | 1149.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 15:15:00 | 1122.20 | 1138.53 | 1144.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 1137.80 | 1135.26 | 1141.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 1137.80 | 1135.26 | 1141.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1140.30 | 1136.16 | 1140.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1140.30 | 1136.16 | 1140.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1138.00 | 1136.53 | 1139.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1145.30 | 1136.53 | 1139.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1138.90 | 1137.00 | 1139.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 1132.40 | 1137.00 | 1139.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 1133.90 | 1127.28 | 1130.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1140.50 | 1134.13 | 1133.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 1140.50 | 1134.13 | 1133.38 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 1127.30 | 1133.50 | 1133.66 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1142.10 | 1132.76 | 1132.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1159.20 | 1143.58 | 1138.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1134.10 | 1143.91 | 1139.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1134.10 | 1143.91 | 1139.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1134.10 | 1143.91 | 1139.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1134.10 | 1143.91 | 1139.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1131.40 | 1141.41 | 1138.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1137.30 | 1141.41 | 1138.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1131.30 | 1139.39 | 1137.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:15:00 | 1139.90 | 1139.39 | 1137.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1134.20 | 1137.79 | 1137.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 1134.00 | 1137.79 | 1137.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 1133.00 | 1136.83 | 1137.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1124.20 | 1134.01 | 1135.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 1071.70 | 1064.37 | 1078.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:00:00 | 1071.70 | 1064.37 | 1078.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1074.50 | 1067.62 | 1077.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1074.50 | 1067.62 | 1077.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1076.00 | 1069.29 | 1077.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 1081.90 | 1072.53 | 1077.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1088.20 | 1075.67 | 1078.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 1089.00 | 1075.67 | 1078.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 1091.30 | 1082.22 | 1081.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1095.80 | 1086.27 | 1083.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 1096.70 | 1097.88 | 1093.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:30:00 | 1097.10 | 1097.88 | 1093.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1100.00 | 1098.30 | 1093.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:45:00 | 1101.10 | 1098.64 | 1094.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1102.20 | 1098.64 | 1094.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1101.40 | 1097.75 | 1096.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:00:00 | 1103.00 | 1098.80 | 1096.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1098.80 | 1098.80 | 1097.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1098.80 | 1098.80 | 1097.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1103.20 | 1099.68 | 1097.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1106.30 | 1100.64 | 1098.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1106.10 | 1101.59 | 1098.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:00:00 | 1107.20 | 1102.72 | 1099.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 1107.80 | 1104.00 | 1100.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1099.30 | 1104.97 | 1102.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 1099.30 | 1104.97 | 1102.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1099.90 | 1103.95 | 1102.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:15:00 | 1087.00 | 1103.95 | 1102.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1079.50 | 1096.35 | 1098.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1084.60 | 1083.52 | 1089.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1084.60 | 1083.52 | 1089.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1088.80 | 1084.57 | 1089.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1088.80 | 1084.57 | 1089.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1079.20 | 1083.50 | 1088.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 1076.40 | 1083.50 | 1088.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 1077.60 | 1081.24 | 1086.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:30:00 | 1075.70 | 1078.23 | 1084.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 1077.40 | 1073.48 | 1078.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1073.70 | 1073.53 | 1077.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1069.40 | 1073.53 | 1077.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1022.58 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1023.72 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1021.91 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1023.53 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1015.93 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1048.90 | 1043.97 | 1054.68 | SL hit (close>ema200) qty=0.50 sl=1043.97 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 1076.20 | 1060.61 | 1058.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 1078.20 | 1064.13 | 1060.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1062.00 | 1063.71 | 1060.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1062.00 | 1063.71 | 1060.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1062.00 | 1063.71 | 1060.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 1062.20 | 1063.71 | 1060.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1065.70 | 1064.10 | 1061.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1060.20 | 1064.10 | 1061.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1073.90 | 1066.06 | 1062.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1084.80 | 1069.25 | 1064.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 1065.50 | 1077.64 | 1078.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1065.50 | 1077.64 | 1078.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 1062.70 | 1073.65 | 1076.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1055.10 | 1052.18 | 1059.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1056.10 | 1052.18 | 1059.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1054.60 | 1053.22 | 1058.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 1054.20 | 1053.22 | 1058.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1070.00 | 1056.33 | 1059.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 1071.70 | 1056.33 | 1059.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1070.90 | 1059.25 | 1060.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 1073.30 | 1059.25 | 1060.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1071.30 | 1061.66 | 1061.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1076.30 | 1066.34 | 1063.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1080.80 | 1082.05 | 1074.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 1080.80 | 1082.05 | 1074.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1081.30 | 1081.89 | 1076.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1073.70 | 1081.89 | 1076.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1075.90 | 1080.69 | 1076.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 12:00:00 | 1083.30 | 1081.22 | 1076.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1084.00 | 1081.40 | 1077.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 15:15:00 | 1086.00 | 1081.12 | 1077.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 1073.20 | 1076.84 | 1076.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 1073.20 | 1076.84 | 1076.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1069.10 | 1075.13 | 1076.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 1072.50 | 1071.19 | 1073.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 1072.50 | 1071.19 | 1073.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1072.50 | 1071.19 | 1073.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 1074.70 | 1071.19 | 1073.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1074.70 | 1071.89 | 1073.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 1074.70 | 1071.89 | 1073.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1072.90 | 1072.09 | 1073.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 1073.90 | 1072.09 | 1073.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1073.10 | 1072.29 | 1073.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1073.10 | 1072.29 | 1073.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1074.70 | 1072.78 | 1073.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1065.10 | 1072.78 | 1073.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 1072.70 | 1056.48 | 1055.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1072.70 | 1056.48 | 1055.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1073.60 | 1059.90 | 1057.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1053.00 | 1061.01 | 1058.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 1053.00 | 1061.01 | 1058.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1053.00 | 1061.01 | 1058.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1053.00 | 1061.01 | 1058.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1080.00 | 1064.81 | 1060.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 1052.20 | 1064.81 | 1060.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1061.00 | 1064.05 | 1060.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 1061.00 | 1064.05 | 1060.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1060.00 | 1063.24 | 1060.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 1055.90 | 1063.24 | 1060.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1054.70 | 1061.53 | 1060.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 1053.10 | 1061.53 | 1060.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1066.00 | 1062.42 | 1060.73 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1053.10 | 1058.96 | 1059.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 1048.30 | 1055.86 | 1057.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1057.80 | 1055.63 | 1057.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 15:15:00 | 1057.80 | 1055.63 | 1057.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1057.80 | 1055.63 | 1057.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1055.80 | 1055.63 | 1057.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1056.60 | 1055.82 | 1057.33 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 1070.40 | 1058.74 | 1058.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1076.40 | 1069.23 | 1064.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 1076.10 | 1076.33 | 1070.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:15:00 | 1083.80 | 1076.33 | 1070.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1097.40 | 1080.54 | 1073.08 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1072.00 | 1076.75 | 1076.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1054.60 | 1068.29 | 1072.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1065.00 | 1064.56 | 1070.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:30:00 | 1068.70 | 1064.56 | 1070.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1050.30 | 1046.19 | 1055.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 1044.00 | 1046.19 | 1055.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1043.00 | 1045.17 | 1052.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 1044.00 | 1044.46 | 1051.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 1044.80 | 1043.54 | 1050.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1041.30 | 1043.09 | 1049.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 1044.40 | 1043.09 | 1049.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | SL hit (close>static) qty=1.00 sl=1058.60 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1068.60 | 1050.52 | 1050.28 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1042.20 | 1052.89 | 1053.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1018.80 | 1046.08 | 1050.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 951.80 | 951.32 | 972.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 964.30 | 951.32 | 972.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 966.60 | 954.38 | 971.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 969.90 | 954.38 | 971.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 969.10 | 959.78 | 967.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 969.40 | 959.78 | 967.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 961.20 | 960.06 | 967.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 957.30 | 959.07 | 966.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 14:30:00 | 960.60 | 952.61 | 959.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 948.30 | 952.61 | 959.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 960.70 | 952.53 | 952.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 960.70 | 952.53 | 952.50 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 943.60 | 952.11 | 952.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 939.00 | 949.49 | 951.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 954.00 | 946.35 | 948.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 954.00 | 946.35 | 948.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 954.00 | 946.35 | 948.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 955.90 | 946.35 | 948.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 959.00 | 948.88 | 949.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 958.90 | 948.88 | 949.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 958.10 | 950.73 | 950.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 975.00 | 956.75 | 953.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 962.70 | 965.72 | 960.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 962.70 | 965.72 | 960.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 962.70 | 965.72 | 960.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 962.70 | 965.72 | 960.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 957.30 | 964.03 | 959.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 960.25 | 964.03 | 959.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 953.55 | 961.94 | 959.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:45:00 | 953.95 | 961.94 | 959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 960.60 | 961.67 | 959.32 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 933.35 | 954.65 | 956.72 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 989.10 | 959.08 | 956.80 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 960.00 | 971.25 | 971.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 954.25 | 964.50 | 967.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 13:15:00 | 927.45 | 924.25 | 940.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 14:00:00 | 927.45 | 924.25 | 940.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 941.70 | 927.74 | 940.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:45:00 | 939.00 | 927.74 | 940.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 940.50 | 930.29 | 940.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 938.00 | 931.57 | 940.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 937.10 | 933.39 | 940.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:30:00 | 938.65 | 936.91 | 940.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 934.95 | 934.64 | 936.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 937.55 | 935.22 | 936.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 926.35 | 935.22 | 936.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 940.05 | 930.37 | 932.79 | SL hit (close>static) qty=1.00 sl=939.95 alert=retest2 |

### Cycle 222 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 901.05 | 881.70 | 880.24 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 880.30 | 881.21 | 881.26 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 888.00 | 882.01 | 881.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 904.75 | 888.49 | 884.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 891.30 | 892.66 | 887.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 891.30 | 892.66 | 887.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 891.30 | 892.66 | 887.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 891.30 | 892.66 | 887.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 886.70 | 891.46 | 887.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 882.55 | 891.46 | 887.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 881.15 | 889.40 | 887.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 880.80 | 889.40 | 887.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 883.55 | 888.39 | 887.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 883.55 | 888.39 | 887.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 882.45 | 887.20 | 886.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 879.50 | 887.20 | 886.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 877.85 | 885.33 | 885.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 872.90 | 882.85 | 884.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 860.00 | 854.77 | 863.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 828.95 | 854.77 | 863.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 845.00 | 834.78 | 841.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 845.00 | 834.78 | 841.17 | SL hit (close>ema400) qty=1.00 sl=841.17 alert=retest1 |

### Cycle 226 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 805.55 | 814.35 | 815.25 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 843.10 | 820.38 | 817.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 857.25 | 842.62 | 835.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 850.00 | 850.98 | 843.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 844.55 | 850.98 | 843.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 846.00 | 849.99 | 843.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 840.15 | 849.99 | 843.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 848.00 | 849.86 | 846.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 851.70 | 849.86 | 846.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:45:00 | 849.75 | 849.93 | 846.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 849.65 | 849.32 | 846.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 836.05 | 845.39 | 845.33 | SL hit (close<static) qty=1.00 sl=837.60 alert=retest2 |

### Cycle 229 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 840.15 | 844.34 | 844.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 833.15 | 842.10 | 843.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 836.10 | 822.60 | 829.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 836.10 | 822.60 | 829.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 836.10 | 822.60 | 829.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 836.10 | 822.60 | 829.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 848.55 | 827.79 | 830.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 841.75 | 827.79 | 830.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 861.70 | 834.58 | 833.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 874.00 | 842.46 | 837.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 857.50 | 865.01 | 855.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 857.50 | 865.01 | 855.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 857.50 | 865.01 | 855.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 857.50 | 865.01 | 855.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 855.00 | 863.01 | 855.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 864.85 | 863.01 | 855.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 859.35 | 862.28 | 855.92 | EMA400 retest candle locked (from upside) |

### Cycle 231 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 839.80 | 852.58 | 852.62 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 13:15:00 | 858.55 | 853.78 | 853.16 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 845.60 | 852.14 | 852.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 839.00 | 849.51 | 851.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 836.35 | 823.36 | 833.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 836.35 | 823.36 | 833.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 836.35 | 823.36 | 833.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 843.50 | 823.36 | 833.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 820.65 | 822.82 | 831.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 816.10 | 822.00 | 829.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:00:00 | 818.00 | 821.20 | 828.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 817.35 | 816.78 | 823.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:00:00 | 817.90 | 817.01 | 822.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 833.85 | 820.38 | 823.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 833.85 | 820.38 | 823.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | SL hit (close>static) qty=1.00 sl=836.35 alert=retest2 |

### Cycle 234 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 842.00 | 827.75 | 826.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 849.90 | 834.36 | 830.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 867.15 | 867.23 | 853.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 867.15 | 867.23 | 853.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 853.95 | 870.29 | 864.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 853.95 | 870.29 | 864.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 868.50 | 869.93 | 864.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 868.95 | 869.93 | 864.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:45:00 | 868.70 | 869.64 | 864.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 869.25 | 869.64 | 864.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 13:00:00 | 871.15 | 869.94 | 865.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 879.35 | 871.82 | 866.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:00:00 | 892.10 | 878.88 | 871.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:00:00 | 888.55 | 880.82 | 873.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 888.50 | 882.40 | 874.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 889.55 | 887.53 | 880.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 890.00 | 891.40 | 885.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 906.15 | 891.40 | 885.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:45:00 | 902.90 | 902.44 | 897.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-24 09:15:00 | 955.85 | 941.82 | 931.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 12:15:00 | 926.60 | 935.21 | 938.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 929.50 | 928.95 | 933.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 11:30:00 | 925.90 | 928.95 | 933.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 939.85 | 930.78 | 932.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 939.85 | 930.78 | 932.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 941.40 | 932.91 | 933.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 941.40 | 932.91 | 933.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 936.20 | 933.87 | 933.64 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 13:15:00 | 931.60 | 933.41 | 933.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 928.00 | 931.71 | 932.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 932.45 | 931.86 | 932.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 932.45 | 931.86 | 932.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 932.45 | 931.86 | 932.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 930.60 | 931.86 | 932.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 932.85 | 932.06 | 932.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 932.85 | 932.06 | 932.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 932.50 | 932.15 | 932.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 932.75 | 932.15 | 932.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 238 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 938.20 | 933.36 | 933.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 945.00 | 936.04 | 934.39 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 649.15 | 2024-04-12 14:15:00 | 636.15 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-04-23 09:15:00 | 686.95 | 2024-04-25 14:15:00 | 680.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-04-23 11:00:00 | 686.90 | 2024-04-25 14:15:00 | 680.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-04-23 12:00:00 | 685.70 | 2024-04-25 14:15:00 | 680.20 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-04-23 13:45:00 | 684.60 | 2024-04-25 14:15:00 | 680.20 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-04-25 09:15:00 | 691.55 | 2024-04-25 14:15:00 | 680.20 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-04-25 09:45:00 | 686.70 | 2024-04-25 14:15:00 | 680.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-02 13:15:00 | 680.65 | 2024-05-03 14:15:00 | 686.25 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-05-02 14:15:00 | 681.30 | 2024-05-03 14:15:00 | 686.25 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-05-02 14:45:00 | 680.95 | 2024-05-03 14:15:00 | 686.25 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-03 10:15:00 | 681.40 | 2024-05-03 14:15:00 | 686.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-06 11:30:00 | 683.95 | 2024-05-07 12:15:00 | 678.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-05-06 13:30:00 | 683.45 | 2024-05-07 12:15:00 | 678.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-05-07 09:15:00 | 687.45 | 2024-05-07 12:15:00 | 678.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-05-18 09:15:00 | 730.60 | 2024-05-18 12:15:00 | 714.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-06-12 09:15:00 | 744.10 | 2024-06-18 14:15:00 | 727.60 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-06-12 09:45:00 | 744.00 | 2024-06-18 14:15:00 | 727.60 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-06-12 12:00:00 | 743.90 | 2024-06-18 14:15:00 | 727.60 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-06-12 14:15:00 | 742.70 | 2024-06-18 14:15:00 | 727.60 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-06-13 09:15:00 | 740.85 | 2024-06-18 14:15:00 | 727.60 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-06-13 09:45:00 | 741.40 | 2024-06-18 14:15:00 | 727.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-07-12 11:30:00 | 722.85 | 2024-07-15 12:15:00 | 730.85 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-07-12 14:30:00 | 722.50 | 2024-07-15 12:15:00 | 730.85 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-12 15:00:00 | 721.50 | 2024-07-15 12:15:00 | 730.85 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-05 12:15:00 | 837.60 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2024-08-06 13:00:00 | 836.00 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2024-08-06 13:30:00 | 836.90 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2024-08-06 14:45:00 | 840.55 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2024-08-09 14:30:00 | 861.80 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-08-12 10:00:00 | 860.00 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-08-13 10:45:00 | 858.05 | 2024-08-13 15:15:00 | 851.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-08-16 09:15:00 | 871.00 | 2024-08-16 09:15:00 | 849.80 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-08-23 12:30:00 | 899.95 | 2024-08-29 12:15:00 | 917.85 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2024-09-04 10:30:00 | 911.05 | 2024-09-04 11:15:00 | 928.55 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-09-23 14:30:00 | 1211.70 | 2024-09-26 12:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-09-25 09:30:00 | 1208.10 | 2024-09-26 12:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-09-25 10:15:00 | 1209.05 | 2024-09-26 12:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-09-25 11:15:00 | 1209.00 | 2024-09-26 12:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-09-25 15:15:00 | 1210.00 | 2024-09-26 12:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-09-26 10:30:00 | 1214.70 | 2024-09-26 12:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1156.50 | 2024-10-03 15:15:00 | 1106.61 | PARTIAL | 0.50 | 4.31% |
| SELL | retest2 | 2024-09-30 10:15:00 | 1157.45 | 2024-10-03 15:15:00 | 1105.80 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2024-10-01 10:00:00 | 1164.85 | 2024-10-04 09:15:00 | 1098.67 | PARTIAL | 0.50 | 5.68% |
| SELL | retest2 | 2024-10-01 13:45:00 | 1164.00 | 2024-10-04 09:15:00 | 1099.58 | PARTIAL | 0.50 | 5.53% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1149.75 | 2024-10-04 09:15:00 | 1092.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1156.50 | 2024-10-04 11:15:00 | 1133.30 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2024-09-30 10:15:00 | 1157.45 | 2024-10-04 11:15:00 | 1133.30 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2024-10-01 10:00:00 | 1164.85 | 2024-10-04 11:15:00 | 1133.30 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2024-10-01 13:45:00 | 1164.00 | 2024-10-04 11:15:00 | 1133.30 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1149.75 | 2024-10-04 11:15:00 | 1133.30 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2024-10-18 15:15:00 | 1172.90 | 2024-10-22 13:15:00 | 1114.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 1173.30 | 2024-10-22 13:15:00 | 1114.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:15:00 | 1172.80 | 2024-10-22 13:15:00 | 1114.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 1173.80 | 2024-10-22 13:15:00 | 1115.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1169.75 | 2024-10-23 09:15:00 | 1111.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:15:00 | 1172.90 | 2024-10-23 10:15:00 | 1140.90 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2024-10-21 09:30:00 | 1173.30 | 2024-10-23 10:15:00 | 1140.90 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2024-10-21 10:15:00 | 1172.80 | 2024-10-23 10:15:00 | 1140.90 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2024-10-21 12:00:00 | 1173.80 | 2024-10-23 10:15:00 | 1140.90 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1169.75 | 2024-10-23 10:15:00 | 1140.90 | STOP_HIT | 0.50 | 2.47% |
| BUY | retest2 | 2024-11-06 15:15:00 | 1276.30 | 2024-11-08 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-11-08 13:45:00 | 1243.90 | 2024-11-08 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-11-08 14:15:00 | 1247.15 | 2024-11-08 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-11-12 14:00:00 | 1224.80 | 2024-11-12 15:15:00 | 1163.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 14:00:00 | 1224.80 | 2024-11-13 09:15:00 | 1102.32 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-03 09:30:00 | 1209.20 | 2024-12-05 15:15:00 | 1205.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-12-05 15:15:00 | 1205.00 | 2024-12-05 15:15:00 | 1205.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-01-09 12:30:00 | 1027.65 | 2025-01-10 09:15:00 | 976.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:15:00 | 1021.05 | 2025-01-10 14:15:00 | 970.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:30:00 | 1027.65 | 2025-01-13 13:15:00 | 924.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 13:15:00 | 1021.05 | 2025-01-13 14:15:00 | 918.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-01 09:15:00 | 985.00 | 2025-02-03 09:15:00 | 942.45 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2025-02-01 10:00:00 | 983.60 | 2025-02-03 09:15:00 | 942.45 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-02-01 12:00:00 | 983.60 | 2025-02-03 09:15:00 | 942.45 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-02-01 14:45:00 | 984.65 | 2025-02-03 09:15:00 | 942.45 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2025-02-12 09:15:00 | 953.50 | 2025-02-14 09:15:00 | 940.50 | PARTIAL | 0.50 | 1.36% |
| SELL | retest2 | 2025-02-12 15:15:00 | 990.00 | 2025-02-14 09:15:00 | 940.64 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-02-12 09:15:00 | 953.50 | 2025-02-14 14:15:00 | 943.95 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-02-12 15:15:00 | 990.00 | 2025-02-14 14:15:00 | 943.95 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-02-13 10:15:00 | 990.15 | 2025-02-17 12:15:00 | 992.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-02-17 10:00:00 | 986.50 | 2025-02-17 12:15:00 | 992.20 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-02-21 09:15:00 | 1005.75 | 2025-02-21 14:15:00 | 968.00 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-02-25 15:00:00 | 936.10 | 2025-02-28 09:15:00 | 889.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 15:00:00 | 936.10 | 2025-02-28 14:15:00 | 900.35 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-03-12 09:15:00 | 885.20 | 2025-03-18 11:15:00 | 888.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-04-01 14:15:00 | 914.95 | 2025-04-07 09:15:00 | 859.05 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest2 | 2025-04-02 09:45:00 | 906.20 | 2025-04-07 09:15:00 | 859.05 | STOP_HIT | 1.00 | -5.20% |
| SELL | retest2 | 2025-04-29 13:15:00 | 903.10 | 2025-05-05 12:15:00 | 914.85 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-02 15:15:00 | 893.90 | 2025-05-05 12:15:00 | 914.85 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-05 10:15:00 | 904.95 | 2025-05-05 12:15:00 | 914.85 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-05-05 10:45:00 | 902.25 | 2025-05-05 12:15:00 | 914.85 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-05-08 09:45:00 | 878.50 | 2025-05-12 11:15:00 | 890.55 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-08 10:45:00 | 878.05 | 2025-05-12 11:15:00 | 890.55 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-05-08 15:15:00 | 870.00 | 2025-05-12 11:15:00 | 890.55 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-22 10:45:00 | 1034.90 | 2025-05-27 10:15:00 | 1138.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-09 10:15:00 | 1135.60 | 2025-06-10 10:15:00 | 1151.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-09 15:15:00 | 1141.30 | 2025-06-10 10:15:00 | 1151.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1128.20 | 2025-06-13 09:15:00 | 1071.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1128.20 | 2025-06-16 09:15:00 | 1130.10 | STOP_HIT | 0.50 | -0.17% |
| SELL | retest2 | 2025-06-13 09:45:00 | 1118.10 | 2025-06-17 11:15:00 | 1130.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-16 10:45:00 | 1125.90 | 2025-06-17 11:15:00 | 1130.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-06-16 11:30:00 | 1111.20 | 2025-06-17 11:15:00 | 1130.40 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-19 09:15:00 | 1130.70 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-19 10:45:00 | 1141.10 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-06-19 13:45:00 | 1130.00 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-20 09:30:00 | 1131.00 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-30 12:15:00 | 1215.50 | 2025-07-04 13:15:00 | 1190.90 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-07-17 14:30:00 | 1201.80 | 2025-07-22 15:15:00 | 1205.90 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-07-17 15:00:00 | 1201.90 | 2025-07-22 15:15:00 | 1205.90 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-18 11:30:00 | 1207.00 | 2025-07-22 15:15:00 | 1205.90 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-07-28 09:15:00 | 1168.60 | 2025-07-28 11:15:00 | 1189.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-07-28 10:30:00 | 1167.20 | 2025-07-28 11:15:00 | 1189.80 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-08-05 11:30:00 | 1129.10 | 2025-08-11 09:15:00 | 1072.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:30:00 | 1129.10 | 2025-08-11 13:15:00 | 1095.10 | STOP_HIT | 0.50 | 3.01% |
| BUY | retest2 | 2025-09-04 14:00:00 | 1061.50 | 2025-09-16 10:15:00 | 1115.80 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1065.70 | 2025-09-16 10:15:00 | 1115.80 | STOP_HIT | 1.00 | 4.70% |
| BUY | retest2 | 2025-09-05 11:00:00 | 1061.00 | 2025-09-16 10:15:00 | 1115.80 | STOP_HIT | 1.00 | 5.16% |
| SELL | retest2 | 2025-09-18 13:00:00 | 1099.30 | 2025-09-18 14:15:00 | 1116.80 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1098.20 | 2025-09-26 09:15:00 | 1043.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1100.00 | 2025-09-26 09:15:00 | 1045.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1098.20 | 2025-09-29 09:15:00 | 1061.40 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1100.00 | 2025-09-29 09:15:00 | 1061.40 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1102.60 | 2025-10-07 10:15:00 | 1087.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-16 13:45:00 | 1070.80 | 2025-10-20 09:15:00 | 1080.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-16 14:45:00 | 1068.40 | 2025-10-20 09:15:00 | 1080.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-17 10:00:00 | 1068.70 | 2025-10-20 09:15:00 | 1080.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1125.70 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1141.00 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-28 10:30:00 | 1127.10 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1127.40 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-29 13:30:00 | 1131.50 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-11-11 10:15:00 | 1132.40 | 2025-11-12 15:15:00 | 1140.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-12 12:45:00 | 1133.90 | 2025-11-12 15:15:00 | 1140.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-27 14:45:00 | 1101.10 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1102.20 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1101.40 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-01 10:00:00 | 1103.00 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1106.30 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-01 14:15:00 | 1106.10 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-01 15:00:00 | 1107.20 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-02 09:30:00 | 1107.80 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-04 12:15:00 | 1076.40 | 2025-12-09 09:15:00 | 1022.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:45:00 | 1077.60 | 2025-12-09 09:15:00 | 1023.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:30:00 | 1075.70 | 2025-12-09 09:15:00 | 1021.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:30:00 | 1077.40 | 2025-12-09 09:15:00 | 1023.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1069.40 | 2025-12-09 09:15:00 | 1015.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 12:15:00 | 1076.40 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-12-04 13:45:00 | 1077.60 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2025-12-04 14:30:00 | 1075.70 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2025-12-05 14:30:00 | 1077.40 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1069.40 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 1.92% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1084.80 | 2025-12-16 10:15:00 | 1065.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-23 12:00:00 | 1083.30 | 2025-12-24 12:15:00 | 1073.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1084.00 | 2025-12-24 12:15:00 | 1073.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-23 15:15:00 | 1086.00 | 2025-12-24 12:15:00 | 1073.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1065.10 | 2025-12-31 13:15:00 | 1072.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-12 15:15:00 | 1044.00 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1043.00 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-13 11:30:00 | 1044.00 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-01-13 13:30:00 | 1044.80 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-01-23 09:30:00 | 957.30 | 2026-01-28 13:15:00 | 960.70 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-23 14:30:00 | 960.60 | 2026-01-28 13:15:00 | 960.70 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-23 15:00:00 | 948.30 | 2026-01-28 13:15:00 | 960.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-10 09:30:00 | 938.00 | 2026-02-12 14:15:00 | 940.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-02-10 11:15:00 | 937.10 | 2026-02-16 15:15:00 | 891.10 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-02-10 14:30:00 | 938.65 | 2026-02-16 15:15:00 | 890.25 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-11 14:30:00 | 934.95 | 2026-02-16 15:15:00 | 891.72 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2026-02-12 09:15:00 | 926.35 | 2026-02-16 15:15:00 | 888.20 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2026-02-10 11:15:00 | 937.10 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2026-02-10 14:30:00 | 938.65 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2026-02-11 14:30:00 | 934.95 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2026-02-12 09:15:00 | 926.35 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2026-02-13 09:15:00 | 924.25 | 2026-02-19 12:15:00 | 878.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 924.25 | 2026-02-20 14:15:00 | 870.90 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest1 | 2026-03-04 09:15:00 | 828.95 | 2026-03-05 15:15:00 | 845.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-06 13:45:00 | 821.20 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-06 14:15:00 | 819.40 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-03-09 09:15:00 | 798.30 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2026-03-09 13:15:00 | 819.95 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-10 09:30:00 | 819.80 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-10 14:30:00 | 819.00 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-12 12:30:00 | 819.15 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-03-20 09:15:00 | 851.70 | 2026-03-20 12:15:00 | 836.05 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-03-20 09:45:00 | 849.75 | 2026-03-20 12:15:00 | 836.05 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-03-20 11:15:00 | 849.65 | 2026-03-20 12:15:00 | 836.05 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-01 13:15:00 | 816.10 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2026-04-01 14:00:00 | 818.00 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-04-02 12:15:00 | 817.35 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-02 13:00:00 | 817.90 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-04-09 11:15:00 | 868.95 | 2026-04-24 09:15:00 | 955.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 11:45:00 | 868.70 | 2026-04-24 09:15:00 | 955.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 12:15:00 | 869.25 | 2026-04-24 09:15:00 | 956.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 13:00:00 | 871.15 | 2026-04-24 09:15:00 | 958.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 11:00:00 | 892.10 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.44% |
| BUY | retest2 | 2026-04-10 12:00:00 | 888.55 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.86% |
| BUY | retest2 | 2026-04-10 12:30:00 | 888.50 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.86% |
| BUY | retest2 | 2026-04-13 10:15:00 | 889.55 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.74% |
| BUY | retest2 | 2026-04-15 09:15:00 | 906.15 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2026-04-16 13:45:00 | 902.90 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 3.19% |

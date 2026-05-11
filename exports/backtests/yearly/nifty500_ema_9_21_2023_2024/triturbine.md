# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 598.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 244 |
| ALERT1 | 156 |
| ALERT2 | 155 |
| ALERT2_SKIP | 109 |
| ALERT3 | 312 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 103 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 82
- **Target hits / Stop hits / Partials:** 6 / 106 / 11
- **Avg / median % per leg:** 0.10% / -0.99%
- **Sum % (uncompounded):** 11.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 20 | 31.7% | 3 | 57 | 3 | -0.30% | -19.2% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 0 | 5 | 3 | 2.67% | 21.4% |
| BUY @ 3rd Alert (retest2) | 55 | 14 | 25.5% | 3 | 52 | 0 | -0.74% | -40.6% |
| SELL (all) | 60 | 21 | 35.0% | 3 | 49 | 8 | 0.52% | 31.1% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.14% | -8.6% |
| SELL @ 3rd Alert (retest2) | 56 | 21 | 37.5% | 3 | 45 | 8 | 0.71% | 39.7% |
| retest1 (combined) | 12 | 6 | 50.0% | 0 | 9 | 3 | 1.07% | 12.8% |
| retest2 (combined) | 111 | 35 | 31.5% | 6 | 97 | 8 | -0.01% | -0.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 09:15:00 | 393.55 | 397.48 | 397.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 12:15:00 | 384.70 | 393.92 | 396.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 391.10 | 389.93 | 393.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 09:15:00 | 391.10 | 389.93 | 393.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 391.10 | 389.93 | 393.19 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 12:15:00 | 393.60 | 387.97 | 387.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 13:15:00 | 397.80 | 389.94 | 388.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 398.70 | 399.28 | 396.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 400.85 | 400.76 | 398.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 400.85 | 400.76 | 398.38 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 14:15:00 | 394.20 | 397.63 | 397.71 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 400.90 | 397.70 | 397.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 09:15:00 | 391.95 | 398.16 | 398.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 10:15:00 | 390.10 | 396.55 | 397.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 09:15:00 | 396.70 | 392.99 | 394.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 396.70 | 392.99 | 394.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 396.70 | 392.99 | 394.79 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 400.00 | 391.39 | 391.30 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 15:15:00 | 390.00 | 391.11 | 391.18 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 393.20 | 391.53 | 391.36 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 13:15:00 | 390.15 | 391.14 | 391.23 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 15:15:00 | 392.35 | 391.36 | 391.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 395.55 | 392.20 | 391.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 13:15:00 | 391.00 | 392.62 | 392.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 13:15:00 | 391.00 | 392.62 | 392.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 391.00 | 392.62 | 392.13 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 14:15:00 | 387.70 | 391.64 | 391.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 12:15:00 | 385.55 | 389.81 | 390.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 09:15:00 | 399.70 | 390.01 | 390.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 399.70 | 390.01 | 390.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 399.70 | 390.01 | 390.30 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 10:15:00 | 398.80 | 391.76 | 391.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 09:15:00 | 406.55 | 399.14 | 395.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 13:15:00 | 401.45 | 402.79 | 398.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 404.90 | 406.36 | 402.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 404.90 | 406.36 | 402.30 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 14:15:00 | 402.80 | 407.97 | 408.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 15:15:00 | 400.00 | 406.38 | 407.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 10:15:00 | 409.65 | 406.84 | 407.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 409.65 | 406.84 | 407.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 409.65 | 406.84 | 407.59 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 12:15:00 | 411.40 | 408.53 | 408.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 413.60 | 410.09 | 409.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 409.70 | 410.56 | 409.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 409.70 | 410.56 | 409.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 409.70 | 410.56 | 409.49 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 12:15:00 | 408.60 | 414.21 | 414.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 14:15:00 | 407.60 | 412.06 | 413.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 11:15:00 | 411.10 | 410.36 | 411.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 11:15:00 | 411.10 | 410.36 | 411.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 411.10 | 410.36 | 411.99 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 404.30 | 401.98 | 401.86 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 11:15:00 | 400.25 | 401.96 | 402.04 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 15:15:00 | 402.50 | 401.78 | 401.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 403.40 | 402.10 | 401.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 10:15:00 | 401.30 | 401.94 | 401.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 10:15:00 | 401.30 | 401.94 | 401.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 401.30 | 401.94 | 401.81 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 399.80 | 401.51 | 401.63 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 14:15:00 | 407.30 | 402.69 | 402.12 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 14:15:00 | 401.15 | 402.05 | 402.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 15:15:00 | 400.65 | 401.77 | 401.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 401.80 | 399.65 | 400.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 401.80 | 399.65 | 400.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 401.80 | 399.65 | 400.39 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 12:15:00 | 400.75 | 398.75 | 398.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 14:15:00 | 403.20 | 399.94 | 399.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 09:15:00 | 399.80 | 400.58 | 399.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 399.80 | 400.58 | 399.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 399.80 | 400.58 | 399.71 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 11:15:00 | 388.30 | 398.25 | 399.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 12:15:00 | 387.40 | 396.08 | 398.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 09:15:00 | 399.35 | 393.92 | 396.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 399.35 | 393.92 | 396.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 399.35 | 393.92 | 396.08 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 399.65 | 396.21 | 396.06 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 14:15:00 | 395.25 | 397.24 | 397.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 09:15:00 | 392.50 | 396.09 | 396.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 09:15:00 | 401.55 | 393.58 | 394.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 401.55 | 393.58 | 394.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 401.55 | 393.58 | 394.59 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 11:15:00 | 399.20 | 395.81 | 395.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 11:15:00 | 402.20 | 400.14 | 398.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 14:15:00 | 405.20 | 406.16 | 403.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 400.25 | 405.11 | 403.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 400.25 | 405.11 | 403.69 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 12:15:00 | 400.50 | 402.93 | 402.93 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 10:15:00 | 406.45 | 403.23 | 402.92 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 12:15:00 | 399.75 | 403.19 | 403.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 10:15:00 | 398.35 | 400.95 | 402.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 400.75 | 399.76 | 400.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 400.75 | 399.76 | 400.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 400.75 | 399.76 | 400.91 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 10:15:00 | 413.95 | 402.51 | 401.06 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 396.45 | 399.91 | 400.14 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 399.05 | 397.51 | 397.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 10:15:00 | 401.30 | 398.27 | 397.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 13:15:00 | 398.15 | 398.66 | 398.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 13:15:00 | 398.15 | 398.66 | 398.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 398.15 | 398.66 | 398.16 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 393.65 | 398.67 | 398.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 10:15:00 | 390.05 | 396.95 | 398.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 393.35 | 392.34 | 394.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 393.35 | 392.34 | 394.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 393.35 | 392.34 | 394.81 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 384.15 | 377.15 | 376.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 10:15:00 | 388.25 | 379.37 | 377.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 14:15:00 | 387.65 | 387.68 | 384.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 13:15:00 | 385.30 | 386.74 | 385.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 385.30 | 386.74 | 385.36 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 382.95 | 386.84 | 387.23 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 13:15:00 | 389.00 | 387.56 | 387.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 390.80 | 388.20 | 387.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 10:15:00 | 387.30 | 389.18 | 388.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 10:15:00 | 387.30 | 389.18 | 388.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 387.30 | 389.18 | 388.43 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 387.55 | 388.35 | 388.35 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 389.95 | 388.53 | 388.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 11:15:00 | 395.95 | 390.45 | 389.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 14:15:00 | 389.00 | 390.45 | 389.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 14:15:00 | 389.00 | 390.45 | 389.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 14:15:00 | 389.00 | 390.45 | 389.64 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 400.50 | 405.21 | 405.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 395.80 | 402.73 | 404.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 399.65 | 396.18 | 399.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 399.65 | 396.18 | 399.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 399.65 | 396.18 | 399.21 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 408.65 | 400.43 | 400.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 11:15:00 | 417.60 | 405.56 | 402.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 14:15:00 | 432.40 | 433.53 | 423.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 450.95 | 437.17 | 426.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 450.95 | 437.17 | 426.57 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-09-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 13:15:00 | 421.00 | 429.18 | 429.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 408.80 | 422.32 | 425.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 417.80 | 411.54 | 414.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 417.80 | 411.54 | 414.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 417.80 | 411.54 | 414.95 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 439.05 | 420.44 | 417.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 10:15:00 | 443.95 | 425.14 | 420.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 439.90 | 440.83 | 434.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 14:15:00 | 438.05 | 440.83 | 437.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 438.05 | 440.83 | 437.85 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 433.00 | 436.51 | 436.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 15:15:00 | 428.45 | 433.06 | 434.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 371.50 | 371.00 | 375.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 377.10 | 372.65 | 374.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 377.10 | 372.65 | 374.07 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 340.90 | 333.59 | 333.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 351.60 | 340.16 | 336.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 385.75 | 387.44 | 378.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 13:15:00 | 376.85 | 383.57 | 379.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 376.85 | 383.57 | 379.21 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 09:15:00 | 410.60 | 415.03 | 415.07 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 10:15:00 | 417.30 | 415.48 | 415.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 14:15:00 | 418.35 | 416.45 | 415.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 12:15:00 | 416.35 | 417.64 | 416.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 12:15:00 | 416.35 | 417.64 | 416.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 416.35 | 417.64 | 416.79 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 410.90 | 415.27 | 415.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 11:15:00 | 409.30 | 412.38 | 414.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 11:15:00 | 419.85 | 411.33 | 412.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 11:15:00 | 419.85 | 411.33 | 412.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 419.85 | 411.33 | 412.24 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 420.05 | 413.08 | 412.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 433.30 | 418.28 | 415.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 12:15:00 | 427.55 | 428.49 | 424.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 433.85 | 432.15 | 428.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 433.85 | 432.15 | 428.37 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 424.15 | 427.86 | 428.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 15:15:00 | 422.95 | 425.70 | 427.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 15:15:00 | 404.35 | 404.17 | 409.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 406.45 | 404.62 | 408.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 406.45 | 404.62 | 408.93 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 412.75 | 408.96 | 408.72 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 11:15:00 | 406.50 | 410.83 | 411.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 406.00 | 409.22 | 410.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 410.00 | 405.75 | 407.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 410.00 | 405.75 | 407.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 410.00 | 405.75 | 407.14 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 11:15:00 | 408.65 | 404.95 | 404.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 14:15:00 | 414.95 | 409.05 | 407.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 12:15:00 | 410.25 | 411.19 | 409.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 421.40 | 413.08 | 410.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 421.40 | 413.08 | 410.43 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 415.60 | 420.34 | 420.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 14:15:00 | 412.45 | 418.76 | 420.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 399.60 | 397.86 | 402.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 399.60 | 397.86 | 402.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 399.60 | 397.86 | 402.07 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 14:15:00 | 403.10 | 401.37 | 401.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 09:15:00 | 407.70 | 402.90 | 401.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 413.20 | 413.27 | 409.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 13:15:00 | 408.60 | 412.34 | 409.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 408.60 | 412.34 | 409.36 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 405.40 | 407.94 | 408.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 396.60 | 404.87 | 406.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 09:15:00 | 395.95 | 395.43 | 398.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 395.95 | 395.43 | 398.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 395.95 | 395.43 | 398.64 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 406.00 | 400.56 | 400.44 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 398.30 | 400.82 | 401.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 394.20 | 398.83 | 400.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 394.90 | 394.26 | 396.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 391.40 | 393.65 | 395.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 391.40 | 393.65 | 395.99 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 381.50 | 376.94 | 376.93 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 13:15:00 | 370.45 | 375.79 | 376.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 353.75 | 366.31 | 370.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 395.90 | 370.42 | 371.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 395.90 | 370.42 | 371.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 395.90 | 370.42 | 371.58 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 399.15 | 376.16 | 374.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 11:15:00 | 411.90 | 383.31 | 377.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 15:15:00 | 462.00 | 462.18 | 444.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 442.30 | 458.12 | 451.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 442.30 | 458.12 | 451.71 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 441.10 | 447.28 | 447.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 430.20 | 443.86 | 446.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 445.80 | 437.21 | 440.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 445.80 | 437.21 | 440.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 445.80 | 437.21 | 440.53 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 460.20 | 443.79 | 441.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 467.95 | 455.26 | 448.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 10:15:00 | 458.25 | 458.87 | 453.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 15:15:00 | 460.00 | 460.53 | 456.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 460.00 | 460.53 | 456.51 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 15:15:00 | 449.00 | 454.36 | 455.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 09:15:00 | 443.10 | 452.11 | 453.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 12:15:00 | 455.85 | 452.00 | 453.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 12:15:00 | 455.85 | 452.00 | 453.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 455.85 | 452.00 | 453.34 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 458.05 | 453.73 | 453.39 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 446.80 | 453.79 | 453.92 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 455.00 | 452.76 | 452.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 14:15:00 | 459.60 | 455.01 | 453.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 11:15:00 | 461.00 | 464.07 | 459.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 11:15:00 | 461.00 | 464.07 | 459.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 461.00 | 464.07 | 459.20 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 479.80 | 489.90 | 490.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 471.60 | 483.50 | 487.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 478.65 | 477.03 | 481.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 478.65 | 477.03 | 481.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 478.65 | 477.03 | 481.16 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 497.20 | 485.18 | 483.61 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 471.95 | 481.34 | 482.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 465.90 | 478.25 | 481.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 15:15:00 | 462.25 | 461.16 | 466.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 466.90 | 462.31 | 466.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 466.90 | 462.31 | 466.96 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 473.50 | 467.74 | 467.38 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 10:15:00 | 460.85 | 466.12 | 466.73 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 09:15:00 | 493.00 | 470.28 | 468.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 10:15:00 | 499.55 | 476.13 | 471.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 10:15:00 | 485.60 | 485.66 | 479.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 482.35 | 484.44 | 480.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 482.35 | 484.44 | 480.10 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 14:15:00 | 550.25 | 553.10 | 553.18 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 09:15:00 | 562.00 | 554.55 | 553.81 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 550.00 | 558.81 | 558.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 15:15:00 | 542.00 | 548.65 | 551.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 559.85 | 550.89 | 552.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 559.85 | 550.89 | 552.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 559.85 | 550.89 | 552.44 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 559.70 | 553.96 | 553.64 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 547.15 | 553.17 | 553.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 15:15:00 | 545.35 | 551.61 | 553.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 545.00 | 542.90 | 546.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 545.00 | 542.90 | 546.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 545.00 | 542.90 | 546.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 521.30 | 532.32 | 538.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 539.95 | 533.76 | 533.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 539.95 | 533.76 | 533.15 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 15:15:00 | 532.05 | 535.23 | 535.37 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 539.60 | 535.51 | 535.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 10:15:00 | 541.50 | 536.92 | 536.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 13:15:00 | 540.50 | 540.58 | 539.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 14:00:00 | 540.50 | 540.58 | 539.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 537.50 | 539.96 | 538.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 537.50 | 539.96 | 538.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 538.45 | 539.66 | 538.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 540.85 | 539.66 | 538.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 550.30 | 541.79 | 539.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:00:00 | 562.00 | 547.10 | 542.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 10:15:00 | 543.80 | 562.49 | 563.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 543.80 | 562.49 | 563.50 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 10:15:00 | 574.75 | 562.48 | 561.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 13:15:00 | 577.60 | 569.14 | 565.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 561.15 | 570.66 | 567.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 561.15 | 570.66 | 567.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 561.15 | 570.66 | 567.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:00:00 | 561.15 | 570.66 | 567.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 560.00 | 568.53 | 566.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:45:00 | 558.20 | 568.53 | 566.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 555.35 | 564.52 | 565.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 549.30 | 560.31 | 562.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 12:15:00 | 553.85 | 552.24 | 555.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 12:45:00 | 554.50 | 552.24 | 555.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 554.00 | 552.79 | 555.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 554.00 | 552.79 | 555.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 558.00 | 553.83 | 555.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 551.95 | 553.83 | 555.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 555.50 | 554.17 | 555.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 555.50 | 554.17 | 555.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 555.05 | 554.34 | 555.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:15:00 | 556.40 | 554.34 | 555.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 554.50 | 554.37 | 555.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:45:00 | 558.80 | 554.37 | 555.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 554.65 | 554.43 | 555.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:30:00 | 556.00 | 554.43 | 555.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 554.05 | 554.35 | 555.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:30:00 | 555.00 | 554.35 | 555.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 557.00 | 554.88 | 555.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 557.00 | 554.88 | 555.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 556.75 | 555.26 | 555.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 570.60 | 555.26 | 555.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 571.15 | 558.44 | 556.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 14:15:00 | 582.95 | 571.43 | 564.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 608.30 | 627.63 | 615.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 608.30 | 627.63 | 615.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 608.30 | 627.63 | 615.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 608.30 | 627.63 | 615.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 605.80 | 623.27 | 614.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 604.80 | 623.27 | 614.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 594.00 | 608.49 | 609.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 588.50 | 602.17 | 606.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 10:15:00 | 595.00 | 590.06 | 596.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 10:15:00 | 595.00 | 590.06 | 596.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 595.00 | 590.06 | 596.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:45:00 | 596.55 | 590.06 | 596.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 597.60 | 591.57 | 596.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 597.60 | 591.57 | 596.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 596.25 | 592.50 | 596.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:00:00 | 596.25 | 592.50 | 596.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 585.00 | 591.00 | 595.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:30:00 | 594.50 | 591.00 | 595.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 592.55 | 589.48 | 593.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 592.55 | 589.48 | 593.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 594.45 | 590.47 | 593.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 595.80 | 590.47 | 593.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 593.65 | 591.11 | 593.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:45:00 | 595.15 | 591.11 | 593.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 597.50 | 592.39 | 593.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 591.65 | 592.39 | 593.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:15:00 | 591.40 | 592.67 | 593.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 562.07 | 576.77 | 583.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:15:00 | 561.83 | 574.99 | 581.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 572.55 | 571.81 | 577.87 | SL hit (close>ema200) qty=0.50 sl=571.81 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 14:15:00 | 593.05 | 575.95 | 573.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 604.65 | 590.01 | 584.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 12:15:00 | 589.85 | 591.34 | 586.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 13:00:00 | 589.85 | 591.34 | 586.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 585.95 | 590.04 | 586.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:00:00 | 585.95 | 590.04 | 586.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 582.00 | 588.43 | 586.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 563.65 | 588.43 | 586.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 552.85 | 581.09 | 583.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 537.70 | 572.41 | 579.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 571.00 | 559.80 | 569.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 571.00 | 559.80 | 569.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 571.00 | 559.80 | 569.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 571.00 | 559.80 | 569.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 587.05 | 565.25 | 570.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 587.05 | 565.25 | 570.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 580.00 | 568.20 | 571.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 13:00:00 | 573.15 | 569.19 | 571.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 10:30:00 | 574.10 | 569.85 | 570.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 562.95 | 557.17 | 557.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 562.95 | 557.17 | 557.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 14:15:00 | 566.40 | 560.60 | 558.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 11:15:00 | 563.00 | 563.07 | 560.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 13:30:00 | 566.75 | 564.27 | 561.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 14:30:00 | 566.15 | 564.43 | 562.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 15:15:00 | 567.50 | 564.43 | 562.02 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 567.45 | 565.52 | 562.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 576.30 | 566.42 | 563.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:15:00 | 571.95 | 571.42 | 567.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:15:00 | 595.09 | 579.58 | 573.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:15:00 | 594.46 | 579.58 | 573.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:15:00 | 595.88 | 579.58 | 573.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 587.90 | 590.21 | 583.31 | SL hit (close<ema200) qty=0.50 sl=590.21 alert=retest1 |

### Cycle 89 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 588.10 | 591.01 | 591.02 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 598.15 | 591.80 | 591.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 607.85 | 598.22 | 595.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 613.70 | 613.76 | 606.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 10:45:00 | 613.50 | 613.76 | 606.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 605.10 | 611.74 | 608.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 605.10 | 611.74 | 608.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 607.00 | 610.79 | 608.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 617.10 | 610.79 | 608.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 14:30:00 | 611.30 | 613.64 | 613.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 606.85 | 611.78 | 612.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 606.85 | 611.78 | 612.41 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 624.00 | 612.47 | 611.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 11:15:00 | 631.65 | 616.31 | 613.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 636.80 | 639.98 | 632.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 10:15:00 | 632.45 | 638.48 | 632.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 632.45 | 638.48 | 632.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 635.00 | 638.48 | 632.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 634.95 | 637.77 | 632.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 13:00:00 | 636.90 | 637.60 | 633.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 637.80 | 637.64 | 633.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 624.50 | 633.40 | 632.72 | SL hit (close<static) qty=1.00 sl=632.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 622.30 | 631.18 | 631.77 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 640.40 | 632.13 | 631.87 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 630.10 | 631.84 | 631.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 620.90 | 629.65 | 630.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 625.95 | 624.77 | 627.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 625.95 | 624.77 | 627.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 625.95 | 624.77 | 627.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:45:00 | 626.15 | 624.77 | 627.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 629.00 | 625.62 | 627.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 631.65 | 626.67 | 628.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 628.00 | 626.94 | 628.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 630.10 | 626.94 | 628.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 629.55 | 627.46 | 628.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:45:00 | 630.30 | 627.46 | 628.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 630.90 | 628.15 | 628.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 628.90 | 628.15 | 628.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 630.45 | 628.61 | 628.60 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 627.85 | 628.46 | 628.53 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 630.00 | 628.77 | 628.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 634.35 | 629.88 | 629.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 15:15:00 | 629.95 | 630.75 | 630.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 15:15:00 | 629.95 | 630.75 | 630.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 629.95 | 630.75 | 630.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 631.05 | 630.75 | 630.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 632.60 | 631.12 | 630.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 14:30:00 | 637.75 | 631.99 | 631.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:30:00 | 636.65 | 635.29 | 633.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 15:00:00 | 641.95 | 635.29 | 633.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 14:15:00 | 622.45 | 631.72 | 632.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 622.45 | 631.72 | 632.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 604.35 | 624.23 | 628.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 602.95 | 602.90 | 609.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 15:00:00 | 602.95 | 602.90 | 609.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 607.50 | 603.70 | 608.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 607.50 | 603.70 | 608.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 604.00 | 603.76 | 608.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 586.80 | 603.81 | 608.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 607.70 | 603.95 | 603.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 09:15:00 | 607.70 | 603.95 | 603.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 613.85 | 608.92 | 606.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 610.05 | 610.99 | 608.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 610.05 | 610.99 | 608.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 607.45 | 610.28 | 608.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 613.40 | 610.28 | 608.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 10:15:00 | 612.40 | 610.20 | 608.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 605.60 | 608.14 | 608.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 605.60 | 608.14 | 608.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 593.85 | 602.97 | 605.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 13:15:00 | 603.00 | 596.62 | 600.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 13:15:00 | 603.00 | 596.62 | 600.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 603.00 | 596.62 | 600.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:00:00 | 603.00 | 596.62 | 600.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 604.00 | 598.10 | 600.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 605.00 | 598.10 | 600.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 606.00 | 600.47 | 601.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 606.00 | 600.47 | 601.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 607.50 | 601.87 | 601.76 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 593.00 | 601.30 | 601.92 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 616.30 | 601.72 | 601.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 10:15:00 | 644.90 | 625.35 | 615.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 680.00 | 684.00 | 665.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:45:00 | 679.40 | 684.00 | 665.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 780.00 | 789.51 | 765.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 11:30:00 | 792.85 | 789.84 | 769.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 13:15:00 | 779.00 | 782.93 | 783.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 779.00 | 782.93 | 783.29 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 790.05 | 784.36 | 783.91 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 769.40 | 781.44 | 782.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 12:15:00 | 766.00 | 776.88 | 780.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 729.20 | 726.54 | 736.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 729.20 | 726.54 | 736.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 729.20 | 726.54 | 736.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 728.10 | 726.54 | 736.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 738.30 | 729.38 | 735.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:45:00 | 737.30 | 729.38 | 735.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 755.35 | 734.57 | 737.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:30:00 | 753.30 | 734.57 | 737.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 750.95 | 740.58 | 739.91 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 732.90 | 739.47 | 739.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 726.50 | 733.85 | 736.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 14:15:00 | 721.60 | 718.57 | 724.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 15:00:00 | 721.60 | 718.57 | 724.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 722.00 | 719.26 | 723.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 725.85 | 719.26 | 723.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 729.00 | 721.20 | 724.35 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 740.50 | 728.66 | 727.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 759.00 | 736.46 | 731.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 12:15:00 | 764.10 | 766.68 | 758.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 13:00:00 | 764.10 | 766.68 | 758.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 760.75 | 765.50 | 759.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 760.75 | 765.50 | 759.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 757.00 | 763.80 | 758.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 757.00 | 763.80 | 758.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 758.00 | 762.64 | 758.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 748.50 | 762.64 | 758.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 745.55 | 759.22 | 757.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 745.55 | 759.22 | 757.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 746.15 | 756.61 | 756.59 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 11:15:00 | 748.40 | 754.96 | 755.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 744.45 | 750.14 | 753.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 745.55 | 744.21 | 748.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 745.55 | 744.21 | 748.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 745.55 | 744.21 | 748.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 745.55 | 744.21 | 748.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 746.25 | 744.62 | 747.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 736.20 | 744.62 | 747.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:15:00 | 743.85 | 743.27 | 745.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 757.60 | 746.23 | 746.27 | SL hit (close>static) qty=1.00 sl=749.70 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 753.40 | 747.66 | 746.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 764.85 | 753.79 | 750.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 15:15:00 | 758.50 | 758.92 | 755.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:15:00 | 776.60 | 758.92 | 755.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 760.00 | 764.57 | 761.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-13 15:15:00 | 760.00 | 764.57 | 761.09 | SL hit (close<ema400) qty=1.00 sl=761.09 alert=retest1 |

### Cycle 113 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 748.80 | 767.63 | 768.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 737.70 | 758.60 | 764.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 740.30 | 736.15 | 743.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 740.30 | 736.15 | 743.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 743.00 | 737.52 | 743.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 738.45 | 737.52 | 743.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 745.05 | 739.03 | 743.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:15:00 | 748.70 | 739.03 | 743.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 745.70 | 740.36 | 743.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 749.60 | 740.36 | 743.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 745.15 | 737.33 | 740.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 745.15 | 737.33 | 740.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 742.20 | 738.31 | 740.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 13:00:00 | 737.55 | 739.13 | 740.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 10:45:00 | 736.60 | 738.64 | 739.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 12:15:00 | 700.67 | 709.75 | 717.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 12:15:00 | 699.77 | 709.75 | 717.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 700.20 | 692.15 | 700.21 | SL hit (close>ema200) qty=0.50 sl=692.15 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 711.85 | 704.53 | 704.13 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 691.40 | 703.14 | 703.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 11:15:00 | 682.65 | 688.77 | 693.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 673.50 | 663.44 | 671.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 673.50 | 663.44 | 671.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 673.50 | 663.44 | 671.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 673.50 | 663.44 | 671.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 691.40 | 669.03 | 673.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 691.40 | 669.03 | 673.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 701.95 | 675.62 | 675.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 701.95 | 675.62 | 675.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 715.00 | 683.49 | 679.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 731.00 | 693.00 | 684.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 741.30 | 744.04 | 730.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 748.75 | 744.04 | 730.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 788.15 | 795.29 | 791.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 788.15 | 795.29 | 791.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 779.55 | 792.14 | 790.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 785.75 | 792.14 | 790.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 781.75 | 788.30 | 788.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 776.00 | 785.84 | 787.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 791.10 | 782.75 | 785.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 791.10 | 782.75 | 785.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 791.10 | 782.75 | 785.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 791.10 | 782.75 | 785.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 790.95 | 784.39 | 785.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:45:00 | 795.75 | 784.39 | 785.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 790.60 | 786.77 | 786.55 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 15:15:00 | 780.45 | 785.51 | 785.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 772.35 | 782.88 | 784.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 750.70 | 747.24 | 757.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 750.70 | 747.24 | 757.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 674.45 | 664.60 | 674.71 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 689.80 | 675.85 | 675.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 695.80 | 679.84 | 677.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 673.80 | 690.88 | 685.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 673.80 | 690.88 | 685.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 673.80 | 690.88 | 685.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 673.80 | 690.88 | 685.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 676.05 | 687.91 | 685.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 683.05 | 687.19 | 685.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 682.90 | 686.09 | 685.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 668.35 | 682.03 | 683.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 668.35 | 682.03 | 683.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 662.80 | 678.19 | 681.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 676.10 | 666.77 | 671.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 676.10 | 666.77 | 671.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 676.10 | 666.77 | 671.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 676.10 | 666.77 | 671.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 680.10 | 669.44 | 671.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:45:00 | 679.70 | 669.44 | 671.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 688.60 | 642.06 | 644.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 696.50 | 642.06 | 644.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 662.65 | 648.39 | 647.11 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 644.90 | 647.46 | 647.71 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 652.40 | 648.36 | 648.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 14:15:00 | 654.95 | 650.36 | 649.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 11:15:00 | 655.05 | 655.47 | 652.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:00:00 | 655.05 | 655.47 | 652.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 651.30 | 654.64 | 652.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:00:00 | 651.30 | 654.64 | 652.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 648.75 | 653.46 | 651.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 648.75 | 653.46 | 651.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 645.30 | 651.83 | 651.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 645.30 | 651.83 | 651.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 646.50 | 650.76 | 650.88 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 675.05 | 655.62 | 653.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 680.70 | 660.64 | 655.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 11:15:00 | 690.90 | 692.61 | 684.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:00:00 | 690.90 | 692.61 | 684.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 686.45 | 690.65 | 684.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:30:00 | 685.05 | 690.65 | 684.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 683.80 | 689.28 | 684.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 683.80 | 689.28 | 684.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 684.80 | 688.38 | 684.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 712.80 | 688.38 | 684.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-25 15:15:00 | 784.08 | 735.43 | 713.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 770.80 | 798.42 | 799.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 10:15:00 | 766.90 | 792.12 | 796.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 12:15:00 | 739.75 | 739.49 | 747.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:30:00 | 739.35 | 739.49 | 747.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 735.05 | 738.34 | 741.59 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 755.50 | 742.22 | 742.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 15:15:00 | 761.95 | 746.17 | 743.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 15:15:00 | 812.50 | 812.76 | 797.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:15:00 | 825.15 | 812.76 | 797.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 803.30 | 809.00 | 803.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 803.30 | 809.00 | 803.60 | SL hit (close<ema400) qty=1.00 sl=803.60 alert=retest1 |

### Cycle 129 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 796.55 | 814.37 | 816.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 11:15:00 | 791.20 | 809.74 | 814.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 734.45 | 733.03 | 743.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 13:00:00 | 734.45 | 733.03 | 743.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 743.90 | 736.48 | 741.64 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 754.30 | 745.14 | 744.49 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 734.90 | 744.48 | 744.88 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 746.10 | 743.99 | 743.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 746.85 | 744.56 | 744.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 765.45 | 771.66 | 764.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 765.45 | 771.66 | 764.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 765.45 | 771.66 | 764.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 766.75 | 771.66 | 764.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 753.85 | 768.10 | 763.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 753.85 | 768.10 | 763.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 752.10 | 764.90 | 762.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 747.65 | 764.90 | 762.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 746.50 | 758.59 | 760.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 736.90 | 749.46 | 753.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 688.75 | 687.08 | 700.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 688.75 | 687.08 | 700.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 696.00 | 687.89 | 695.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 694.95 | 687.89 | 695.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 689.60 | 688.24 | 695.20 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 702.90 | 697.66 | 697.43 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 11:15:00 | 694.20 | 696.96 | 697.15 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 701.60 | 697.52 | 697.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 702.25 | 698.46 | 697.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 697.40 | 698.33 | 697.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 697.40 | 698.33 | 697.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 697.40 | 698.33 | 697.85 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 688.10 | 696.28 | 696.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 684.50 | 692.53 | 695.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 692.40 | 688.62 | 692.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 692.40 | 688.62 | 692.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 692.40 | 688.62 | 692.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 695.40 | 688.62 | 692.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 706.40 | 692.17 | 693.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 706.40 | 692.17 | 693.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 703.60 | 694.46 | 694.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 716.80 | 698.93 | 696.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 696.75 | 706.24 | 701.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 696.75 | 706.24 | 701.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 696.75 | 706.24 | 701.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 696.75 | 706.24 | 701.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 697.10 | 704.41 | 701.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 701.70 | 704.41 | 701.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 695.50 | 702.63 | 700.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 697.85 | 702.63 | 700.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 690.45 | 698.60 | 698.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 685.35 | 695.95 | 697.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 12:15:00 | 674.70 | 674.38 | 680.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:30:00 | 675.45 | 674.38 | 680.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 655.45 | 666.24 | 674.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 653.70 | 660.91 | 669.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 621.01 | 645.83 | 658.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 10:15:00 | 588.33 | 616.88 | 636.60 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 140 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 638.05 | 623.75 | 622.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 650.80 | 635.41 | 629.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 648.00 | 656.94 | 647.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 648.00 | 656.94 | 647.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 648.00 | 656.94 | 647.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 648.00 | 656.94 | 647.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 660.90 | 657.73 | 648.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 674.25 | 658.23 | 650.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 11:15:00 | 644.65 | 653.49 | 650.26 | SL hit (close<static) qty=1.00 sl=644.70 alert=retest2 |

### Cycle 141 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 635.75 | 647.98 | 648.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 593.45 | 633.33 | 641.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 588.30 | 575.80 | 584.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 588.30 | 575.80 | 584.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 588.30 | 575.80 | 584.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:15:00 | 570.85 | 578.05 | 582.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 555.45 | 569.75 | 575.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 573.30 | 567.04 | 566.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 573.30 | 567.04 | 566.57 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 562.35 | 569.20 | 569.37 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 14:15:00 | 586.30 | 571.62 | 570.20 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 558.95 | 568.31 | 568.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 10:15:00 | 549.25 | 564.50 | 567.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 13:15:00 | 553.55 | 547.68 | 553.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 13:15:00 | 553.55 | 547.68 | 553.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 553.55 | 547.68 | 553.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 551.40 | 547.68 | 553.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 566.85 | 551.51 | 554.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:45:00 | 570.15 | 551.51 | 554.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 565.90 | 554.39 | 555.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 562.05 | 554.39 | 555.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 567.00 | 556.91 | 556.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 567.00 | 556.91 | 556.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 577.20 | 565.12 | 561.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 577.05 | 577.90 | 570.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 14:45:00 | 577.30 | 577.90 | 570.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 593.60 | 581.09 | 573.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:30:00 | 599.95 | 584.38 | 575.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:45:00 | 601.90 | 587.88 | 577.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 10:15:00 | 566.45 | 582.30 | 580.08 | SL hit (close<static) qty=1.00 sl=570.55 alert=retest2 |

### Cycle 147 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 569.85 | 578.12 | 578.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 559.95 | 571.52 | 575.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 492.20 | 482.44 | 496.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 492.20 | 482.44 | 496.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 492.20 | 482.44 | 496.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 492.20 | 482.44 | 496.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 485.30 | 483.01 | 495.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 481.65 | 483.01 | 495.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:45:00 | 483.25 | 482.76 | 494.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 500.00 | 488.34 | 492.27 | SL hit (close>static) qty=1.00 sl=498.65 alert=retest2 |

### Cycle 148 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 502.45 | 494.55 | 494.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 15:15:00 | 504.30 | 496.50 | 495.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 577.55 | 580.84 | 557.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 577.55 | 580.84 | 557.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 561.00 | 573.93 | 562.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 539.55 | 573.93 | 562.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 530.80 | 565.31 | 559.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 530.80 | 565.31 | 559.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 536.15 | 559.48 | 557.78 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 531.75 | 553.93 | 555.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 528.60 | 548.86 | 552.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 525.40 | 522.73 | 532.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:00:00 | 525.40 | 522.73 | 532.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 537.85 | 517.56 | 520.28 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 544.90 | 523.03 | 522.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 556.90 | 529.80 | 525.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 558.40 | 561.29 | 552.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 558.40 | 561.29 | 552.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 566.35 | 565.41 | 560.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:30:00 | 565.90 | 565.41 | 560.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 561.40 | 571.86 | 568.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 561.40 | 571.86 | 568.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 558.20 | 569.13 | 567.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 560.40 | 569.13 | 567.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 554.20 | 566.14 | 566.44 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 13:15:00 | 567.05 | 564.86 | 564.72 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 561.00 | 564.09 | 564.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 552.90 | 561.66 | 563.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 556.05 | 555.90 | 559.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 556.05 | 555.90 | 559.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 556.05 | 555.90 | 559.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 556.05 | 555.90 | 559.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 558.00 | 556.32 | 559.21 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 574.95 | 562.77 | 561.60 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 549.00 | 561.29 | 561.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 541.70 | 557.37 | 559.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 542.45 | 539.61 | 545.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 15:00:00 | 542.45 | 539.61 | 545.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 489.35 | 482.67 | 489.05 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 508.70 | 494.46 | 492.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 513.55 | 505.45 | 499.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 500.85 | 509.29 | 505.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 500.85 | 509.29 | 505.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 500.85 | 509.29 | 505.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:45:00 | 516.00 | 509.93 | 507.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 516.45 | 509.93 | 507.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:00:00 | 515.50 | 511.04 | 508.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:00:00 | 514.15 | 511.66 | 508.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 508.60 | 511.53 | 509.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:45:00 | 507.80 | 511.53 | 509.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 510.20 | 511.26 | 509.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:30:00 | 509.15 | 511.26 | 509.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 509.05 | 510.82 | 509.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:30:00 | 511.30 | 510.82 | 509.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 510.45 | 510.75 | 509.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:15:00 | 510.70 | 510.75 | 509.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 15:15:00 | 510.80 | 510.39 | 509.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 502.45 | 517.76 | 517.40 | SL hit (close<static) qty=1.00 sl=508.75 alert=retest2 |

### Cycle 157 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 499.50 | 514.11 | 515.77 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 520.60 | 513.64 | 513.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 524.30 | 517.05 | 515.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 11:15:00 | 524.05 | 524.96 | 521.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 12:00:00 | 524.05 | 524.96 | 521.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 524.95 | 524.96 | 522.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 523.40 | 524.96 | 522.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 520.30 | 523.87 | 522.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 520.30 | 523.87 | 522.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 513.05 | 521.70 | 521.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 510.40 | 521.70 | 521.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 510.85 | 519.53 | 520.42 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 525.25 | 519.16 | 518.37 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 511.15 | 518.32 | 519.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 509.65 | 516.59 | 518.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 516.35 | 511.84 | 514.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 516.35 | 511.84 | 514.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 516.35 | 511.84 | 514.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 516.35 | 511.84 | 514.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 515.90 | 512.65 | 514.74 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 529.40 | 517.62 | 516.42 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 514.25 | 518.76 | 518.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 13:15:00 | 510.60 | 516.32 | 517.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 519.40 | 516.69 | 517.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 15:15:00 | 519.40 | 516.69 | 517.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 519.40 | 516.69 | 517.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 570.00 | 516.69 | 517.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 567.00 | 526.75 | 522.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 580.45 | 557.85 | 549.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 600.25 | 602.75 | 586.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 600.25 | 602.75 | 586.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 599.70 | 603.79 | 598.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 599.70 | 603.79 | 598.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 603.90 | 603.81 | 599.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 591.50 | 603.81 | 599.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 590.30 | 601.11 | 598.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 589.80 | 601.11 | 598.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 592.90 | 599.47 | 598.04 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 590.30 | 596.36 | 596.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 586.45 | 594.38 | 595.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 576.45 | 576.00 | 583.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 585.95 | 576.00 | 583.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 580.30 | 576.86 | 582.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 573.20 | 576.39 | 580.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 569.80 | 575.11 | 579.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 585.00 | 580.45 | 580.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 585.00 | 580.45 | 580.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 590.35 | 582.43 | 581.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 592.05 | 600.87 | 595.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 592.05 | 600.87 | 595.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 592.05 | 600.87 | 595.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 592.05 | 600.87 | 595.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 597.40 | 600.18 | 595.71 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 585.00 | 593.06 | 593.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 581.60 | 587.17 | 590.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 586.35 | 580.51 | 583.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 586.35 | 580.51 | 583.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 586.35 | 580.51 | 583.55 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 590.25 | 585.76 | 585.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 593.90 | 588.98 | 587.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 588.70 | 589.61 | 587.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 588.70 | 589.61 | 587.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 588.70 | 589.61 | 587.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 587.70 | 589.61 | 587.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 590.00 | 589.69 | 588.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 588.00 | 589.69 | 588.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 589.15 | 589.67 | 588.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:00:00 | 594.80 | 589.60 | 588.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 602.20 | 606.15 | 606.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 602.20 | 606.15 | 606.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 598.85 | 604.69 | 605.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 11:15:00 | 605.30 | 603.59 | 604.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 11:15:00 | 605.30 | 603.59 | 604.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 605.30 | 603.59 | 604.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:00:00 | 605.30 | 603.59 | 604.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 606.15 | 604.10 | 604.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:45:00 | 606.15 | 604.10 | 604.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 602.85 | 603.85 | 604.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:30:00 | 601.95 | 603.85 | 604.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 603.50 | 600.33 | 601.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 602.00 | 600.33 | 601.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 600.70 | 600.40 | 601.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 595.40 | 600.52 | 601.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 612.10 | 602.84 | 602.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 612.10 | 602.84 | 602.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 10:15:00 | 615.40 | 608.31 | 605.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 611.80 | 613.04 | 609.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 611.80 | 613.04 | 609.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 611.80 | 613.04 | 609.82 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 10:15:00 | 608.25 | 608.88 | 608.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 603.30 | 607.21 | 608.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 605.60 | 601.16 | 603.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 605.60 | 601.16 | 603.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 605.60 | 601.16 | 603.35 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 615.50 | 606.24 | 605.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 634.85 | 614.35 | 611.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 614.20 | 617.07 | 613.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 614.20 | 617.07 | 613.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 614.20 | 617.07 | 613.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 614.20 | 617.07 | 613.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 611.45 | 615.95 | 613.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 612.95 | 615.95 | 613.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 606.65 | 614.09 | 612.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 606.65 | 614.09 | 612.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 610.10 | 611.83 | 611.88 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 613.45 | 611.97 | 611.90 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 607.45 | 611.33 | 611.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 606.10 | 609.08 | 610.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 15:15:00 | 609.40 | 609.15 | 610.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 610.10 | 609.34 | 610.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 610.10 | 609.34 | 610.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 610.10 | 609.34 | 610.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 618.40 | 611.15 | 610.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 12:15:00 | 622.45 | 614.77 | 612.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 644.45 | 644.89 | 637.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 12:00:00 | 644.45 | 644.89 | 637.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 637.55 | 642.44 | 637.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:45:00 | 640.30 | 642.44 | 637.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 636.00 | 641.15 | 637.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 650.55 | 641.15 | 637.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 638.90 | 640.02 | 638.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:00:00 | 639.10 | 639.84 | 638.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:30:00 | 640.80 | 639.41 | 638.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 639.10 | 639.35 | 638.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 646.85 | 639.35 | 638.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 646.20 | 645.07 | 644.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 645.80 | 644.98 | 644.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 652.95 | 653.82 | 653.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 652.95 | 653.82 | 653.82 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 657.05 | 653.85 | 653.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 10:15:00 | 665.80 | 656.24 | 654.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 14:15:00 | 665.35 | 665.41 | 662.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 15:00:00 | 665.35 | 665.41 | 662.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 661.35 | 664.90 | 663.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 661.35 | 664.90 | 663.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 661.00 | 664.12 | 662.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 661.00 | 664.12 | 662.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 660.80 | 663.45 | 662.66 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 661.00 | 662.14 | 662.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 646.55 | 659.02 | 660.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 617.25 | 616.74 | 625.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 617.70 | 616.74 | 625.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 610.15 | 607.90 | 612.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:30:00 | 606.25 | 608.13 | 611.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 604.20 | 609.16 | 611.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-05 09:15:00 | 545.62 | 583.57 | 591.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 526.75 | 518.92 | 517.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 528.85 | 522.32 | 519.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 524.00 | 524.75 | 522.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 524.00 | 524.75 | 522.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 524.00 | 524.75 | 522.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 521.75 | 524.75 | 522.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 527.80 | 529.62 | 526.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 527.00 | 529.62 | 526.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 538.50 | 531.39 | 527.59 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 529.60 | 531.89 | 531.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 525.50 | 529.59 | 530.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 519.70 | 519.46 | 522.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 12:30:00 | 518.70 | 519.46 | 522.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 522.90 | 520.33 | 521.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 522.60 | 520.33 | 521.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 525.45 | 521.35 | 522.20 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 527.40 | 522.97 | 522.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 530.20 | 524.42 | 523.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 526.85 | 527.14 | 525.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 526.85 | 527.14 | 525.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 526.85 | 527.14 | 525.39 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 515.05 | 524.87 | 525.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 513.00 | 518.09 | 521.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 515.35 | 513.27 | 516.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 515.35 | 513.27 | 516.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 515.35 | 513.27 | 516.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 508.45 | 511.88 | 514.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 509.45 | 511.10 | 513.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 509.05 | 510.51 | 513.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 519.55 | 514.19 | 513.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 519.55 | 514.19 | 513.95 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 511.30 | 514.03 | 514.28 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 526.65 | 516.65 | 515.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 537.30 | 521.80 | 517.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 539.15 | 540.36 | 533.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 10:00:00 | 539.15 | 540.36 | 533.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 537.30 | 539.85 | 537.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 537.30 | 539.85 | 537.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 541.70 | 540.22 | 537.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 543.00 | 540.18 | 538.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 535.60 | 538.94 | 537.88 | SL hit (close<static) qty=1.00 sl=536.80 alert=retest2 |

### Cycle 187 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 532.40 | 536.71 | 537.00 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 538.50 | 536.86 | 536.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 544.15 | 538.32 | 537.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 13:15:00 | 539.30 | 539.70 | 538.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 539.30 | 539.70 | 538.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 539.80 | 539.72 | 538.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 540.15 | 539.72 | 538.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 539.25 | 539.63 | 538.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 539.25 | 539.32 | 538.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 533.25 | 538.11 | 538.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 533.25 | 538.11 | 538.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 532.75 | 537.04 | 537.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 529.50 | 534.50 | 536.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 515.55 | 515.08 | 519.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 15:00:00 | 515.55 | 515.08 | 519.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 522.00 | 516.93 | 519.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 522.00 | 516.93 | 519.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 520.00 | 517.55 | 519.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 518.40 | 517.75 | 519.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 517.50 | 517.89 | 519.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:00:00 | 519.10 | 518.13 | 519.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 518.85 | 518.27 | 519.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 520.00 | 518.62 | 519.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 518.20 | 518.62 | 519.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 518.05 | 518.79 | 519.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 517.35 | 518.70 | 519.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 518.10 | 518.28 | 518.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 520.00 | 518.62 | 519.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 520.00 | 518.62 | 519.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 518.00 | 518.50 | 518.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 522.90 | 518.50 | 518.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 522.50 | 519.30 | 519.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 522.50 | 519.30 | 519.29 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 519.75 | 521.49 | 521.59 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 523.45 | 521.77 | 521.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 09:15:00 | 531.90 | 524.16 | 522.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 526.25 | 526.84 | 524.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 526.25 | 526.84 | 524.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 526.00 | 526.67 | 524.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 526.00 | 526.67 | 524.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 526.00 | 526.66 | 525.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 531.00 | 526.00 | 525.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:45:00 | 532.20 | 527.73 | 526.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 529.00 | 529.37 | 527.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 520.50 | 527.01 | 527.00 | SL hit (close<static) qty=1.00 sl=524.40 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 524.45 | 526.50 | 526.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 12:15:00 | 516.80 | 522.24 | 523.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 524.00 | 522.57 | 523.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 524.00 | 522.57 | 523.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 524.00 | 522.57 | 523.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 524.45 | 522.57 | 523.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 523.50 | 522.76 | 523.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 524.90 | 522.76 | 523.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 519.80 | 522.16 | 523.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 522.85 | 522.16 | 523.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 522.50 | 522.23 | 523.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 522.90 | 522.23 | 523.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 523.20 | 522.47 | 523.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 523.25 | 522.47 | 523.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 523.00 | 522.57 | 523.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 523.00 | 522.57 | 523.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 521.70 | 522.40 | 523.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 523.00 | 522.40 | 523.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 522.10 | 522.34 | 522.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 521.40 | 522.34 | 522.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 519.40 | 521.75 | 522.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 516.70 | 521.23 | 521.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 517.80 | 519.65 | 520.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:30:00 | 517.80 | 518.86 | 520.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 530.00 | 521.66 | 521.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 530.00 | 521.66 | 521.17 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 10:15:00 | 522.30 | 524.97 | 525.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 521.00 | 522.66 | 523.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 523.10 | 521.54 | 522.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 523.10 | 521.54 | 522.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 523.10 | 521.54 | 522.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 523.10 | 521.54 | 522.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 522.05 | 521.64 | 522.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 523.80 | 521.64 | 522.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 524.20 | 522.15 | 522.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 524.20 | 522.15 | 522.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 524.70 | 522.66 | 522.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 525.40 | 522.66 | 522.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 527.00 | 523.53 | 523.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 535.00 | 526.25 | 524.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 528.00 | 531.53 | 528.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 528.00 | 531.53 | 528.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 528.00 | 531.53 | 528.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 527.00 | 531.53 | 528.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 528.55 | 530.93 | 528.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 528.55 | 530.93 | 528.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 529.25 | 530.60 | 528.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 535.20 | 530.60 | 528.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 527.65 | 534.15 | 534.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 527.65 | 534.15 | 534.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 525.70 | 532.46 | 533.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 518.90 | 518.42 | 523.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 518.90 | 518.42 | 523.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 518.90 | 518.42 | 523.18 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 553.50 | 529.80 | 526.66 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 536.50 | 538.96 | 539.13 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 10:15:00 | 547.90 | 540.96 | 540.02 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 540.20 | 541.15 | 541.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 538.05 | 539.91 | 540.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 540.45 | 540.01 | 540.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 12:15:00 | 540.45 | 540.01 | 540.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 540.45 | 540.01 | 540.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:45:00 | 541.20 | 540.01 | 540.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 540.30 | 540.07 | 540.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 540.30 | 540.07 | 540.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 540.40 | 540.14 | 540.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 537.10 | 539.54 | 540.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 543.50 | 539.66 | 539.86 | SL hit (close>static) qty=1.00 sl=542.65 alert=retest2 |

### Cycle 202 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 543.00 | 539.89 | 539.60 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 534.00 | 540.03 | 540.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 526.00 | 534.28 | 536.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 527.60 | 526.61 | 529.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 13:15:00 | 527.60 | 526.61 | 529.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 527.60 | 526.61 | 529.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 527.60 | 526.61 | 529.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 526.70 | 526.63 | 529.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 528.65 | 526.63 | 529.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 536.80 | 527.86 | 529.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 535.00 | 527.86 | 529.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 537.85 | 529.86 | 530.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 537.85 | 529.86 | 530.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 540.85 | 532.06 | 531.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 12:15:00 | 542.20 | 534.08 | 532.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 538.50 | 539.13 | 535.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 538.50 | 539.13 | 535.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 537.70 | 539.09 | 536.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 536.85 | 539.09 | 536.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 536.80 | 538.59 | 537.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:00:00 | 538.75 | 538.62 | 537.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:30:00 | 538.40 | 538.80 | 537.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 533.85 | 537.91 | 537.91 | SL hit (close<static) qty=1.00 sl=534.15 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 535.25 | 537.38 | 537.67 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 541.80 | 538.23 | 537.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 11:15:00 | 546.45 | 539.87 | 538.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 540.55 | 544.68 | 542.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 540.55 | 544.68 | 542.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 540.55 | 544.68 | 542.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 540.15 | 544.68 | 542.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 538.25 | 543.39 | 541.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 538.45 | 543.39 | 541.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 536.45 | 540.17 | 540.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 533.05 | 538.75 | 539.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 533.35 | 527.44 | 531.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 533.35 | 527.44 | 531.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 533.35 | 527.44 | 531.17 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 534.80 | 532.01 | 531.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 539.00 | 533.41 | 532.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 528.00 | 532.33 | 532.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 528.00 | 532.33 | 532.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 528.00 | 532.33 | 532.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 528.00 | 532.33 | 532.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 530.15 | 531.89 | 531.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 526.70 | 529.50 | 530.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 527.25 | 527.05 | 528.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 527.25 | 527.05 | 528.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 527.25 | 527.05 | 528.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 527.00 | 527.05 | 528.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 524.50 | 523.66 | 525.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 524.65 | 523.66 | 525.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 524.50 | 523.83 | 525.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:15:00 | 524.45 | 523.83 | 525.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 529.00 | 524.86 | 525.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 529.00 | 524.86 | 525.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 524.10 | 524.71 | 525.48 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 527.00 | 525.99 | 525.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 531.55 | 527.10 | 526.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 542.50 | 544.23 | 541.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 15:15:00 | 542.50 | 544.23 | 541.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 542.50 | 544.23 | 541.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 539.80 | 544.23 | 541.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 543.25 | 544.03 | 542.06 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 535.15 | 541.86 | 541.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 529.75 | 539.44 | 540.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 538.00 | 535.69 | 537.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 538.00 | 535.69 | 537.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 538.00 | 535.69 | 537.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 538.00 | 535.69 | 537.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 539.75 | 536.50 | 538.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:00:00 | 537.00 | 537.06 | 538.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 537.05 | 537.84 | 538.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 540.10 | 537.71 | 537.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 540.10 | 537.71 | 537.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 541.25 | 539.37 | 538.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 541.00 | 541.03 | 539.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 541.00 | 541.03 | 539.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 538.75 | 540.57 | 539.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 538.75 | 540.57 | 539.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 540.10 | 540.48 | 539.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 533.25 | 540.48 | 539.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 530.45 | 538.47 | 538.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 528.00 | 536.38 | 537.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 533.20 | 533.08 | 535.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:45:00 | 533.15 | 533.08 | 535.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 531.10 | 532.70 | 535.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 530.00 | 531.84 | 534.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 528.40 | 531.84 | 534.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 530.00 | 530.57 | 532.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 503.50 | 517.75 | 523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 501.98 | 517.75 | 523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 503.50 | 517.75 | 523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 515.00 | 513.82 | 518.67 | SL hit (close>ema200) qty=0.50 sl=513.82 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 474.70 | 470.87 | 470.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 480.10 | 472.72 | 471.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 471.80 | 474.41 | 472.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 471.80 | 474.41 | 472.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 471.80 | 474.41 | 472.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:00:00 | 471.80 | 474.41 | 472.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 480.25 | 475.58 | 473.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:15:00 | 481.70 | 475.58 | 473.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 483.55 | 481.85 | 477.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 529.87 | 508.42 | 499.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 496.70 | 512.27 | 514.05 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 512.05 | 504.61 | 504.11 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 09:15:00 | 500.00 | 504.23 | 504.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 15:15:00 | 498.05 | 501.01 | 502.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 15:15:00 | 491.90 | 491.37 | 495.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 487.55 | 490.12 | 494.46 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 13:00:00 | 487.80 | 489.07 | 493.20 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:15:00 | 482.50 | 489.26 | 492.28 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 493.60 | 487.02 | 489.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 493.60 | 487.02 | 489.90 | SL hit (close>ema400) qty=1.00 sl=489.90 alert=retest1 |

### Cycle 218 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 491.20 | 490.38 | 490.31 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 487.60 | 490.35 | 490.36 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 491.05 | 490.49 | 490.42 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 487.00 | 489.91 | 490.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 15:15:00 | 485.00 | 487.98 | 489.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 490.05 | 487.61 | 488.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 490.05 | 487.61 | 488.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 490.05 | 487.61 | 488.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 489.55 | 487.61 | 488.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 488.60 | 487.80 | 488.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 489.65 | 487.80 | 488.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 490.05 | 488.25 | 488.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 490.05 | 488.25 | 488.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 490.05 | 488.61 | 488.45 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 486.00 | 488.09 | 488.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 484.80 | 487.43 | 487.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 488.50 | 487.65 | 487.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 488.50 | 487.65 | 487.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 488.50 | 487.65 | 487.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 488.50 | 487.65 | 487.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 486.25 | 487.37 | 487.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 485.50 | 487.70 | 487.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 491.25 | 488.41 | 488.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 491.25 | 488.41 | 488.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 493.95 | 489.52 | 488.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 484.00 | 488.65 | 488.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 484.00 | 488.65 | 488.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 484.00 | 488.65 | 488.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 480.75 | 488.65 | 488.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 484.15 | 487.75 | 488.11 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 491.90 | 488.28 | 487.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 493.95 | 490.42 | 489.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 489.85 | 490.88 | 489.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 489.85 | 490.88 | 489.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 489.85 | 490.88 | 489.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 489.85 | 490.88 | 489.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 492.20 | 491.14 | 489.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:00:00 | 493.75 | 491.66 | 490.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 488.85 | 492.31 | 491.44 | SL hit (close<static) qty=1.00 sl=489.80 alert=retest2 |

### Cycle 227 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 486.45 | 490.25 | 490.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 477.55 | 486.87 | 488.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 463.35 | 461.85 | 468.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 463.35 | 461.85 | 468.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 468.75 | 463.23 | 468.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 469.80 | 463.23 | 468.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 469.50 | 464.49 | 468.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 470.10 | 464.49 | 468.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 470.50 | 465.69 | 468.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 470.50 | 465.69 | 468.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 469.00 | 466.35 | 468.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 466.50 | 466.35 | 468.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 471.20 | 469.77 | 469.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 471.20 | 469.77 | 469.62 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 462.60 | 469.18 | 469.46 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 472.05 | 468.12 | 467.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 475.00 | 470.40 | 469.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 467.00 | 470.62 | 469.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 467.00 | 470.62 | 469.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 467.00 | 470.62 | 469.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 468.65 | 470.23 | 469.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 469.00 | 469.98 | 469.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 469.00 | 469.98 | 469.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 462.20 | 468.44 | 468.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 462.20 | 468.44 | 468.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 461.55 | 465.65 | 467.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 457.05 | 456.62 | 460.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 457.05 | 456.62 | 460.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 464.70 | 458.23 | 461.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 464.70 | 458.23 | 461.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 461.00 | 458.79 | 461.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 465.40 | 460.03 | 461.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 466.00 | 461.22 | 461.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 466.00 | 461.22 | 461.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 467.05 | 463.15 | 462.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 468.90 | 465.46 | 464.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 465.55 | 470.49 | 467.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 465.55 | 470.49 | 467.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 465.55 | 470.49 | 467.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 464.85 | 470.49 | 467.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 464.00 | 469.19 | 467.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:30:00 | 467.70 | 469.20 | 467.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 465.90 | 467.99 | 467.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 459.45 | 465.75 | 466.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 459.45 | 465.75 | 466.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 457.55 | 464.11 | 465.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 442.20 | 441.02 | 448.91 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:30:00 | 437.50 | 439.65 | 447.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 444.40 | 441.77 | 446.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 449.15 | 441.77 | 446.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 454.25 | 445.17 | 447.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 454.25 | 445.17 | 447.06 | SL hit (close>ema400) qty=1.00 sl=447.06 alert=retest1 |

### Cycle 234 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 455.75 | 448.74 | 448.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 464.60 | 453.92 | 451.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 450.85 | 453.31 | 451.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 450.85 | 453.31 | 451.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 450.85 | 453.31 | 451.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 449.85 | 453.31 | 451.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 448.30 | 452.31 | 450.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 448.30 | 452.31 | 450.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 445.80 | 451.01 | 450.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 447.50 | 451.01 | 450.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 454.65 | 451.73 | 450.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:30:00 | 453.25 | 451.73 | 450.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 452.90 | 451.97 | 451.03 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 434.60 | 448.18 | 449.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 432.55 | 445.05 | 447.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 452.25 | 443.42 | 445.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 452.25 | 443.42 | 445.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 452.25 | 443.42 | 445.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 453.35 | 443.42 | 445.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 451.25 | 444.98 | 445.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 451.25 | 444.98 | 445.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 452.05 | 446.40 | 446.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 454.40 | 448.00 | 447.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 444.75 | 449.33 | 448.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 444.75 | 449.33 | 448.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 444.75 | 449.33 | 448.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 455.85 | 448.97 | 448.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 455.00 | 455.93 | 452.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 457.40 | 455.93 | 452.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 455.10 | 459.94 | 460.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 237 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 455.10 | 459.94 | 460.05 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 465.85 | 460.21 | 460.10 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 458.85 | 460.62 | 460.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 457.70 | 460.04 | 460.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 13:15:00 | 461.75 | 460.38 | 460.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 13:15:00 | 461.75 | 460.38 | 460.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 461.75 | 460.38 | 460.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:00:00 | 461.75 | 460.38 | 460.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 455.55 | 459.41 | 460.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 455.55 | 459.41 | 460.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 459.55 | 459.44 | 460.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 470.45 | 459.44 | 460.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 240 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 486.80 | 464.91 | 462.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 489.75 | 479.21 | 474.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 13:15:00 | 561.75 | 564.41 | 544.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 13:45:00 | 561.00 | 564.41 | 544.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 568.00 | 575.09 | 569.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 563.30 | 575.09 | 569.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 559.20 | 571.91 | 568.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 559.20 | 571.91 | 568.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 241 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 556.75 | 564.52 | 565.28 | EMA200 below EMA400 |

### Cycle 242 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 570.00 | 566.48 | 566.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 582.90 | 569.76 | 567.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 575.60 | 578.83 | 575.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 575.60 | 578.83 | 575.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 580.65 | 579.19 | 576.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 582.00 | 579.19 | 576.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 573.50 | 579.75 | 578.18 | SL hit (close<static) qty=1.00 sl=574.40 alert=retest2 |

### Cycle 243 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 567.60 | 575.91 | 576.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 565.85 | 573.90 | 575.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 573.10 | 572.79 | 574.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 573.10 | 572.79 | 574.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 573.10 | 572.79 | 574.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 576.35 | 572.79 | 574.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 572.40 | 572.71 | 574.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 575.00 | 572.71 | 574.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 572.05 | 572.58 | 574.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:15:00 | 579.30 | 572.58 | 574.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 574.50 | 572.96 | 574.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 575.85 | 572.96 | 574.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 575.25 | 573.42 | 574.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 578.15 | 573.42 | 574.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 575.30 | 573.80 | 574.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 573.30 | 573.80 | 574.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 576.85 | 574.41 | 574.56 | SL hit (close>static) qty=1.00 sl=575.90 alert=retest2 |

### Cycle 244 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 568.00 | 562.41 | 562.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 576.75 | 566.89 | 564.23 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 13:15:00 | 521.30 | 2024-04-24 10:15:00 | 539.95 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-05-02 12:00:00 | 562.00 | 2024-05-07 10:15:00 | 543.80 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-05-24 13:15:00 | 591.65 | 2024-05-28 09:15:00 | 562.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 14:15:00 | 591.40 | 2024-05-28 10:15:00 | 561.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 13:15:00 | 591.65 | 2024-05-28 14:15:00 | 572.55 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2024-05-24 14:15:00 | 591.40 | 2024-05-28 14:15:00 | 572.55 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2024-06-05 13:00:00 | 573.15 | 2024-06-12 11:15:00 | 562.95 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2024-06-06 10:30:00 | 574.10 | 2024-06-12 11:15:00 | 562.95 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest1 | 2024-06-13 13:30:00 | 566.75 | 2024-06-19 09:15:00 | 595.09 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-13 14:30:00 | 566.15 | 2024-06-19 09:15:00 | 594.46 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-13 15:15:00 | 567.50 | 2024-06-19 09:15:00 | 595.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-13 13:30:00 | 566.75 | 2024-06-20 09:15:00 | 587.90 | STOP_HIT | 0.50 | 3.73% |
| BUY | retest1 | 2024-06-13 14:30:00 | 566.15 | 2024-06-20 09:15:00 | 587.90 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest1 | 2024-06-13 15:15:00 | 567.50 | 2024-06-20 09:15:00 | 587.90 | STOP_HIT | 0.50 | 3.59% |
| BUY | retest2 | 2024-06-14 11:15:00 | 576.30 | 2024-06-24 14:15:00 | 588.10 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2024-06-18 10:15:00 | 571.95 | 2024-06-24 14:15:00 | 588.10 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2024-06-28 09:15:00 | 617.10 | 2024-07-02 09:15:00 | 606.85 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-07-01 14:30:00 | 611.30 | 2024-07-02 09:15:00 | 606.85 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-05 13:00:00 | 636.90 | 2024-07-08 10:15:00 | 624.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-07-05 14:00:00 | 637.80 | 2024-07-08 10:15:00 | 624.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-07-15 14:30:00 | 637.75 | 2024-07-18 14:15:00 | 622.45 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-07-16 14:30:00 | 636.65 | 2024-07-18 14:15:00 | 622.45 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-07-16 15:00:00 | 641.95 | 2024-07-18 14:15:00 | 622.45 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-07-23 12:15:00 | 586.80 | 2024-07-25 09:15:00 | 607.70 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-07-29 09:15:00 | 613.40 | 2024-07-30 09:15:00 | 605.60 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-29 10:15:00 | 612.40 | 2024-07-30 09:15:00 | 605.60 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-08-16 11:30:00 | 792.85 | 2024-08-20 13:15:00 | 779.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-09-10 09:15:00 | 736.20 | 2024-09-11 09:15:00 | 757.60 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-09-10 15:15:00 | 743.85 | 2024-09-11 09:15:00 | 757.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest1 | 2024-09-13 09:15:00 | 776.60 | 2024-09-13 15:15:00 | 760.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-09-16 09:15:00 | 777.15 | 2024-09-17 11:15:00 | 758.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-09-23 13:00:00 | 737.55 | 2024-09-27 12:15:00 | 700.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 10:45:00 | 736.60 | 2024-09-27 12:15:00 | 699.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-23 13:00:00 | 737.55 | 2024-10-01 09:15:00 | 700.20 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2024-09-24 10:45:00 | 736.60 | 2024-10-01 09:15:00 | 700.20 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2024-11-04 11:45:00 | 683.05 | 2024-11-05 09:15:00 | 668.35 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-11-04 15:15:00 | 682.90 | 2024-11-05 09:15:00 | 668.35 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-11-25 09:15:00 | 712.80 | 2024-11-25 15:15:00 | 784.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-12-12 09:15:00 | 825.15 | 2024-12-13 09:15:00 | 803.30 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-12-16 13:15:00 | 818.50 | 2024-12-19 10:15:00 | 796.55 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-12-17 10:00:00 | 823.00 | 2024-12-19 10:15:00 | 796.55 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-01-24 13:30:00 | 653.70 | 2025-01-27 10:15:00 | 621.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 653.70 | 2025-01-28 10:15:00 | 588.33 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-03 09:15:00 | 674.25 | 2025-02-03 11:15:00 | 644.65 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2025-02-10 10:15:00 | 570.85 | 2025-02-12 14:15:00 | 573.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-02-11 09:15:00 | 555.45 | 2025-02-12 14:15:00 | 573.30 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-02-19 09:15:00 | 562.05 | 2025-02-19 09:15:00 | 567.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-21 10:30:00 | 599.95 | 2025-02-24 10:15:00 | 566.45 | STOP_HIT | 1.00 | -5.58% |
| BUY | retest2 | 2025-02-21 11:45:00 | 601.90 | 2025-02-24 10:15:00 | 566.45 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest2 | 2025-03-04 11:15:00 | 481.65 | 2025-03-05 10:15:00 | 500.00 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-03-04 11:45:00 | 483.25 | 2025-03-05 10:15:00 | 500.00 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-04-21 10:45:00 | 516.00 | 2025-04-25 09:15:00 | 502.45 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-04-21 11:15:00 | 516.45 | 2025-04-25 09:15:00 | 502.45 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-04-21 12:00:00 | 515.50 | 2025-04-25 10:15:00 | 499.50 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-04-21 13:00:00 | 514.15 | 2025-04-25 10:15:00 | 499.50 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-04-22 13:15:00 | 510.70 | 2025-04-25 10:15:00 | 499.50 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-04-22 15:15:00 | 510.80 | 2025-04-25 10:15:00 | 499.50 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-05-22 13:30:00 | 573.20 | 2025-05-26 09:15:00 | 585.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-05-23 09:45:00 | 569.80 | 2025-05-26 09:15:00 | 585.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-06-06 12:00:00 | 594.80 | 2025-06-12 15:15:00 | 602.20 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2025-06-17 09:15:00 | 595.40 | 2025-06-17 09:15:00 | 612.10 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-07-08 09:15:00 | 650.55 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-07-08 11:45:00 | 638.90 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-07-08 13:00:00 | 639.10 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2025-07-08 14:30:00 | 640.80 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2025-07-09 09:15:00 | 646.85 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-07-11 11:30:00 | 646.20 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-07-14 09:30:00 | 645.80 | 2025-07-17 14:15:00 | 652.95 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-07-30 12:30:00 | 606.25 | 2025-08-05 09:15:00 | 545.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 604.20 | 2025-08-05 09:15:00 | 543.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-08 13:45:00 | 508.45 | 2025-09-10 09:15:00 | 519.55 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-09-08 14:45:00 | 509.45 | 2025-09-10 09:15:00 | 519.55 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-09 10:00:00 | 509.05 | 2025-09-10 09:15:00 | 519.55 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-09-18 09:15:00 | 543.00 | 2025-09-18 10:15:00 | 535.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-29 11:30:00 | 518.40 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-29 12:30:00 | 517.50 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-29 14:00:00 | 519.10 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-09-29 15:00:00 | 518.85 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-30 09:15:00 | 518.20 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-30 10:15:00 | 518.05 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-30 11:15:00 | 517.35 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-30 13:30:00 | 518.10 | 2025-10-01 09:15:00 | 522.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-09 14:15:00 | 531.00 | 2025-10-13 09:15:00 | 520.50 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-10 09:45:00 | 532.20 | 2025-10-13 09:15:00 | 520.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-10-10 14:15:00 | 529.00 | 2025-10-13 09:15:00 | 520.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-10-20 09:15:00 | 516.70 | 2025-10-21 14:15:00 | 530.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-20 12:15:00 | 517.80 | 2025-10-21 14:15:00 | 530.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-10-20 14:30:00 | 517.80 | 2025-10-21 14:15:00 | 530.00 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-10-31 09:15:00 | 535.20 | 2025-11-06 10:15:00 | 527.65 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-24 09:30:00 | 537.10 | 2025-11-24 14:15:00 | 543.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-11-25 09:30:00 | 538.00 | 2025-11-26 09:15:00 | 543.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-25 10:30:00 | 538.45 | 2025-11-26 09:15:00 | 543.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-26 09:15:00 | 538.10 | 2025-11-26 09:15:00 | 543.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-05 11:00:00 | 538.75 | 2025-12-08 11:15:00 | 533.85 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-05 12:30:00 | 538.40 | 2025-12-08 11:15:00 | 533.85 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-31 13:00:00 | 537.00 | 2026-01-02 12:15:00 | 540.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-01 09:15:00 | 537.05 | 2026-01-02 12:15:00 | 540.10 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-01-08 10:30:00 | 530.00 | 2026-01-12 09:15:00 | 503.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 528.40 | 2026-01-12 09:15:00 | 501.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 530.00 | 2026-01-12 09:15:00 | 503.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:30:00 | 530.00 | 2026-01-12 15:15:00 | 515.00 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2026-01-08 11:00:00 | 528.40 | 2026-01-12 15:15:00 | 515.00 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2026-01-08 15:15:00 | 530.00 | 2026-01-12 15:15:00 | 515.00 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest2 | 2026-01-29 12:15:00 | 481.70 | 2026-02-03 09:15:00 | 529.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 483.55 | 2026-02-03 09:15:00 | 531.91 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-02-12 10:45:00 | 487.55 | 2026-02-13 12:15:00 | 493.60 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest1 | 2026-02-12 13:00:00 | 487.80 | 2026-02-13 12:15:00 | 493.60 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest1 | 2026-02-13 09:15:00 | 482.50 | 2026-02-13 12:15:00 | 493.60 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-02-23 12:30:00 | 485.50 | 2026-02-23 13:15:00 | 491.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-02-26 12:00:00 | 493.75 | 2026-02-27 10:15:00 | 488.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-03-06 09:15:00 | 466.50 | 2026-03-06 13:15:00 | 471.20 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-03-12 11:00:00 | 468.65 | 2026-03-13 09:15:00 | 462.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-03-12 11:30:00 | 469.00 | 2026-03-13 09:15:00 | 462.20 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-03-12 12:00:00 | 469.00 | 2026-03-13 09:15:00 | 462.20 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-03-19 11:30:00 | 467.70 | 2026-03-20 10:15:00 | 459.45 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-03-20 09:15:00 | 465.90 | 2026-03-20 10:15:00 | 459.45 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest1 | 2026-03-24 10:30:00 | 437.50 | 2026-03-25 09:15:00 | 454.25 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-04-02 11:15:00 | 455.85 | 2026-04-09 14:15:00 | 455.10 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-04-06 09:30:00 | 455.00 | 2026-04-09 14:15:00 | 455.10 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2026-04-06 10:15:00 | 457.40 | 2026-04-09 14:15:00 | 455.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-04-28 15:15:00 | 582.00 | 2026-04-29 13:15:00 | 573.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-05-04 13:15:00 | 573.30 | 2026-05-04 13:15:00 | 576.85 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-05-04 14:45:00 | 571.25 | 2026-05-07 13:15:00 | 568.00 | STOP_HIT | 1.00 | 0.57% |

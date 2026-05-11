# Zydus Wellness Ltd. (ZYDUSWELL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 152 |
| ALERT1 | 101 |
| ALERT2 | 99 |
| ALERT2_SKIP | 52 |
| ALERT3 | 249 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 115 |
| PARTIAL | 15 |
| TARGET_HIT | 3 |
| STOP_HIT | 115 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 88
- **Target hits / Stop hits / Partials:** 3 / 114 / 15
- **Avg / median % per leg:** 0.10% / -0.90%
- **Sum % (uncompounded):** 13.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 12 | 19.7% | 3 | 58 | 0 | -0.35% | -21.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.06% | -2.1% |
| BUY @ 3rd Alert (retest2) | 59 | 12 | 20.3% | 3 | 56 | 0 | -0.32% | -19.1% |
| SELL (all) | 71 | 32 | 45.1% | 0 | 56 | 15 | 0.49% | 34.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 71 | 32 | 45.1% | 0 | 56 | 15 | 0.49% | 34.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.06% | -2.1% |
| retest2 (combined) | 130 | 44 | 33.8% | 3 | 112 | 15 | 0.12% | 15.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 14:15:00 | 337.61 | 339.52 | 339.75 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 346.26 | 340.04 | 339.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 352.97 | 345.92 | 343.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 348.32 | 350.32 | 347.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 348.32 | 350.32 | 347.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 345.23 | 349.47 | 347.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 345.87 | 349.47 | 347.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 346.00 | 348.78 | 347.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:30:00 | 348.22 | 348.19 | 347.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:15:00 | 348.98 | 348.19 | 347.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:30:00 | 348.00 | 348.19 | 347.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 344.32 | 347.40 | 347.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 15:15:00 | 344.32 | 347.40 | 347.44 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 353.00 | 348.52 | 347.95 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 346.20 | 348.33 | 348.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 342.00 | 344.48 | 345.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 346.20 | 344.63 | 345.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 346.20 | 344.63 | 345.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 346.20 | 344.63 | 345.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 346.20 | 344.63 | 345.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 350.52 | 345.81 | 345.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 13:15:00 | 352.13 | 347.81 | 346.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 356.63 | 356.72 | 353.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 09:45:00 | 355.40 | 356.72 | 353.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 350.64 | 354.90 | 353.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:30:00 | 348.96 | 354.90 | 353.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 346.39 | 353.19 | 353.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:45:00 | 345.41 | 353.19 | 353.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 347.94 | 352.14 | 352.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 345.74 | 350.86 | 351.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 347.46 | 347.38 | 349.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 347.46 | 347.38 | 349.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 352.00 | 348.30 | 349.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 352.00 | 348.30 | 349.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 345.30 | 347.70 | 349.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 347.68 | 347.70 | 349.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 347.00 | 347.56 | 349.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 345.41 | 347.56 | 349.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:45:00 | 343.60 | 346.65 | 348.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 344.00 | 342.38 | 344.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 360.01 | 346.17 | 345.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 360.01 | 346.17 | 345.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 374.95 | 351.92 | 348.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 11:15:00 | 375.46 | 376.64 | 370.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 12:00:00 | 375.46 | 376.64 | 370.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 370.56 | 374.88 | 372.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 12:00:00 | 370.56 | 374.88 | 372.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 370.40 | 373.98 | 372.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 12:45:00 | 370.75 | 373.98 | 372.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 383.05 | 376.84 | 374.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 389.63 | 381.84 | 378.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 14:15:00 | 372.17 | 378.44 | 379.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 14:15:00 | 372.17 | 378.44 | 379.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 14:15:00 | 369.89 | 374.65 | 376.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 368.60 | 367.62 | 369.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 370.83 | 368.26 | 369.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 370.83 | 368.26 | 369.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:30:00 | 370.27 | 368.26 | 369.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 373.92 | 369.39 | 370.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:30:00 | 375.97 | 369.39 | 370.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 377.00 | 371.68 | 371.08 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 369.78 | 371.25 | 371.33 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 372.90 | 371.41 | 371.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 374.06 | 371.94 | 371.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 370.17 | 371.76 | 371.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 370.17 | 371.76 | 371.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 370.17 | 371.76 | 371.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:30:00 | 369.79 | 371.76 | 371.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 370.04 | 371.42 | 371.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 368.12 | 370.76 | 371.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 362.58 | 361.71 | 364.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 362.58 | 361.71 | 364.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 362.58 | 361.71 | 364.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 364.99 | 361.71 | 364.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 371.98 | 362.02 | 362.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 373.43 | 362.02 | 362.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 373.35 | 364.29 | 363.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 375.06 | 366.44 | 364.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 13:15:00 | 415.26 | 416.03 | 409.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:45:00 | 415.59 | 416.03 | 409.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 412.39 | 415.95 | 413.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 411.75 | 415.95 | 413.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 418.57 | 416.47 | 414.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:00:00 | 426.98 | 420.25 | 417.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:45:00 | 424.00 | 424.75 | 420.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 10:45:00 | 428.40 | 425.39 | 421.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 14:15:00 | 427.75 | 430.61 | 430.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 427.75 | 430.61 | 430.71 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 437.63 | 431.66 | 431.15 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 426.00 | 430.47 | 430.96 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 436.00 | 431.31 | 431.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-19 15:15:00 | 439.00 | 433.44 | 432.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 15:15:00 | 444.96 | 447.48 | 441.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 445.57 | 447.48 | 441.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 448.00 | 447.59 | 442.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 453.20 | 447.59 | 442.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:45:00 | 453.60 | 450.86 | 445.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 13:15:00 | 454.60 | 450.86 | 445.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 14:15:00 | 476.40 | 479.60 | 479.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 476.40 | 479.60 | 479.65 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 15:15:00 | 480.00 | 479.68 | 479.68 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 477.26 | 479.20 | 479.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 470.00 | 477.36 | 478.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 11:15:00 | 473.91 | 469.64 | 472.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 11:15:00 | 473.91 | 469.64 | 472.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 473.91 | 469.64 | 472.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:45:00 | 474.69 | 469.64 | 472.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 464.04 | 468.52 | 472.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:30:00 | 469.60 | 468.52 | 472.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 453.35 | 450.00 | 456.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 449.82 | 451.00 | 455.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 464.00 | 457.17 | 456.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 464.00 | 457.17 | 456.28 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 459.12 | 460.86 | 461.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 457.86 | 460.26 | 460.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 15:15:00 | 445.80 | 445.38 | 448.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:15:00 | 444.67 | 445.38 | 448.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 444.20 | 445.14 | 448.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 442.06 | 443.77 | 446.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:00:00 | 440.80 | 443.77 | 446.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 442.29 | 444.24 | 446.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:45:00 | 439.00 | 443.45 | 445.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 447.83 | 444.33 | 445.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 447.83 | 444.33 | 445.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 450.44 | 445.55 | 446.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 451.83 | 445.55 | 446.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-20 13:15:00 | 451.90 | 447.76 | 447.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 13:15:00 | 451.90 | 447.76 | 447.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 463.80 | 450.97 | 448.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 458.28 | 460.43 | 457.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 458.28 | 460.43 | 457.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 457.87 | 459.89 | 457.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 457.87 | 459.89 | 457.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 456.02 | 459.12 | 457.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 456.02 | 459.12 | 457.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 457.16 | 458.73 | 457.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 455.39 | 458.73 | 457.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 456.00 | 458.18 | 457.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 456.00 | 458.18 | 457.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 456.28 | 457.80 | 457.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 457.94 | 457.80 | 457.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 453.70 | 456.98 | 457.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 11:15:00 | 451.20 | 455.36 | 456.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 452.87 | 451.66 | 453.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 10:15:00 | 452.87 | 451.66 | 453.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 452.87 | 451.66 | 453.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 452.87 | 451.66 | 453.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 452.99 | 451.92 | 453.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 452.99 | 451.92 | 453.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 452.70 | 451.22 | 452.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 451.95 | 451.22 | 452.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 452.90 | 451.56 | 452.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 454.11 | 451.56 | 452.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 453.60 | 451.97 | 452.87 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 455.76 | 453.42 | 453.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 456.45 | 454.02 | 453.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 452.06 | 454.61 | 454.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 452.06 | 454.61 | 454.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 452.06 | 454.61 | 454.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 452.69 | 454.61 | 454.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 451.49 | 453.99 | 453.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:30:00 | 451.11 | 453.99 | 453.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 451.49 | 453.49 | 453.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 450.24 | 452.04 | 452.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 453.04 | 452.24 | 452.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 453.04 | 452.24 | 452.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 453.04 | 452.24 | 452.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 453.04 | 452.24 | 452.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 454.20 | 452.63 | 453.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:00:00 | 454.20 | 452.63 | 453.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 456.18 | 453.34 | 453.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 460.00 | 454.67 | 453.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 452.87 | 455.42 | 454.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 10:15:00 | 452.87 | 455.42 | 454.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 452.87 | 455.42 | 454.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 453.16 | 455.42 | 454.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 453.76 | 455.08 | 454.57 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 445.18 | 453.10 | 453.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 440.20 | 450.52 | 452.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 448.00 | 446.95 | 450.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 448.00 | 446.95 | 450.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 448.00 | 446.95 | 450.08 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 458.22 | 451.44 | 450.56 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 444.43 | 451.63 | 452.35 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 13:15:00 | 450.72 | 450.09 | 450.01 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 449.50 | 449.94 | 449.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 447.95 | 449.55 | 449.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 14:15:00 | 442.44 | 442.00 | 443.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 442.44 | 442.00 | 443.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 437.41 | 441.09 | 443.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:15:00 | 435.94 | 440.41 | 442.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 13:15:00 | 414.14 | 422.64 | 429.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 12:15:00 | 422.42 | 419.20 | 424.01 | SL hit (close>ema200) qty=0.50 sl=419.20 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 390.03 | 387.41 | 387.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 11:15:00 | 391.69 | 389.51 | 388.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 389.20 | 390.88 | 389.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 389.20 | 390.88 | 389.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 389.20 | 390.88 | 389.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 387.34 | 390.88 | 389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 389.00 | 390.50 | 389.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 389.93 | 390.50 | 389.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 389.57 | 390.32 | 389.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 389.52 | 390.32 | 389.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 391.39 | 390.53 | 389.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 389.99 | 390.53 | 389.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 392.00 | 390.76 | 390.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 392.00 | 390.76 | 390.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 390.00 | 390.84 | 390.31 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 11:15:00 | 387.20 | 389.75 | 389.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 385.46 | 388.90 | 389.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 395.31 | 389.16 | 389.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 395.31 | 389.16 | 389.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 395.31 | 389.16 | 389.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 395.31 | 389.16 | 389.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 397.66 | 390.86 | 390.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 403.60 | 394.56 | 392.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 391.31 | 397.99 | 396.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 391.31 | 397.99 | 396.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 391.31 | 397.99 | 396.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 391.31 | 397.99 | 396.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 388.97 | 396.19 | 395.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 389.56 | 396.19 | 395.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 395.10 | 395.78 | 395.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 395.10 | 395.78 | 395.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 397.99 | 396.22 | 395.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:30:00 | 397.25 | 396.22 | 395.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 393.50 | 395.80 | 395.71 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 393.15 | 395.27 | 395.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 14:15:00 | 391.99 | 393.92 | 394.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 14:15:00 | 369.81 | 368.52 | 373.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 15:00:00 | 369.81 | 368.52 | 373.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 372.19 | 369.87 | 373.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 372.19 | 369.87 | 373.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 362.82 | 361.46 | 365.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:30:00 | 362.63 | 361.46 | 365.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 365.43 | 362.26 | 365.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 365.43 | 362.26 | 365.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 366.63 | 363.13 | 365.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 367.06 | 363.13 | 365.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 368.00 | 364.11 | 365.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 368.00 | 364.11 | 365.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 369.58 | 366.87 | 366.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 373.70 | 369.58 | 368.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 10:15:00 | 389.61 | 389.82 | 386.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:30:00 | 389.07 | 389.82 | 386.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 388.15 | 389.53 | 387.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 388.15 | 389.53 | 387.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 388.98 | 389.42 | 387.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 389.81 | 389.53 | 387.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 389.12 | 390.05 | 388.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:15:00 | 390.00 | 390.05 | 388.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 390.00 | 390.04 | 388.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 391.14 | 390.04 | 388.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 10:45:00 | 393.00 | 391.14 | 389.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 385.96 | 392.68 | 392.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 385.96 | 392.68 | 392.97 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 398.94 | 393.47 | 393.25 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 383.43 | 391.46 | 392.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 378.00 | 384.61 | 387.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 378.03 | 376.97 | 381.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:15:00 | 383.06 | 376.97 | 381.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 389.13 | 379.40 | 381.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 389.13 | 379.40 | 381.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 379.44 | 379.41 | 381.65 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 13:15:00 | 389.50 | 383.18 | 382.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 14:15:00 | 392.61 | 385.07 | 383.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 10:15:00 | 383.56 | 385.76 | 384.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 10:15:00 | 383.56 | 385.76 | 384.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 383.56 | 385.76 | 384.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:30:00 | 382.85 | 385.76 | 384.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 386.03 | 385.81 | 384.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 12:15:00 | 386.66 | 385.81 | 384.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 15:15:00 | 382.33 | 384.61 | 384.43 | SL hit (close<static) qty=1.00 sl=383.19 alert=retest2 |

### Cycle 43 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 385.42 | 386.22 | 386.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 14:15:00 | 382.22 | 384.24 | 385.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 15:15:00 | 385.80 | 384.55 | 385.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 15:15:00 | 385.80 | 384.55 | 385.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 385.80 | 384.55 | 385.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 390.70 | 384.55 | 385.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 388.39 | 385.32 | 385.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 387.80 | 385.32 | 385.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 388.58 | 385.97 | 385.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 389.46 | 386.67 | 386.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 383.38 | 386.68 | 386.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 383.38 | 386.68 | 386.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 383.38 | 386.68 | 386.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 383.38 | 386.68 | 386.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 15:15:00 | 380.22 | 385.39 | 385.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 378.67 | 384.04 | 385.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 13:15:00 | 383.91 | 383.19 | 384.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 13:15:00 | 383.91 | 383.19 | 384.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 383.91 | 383.19 | 384.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:30:00 | 383.82 | 383.19 | 384.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 386.54 | 383.86 | 384.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 386.54 | 383.86 | 384.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 387.46 | 384.58 | 384.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 389.52 | 384.58 | 384.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 391.20 | 385.90 | 385.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 11:15:00 | 395.23 | 389.13 | 386.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 398.39 | 400.74 | 396.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 398.39 | 400.74 | 396.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 398.39 | 400.74 | 396.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 398.39 | 400.74 | 396.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 405.47 | 401.69 | 397.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 11:15:00 | 409.21 | 403.83 | 400.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:45:00 | 408.55 | 405.02 | 402.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:30:00 | 408.00 | 406.90 | 403.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 407.90 | 412.53 | 413.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 407.90 | 412.53 | 413.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 405.40 | 409.28 | 410.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 411.00 | 407.47 | 408.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 411.00 | 407.47 | 408.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 411.00 | 407.47 | 408.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 412.00 | 407.47 | 408.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 409.11 | 407.79 | 408.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:45:00 | 408.86 | 408.18 | 408.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 411.29 | 408.96 | 409.21 | SL hit (close>static) qty=1.00 sl=411.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 415.96 | 410.36 | 409.82 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 405.81 | 409.46 | 409.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 404.58 | 408.49 | 409.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 10:15:00 | 404.97 | 404.97 | 406.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 11:00:00 | 404.97 | 404.97 | 406.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 407.00 | 403.54 | 405.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 411.10 | 403.54 | 405.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 407.90 | 404.41 | 405.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:45:00 | 410.55 | 404.41 | 405.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 411.78 | 405.88 | 405.86 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 402.58 | 405.99 | 406.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 397.53 | 404.30 | 405.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 397.01 | 396.71 | 399.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:45:00 | 397.13 | 396.71 | 399.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 391.86 | 395.75 | 397.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 388.76 | 393.01 | 396.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 387.98 | 380.84 | 380.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 387.98 | 380.84 | 380.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 12:15:00 | 389.24 | 383.43 | 381.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 15:15:00 | 394.40 | 394.46 | 390.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:15:00 | 394.40 | 394.46 | 390.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 396.87 | 398.38 | 396.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:45:00 | 395.65 | 398.38 | 396.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 396.00 | 397.91 | 396.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 391.83 | 397.91 | 396.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 391.76 | 396.68 | 395.77 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 386.60 | 394.66 | 394.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 386.00 | 391.83 | 393.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 391.64 | 390.29 | 391.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 13:15:00 | 391.64 | 390.29 | 391.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 391.64 | 390.29 | 391.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 391.64 | 390.29 | 391.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 393.54 | 390.94 | 391.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 389.73 | 391.68 | 392.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 390.58 | 391.57 | 391.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 390.14 | 390.81 | 391.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:00:00 | 390.63 | 390.37 | 390.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 391.20 | 390.54 | 390.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:00:00 | 389.00 | 390.23 | 390.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:30:00 | 387.98 | 390.00 | 390.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 385.43 | 390.28 | 390.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 371.10 | 379.90 | 384.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 370.24 | 377.76 | 382.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 371.05 | 377.76 | 382.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 370.63 | 377.76 | 382.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 369.55 | 377.76 | 382.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 368.58 | 377.76 | 382.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 376.00 | 373.35 | 376.53 | SL hit (close>ema200) qty=0.50 sl=373.35 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 379.52 | 377.70 | 377.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 15:15:00 | 382.40 | 378.64 | 378.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 378.06 | 381.42 | 380.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 378.06 | 381.42 | 380.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 378.06 | 381.42 | 380.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 378.06 | 381.42 | 380.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 377.94 | 380.72 | 380.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 377.94 | 380.72 | 380.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 377.15 | 379.52 | 379.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 376.20 | 378.86 | 379.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 13:15:00 | 375.74 | 374.88 | 376.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 13:15:00 | 375.74 | 374.88 | 376.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 375.74 | 374.88 | 376.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 376.16 | 374.88 | 376.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 375.87 | 375.08 | 376.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 376.80 | 375.08 | 376.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 376.50 | 375.36 | 376.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 379.60 | 375.36 | 376.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 375.76 | 375.44 | 376.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 373.66 | 375.44 | 376.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 368.68 | 373.82 | 374.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 354.98 | 358.41 | 361.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 350.25 | 358.41 | 361.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 349.13 | 348.77 | 353.26 | SL hit (close>ema200) qty=0.50 sl=348.77 alert=retest2 |

### Cycle 56 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 362.62 | 354.58 | 354.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 365.24 | 359.87 | 357.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 364.93 | 366.40 | 364.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:45:00 | 365.24 | 366.40 | 364.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 363.50 | 366.16 | 364.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 364.36 | 366.16 | 364.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 365.06 | 365.94 | 364.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 365.86 | 364.95 | 364.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 351.13 | 366.02 | 366.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 12:15:00 | 351.13 | 366.02 | 366.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 350.45 | 355.54 | 356.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 352.61 | 352.37 | 354.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 352.61 | 352.37 | 354.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 348.00 | 351.50 | 353.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 344.50 | 351.50 | 353.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 327.27 | 335.02 | 338.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 335.10 | 333.80 | 336.51 | SL hit (close>ema200) qty=0.50 sl=333.80 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 333.70 | 331.62 | 331.51 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 325.77 | 330.48 | 331.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 10:15:00 | 322.96 | 328.97 | 330.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 327.97 | 325.86 | 327.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 12:15:00 | 327.97 | 325.86 | 327.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 327.97 | 325.86 | 327.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:30:00 | 330.33 | 325.86 | 327.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 327.36 | 326.16 | 327.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 324.33 | 326.30 | 327.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 308.11 | 310.89 | 315.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 310.15 | 306.85 | 311.52 | SL hit (close>ema200) qty=0.50 sl=306.85 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 314.60 | 309.80 | 309.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 315.56 | 311.67 | 310.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 15:15:00 | 311.30 | 311.60 | 310.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:15:00 | 322.04 | 311.60 | 310.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 320.20 | 322.16 | 319.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:30:00 | 324.24 | 322.72 | 320.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:45:00 | 323.23 | 323.36 | 321.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 318.05 | 322.30 | 321.29 | SL hit (close<ema400) qty=1.00 sl=321.29 alert=retest1 |

### Cycle 61 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 317.82 | 320.60 | 320.64 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 322.51 | 320.84 | 320.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 10:15:00 | 324.31 | 322.06 | 321.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 13:15:00 | 322.80 | 322.98 | 321.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 13:45:00 | 322.80 | 322.98 | 321.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 322.67 | 322.92 | 322.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 322.67 | 322.92 | 322.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 322.80 | 322.89 | 322.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 318.50 | 322.89 | 322.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 321.54 | 322.62 | 322.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 322.44 | 322.42 | 322.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 12:15:00 | 319.53 | 321.51 | 321.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 319.53 | 321.51 | 321.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 318.06 | 320.39 | 321.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 12:15:00 | 320.00 | 318.57 | 319.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 12:15:00 | 320.00 | 318.57 | 319.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 320.00 | 318.57 | 319.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 320.00 | 318.57 | 319.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 320.80 | 319.01 | 319.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 320.80 | 319.01 | 319.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 322.00 | 319.61 | 320.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:30:00 | 323.30 | 319.61 | 320.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 325.84 | 320.76 | 320.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 329.00 | 323.11 | 321.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 12:15:00 | 328.48 | 328.63 | 325.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 13:00:00 | 328.48 | 328.63 | 325.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 331.76 | 332.26 | 331.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 330.98 | 332.26 | 331.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 331.70 | 332.15 | 331.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:45:00 | 333.08 | 332.15 | 331.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 331.20 | 331.96 | 331.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:00:00 | 331.20 | 331.96 | 331.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 331.28 | 331.82 | 331.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:45:00 | 331.23 | 331.82 | 331.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 336.42 | 332.74 | 331.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 12:30:00 | 336.62 | 334.41 | 333.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 330.82 | 333.47 | 333.03 | SL hit (close<static) qty=1.00 sl=331.26 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 328.58 | 332.49 | 332.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 327.30 | 331.45 | 332.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 331.78 | 327.14 | 328.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 331.78 | 327.14 | 328.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 331.78 | 327.14 | 328.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 331.78 | 327.14 | 328.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 335.22 | 328.76 | 329.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 335.22 | 328.76 | 329.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 340.32 | 331.07 | 330.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 341.56 | 334.15 | 331.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 336.14 | 336.86 | 334.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 15:00:00 | 336.14 | 336.86 | 334.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 333.20 | 336.13 | 334.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 336.46 | 336.13 | 334.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 338.69 | 336.64 | 334.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 340.76 | 337.25 | 336.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 13:15:00 | 340.70 | 340.38 | 338.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 13:45:00 | 340.49 | 340.43 | 338.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 337.10 | 338.35 | 338.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 337.10 | 338.35 | 338.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 329.52 | 335.91 | 337.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 10:15:00 | 337.10 | 336.14 | 337.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 10:15:00 | 337.10 | 336.14 | 337.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 337.10 | 336.14 | 337.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 11:00:00 | 337.10 | 336.14 | 337.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 335.82 | 336.08 | 337.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 11:30:00 | 340.69 | 336.08 | 337.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 338.45 | 336.55 | 337.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 13:00:00 | 338.45 | 336.55 | 337.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 334.80 | 336.20 | 336.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 333.18 | 336.16 | 336.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 344.60 | 337.37 | 337.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 344.60 | 337.37 | 337.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 350.00 | 342.82 | 340.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 342.89 | 344.47 | 341.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 09:30:00 | 344.19 | 344.47 | 341.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 347.60 | 346.85 | 344.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 352.50 | 346.85 | 344.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 11:00:00 | 348.30 | 349.23 | 347.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 11:30:00 | 348.06 | 349.37 | 347.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 09:15:00 | 350.64 | 349.20 | 348.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 352.66 | 352.57 | 350.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:30:00 | 355.00 | 352.99 | 351.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 355.08 | 352.99 | 351.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 355.52 | 353.50 | 351.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 350.00 | 352.08 | 351.94 | SL hit (close<static) qty=1.00 sl=350.36 alert=retest2 |

### Cycle 69 — SELL (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 12:15:00 | 348.64 | 351.39 | 351.64 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 354.36 | 351.66 | 351.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 12:15:00 | 358.30 | 352.99 | 352.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 354.42 | 354.83 | 353.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 354.42 | 354.83 | 353.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 354.70 | 354.98 | 353.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 353.80 | 354.98 | 353.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 353.54 | 354.69 | 353.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 353.56 | 354.69 | 353.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 353.78 | 354.51 | 353.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:45:00 | 354.20 | 354.51 | 353.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 352.80 | 354.17 | 353.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 354.78 | 354.34 | 353.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 352.90 | 354.80 | 354.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 352.90 | 354.80 | 354.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 350.20 | 353.37 | 354.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 354.66 | 353.63 | 354.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 354.66 | 353.63 | 354.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 354.66 | 353.63 | 354.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 354.66 | 353.63 | 354.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 354.02 | 353.70 | 354.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 354.02 | 353.70 | 354.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 354.60 | 353.88 | 354.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 354.60 | 353.88 | 354.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 353.58 | 353.82 | 354.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 354.16 | 353.82 | 354.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 361.08 | 354.57 | 354.34 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 353.54 | 356.51 | 356.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 351.80 | 355.01 | 355.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 339.36 | 338.94 | 342.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:00:00 | 339.36 | 338.94 | 342.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 338.60 | 338.33 | 340.96 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 15:15:00 | 343.60 | 341.90 | 341.77 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 337.16 | 340.96 | 341.36 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 344.10 | 341.28 | 341.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 354.60 | 343.95 | 342.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 357.30 | 358.09 | 353.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 357.58 | 358.09 | 353.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 359.40 | 360.52 | 358.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 360.12 | 360.52 | 358.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 359.82 | 360.38 | 359.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 359.66 | 360.38 | 359.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 360.00 | 360.30 | 359.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 360.74 | 360.30 | 359.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 361.56 | 360.55 | 359.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 363.02 | 360.55 | 359.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 365.00 | 361.68 | 360.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-21 10:15:00 | 399.32 | 385.25 | 376.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 389.18 | 391.07 | 391.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 388.60 | 390.28 | 390.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 391.50 | 390.12 | 390.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 391.50 | 390.12 | 390.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 391.50 | 390.12 | 390.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 391.50 | 390.12 | 390.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 392.26 | 390.55 | 390.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 393.52 | 390.55 | 390.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 392.72 | 390.99 | 390.91 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 389.70 | 390.63 | 390.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 388.64 | 390.04 | 390.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 12:15:00 | 389.98 | 389.95 | 390.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 12:30:00 | 389.78 | 389.95 | 390.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 390.34 | 390.03 | 390.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:30:00 | 389.42 | 390.02 | 390.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 397.46 | 391.50 | 390.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 397.46 | 391.50 | 390.90 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 388.52 | 391.00 | 391.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 384.70 | 388.67 | 390.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 382.86 | 380.16 | 382.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 382.86 | 380.16 | 382.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 382.86 | 380.16 | 382.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 382.86 | 380.16 | 382.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 381.62 | 380.46 | 382.71 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 389.00 | 384.56 | 384.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 394.50 | 387.35 | 385.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 13:15:00 | 389.22 | 390.16 | 387.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:00:00 | 389.22 | 390.16 | 387.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 385.80 | 389.14 | 387.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 390.40 | 389.26 | 387.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 13:15:00 | 385.50 | 387.81 | 387.54 | SL hit (close<static) qty=1.00 sl=385.80 alert=retest2 |

### Cycle 83 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 387.38 | 390.02 | 390.18 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 15:15:00 | 392.80 | 390.57 | 390.42 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 384.38 | 389.33 | 389.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 379.80 | 383.98 | 386.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 382.54 | 382.12 | 384.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 382.54 | 382.12 | 384.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 387.10 | 383.17 | 384.49 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 388.00 | 385.46 | 385.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 389.78 | 386.70 | 385.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 411.02 | 411.31 | 404.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 15:00:00 | 411.02 | 411.31 | 404.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 406.20 | 411.00 | 407.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 406.20 | 411.00 | 407.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 405.88 | 409.97 | 407.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 404.52 | 409.97 | 407.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 400.20 | 406.36 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 402.40 | 406.36 | 405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 402.00 | 405.49 | 405.59 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 408.18 | 406.11 | 405.83 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 403.32 | 405.26 | 405.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 11:15:00 | 399.48 | 404.10 | 404.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 403.04 | 401.76 | 403.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 403.04 | 401.76 | 403.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 403.04 | 401.76 | 403.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 403.60 | 401.76 | 403.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 404.14 | 402.23 | 403.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:30:00 | 404.98 | 402.23 | 403.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 405.12 | 402.81 | 403.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 405.12 | 402.81 | 403.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 406.02 | 403.54 | 403.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 406.00 | 403.54 | 403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 407.32 | 404.30 | 404.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 408.72 | 406.14 | 405.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 405.60 | 407.43 | 406.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 405.60 | 407.43 | 406.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 405.60 | 407.43 | 406.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 405.60 | 407.43 | 406.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 405.72 | 407.09 | 406.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 410.54 | 407.09 | 406.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 12:15:00 | 403.82 | 406.29 | 406.20 | SL hit (close<static) qty=1.00 sl=405.06 alert=retest2 |

### Cycle 91 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 403.06 | 405.64 | 405.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 399.84 | 403.34 | 404.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 398.98 | 398.35 | 400.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 398.98 | 398.35 | 400.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 398.98 | 398.35 | 400.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 401.32 | 398.35 | 400.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 401.94 | 399.07 | 400.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 402.22 | 399.07 | 400.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 399.24 | 399.10 | 400.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 397.66 | 399.22 | 399.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:45:00 | 397.82 | 398.99 | 399.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 397.60 | 398.62 | 399.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 397.50 | 397.29 | 398.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 396.80 | 396.05 | 397.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 402.00 | 397.70 | 397.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 402.00 | 397.70 | 397.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 405.70 | 399.97 | 398.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 402.30 | 402.53 | 400.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:45:00 | 402.24 | 402.53 | 400.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 401.40 | 402.21 | 401.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 406.46 | 402.21 | 401.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 403.96 | 403.97 | 402.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 404.68 | 403.86 | 402.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 404.90 | 404.07 | 403.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 402.80 | 403.90 | 403.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 402.14 | 403.90 | 403.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 402.84 | 403.69 | 403.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:30:00 | 402.60 | 403.69 | 403.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 401.82 | 403.05 | 402.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 405.80 | 403.05 | 402.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 401.68 | 403.38 | 403.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 401.68 | 403.38 | 403.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 400.80 | 402.87 | 403.33 | Break + close below crossover candle low |

### Cycle 94 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 407.76 | 403.68 | 403.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 12:15:00 | 412.98 | 406.68 | 405.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 11:15:00 | 417.54 | 418.05 | 414.88 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 13:15:00 | 419.32 | 418.14 | 415.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 419.06 | 418.32 | 416.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 415.60 | 417.78 | 416.16 | SL hit (close<ema400) qty=1.00 sl=416.16 alert=retest1 |

### Cycle 95 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 410.40 | 415.35 | 415.47 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 417.98 | 415.65 | 415.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 430.20 | 419.56 | 417.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 423.80 | 425.22 | 422.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 423.80 | 425.22 | 422.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 419.56 | 424.09 | 421.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 419.56 | 424.09 | 421.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 420.24 | 423.32 | 421.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:30:00 | 421.42 | 422.89 | 421.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:30:00 | 422.10 | 422.78 | 421.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 418.38 | 421.10 | 421.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 418.38 | 421.10 | 421.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 415.18 | 419.91 | 420.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 13:15:00 | 409.24 | 404.87 | 407.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 409.24 | 404.87 | 407.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 409.24 | 404.87 | 407.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:00:00 | 409.24 | 404.87 | 407.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 410.12 | 405.92 | 407.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 410.12 | 405.92 | 407.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 407.80 | 406.29 | 407.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:30:00 | 402.64 | 406.91 | 407.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 416.06 | 408.74 | 408.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 416.06 | 408.74 | 408.17 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 403.28 | 409.65 | 409.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 401.18 | 407.96 | 409.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 10:15:00 | 392.84 | 389.04 | 392.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 10:15:00 | 392.84 | 389.04 | 392.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 392.84 | 389.04 | 392.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 392.84 | 389.04 | 392.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 392.30 | 389.69 | 392.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 390.38 | 389.69 | 392.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:30:00 | 391.00 | 388.97 | 389.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 392.48 | 389.32 | 389.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 392.48 | 389.32 | 389.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 394.60 | 390.38 | 389.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 389.56 | 390.90 | 390.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 389.56 | 390.90 | 390.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 389.56 | 390.90 | 390.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 389.56 | 390.90 | 390.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 388.40 | 390.40 | 390.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 391.22 | 390.40 | 390.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 391.80 | 390.60 | 390.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 391.18 | 390.60 | 390.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 388.60 | 390.20 | 390.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 388.60 | 390.20 | 390.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 386.20 | 389.40 | 389.71 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 397.40 | 389.56 | 389.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 397.42 | 392.18 | 390.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 400.02 | 401.21 | 397.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 400.02 | 401.21 | 397.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 403.94 | 401.76 | 398.17 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 396.84 | 399.30 | 399.36 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 15:15:00 | 405.60 | 399.11 | 398.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 10:15:00 | 406.10 | 401.48 | 399.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 404.62 | 404.76 | 402.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:30:00 | 405.54 | 404.76 | 402.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 404.30 | 404.67 | 402.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:30:00 | 403.50 | 404.67 | 402.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 403.90 | 404.46 | 402.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 403.14 | 404.46 | 402.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 404.32 | 404.43 | 402.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 403.70 | 404.43 | 402.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 401.84 | 403.79 | 402.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 416.20 | 403.79 | 402.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-03 11:15:00 | 457.82 | 452.38 | 440.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 500.60 | 516.92 | 518.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 498.40 | 513.21 | 516.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 476.20 | 475.19 | 485.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 476.20 | 475.19 | 485.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 486.40 | 477.44 | 485.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 486.40 | 477.44 | 485.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 486.10 | 479.17 | 485.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 479.60 | 479.17 | 485.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 490.00 | 479.13 | 479.38 | SL hit (close>static) qty=1.00 sl=489.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 459.20 | 454.95 | 454.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 461.70 | 457.14 | 455.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 13:15:00 | 457.50 | 458.58 | 456.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:45:00 | 457.55 | 458.58 | 456.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 455.00 | 457.87 | 456.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 455.00 | 457.87 | 456.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 456.00 | 457.49 | 456.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 461.80 | 457.49 | 456.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 451.30 | 455.84 | 455.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 446.35 | 452.60 | 454.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 456.10 | 451.76 | 453.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 456.10 | 451.76 | 453.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 456.10 | 451.76 | 453.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 456.10 | 451.76 | 453.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 456.00 | 452.60 | 453.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:45:00 | 453.25 | 452.63 | 453.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 464.35 | 455.78 | 454.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 464.35 | 455.78 | 454.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 476.25 | 461.05 | 457.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 469.15 | 469.64 | 465.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 469.15 | 469.64 | 465.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 462.95 | 468.30 | 465.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 459.55 | 468.30 | 465.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 461.65 | 466.97 | 464.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 461.80 | 466.97 | 464.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 467.80 | 465.48 | 464.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 463.85 | 465.48 | 464.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 476.15 | 467.61 | 465.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 482.30 | 471.28 | 467.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 483.45 | 476.47 | 471.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 481.25 | 485.77 | 483.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 482.80 | 488.00 | 488.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 482.80 | 488.00 | 488.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 481.80 | 486.76 | 488.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 484.95 | 482.48 | 485.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 484.95 | 482.48 | 485.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 484.95 | 482.48 | 485.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 489.00 | 482.48 | 485.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 484.65 | 482.92 | 485.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 484.65 | 482.92 | 485.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 479.20 | 482.17 | 484.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 478.15 | 480.67 | 482.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 454.24 | 464.06 | 471.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 449.60 | 445.54 | 453.23 | SL hit (close>ema200) qty=0.50 sl=445.54 alert=retest2 |

### Cycle 110 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 462.00 | 454.85 | 454.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 463.85 | 460.51 | 458.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 461.10 | 461.27 | 459.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 13:00:00 | 461.10 | 461.27 | 459.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 457.30 | 460.56 | 459.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 457.30 | 460.56 | 459.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 456.00 | 459.65 | 458.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 455.05 | 459.65 | 458.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 454.30 | 457.72 | 458.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 453.70 | 456.16 | 457.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 15:15:00 | 456.00 | 455.69 | 456.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 15:15:00 | 456.00 | 455.69 | 456.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 456.00 | 455.69 | 456.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 450.80 | 455.69 | 456.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 451.10 | 453.96 | 454.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 12:00:00 | 452.55 | 452.56 | 453.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 455.45 | 453.65 | 453.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 455.45 | 453.65 | 453.56 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 450.50 | 453.47 | 453.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 448.70 | 452.52 | 453.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 449.90 | 441.90 | 445.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 449.90 | 441.90 | 445.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 449.90 | 441.90 | 445.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 449.90 | 441.90 | 445.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 451.80 | 443.88 | 446.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 442.40 | 443.88 | 446.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 420.28 | 426.70 | 429.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 421.75 | 421.49 | 425.42 | SL hit (close>ema200) qty=0.50 sl=421.49 alert=retest2 |

### Cycle 114 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 419.90 | 415.76 | 415.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 422.95 | 417.19 | 415.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 424.25 | 424.80 | 421.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:45:00 | 422.85 | 424.80 | 421.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 419.15 | 423.86 | 421.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 418.50 | 423.86 | 421.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 419.05 | 422.90 | 421.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 418.50 | 422.90 | 421.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 425.85 | 424.50 | 422.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 15:15:00 | 429.50 | 425.03 | 423.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 424.45 | 425.55 | 425.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 424.45 | 425.55 | 425.57 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 427.40 | 425.46 | 425.36 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 424.60 | 425.33 | 425.36 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 426.05 | 425.47 | 425.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 430.00 | 426.57 | 425.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 423.85 | 427.39 | 426.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 423.85 | 427.39 | 426.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 423.85 | 427.39 | 426.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 424.00 | 427.39 | 426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 423.80 | 426.67 | 426.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 422.75 | 426.67 | 426.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 425.95 | 426.53 | 426.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 425.95 | 426.53 | 426.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 424.75 | 426.17 | 426.19 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 14:15:00 | 428.90 | 426.72 | 426.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 430.60 | 427.70 | 426.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 426.85 | 427.93 | 427.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 12:15:00 | 426.85 | 427.93 | 427.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 426.85 | 427.93 | 427.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:45:00 | 427.55 | 427.93 | 427.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 426.90 | 427.72 | 427.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 426.20 | 427.72 | 427.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 427.80 | 427.49 | 427.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 424.70 | 427.49 | 427.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 424.65 | 426.93 | 426.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 422.80 | 425.63 | 426.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 12:15:00 | 425.80 | 425.67 | 426.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 12:45:00 | 424.70 | 425.67 | 426.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 425.75 | 425.68 | 426.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:30:00 | 426.50 | 425.68 | 426.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 423.25 | 425.20 | 425.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:15:00 | 425.25 | 425.20 | 425.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 425.25 | 425.21 | 425.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:00:00 | 422.00 | 424.92 | 425.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 422.20 | 425.13 | 425.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 422.25 | 425.23 | 425.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 460.10 | 431.73 | 428.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 460.10 | 431.73 | 428.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 481.00 | 453.14 | 440.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 451.90 | 452.89 | 441.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 15:00:00 | 451.90 | 452.89 | 441.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 456.60 | 461.12 | 453.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 456.65 | 461.12 | 453.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 484.90 | 478.31 | 473.47 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 460.20 | 470.98 | 472.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 456.55 | 464.39 | 468.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 442.50 | 441.80 | 449.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 442.50 | 441.80 | 449.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 442.00 | 439.64 | 442.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 442.00 | 439.64 | 442.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 442.15 | 440.14 | 442.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 442.15 | 440.14 | 442.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 444.40 | 440.99 | 442.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 15:00:00 | 444.40 | 440.99 | 442.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 443.00 | 441.39 | 442.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 439.70 | 441.39 | 442.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 439.80 | 439.28 | 440.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 441.15 | 439.28 | 440.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 436.00 | 438.63 | 440.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:45:00 | 432.00 | 437.49 | 439.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 410.40 | 420.46 | 426.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 420.30 | 419.93 | 424.90 | SL hit (close>ema200) qty=0.50 sl=419.93 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 431.15 | 426.25 | 426.02 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 422.60 | 426.57 | 426.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 420.90 | 425.44 | 426.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 428.75 | 424.16 | 425.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 428.75 | 424.16 | 425.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 428.75 | 424.16 | 425.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 426.70 | 424.16 | 425.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 424.20 | 424.17 | 425.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 12:30:00 | 420.20 | 423.29 | 424.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 431.00 | 424.00 | 424.52 | SL hit (close>static) qty=1.00 sl=428.75 alert=retest2 |

### Cycle 126 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 430.90 | 424.07 | 423.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 434.05 | 427.04 | 424.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 440.05 | 442.33 | 436.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:30:00 | 441.00 | 442.33 | 436.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 439.50 | 441.77 | 436.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 441.25 | 441.66 | 437.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 431.25 | 439.15 | 436.92 | SL hit (close<static) qty=1.00 sl=436.85 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 428.30 | 434.36 | 435.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 12:15:00 | 419.05 | 429.68 | 432.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 11:15:00 | 404.75 | 392.48 | 396.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 11:15:00 | 404.75 | 392.48 | 396.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 404.75 | 392.48 | 396.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 404.75 | 392.48 | 396.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 403.55 | 394.69 | 397.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:30:00 | 404.75 | 394.69 | 397.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 408.00 | 399.16 | 398.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 412.80 | 403.19 | 400.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 411.60 | 413.49 | 410.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 411.60 | 413.49 | 410.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 411.60 | 413.49 | 410.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 409.85 | 413.49 | 410.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 409.00 | 412.59 | 410.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 409.00 | 412.59 | 410.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 410.70 | 412.22 | 410.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:30:00 | 410.25 | 412.22 | 410.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 412.25 | 412.22 | 410.69 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 399.70 | 408.34 | 409.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 396.30 | 399.95 | 401.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 400.35 | 399.45 | 400.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 400.35 | 399.45 | 400.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 400.35 | 399.45 | 400.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 400.00 | 399.45 | 400.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 399.85 | 399.53 | 400.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 400.55 | 399.53 | 400.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 401.30 | 399.89 | 400.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 401.30 | 399.89 | 400.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 400.45 | 400.00 | 400.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 398.50 | 400.00 | 400.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 402.15 | 400.19 | 400.77 | SL hit (close>static) qty=1.00 sl=401.30 alert=retest2 |

### Cycle 130 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 406.45 | 401.87 | 401.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 409.20 | 403.34 | 402.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 402.70 | 404.99 | 403.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 402.70 | 404.99 | 403.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 402.70 | 404.99 | 403.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 400.25 | 404.99 | 403.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 401.00 | 404.19 | 403.27 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 397.15 | 401.77 | 402.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 393.60 | 397.56 | 399.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 384.95 | 384.49 | 388.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 384.95 | 384.49 | 388.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 384.95 | 384.49 | 388.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 387.55 | 384.49 | 388.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 379.80 | 378.80 | 380.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 379.80 | 378.80 | 380.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 379.40 | 378.92 | 380.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 14:15:00 | 378.40 | 378.92 | 380.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 382.85 | 379.71 | 380.69 | SL hit (close>static) qty=1.00 sl=380.50 alert=retest2 |

### Cycle 132 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 385.35 | 381.36 | 381.30 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 372.85 | 380.00 | 380.86 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 386.80 | 379.82 | 378.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 388.90 | 384.83 | 382.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 13:15:00 | 385.00 | 385.47 | 383.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 14:00:00 | 385.00 | 385.47 | 383.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 390.00 | 387.47 | 384.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 389.00 | 387.47 | 384.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 412.05 | 408.50 | 398.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 15:15:00 | 425.00 | 414.87 | 406.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:45:00 | 426.45 | 418.69 | 413.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:45:00 | 425.70 | 420.83 | 415.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 15:15:00 | 412.10 | 418.20 | 418.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 412.10 | 418.20 | 418.92 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 424.00 | 419.72 | 419.32 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 407.05 | 418.48 | 419.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 406.10 | 416.00 | 417.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 414.75 | 412.47 | 415.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 414.75 | 412.47 | 415.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 414.75 | 412.47 | 415.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 414.75 | 412.47 | 415.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 415.80 | 413.13 | 415.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 421.00 | 413.13 | 415.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 421.15 | 414.74 | 415.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:15:00 | 424.80 | 414.74 | 415.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 430.70 | 417.93 | 417.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 435.55 | 421.45 | 418.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 449.25 | 449.81 | 441.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 441.00 | 448.05 | 441.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 441.00 | 448.05 | 441.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 441.00 | 448.05 | 441.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 439.10 | 446.26 | 441.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 439.10 | 446.26 | 441.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 448.05 | 446.62 | 441.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:30:00 | 434.85 | 446.62 | 441.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 445.00 | 446.29 | 442.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 438.40 | 446.29 | 442.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 436.60 | 444.35 | 441.61 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 430.35 | 439.69 | 439.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 425.00 | 431.94 | 435.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 438.20 | 433.19 | 435.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 438.20 | 433.19 | 435.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 438.20 | 433.19 | 435.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 432.05 | 432.96 | 435.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 431.10 | 435.69 | 436.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 444.20 | 435.85 | 435.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 444.20 | 435.85 | 435.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 484.30 | 446.29 | 440.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 11:15:00 | 496.30 | 496.61 | 478.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:45:00 | 494.65 | 496.61 | 478.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 479.50 | 492.20 | 483.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 526.00 | 490.14 | 488.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 509.75 | 512.45 | 509.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 499.80 | 506.54 | 507.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 499.80 | 506.54 | 507.35 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 515.15 | 508.44 | 507.71 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 499.90 | 506.50 | 507.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 10:15:00 | 498.15 | 502.42 | 504.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 512.20 | 502.60 | 503.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 512.20 | 502.60 | 503.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 512.20 | 502.60 | 503.46 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 509.55 | 505.11 | 504.52 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 502.10 | 505.32 | 505.51 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 508.95 | 506.04 | 505.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 10:15:00 | 510.70 | 506.97 | 506.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 507.10 | 507.29 | 506.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 13:15:00 | 507.10 | 507.29 | 506.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 507.10 | 507.29 | 506.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:15:00 | 505.50 | 507.29 | 506.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 498.40 | 505.51 | 505.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 496.35 | 502.04 | 504.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 13:15:00 | 489.80 | 488.83 | 493.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:45:00 | 490.55 | 488.83 | 493.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 492.70 | 489.61 | 493.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 492.70 | 489.61 | 493.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 494.00 | 490.48 | 493.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 493.40 | 490.48 | 493.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 491.90 | 490.77 | 493.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:15:00 | 485.15 | 490.46 | 492.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 487.00 | 489.78 | 491.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 499.80 | 491.96 | 492.03 | SL hit (close>static) qty=1.00 sl=495.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 499.85 | 493.54 | 492.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 502.00 | 496.20 | 494.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 14:15:00 | 508.70 | 510.20 | 505.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 15:00:00 | 508.70 | 510.20 | 505.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 510.80 | 509.88 | 506.29 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 502.60 | 505.29 | 505.42 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 511.00 | 506.44 | 505.93 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 15:15:00 | 502.10 | 505.87 | 506.38 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 509.95 | 506.72 | 506.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 515.10 | 508.39 | 507.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 10:30:00 | 348.22 | 2024-05-17 15:15:00 | 344.32 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-05-17 11:15:00 | 348.98 | 2024-05-17 15:15:00 | 344.32 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-05-17 12:30:00 | 348.00 | 2024-05-17 15:15:00 | 344.32 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-06-03 10:15:00 | 345.41 | 2024-06-05 09:15:00 | 360.01 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2024-06-03 11:45:00 | 343.60 | 2024-06-05 09:15:00 | 360.01 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2024-06-04 15:15:00 | 344.00 | 2024-06-05 09:15:00 | 360.01 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2024-06-12 09:15:00 | 389.63 | 2024-06-13 14:15:00 | 372.17 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2024-07-10 13:00:00 | 426.98 | 2024-07-16 14:15:00 | 427.75 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-07-11 09:45:00 | 424.00 | 2024-07-16 14:15:00 | 427.75 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-07-11 10:45:00 | 428.40 | 2024-07-16 14:15:00 | 427.75 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-07-23 10:15:00 | 453.20 | 2024-07-31 14:15:00 | 476.40 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2024-07-23 12:45:00 | 453.60 | 2024-07-31 14:15:00 | 476.40 | STOP_HIT | 1.00 | 5.03% |
| BUY | retest2 | 2024-07-23 13:15:00 | 454.60 | 2024-07-31 14:15:00 | 476.40 | STOP_HIT | 1.00 | 4.80% |
| SELL | retest2 | 2024-08-06 13:30:00 | 449.82 | 2024-08-07 13:15:00 | 464.00 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-08-19 13:30:00 | 442.06 | 2024-08-20 13:15:00 | 451.90 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-08-19 14:00:00 | 440.80 | 2024-08-20 13:15:00 | 451.90 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-08-20 09:15:00 | 442.29 | 2024-08-20 13:15:00 | 451.90 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-08-20 09:45:00 | 439.00 | 2024-08-20 13:15:00 | 451.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-09-17 11:15:00 | 435.94 | 2024-09-19 13:15:00 | 414.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 11:15:00 | 435.94 | 2024-09-20 12:15:00 | 422.42 | STOP_HIT | 0.50 | 3.10% |
| BUY | retest2 | 2024-11-07 09:15:00 | 391.14 | 2024-11-11 09:15:00 | 385.96 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-07 10:45:00 | 393.00 | 2024-11-11 09:15:00 | 385.96 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-11-18 12:15:00 | 386.66 | 2024-11-18 15:15:00 | 382.33 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-11-19 09:15:00 | 390.00 | 2024-11-22 09:15:00 | 385.42 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-11-21 13:15:00 | 386.87 | 2024-11-22 09:15:00 | 385.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-11-22 09:15:00 | 387.16 | 2024-11-22 09:15:00 | 385.42 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-12-02 11:15:00 | 409.21 | 2024-12-06 09:15:00 | 407.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-12-02 12:45:00 | 408.55 | 2024-12-06 09:15:00 | 407.90 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-12-02 14:30:00 | 408.00 | 2024-12-06 09:15:00 | 407.90 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-12-10 11:45:00 | 408.86 | 2024-12-10 13:15:00 | 411.29 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-12-20 12:45:00 | 388.76 | 2024-12-31 10:15:00 | 387.98 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-01-08 09:15:00 | 389.73 | 2025-01-13 11:15:00 | 371.10 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2025-01-08 10:30:00 | 390.58 | 2025-01-13 12:15:00 | 370.24 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-01-08 11:30:00 | 390.14 | 2025-01-13 12:15:00 | 371.05 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-01-09 11:00:00 | 390.63 | 2025-01-13 12:15:00 | 370.63 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2025-01-09 13:00:00 | 389.00 | 2025-01-13 12:15:00 | 369.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:30:00 | 387.98 | 2025-01-13 12:15:00 | 368.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 389.73 | 2025-01-14 15:15:00 | 376.00 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2025-01-08 10:30:00 | 390.58 | 2025-01-14 15:15:00 | 376.00 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-01-08 11:30:00 | 390.14 | 2025-01-14 15:15:00 | 376.00 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-01-09 11:00:00 | 390.63 | 2025-01-14 15:15:00 | 376.00 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2025-01-09 13:00:00 | 389.00 | 2025-01-14 15:15:00 | 376.00 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2025-01-09 14:30:00 | 387.98 | 2025-01-14 15:15:00 | 376.00 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-01-10 09:15:00 | 385.43 | 2025-01-15 14:15:00 | 379.52 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2025-01-21 10:15:00 | 373.66 | 2025-01-27 09:15:00 | 354.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 368.68 | 2025-01-27 09:15:00 | 350.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 373.66 | 2025-01-28 12:15:00 | 349.13 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-01-22 09:15:00 | 368.68 | 2025-01-28 12:15:00 | 349.13 | STOP_HIT | 0.50 | 5.30% |
| BUY | retest2 | 2025-02-01 14:45:00 | 365.86 | 2025-02-04 12:15:00 | 351.13 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-02-12 09:15:00 | 344.50 | 2025-02-17 09:15:00 | 327.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 09:15:00 | 344.50 | 2025-02-17 14:15:00 | 335.10 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-02-27 09:15:00 | 324.33 | 2025-03-03 09:15:00 | 308.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 324.33 | 2025-03-03 14:15:00 | 310.15 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest1 | 2025-03-06 09:15:00 | 322.04 | 2025-03-10 15:15:00 | 318.05 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-03-10 10:30:00 | 324.24 | 2025-03-11 09:15:00 | 317.30 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-03-10 14:45:00 | 323.23 | 2025-03-11 09:15:00 | 317.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-03-13 11:15:00 | 322.44 | 2025-03-13 12:15:00 | 319.53 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-03-25 12:30:00 | 336.62 | 2025-03-26 09:15:00 | 330.82 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-04-02 12:15:00 | 340.76 | 2025-04-04 11:15:00 | 337.10 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-04-03 13:15:00 | 340.70 | 2025-04-04 11:15:00 | 337.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-04-03 13:45:00 | 340.49 | 2025-04-04 11:15:00 | 337.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-07 15:15:00 | 333.18 | 2025-04-08 09:15:00 | 344.60 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2025-04-11 09:15:00 | 352.50 | 2025-04-21 11:15:00 | 350.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-04-15 11:00:00 | 348.30 | 2025-04-21 11:15:00 | 350.00 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-04-15 11:30:00 | 348.06 | 2025-04-21 11:15:00 | 350.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-04-16 09:15:00 | 350.64 | 2025-04-21 12:15:00 | 348.64 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-04-17 10:30:00 | 355.00 | 2025-04-21 12:15:00 | 348.64 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-04-17 11:15:00 | 355.08 | 2025-04-21 12:15:00 | 348.64 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-04-17 12:00:00 | 355.52 | 2025-04-21 12:15:00 | 348.64 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-04-24 09:30:00 | 354.78 | 2025-04-25 12:15:00 | 352.90 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-16 10:15:00 | 363.02 | 2025-05-21 10:15:00 | 399.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-16 11:30:00 | 365.00 | 2025-05-21 10:15:00 | 401.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-30 14:30:00 | 389.42 | 2025-06-02 09:15:00 | 397.46 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-06-10 09:30:00 | 390.40 | 2025-06-10 13:15:00 | 385.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-06-11 09:15:00 | 390.54 | 2025-06-12 14:15:00 | 387.38 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-12 12:30:00 | 390.92 | 2025-06-12 14:15:00 | 387.38 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-30 09:15:00 | 410.54 | 2025-06-30 12:15:00 | 403.82 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-07-07 09:30:00 | 397.66 | 2025-07-09 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-07 10:45:00 | 397.82 | 2025-07-09 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-07 11:30:00 | 397.60 | 2025-07-09 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-08 09:30:00 | 397.50 | 2025-07-09 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-07-11 09:15:00 | 406.46 | 2025-07-16 13:15:00 | 401.68 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-11 14:30:00 | 403.96 | 2025-07-16 13:15:00 | 401.68 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-14 10:15:00 | 404.68 | 2025-07-16 13:15:00 | 401.68 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-14 11:00:00 | 404.90 | 2025-07-16 13:15:00 | 401.68 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-15 09:15:00 | 405.80 | 2025-07-16 13:15:00 | 401.68 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest1 | 2025-07-21 13:15:00 | 419.32 | 2025-07-22 10:15:00 | 415.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-25 12:30:00 | 421.42 | 2025-07-28 09:15:00 | 418.38 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-25 13:30:00 | 422.10 | 2025-07-28 09:15:00 | 418.38 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-01 09:30:00 | 402.64 | 2025-08-01 10:15:00 | 416.06 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-08-08 12:15:00 | 390.38 | 2025-08-13 10:15:00 | 392.48 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-08-12 13:30:00 | 391.00 | 2025-08-13 10:15:00 | 392.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-01 09:15:00 | 416.20 | 2025-09-03 11:15:00 | 457.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 479.60 | 2025-09-29 14:15:00 | 490.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-09-29 15:15:00 | 477.00 | 2025-10-06 11:15:00 | 453.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 15:15:00 | 477.00 | 2025-10-07 10:15:00 | 455.75 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-10-15 11:45:00 | 453.25 | 2025-10-15 14:15:00 | 464.35 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-10-20 12:30:00 | 482.30 | 2025-10-31 10:15:00 | 482.80 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-10-21 13:45:00 | 483.45 | 2025-10-31 10:15:00 | 482.80 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-10-24 14:00:00 | 481.25 | 2025-10-31 10:15:00 | 482.80 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-04 09:15:00 | 478.15 | 2025-11-06 09:15:00 | 454.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 478.15 | 2025-11-07 14:15:00 | 449.60 | STOP_HIT | 0.50 | 5.97% |
| SELL | retest2 | 2025-11-17 09:15:00 | 450.80 | 2025-11-19 14:15:00 | 455.45 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-18 09:15:00 | 451.10 | 2025-11-19 14:15:00 | 455.45 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-11-19 12:00:00 | 452.55 | 2025-11-19 14:15:00 | 455.45 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-25 09:15:00 | 442.40 | 2025-12-03 09:15:00 | 420.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 09:15:00 | 442.40 | 2025-12-03 14:15:00 | 421.75 | STOP_HIT | 0.50 | 4.67% |
| BUY | retest2 | 2025-12-12 15:15:00 | 429.50 | 2025-12-18 10:15:00 | 424.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-29 13:00:00 | 422.00 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest2 | 2025-12-30 09:15:00 | 422.20 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -8.98% |
| SELL | retest2 | 2025-12-30 15:15:00 | 422.25 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -8.96% |
| SELL | retest2 | 2026-01-19 09:45:00 | 432.00 | 2026-01-21 09:15:00 | 410.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:45:00 | 432.00 | 2026-01-21 11:15:00 | 420.30 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2026-01-27 12:30:00 | 420.20 | 2026-01-27 15:15:00 | 431.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-01-28 09:15:00 | 421.35 | 2026-01-29 13:15:00 | 426.85 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-01-28 14:00:00 | 421.75 | 2026-01-29 13:15:00 | 426.85 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-28 15:00:00 | 421.55 | 2026-01-29 13:15:00 | 426.85 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-29 09:45:00 | 421.25 | 2026-01-29 14:15:00 | 430.90 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-01-29 12:15:00 | 420.55 | 2026-01-29 14:15:00 | 430.90 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-01-29 12:45:00 | 419.60 | 2026-01-29 14:15:00 | 430.90 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-02-01 15:00:00 | 441.25 | 2026-02-02 09:15:00 | 431.25 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-02-20 15:15:00 | 398.50 | 2026-02-23 09:15:00 | 402.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-03-05 14:15:00 | 378.40 | 2026-03-05 14:15:00 | 382.85 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-03-16 15:15:00 | 425.00 | 2026-03-19 15:15:00 | 412.10 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-17 14:45:00 | 426.45 | 2026-03-19 15:15:00 | 412.10 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2026-03-18 09:45:00 | 425.70 | 2026-03-19 15:15:00 | 412.10 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2026-04-01 11:00:00 | 432.05 | 2026-04-02 14:15:00 | 444.20 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-04-01 15:15:00 | 431.10 | 2026-04-02 14:15:00 | 444.20 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-04-13 09:15:00 | 526.00 | 2026-04-16 12:15:00 | 499.80 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest2 | 2026-04-16 09:15:00 | 509.75 | 2026-04-16 12:15:00 | 499.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-04-28 12:15:00 | 485.15 | 2026-04-29 12:15:00 | 499.80 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-04-29 10:15:00 | 487.00 | 2026-04-29 12:15:00 | 499.80 | STOP_HIT | 1.00 | -2.63% |

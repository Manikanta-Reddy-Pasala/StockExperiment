# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2026-01-19 09:15:00 → 2026-05-08 15:15:00 (518 bars)
- **Last close:** 387.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 21 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT2_SKIP | 5 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 15 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 16
- **Target hits / Stop hits / Partials:** 0 / 18 / 2
- **Avg / median % per leg:** -0.60% / -1.63%
- **Sum % (uncompounded):** -11.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.73% | -8.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.10% | -2.1% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.64% | -6.6% |
| SELL (all) | 15 | 4 | 26.7% | 0 | 13 | 2 | -0.22% | -3.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.66% | -7.3% |
| SELL @ 3rd Alert (retest2) | 13 | 4 | 30.8% | 0 | 11 | 2 | 0.31% | 4.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.14% | -9.4% |
| retest2 (combined) | 17 | 4 | 23.5% | 0 | 15 | 2 | -0.15% | -2.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 431.90 | 425.69 | 425.29 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 425.25 | 428.64 | 428.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 421.65 | 426.53 | 427.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 13:15:00 | 427.65 | 423.68 | 425.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 427.65 | 423.68 | 425.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 427.65 | 423.68 | 425.79 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 447.70 | 430.71 | 428.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 13:15:00 | 448.20 | 440.11 | 434.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 10:15:00 | 445.15 | 445.42 | 439.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 468.60 | 457.02 | 451.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 458.75 | 462.83 | 461.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 458.75 | 462.83 | 461.47 | SL hit (close<ema400) qty=1.00 sl=461.47 alert=retest1 |

### Cycle 4 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 458.50 | 461.17 | 461.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 455.75 | 460.08 | 460.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 449.20 | 448.97 | 451.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 449.20 | 448.97 | 451.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 452.00 | 449.57 | 451.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 452.00 | 449.57 | 451.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 451.90 | 450.04 | 451.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 447.00 | 450.04 | 451.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 454.30 | 450.57 | 451.05 | SL hit (close>static) qty=1.00 sl=452.60 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 454.70 | 451.40 | 451.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 455.35 | 453.51 | 452.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 446.85 | 453.55 | 453.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 446.85 | 453.55 | 453.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 446.85 | 453.55 | 453.00 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 447.45 | 452.33 | 452.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 444.90 | 450.85 | 451.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 435.50 | 433.59 | 438.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 440.35 | 434.94 | 438.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 440.35 | 434.94 | 438.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 440.35 | 434.94 | 438.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 438.50 | 435.65 | 438.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 438.10 | 435.65 | 438.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:45:00 | 437.60 | 436.02 | 438.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:30:00 | 437.15 | 436.99 | 438.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 441.90 | 438.63 | 439.10 | SL hit (close>static) qty=1.00 sl=441.40 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 441.45 | 439.61 | 439.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 447.45 | 441.50 | 440.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 439.00 | 441.96 | 440.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 439.00 | 441.96 | 440.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 439.00 | 441.96 | 440.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 439.00 | 441.96 | 440.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 438.30 | 441.23 | 440.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:15:00 | 435.40 | 441.23 | 440.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 434.15 | 439.81 | 440.03 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 443.85 | 440.00 | 439.50 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 438.50 | 439.41 | 439.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 429.85 | 437.50 | 438.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 409.25 | 408.46 | 417.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 402.25 | 408.46 | 417.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 401.60 | 407.32 | 416.39 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 416.65 | 408.43 | 413.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 416.65 | 408.43 | 413.95 | SL hit (close>ema400) qty=1.00 sl=413.95 alert=retest1 |

### Cycle 11 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 336.10 | 330.91 | 330.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 344.75 | 333.68 | 331.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 340.75 | 342.22 | 339.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 340.75 | 342.22 | 339.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 340.75 | 342.22 | 339.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 340.75 | 342.22 | 339.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 338.35 | 341.13 | 339.40 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 335.50 | 338.49 | 338.62 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 344.40 | 339.06 | 338.72 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 335.00 | 338.25 | 338.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 319.50 | 333.98 | 336.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 12:15:00 | 326.70 | 324.13 | 328.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 13:00:00 | 326.70 | 324.13 | 328.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 327.50 | 324.80 | 328.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:30:00 | 326.00 | 324.80 | 328.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 327.65 | 325.37 | 327.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:30:00 | 328.95 | 325.37 | 327.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 328.85 | 326.07 | 328.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 324.20 | 326.07 | 328.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 332.00 | 325.78 | 326.68 | SL hit (close>static) qty=1.00 sl=330.60 alert=retest2 |

### Cycle 15 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 358.45 | 332.93 | 329.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 365.70 | 343.76 | 335.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 357.40 | 357.74 | 349.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 356.50 | 357.74 | 349.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 348.50 | 357.52 | 355.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 349.80 | 357.52 | 355.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 12:15:00 | 347.35 | 352.38 | 353.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 347.35 | 352.38 | 353.01 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 365.35 | 353.90 | 353.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 368.60 | 360.48 | 356.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 369.55 | 370.33 | 367.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:45:00 | 369.05 | 370.33 | 367.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 382.20 | 382.58 | 378.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 384.10 | 382.61 | 378.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 374.45 | 381.47 | 380.13 | SL hit (close<static) qty=1.00 sl=376.30 alert=retest2 |

### Cycle 18 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 374.30 | 378.55 | 378.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 371.90 | 374.85 | 376.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 374.80 | 373.75 | 375.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 376.20 | 373.75 | 375.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 381.75 | 375.35 | 376.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 383.95 | 375.35 | 376.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 381.70 | 376.62 | 376.56 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 372.25 | 378.72 | 378.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 370.10 | 377.00 | 378.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 375.55 | 375.00 | 376.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 14:45:00 | 376.30 | 375.00 | 376.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 376.10 | 375.16 | 376.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 372.90 | 375.60 | 376.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 373.25 | 374.95 | 375.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 369.15 | 374.85 | 375.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 13:15:00 | 373.35 | 372.55 | 374.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 373.40 | 372.72 | 374.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 381.15 | 375.85 | 375.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 381.15 | 375.85 | 375.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 382.20 | 377.12 | 375.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 391.40 | 393.77 | 388.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 391.75 | 393.77 | 388.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 388.85 | 392.40 | 388.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 388.85 | 392.40 | 388.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 388.45 | 391.61 | 388.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 388.45 | 391.61 | 388.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 385.30 | 390.35 | 388.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 385.30 | 390.35 | 388.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 386.45 | 389.57 | 388.36 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-05 09:15:00 | 468.60 | 2026-02-10 09:15:00 | 458.75 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-11 09:15:00 | 465.65 | 2026-02-12 09:15:00 | 458.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-02-11 10:15:00 | 467.00 | 2026-02-12 09:15:00 | 458.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-02-17 09:15:00 | 447.00 | 2026-02-17 14:15:00 | 454.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-23 12:15:00 | 438.10 | 2026-02-24 10:15:00 | 441.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-23 12:45:00 | 437.60 | 2026-02-24 10:15:00 | 441.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-23 14:30:00 | 437.15 | 2026-02-24 10:15:00 | 441.90 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest1 | 2026-03-05 10:15:00 | 402.25 | 2026-03-05 14:15:00 | 416.65 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest1 | 2026-03-05 11:15:00 | 401.60 | 2026-03-05 14:15:00 | 416.65 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2026-03-06 12:15:00 | 409.30 | 2026-03-09 09:15:00 | 388.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 405.85 | 2026-03-09 09:15:00 | 385.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 409.30 | 2026-03-10 13:15:00 | 385.80 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2026-03-06 14:45:00 | 405.85 | 2026-03-10 13:15:00 | 385.80 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2026-04-07 09:15:00 | 324.20 | 2026-04-07 14:15:00 | 332.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-04-13 10:15:00 | 349.80 | 2026-04-13 12:15:00 | 347.35 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-04-22 11:15:00 | 384.10 | 2026-04-23 09:15:00 | 374.45 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-05-04 13:15:00 | 372.90 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-05-04 15:00:00 | 373.25 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-05-05 09:15:00 | 369.15 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-05-05 13:15:00 | 373.35 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -2.09% |

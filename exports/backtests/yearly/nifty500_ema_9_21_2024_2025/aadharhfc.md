# Aadhar Housing Finance Ltd. (AADHARHFC)

## Backtest Summary

- **Window:** 2024-05-15 09:15:00 → 2026-05-08 15:15:00 (3430 bars)
- **Last close:** 502.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 192 |
| ALERT1 | 114 |
| ALERT2 | 113 |
| ALERT2_SKIP | 69 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 82 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 70
- **Target hits / Stop hits / Partials:** 3 / 84 / 6
- **Avg / median % per leg:** -0.06% / -0.78%
- **Sum % (uncompounded):** -5.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 9 | 21.4% | 3 | 39 | 0 | 0.01% | 0.3% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.22% | -0.9% |
| BUY @ 3rd Alert (retest2) | 38 | 8 | 21.1% | 3 | 35 | 0 | 0.03% | 1.2% |
| SELL (all) | 51 | 14 | 27.5% | 0 | 45 | 6 | -0.11% | -5.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.78% | -0.8% |
| SELL @ 3rd Alert (retest2) | 50 | 14 | 28.0% | 0 | 44 | 6 | -0.10% | -5.1% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.33% | -1.7% |
| retest2 (combined) | 88 | 22 | 25.0% | 3 | 79 | 6 | -0.04% | -3.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 337.65 | 344.61 | 344.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 10:15:00 | 336.70 | 340.87 | 342.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 343.00 | 340.27 | 341.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 343.00 | 340.27 | 341.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 343.00 | 340.27 | 341.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 343.00 | 340.27 | 341.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 342.35 | 340.68 | 341.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 343.20 | 340.68 | 341.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 339.60 | 340.47 | 341.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 338.25 | 340.04 | 341.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:15:00 | 336.75 | 339.48 | 340.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 15:15:00 | 340.50 | 339.42 | 339.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 15:15:00 | 340.50 | 339.42 | 339.31 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 335.05 | 338.55 | 338.93 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 15:15:00 | 343.00 | 338.44 | 338.40 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 338.00 | 338.35 | 338.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 10:15:00 | 335.00 | 337.68 | 338.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 10:15:00 | 334.35 | 333.68 | 335.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 11:00:00 | 334.35 | 333.68 | 335.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 336.80 | 334.31 | 335.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:00:00 | 336.80 | 334.31 | 335.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 335.50 | 334.55 | 335.49 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 15:15:00 | 338.05 | 336.14 | 336.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 341.50 | 337.21 | 336.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 339.05 | 339.87 | 338.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 15:00:00 | 339.05 | 339.87 | 338.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 340.50 | 340.10 | 338.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 347.50 | 341.51 | 340.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 12:30:00 | 346.00 | 343.33 | 341.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 13:45:00 | 344.10 | 343.66 | 341.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 335.05 | 342.92 | 342.05 | SL hit (close<static) qty=1.00 sl=338.45 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 316.30 | 337.60 | 339.71 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 351.00 | 338.36 | 336.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 358.00 | 346.77 | 342.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 388.05 | 389.33 | 378.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 388.05 | 389.33 | 378.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 381.70 | 387.80 | 378.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:45:00 | 376.60 | 387.80 | 378.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 379.95 | 386.23 | 378.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 381.75 | 386.23 | 378.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 380.50 | 385.08 | 378.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 12:00:00 | 391.00 | 385.47 | 379.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-14 10:15:00 | 430.10 | 402.51 | 393.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 402.60 | 409.32 | 409.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 11:15:00 | 401.75 | 406.48 | 407.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 396.90 | 396.11 | 400.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 14:00:00 | 396.90 | 396.11 | 400.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 395.65 | 396.04 | 399.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 398.80 | 396.04 | 399.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 393.05 | 390.24 | 392.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 396.00 | 390.24 | 392.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 402.00 | 392.59 | 393.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 401.20 | 392.59 | 393.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 400.85 | 394.24 | 394.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 12:15:00 | 404.60 | 397.52 | 395.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 408.25 | 411.28 | 405.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 408.25 | 411.28 | 405.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 408.25 | 411.28 | 405.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 408.25 | 411.28 | 405.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 410.00 | 411.40 | 407.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:30:00 | 415.10 | 409.69 | 407.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 430.20 | 433.84 | 433.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 430.20 | 433.84 | 433.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 426.95 | 431.88 | 432.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 431.75 | 431.25 | 432.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 431.75 | 431.25 | 432.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 431.75 | 431.25 | 432.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 431.75 | 431.25 | 432.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 429.75 | 430.95 | 432.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:30:00 | 432.20 | 430.95 | 432.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 421.80 | 428.81 | 430.99 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 434.50 | 427.76 | 427.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 12:15:00 | 443.10 | 432.64 | 429.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 433.00 | 435.28 | 432.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 433.00 | 435.28 | 432.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 433.00 | 435.28 | 432.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 427.60 | 435.28 | 432.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 431.75 | 434.57 | 432.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 429.30 | 434.57 | 432.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 437.15 | 435.09 | 432.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:45:00 | 437.70 | 435.80 | 433.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 14:00:00 | 446.00 | 437.84 | 434.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-18 09:15:00 | 481.47 | 460.64 | 450.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 445.55 | 453.88 | 454.92 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 476.00 | 457.68 | 456.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 478.30 | 464.58 | 459.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 470.60 | 474.13 | 468.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 470.60 | 474.13 | 468.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 470.60 | 474.13 | 468.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 465.05 | 474.13 | 468.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 465.40 | 472.39 | 468.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 466.40 | 472.39 | 468.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 458.20 | 469.55 | 467.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 458.20 | 469.55 | 467.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 09:15:00 | 459.00 | 465.81 | 466.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 10:15:00 | 455.70 | 460.65 | 462.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 451.95 | 447.16 | 451.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 451.95 | 447.16 | 451.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 451.95 | 447.16 | 451.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:15:00 | 456.45 | 447.16 | 451.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 456.50 | 449.03 | 451.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:30:00 | 457.50 | 449.03 | 451.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 461.30 | 451.48 | 452.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 461.30 | 451.48 | 452.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 14:15:00 | 459.00 | 454.35 | 453.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 15:15:00 | 460.05 | 455.49 | 454.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 10:15:00 | 452.65 | 455.51 | 454.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 10:15:00 | 452.65 | 455.51 | 454.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 452.65 | 455.51 | 454.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 452.65 | 455.51 | 454.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 447.05 | 453.82 | 453.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 10:15:00 | 444.60 | 449.78 | 451.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 15:15:00 | 450.00 | 447.41 | 449.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 15:15:00 | 450.00 | 447.41 | 449.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 450.00 | 447.41 | 449.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:00:00 | 441.55 | 445.73 | 448.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 13:15:00 | 441.50 | 445.12 | 447.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 419.47 | 428.70 | 435.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 419.42 | 428.70 | 435.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 416.35 | 413.47 | 422.79 | SL hit (close>ema200) qty=0.50 sl=413.47 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 424.95 | 416.47 | 416.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 428.05 | 423.18 | 420.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 422.40 | 424.84 | 422.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 422.40 | 424.84 | 422.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 422.40 | 424.84 | 422.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 422.40 | 424.84 | 422.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 429.20 | 425.71 | 422.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 423.55 | 425.71 | 422.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 422.85 | 425.14 | 422.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:45:00 | 423.65 | 425.14 | 422.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 421.50 | 424.41 | 422.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:00:00 | 421.50 | 424.41 | 422.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 422.80 | 424.09 | 422.68 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 405.35 | 419.58 | 420.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 11:15:00 | 401.75 | 413.55 | 417.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 399.45 | 393.81 | 399.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 11:15:00 | 399.45 | 393.81 | 399.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 399.45 | 393.81 | 399.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 399.45 | 393.81 | 399.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 395.15 | 394.08 | 399.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:15:00 | 393.25 | 394.39 | 398.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:30:00 | 393.50 | 393.57 | 397.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:45:00 | 393.75 | 393.53 | 397.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 393.40 | 393.88 | 396.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 394.40 | 394.05 | 395.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 12:45:00 | 391.60 | 393.93 | 395.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 09:45:00 | 391.25 | 392.31 | 394.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 399.60 | 393.77 | 394.55 | SL hit (close>static) qty=1.00 sl=399.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 403.10 | 395.64 | 395.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 406.00 | 399.68 | 397.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 406.40 | 406.51 | 403.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 406.40 | 406.51 | 403.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 406.40 | 406.51 | 403.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 404.15 | 406.51 | 403.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 403.30 | 405.51 | 403.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 403.30 | 405.51 | 403.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 399.90 | 404.38 | 403.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 400.95 | 404.38 | 403.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 400.90 | 403.69 | 403.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:30:00 | 400.45 | 403.69 | 403.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 395.60 | 401.48 | 402.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 389.90 | 396.58 | 398.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 10:15:00 | 391.35 | 390.14 | 393.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 391.35 | 390.14 | 393.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 391.35 | 390.14 | 393.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 392.05 | 390.14 | 393.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 384.00 | 385.40 | 389.05 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 406.70 | 391.25 | 389.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 11:15:00 | 423.60 | 400.54 | 393.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 426.05 | 437.22 | 430.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 426.05 | 437.22 | 430.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 426.05 | 437.22 | 430.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 425.60 | 437.22 | 430.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 428.20 | 435.42 | 430.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 15:15:00 | 429.00 | 429.59 | 428.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-13 09:15:00 | 471.90 | 454.07 | 445.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 453.80 | 458.95 | 459.33 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 12:15:00 | 464.00 | 459.87 | 459.51 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 456.70 | 459.22 | 459.30 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 462.05 | 459.43 | 459.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 15:15:00 | 465.15 | 461.31 | 460.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 10:15:00 | 501.90 | 503.06 | 492.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:45:00 | 501.35 | 503.06 | 492.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 494.00 | 502.69 | 497.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 494.00 | 502.69 | 497.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 494.00 | 500.95 | 497.10 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 478.00 | 492.40 | 493.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 470.85 | 485.78 | 489.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 446.50 | 446.34 | 454.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 446.50 | 446.34 | 454.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 446.55 | 435.57 | 441.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 446.55 | 435.57 | 441.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 457.05 | 439.86 | 442.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 457.05 | 439.86 | 442.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 451.30 | 444.48 | 444.30 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 12:15:00 | 440.40 | 444.52 | 444.88 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 458.00 | 444.85 | 443.61 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 446.65 | 449.73 | 450.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 444.45 | 448.22 | 449.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 437.40 | 437.31 | 441.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:15:00 | 439.45 | 437.31 | 441.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 435.75 | 437.45 | 440.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 431.70 | 436.26 | 439.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 15:15:00 | 447.00 | 434.54 | 436.94 | SL hit (close>static) qty=1.00 sl=445.20 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 424.05 | 420.53 | 420.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 425.60 | 421.97 | 421.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 452.10 | 452.12 | 443.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 452.10 | 452.12 | 443.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 442.05 | 448.87 | 444.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:30:00 | 441.70 | 448.87 | 444.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 438.40 | 446.78 | 444.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:45:00 | 439.60 | 446.78 | 444.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 445.80 | 444.89 | 444.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 445.80 | 444.89 | 444.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 448.55 | 445.62 | 444.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 13:00:00 | 456.90 | 449.72 | 448.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:45:00 | 455.95 | 462.32 | 461.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 11:15:00 | 455.30 | 459.70 | 459.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 455.30 | 459.70 | 459.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 439.75 | 455.71 | 458.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 444.50 | 430.07 | 434.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 444.50 | 430.07 | 434.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 444.50 | 430.07 | 434.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 444.50 | 430.07 | 434.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 439.60 | 431.97 | 435.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:30:00 | 435.05 | 432.66 | 435.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:15:00 | 435.50 | 433.61 | 435.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 435.90 | 434.32 | 435.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 413.30 | 423.23 | 426.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 413.72 | 423.23 | 426.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 414.10 | 423.23 | 426.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-21 11:15:00 | 424.90 | 423.09 | 425.66 | SL hit (close>ema200) qty=0.50 sl=423.09 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 10:15:00 | 425.90 | 412.43 | 412.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 434.40 | 421.85 | 417.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 421.05 | 423.01 | 419.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 12:15:00 | 421.05 | 423.01 | 419.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 421.05 | 423.01 | 419.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 421.05 | 423.01 | 419.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 419.90 | 422.39 | 419.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 419.90 | 422.39 | 419.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 418.05 | 421.52 | 419.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 418.05 | 421.52 | 419.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 418.00 | 420.82 | 419.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 421.05 | 420.82 | 419.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 431.45 | 434.97 | 435.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 431.45 | 434.97 | 435.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 428.00 | 430.51 | 432.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 429.15 | 427.72 | 429.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 429.15 | 427.72 | 429.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 429.15 | 427.72 | 429.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:15:00 | 432.20 | 427.72 | 429.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 428.55 | 427.89 | 429.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:30:00 | 427.40 | 426.89 | 428.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 430.40 | 425.34 | 425.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 11:15:00 | 430.40 | 425.34 | 425.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 12:15:00 | 434.15 | 427.10 | 426.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 09:15:00 | 435.15 | 436.29 | 433.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 435.15 | 436.29 | 433.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 435.15 | 436.29 | 433.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 436.65 | 436.29 | 433.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 430.00 | 435.03 | 432.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 430.00 | 435.03 | 432.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 432.00 | 434.43 | 432.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 433.75 | 433.95 | 432.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 437.00 | 434.33 | 433.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 430.00 | 432.99 | 433.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 430.00 | 432.99 | 433.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 15:15:00 | 428.50 | 431.69 | 432.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 432.35 | 431.82 | 432.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 432.35 | 431.82 | 432.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 432.35 | 431.82 | 432.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 432.35 | 431.82 | 432.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 434.35 | 432.33 | 432.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 434.35 | 432.33 | 432.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 444.65 | 434.79 | 433.72 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 424.20 | 433.39 | 433.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 15:15:00 | 422.00 | 423.75 | 425.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 11:15:00 | 423.75 | 423.48 | 424.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 11:45:00 | 423.40 | 423.48 | 424.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 424.80 | 423.74 | 424.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:45:00 | 424.85 | 423.74 | 424.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 424.30 | 423.85 | 424.75 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 425.35 | 425.17 | 425.17 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 424.45 | 425.02 | 425.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 424.05 | 424.87 | 425.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 425.00 | 424.68 | 424.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 11:15:00 | 425.00 | 424.68 | 424.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 425.00 | 424.68 | 424.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:45:00 | 425.00 | 424.68 | 424.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 425.40 | 424.83 | 424.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 425.40 | 424.83 | 424.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 424.80 | 424.82 | 424.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 424.80 | 424.82 | 424.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 436.85 | 427.23 | 426.01 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 422.90 | 425.60 | 425.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 14:15:00 | 419.25 | 423.22 | 424.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 421.90 | 421.43 | 422.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 15:15:00 | 421.90 | 421.43 | 422.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 421.90 | 421.43 | 422.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 425.00 | 421.43 | 422.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 426.10 | 422.37 | 422.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 426.10 | 422.37 | 422.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 428.70 | 423.63 | 423.40 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 421.25 | 424.21 | 424.37 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 425.00 | 423.85 | 423.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 15:15:00 | 425.70 | 424.22 | 423.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 423.00 | 423.98 | 423.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 423.00 | 423.98 | 423.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 423.00 | 423.98 | 423.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 423.00 | 423.98 | 423.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 421.70 | 423.52 | 423.65 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 433.80 | 424.58 | 423.68 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 420.70 | 423.26 | 423.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 402.60 | 418.64 | 421.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 400.75 | 397.90 | 401.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 400.75 | 397.90 | 401.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 400.75 | 397.90 | 401.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 399.35 | 398.29 | 401.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 398.60 | 397.98 | 398.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 11:00:00 | 399.55 | 398.30 | 398.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 404.85 | 399.61 | 399.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 404.85 | 399.61 | 399.31 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 393.65 | 399.59 | 399.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 389.90 | 393.47 | 395.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 371.05 | 366.34 | 375.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 371.05 | 366.34 | 375.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 371.05 | 366.34 | 375.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:45:00 | 375.00 | 366.34 | 375.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 394.45 | 373.55 | 376.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:45:00 | 388.95 | 373.55 | 376.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 385.00 | 375.84 | 377.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 381.75 | 375.84 | 377.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 391.25 | 380.42 | 379.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 391.25 | 380.42 | 379.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 395.10 | 387.23 | 382.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 392.15 | 392.58 | 387.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 392.15 | 392.58 | 387.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 387.90 | 391.53 | 388.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:45:00 | 392.30 | 390.45 | 388.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 393.10 | 390.45 | 388.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 393.60 | 395.26 | 394.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 392.00 | 394.00 | 394.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 09:15:00 | 392.00 | 394.00 | 394.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 390.70 | 393.34 | 393.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 392.35 | 391.62 | 392.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 14:15:00 | 392.35 | 391.62 | 392.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 392.35 | 391.62 | 392.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:30:00 | 391.75 | 391.62 | 392.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 392.00 | 391.70 | 392.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 396.20 | 391.70 | 392.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 393.75 | 392.11 | 392.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 396.20 | 392.11 | 392.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 395.20 | 393.05 | 393.05 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 391.30 | 392.82 | 392.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 15:15:00 | 389.55 | 392.17 | 392.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 09:15:00 | 395.15 | 392.76 | 392.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 395.15 | 392.76 | 392.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 395.15 | 392.76 | 392.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 395.15 | 392.76 | 392.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 394.20 | 393.05 | 393.00 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 389.40 | 392.32 | 392.68 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 409.15 | 395.37 | 393.81 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 385.15 | 395.20 | 396.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 384.50 | 391.43 | 394.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 383.85 | 383.47 | 387.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 14:45:00 | 384.00 | 383.47 | 387.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 385.40 | 382.66 | 385.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:00:00 | 385.40 | 382.66 | 385.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 388.50 | 383.83 | 385.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 388.50 | 383.83 | 385.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 388.95 | 384.85 | 385.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 387.55 | 384.85 | 385.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 390.00 | 385.88 | 386.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:15:00 | 391.80 | 385.88 | 386.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 392.70 | 387.24 | 386.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 11:15:00 | 394.10 | 388.62 | 387.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 387.40 | 390.29 | 388.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 387.40 | 390.29 | 388.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 387.40 | 390.29 | 388.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 389.35 | 390.29 | 388.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 380.70 | 388.37 | 388.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:45:00 | 379.95 | 388.37 | 388.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 380.10 | 386.72 | 387.45 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 10:15:00 | 392.80 | 387.48 | 387.11 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 11:15:00 | 383.85 | 387.66 | 387.90 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 390.05 | 388.16 | 388.00 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 385.95 | 387.85 | 387.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 15:15:00 | 384.05 | 386.58 | 387.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 09:15:00 | 382.25 | 381.79 | 383.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 382.25 | 381.79 | 383.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 382.25 | 381.79 | 383.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:30:00 | 376.95 | 379.49 | 381.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 10:15:00 | 376.45 | 379.49 | 381.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 376.20 | 379.10 | 381.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 11:45:00 | 376.55 | 378.53 | 380.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 379.45 | 371.48 | 374.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:45:00 | 380.25 | 371.48 | 374.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 379.00 | 372.99 | 374.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 386.00 | 377.25 | 376.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 386.00 | 377.25 | 376.41 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 14:15:00 | 377.50 | 379.54 | 379.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 10:15:00 | 374.25 | 378.13 | 378.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 12:15:00 | 377.65 | 377.14 | 378.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 12:15:00 | 377.65 | 377.14 | 378.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 377.65 | 377.14 | 378.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 377.65 | 377.14 | 378.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 379.95 | 377.70 | 378.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:30:00 | 380.00 | 377.70 | 378.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 380.00 | 378.16 | 378.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:30:00 | 380.15 | 378.16 | 378.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 382.50 | 379.32 | 379.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 388.00 | 382.88 | 381.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 417.10 | 417.41 | 407.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 417.10 | 417.41 | 407.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 423.45 | 428.81 | 423.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:45:00 | 423.60 | 428.81 | 423.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 423.65 | 427.78 | 423.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 423.50 | 427.78 | 423.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 424.25 | 427.07 | 423.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 424.00 | 427.07 | 423.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 424.60 | 426.58 | 423.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:45:00 | 422.50 | 426.58 | 423.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 425.95 | 426.45 | 424.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 423.85 | 426.45 | 424.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 425.10 | 426.18 | 424.13 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 416.00 | 421.92 | 422.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 15:15:00 | 413.25 | 420.19 | 421.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 422.65 | 420.68 | 421.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 422.65 | 420.68 | 421.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 422.65 | 420.68 | 421.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 423.25 | 420.68 | 421.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 425.00 | 421.54 | 422.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 425.00 | 421.54 | 422.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 421.30 | 421.50 | 422.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 424.45 | 421.50 | 422.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 421.35 | 421.41 | 421.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:30:00 | 418.15 | 420.47 | 421.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 423.70 | 420.64 | 421.33 | SL hit (close>static) qty=1.00 sl=421.95 alert=retest2 |

### Cycle 70 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 424.75 | 421.99 | 421.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 428.45 | 423.78 | 422.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 427.00 | 435.81 | 431.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 427.00 | 435.81 | 431.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 427.00 | 435.81 | 431.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 427.00 | 435.81 | 431.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 422.65 | 433.18 | 430.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:30:00 | 422.20 | 433.18 | 430.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 13:15:00 | 428.30 | 429.39 | 429.39 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 14:15:00 | 430.05 | 429.52 | 429.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 439.65 | 431.72 | 430.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 434.30 | 434.83 | 432.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:30:00 | 435.95 | 434.83 | 432.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 73 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 414.75 | 430.81 | 430.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 15:15:00 | 410.00 | 426.65 | 429.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 15:15:00 | 404.00 | 403.88 | 408.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 09:15:00 | 413.00 | 403.88 | 408.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 408.00 | 404.70 | 408.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:30:00 | 414.15 | 404.70 | 408.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 418.40 | 407.44 | 409.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 418.40 | 407.44 | 409.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 418.25 | 409.60 | 410.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 419.10 | 409.60 | 410.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 419.15 | 411.51 | 410.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 423.15 | 415.18 | 412.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 419.90 | 420.05 | 416.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 419.90 | 420.05 | 416.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 441.45 | 451.24 | 444.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 441.45 | 451.24 | 444.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 439.65 | 448.93 | 444.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:30:00 | 440.40 | 448.93 | 444.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 422.05 | 443.14 | 442.78 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 427.45 | 440.00 | 441.39 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 454.25 | 440.99 | 439.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 458.00 | 451.86 | 446.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 475.40 | 482.75 | 477.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 475.40 | 482.75 | 477.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 475.40 | 482.75 | 477.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:45:00 | 474.85 | 482.75 | 477.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 477.20 | 481.64 | 477.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 478.20 | 481.64 | 477.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 473.50 | 480.01 | 477.48 | SL hit (close<static) qty=1.00 sl=473.60 alert=retest2 |

### Cycle 77 — SELL (started 2025-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 13:15:00 | 472.10 | 476.96 | 477.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 15:15:00 | 472.00 | 475.19 | 476.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 13:15:00 | 481.65 | 474.28 | 475.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 13:15:00 | 481.65 | 474.28 | 475.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 481.65 | 474.28 | 475.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 481.65 | 474.28 | 475.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 477.95 | 475.02 | 475.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:45:00 | 486.90 | 475.02 | 475.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 478.15 | 475.64 | 475.59 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 09:15:00 | 474.80 | 475.47 | 475.52 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 476.25 | 475.61 | 475.57 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 474.20 | 475.33 | 475.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 14:15:00 | 467.50 | 473.89 | 474.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 469.75 | 462.64 | 466.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 469.75 | 462.64 | 466.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 469.75 | 462.64 | 466.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 469.75 | 462.64 | 466.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 465.20 | 463.15 | 466.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:30:00 | 471.90 | 463.15 | 466.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 467.35 | 463.99 | 466.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 467.35 | 463.99 | 466.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 466.40 | 464.47 | 466.41 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 473.25 | 468.28 | 467.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 11:15:00 | 479.55 | 471.36 | 469.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 472.25 | 473.85 | 471.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 472.25 | 473.85 | 471.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 472.25 | 473.85 | 471.62 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 465.00 | 469.87 | 470.48 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 473.15 | 470.54 | 470.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 476.00 | 471.63 | 470.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 470.75 | 473.55 | 472.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 470.75 | 473.55 | 472.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 470.75 | 473.55 | 472.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 469.05 | 473.55 | 472.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 466.35 | 472.11 | 471.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 466.35 | 472.11 | 471.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 466.55 | 471.00 | 471.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 461.40 | 469.08 | 470.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 15:15:00 | 467.05 | 466.40 | 468.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 15:15:00 | 467.05 | 466.40 | 468.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 467.05 | 466.40 | 468.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 453.80 | 466.40 | 468.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 11:15:00 | 457.05 | 449.15 | 448.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 457.05 | 449.15 | 448.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 461.75 | 453.70 | 451.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 453.90 | 454.80 | 452.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 453.90 | 454.80 | 452.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 453.90 | 454.80 | 452.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 455.40 | 454.80 | 452.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 450.90 | 454.02 | 452.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 450.90 | 454.02 | 452.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 452.30 | 453.68 | 452.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:15:00 | 452.85 | 451.85 | 451.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 10:15:00 | 449.95 | 451.47 | 451.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 449.95 | 451.47 | 451.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 13:15:00 | 447.55 | 450.41 | 450.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 452.85 | 449.91 | 450.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 452.85 | 449.91 | 450.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 452.85 | 449.91 | 450.52 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 456.10 | 451.95 | 451.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 15:15:00 | 458.40 | 454.62 | 452.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 451.05 | 453.91 | 452.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 451.05 | 453.91 | 452.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 451.05 | 453.91 | 452.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 452.00 | 453.91 | 452.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 452.60 | 453.65 | 452.74 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 445.85 | 451.07 | 451.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 444.55 | 448.95 | 450.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 441.65 | 441.40 | 443.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 10:45:00 | 441.35 | 441.40 | 443.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 443.60 | 439.10 | 441.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 443.60 | 439.10 | 441.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 445.45 | 440.37 | 441.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 445.45 | 440.37 | 441.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 446.00 | 442.35 | 442.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 449.70 | 444.76 | 443.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 448.20 | 448.40 | 446.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 448.20 | 448.40 | 446.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 448.20 | 448.40 | 446.40 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 439.40 | 446.18 | 446.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 438.00 | 443.55 | 444.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 440.15 | 437.85 | 441.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 440.15 | 437.85 | 441.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 440.15 | 437.85 | 441.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 440.15 | 437.85 | 441.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 440.10 | 438.45 | 440.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 442.65 | 438.45 | 440.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 444.60 | 439.68 | 441.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 444.60 | 439.68 | 441.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 443.60 | 440.47 | 441.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 443.85 | 440.47 | 441.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 443.55 | 441.69 | 441.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 442.85 | 441.69 | 441.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 444.60 | 442.27 | 442.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 444.60 | 442.27 | 442.02 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 440.65 | 442.43 | 442.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 439.50 | 441.84 | 442.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 442.65 | 441.47 | 441.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 442.65 | 441.47 | 441.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 442.65 | 441.47 | 441.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:45:00 | 442.85 | 441.47 | 441.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 440.05 | 441.18 | 441.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 439.80 | 441.18 | 441.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 439.00 | 440.75 | 441.55 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 444.75 | 441.78 | 441.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 448.20 | 444.73 | 443.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 451.80 | 452.18 | 449.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 449.80 | 452.25 | 451.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 449.80 | 452.25 | 451.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 449.85 | 452.25 | 451.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 446.90 | 451.18 | 450.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 446.90 | 451.18 | 450.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 446.55 | 449.84 | 450.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 444.40 | 448.75 | 449.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 440.50 | 438.10 | 440.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 440.50 | 438.10 | 440.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 440.50 | 438.10 | 440.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 441.00 | 438.10 | 440.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 440.70 | 438.62 | 440.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 440.70 | 438.62 | 440.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 443.90 | 439.68 | 441.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 444.50 | 439.68 | 441.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 445.45 | 440.83 | 441.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 445.45 | 440.83 | 441.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 445.00 | 442.09 | 441.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 10:15:00 | 445.85 | 443.18 | 442.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 443.00 | 444.46 | 443.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 15:15:00 | 443.00 | 444.46 | 443.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 443.00 | 444.46 | 443.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 442.60 | 444.46 | 443.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 444.00 | 444.36 | 443.57 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 439.95 | 442.89 | 443.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 438.55 | 440.60 | 441.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 439.10 | 438.52 | 439.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 439.10 | 438.52 | 439.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 438.40 | 438.36 | 439.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:15:00 | 444.40 | 438.36 | 439.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 446.45 | 439.97 | 439.93 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 438.00 | 440.29 | 440.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 436.85 | 439.60 | 440.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 437.20 | 437.06 | 438.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 437.20 | 437.06 | 438.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 441.80 | 438.13 | 438.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 442.75 | 438.13 | 438.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 441.90 | 438.88 | 439.18 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 441.70 | 439.45 | 439.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 443.50 | 440.69 | 440.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 444.55 | 444.96 | 443.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 444.55 | 444.96 | 443.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 443.80 | 444.70 | 443.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 443.80 | 444.70 | 443.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 462.45 | 460.03 | 456.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:30:00 | 460.90 | 460.03 | 456.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 461.55 | 462.32 | 459.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 459.35 | 462.32 | 459.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 460.80 | 463.96 | 461.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 460.80 | 463.96 | 461.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 460.10 | 463.19 | 461.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 460.10 | 463.19 | 461.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 454.70 | 459.95 | 460.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 452.65 | 458.49 | 459.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 457.65 | 456.96 | 458.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 14:00:00 | 457.65 | 456.96 | 458.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 454.85 | 456.54 | 458.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:30:00 | 459.15 | 456.54 | 458.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 455.00 | 455.81 | 457.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 456.80 | 455.81 | 457.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 449.80 | 449.65 | 451.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:30:00 | 448.70 | 449.54 | 451.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:00:00 | 448.75 | 449.54 | 451.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 453.95 | 451.89 | 451.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 453.95 | 451.89 | 451.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 459.70 | 453.45 | 452.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 495.10 | 495.15 | 485.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:45:00 | 495.50 | 495.15 | 485.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 516.25 | 518.88 | 513.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 514.00 | 518.88 | 513.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 515.30 | 518.16 | 513.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 513.65 | 518.16 | 513.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 513.65 | 516.78 | 513.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 514.15 | 516.78 | 513.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 514.45 | 516.31 | 513.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 517.20 | 516.49 | 514.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 511.85 | 515.51 | 513.99 | SL hit (close<static) qty=1.00 sl=513.60 alert=retest2 |

### Cycle 103 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 510.95 | 512.91 | 513.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 505.10 | 511.09 | 512.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 523.70 | 506.10 | 507.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 523.70 | 506.10 | 507.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 523.70 | 506.10 | 507.85 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 521.05 | 509.09 | 509.05 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 505.60 | 510.60 | 510.66 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 511.45 | 510.41 | 510.29 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 506.80 | 509.69 | 509.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 506.00 | 508.16 | 509.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 507.85 | 507.56 | 508.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 507.85 | 507.56 | 508.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 506.65 | 507.38 | 508.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:30:00 | 506.95 | 507.38 | 508.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 506.60 | 507.07 | 508.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:45:00 | 508.50 | 507.07 | 508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 508.95 | 507.44 | 508.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 505.75 | 507.08 | 507.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 503.00 | 502.30 | 504.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 503.00 | 502.30 | 504.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 504.55 | 502.75 | 504.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 504.55 | 502.75 | 504.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 503.95 | 502.99 | 504.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 499.65 | 502.99 | 504.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:30:00 | 500.75 | 501.23 | 503.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 11:15:00 | 505.00 | 499.43 | 499.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 11:15:00 | 505.00 | 499.43 | 499.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 506.40 | 501.90 | 500.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 505.00 | 505.51 | 503.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 505.00 | 505.51 | 503.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 502.00 | 504.80 | 503.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 509.90 | 504.80 | 503.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:00:00 | 506.75 | 505.79 | 504.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 506.40 | 505.49 | 504.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 506.00 | 505.48 | 505.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 505.00 | 505.38 | 505.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:15:00 | 504.90 | 505.38 | 505.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 503.40 | 504.99 | 504.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 503.40 | 504.99 | 504.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 503.95 | 504.78 | 504.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 503.95 | 504.78 | 504.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 12:15:00 | 500.85 | 503.99 | 504.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 12:15:00 | 502.00 | 500.27 | 501.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 12:15:00 | 502.00 | 500.27 | 501.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 502.00 | 500.27 | 501.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 502.00 | 500.27 | 501.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 501.30 | 500.47 | 501.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 502.95 | 500.47 | 501.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 501.30 | 500.64 | 501.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 501.30 | 500.64 | 501.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 500.00 | 500.51 | 501.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 503.95 | 500.51 | 501.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 504.95 | 501.40 | 501.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:15:00 | 506.20 | 501.40 | 501.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 507.25 | 502.57 | 502.36 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 502.00 | 502.99 | 503.08 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 505.05 | 503.37 | 503.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 506.70 | 504.04 | 503.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 514.25 | 515.77 | 512.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:15:00 | 519.35 | 515.77 | 512.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 11:15:00 | 518.00 | 515.70 | 512.84 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 12:15:00 | 519.90 | 515.78 | 513.14 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 516.55 | 522.06 | 519.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 516.55 | 522.06 | 519.20 | SL hit (close<ema400) qty=1.00 sl=519.20 alert=retest1 |

### Cycle 113 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 508.20 | 516.81 | 517.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 504.05 | 511.72 | 514.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 506.15 | 501.52 | 505.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 506.15 | 501.52 | 505.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 506.15 | 501.52 | 505.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 505.05 | 501.52 | 505.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 502.55 | 501.73 | 505.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 11:45:00 | 499.40 | 501.88 | 505.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 498.10 | 503.30 | 504.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 508.60 | 504.32 | 504.55 | SL hit (close>static) qty=1.00 sl=508.35 alert=retest2 |

### Cycle 114 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 513.00 | 506.06 | 505.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 516.50 | 509.87 | 507.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 511.95 | 513.94 | 511.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 511.95 | 513.94 | 511.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 511.95 | 513.94 | 511.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 511.95 | 513.94 | 511.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 510.90 | 513.33 | 511.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 510.90 | 513.33 | 511.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 507.05 | 512.08 | 511.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 507.25 | 512.08 | 511.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 509.30 | 511.52 | 510.85 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 506.05 | 509.71 | 510.10 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 512.75 | 510.71 | 510.44 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 510.10 | 511.04 | 511.07 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 513.10 | 511.35 | 511.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 517.50 | 512.58 | 511.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 512.50 | 512.58 | 511.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 512.50 | 512.58 | 511.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 512.85 | 512.66 | 512.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 512.35 | 512.66 | 512.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 512.45 | 512.62 | 512.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 518.30 | 512.62 | 512.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 508.30 | 513.88 | 514.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 508.30 | 513.88 | 514.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 506.00 | 511.43 | 512.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 506.40 | 504.70 | 508.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 506.40 | 504.70 | 508.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 506.40 | 504.70 | 508.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 506.40 | 504.70 | 508.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 509.05 | 505.94 | 508.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 510.00 | 505.94 | 508.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 509.00 | 506.55 | 508.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 509.00 | 506.55 | 508.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 514.00 | 508.04 | 508.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 514.00 | 508.04 | 508.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 515.70 | 509.57 | 509.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 518.55 | 512.47 | 510.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 533.75 | 534.41 | 527.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 12:00:00 | 533.75 | 534.41 | 527.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 533.10 | 536.58 | 533.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 533.10 | 536.58 | 533.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 531.15 | 535.50 | 533.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 532.20 | 535.50 | 533.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 533.00 | 535.00 | 533.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 539.00 | 535.20 | 533.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 527.20 | 532.63 | 533.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 527.20 | 532.63 | 533.32 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 535.15 | 533.13 | 532.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 542.75 | 535.68 | 534.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 537.05 | 538.64 | 536.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 15:15:00 | 537.05 | 538.64 | 536.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 537.05 | 538.64 | 536.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 536.90 | 538.64 | 536.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 538.40 | 538.59 | 536.92 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 531.55 | 535.34 | 535.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 525.90 | 532.96 | 534.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 524.70 | 522.09 | 526.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 524.70 | 522.09 | 526.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 516.40 | 520.95 | 525.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 515.85 | 520.95 | 525.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:00:00 | 516.00 | 515.31 | 519.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 526.15 | 512.46 | 512.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 526.15 | 512.46 | 512.33 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 512.05 | 513.24 | 513.33 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 514.60 | 513.23 | 513.23 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 513.00 | 513.18 | 513.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 512.00 | 512.95 | 513.10 | Break + close below crossover candle low |

### Cycle 128 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 518.75 | 514.11 | 513.61 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 511.80 | 515.68 | 515.69 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 517.00 | 514.98 | 514.95 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 514.10 | 514.80 | 514.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 09:15:00 | 513.15 | 514.47 | 514.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 13:15:00 | 506.60 | 506.57 | 508.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:30:00 | 506.50 | 506.57 | 508.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 507.85 | 506.29 | 508.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 508.95 | 506.29 | 508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 508.15 | 506.66 | 508.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 508.15 | 506.66 | 508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 509.00 | 507.13 | 508.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 508.55 | 507.13 | 508.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 512.15 | 508.14 | 508.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:45:00 | 512.00 | 508.14 | 508.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 513.65 | 509.24 | 508.99 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 506.45 | 509.38 | 509.68 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 510.95 | 509.90 | 509.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 516.50 | 511.22 | 510.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 514.00 | 517.48 | 515.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 514.00 | 517.48 | 515.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 514.00 | 517.48 | 515.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 514.00 | 517.48 | 515.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 515.65 | 517.11 | 515.09 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 511.20 | 513.70 | 513.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 510.00 | 512.13 | 512.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 513.05 | 512.31 | 512.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 513.05 | 512.31 | 512.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 513.05 | 512.31 | 512.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 513.05 | 512.31 | 512.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 512.35 | 512.32 | 512.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 511.40 | 512.26 | 512.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 514.50 | 512.89 | 513.01 | SL hit (close>static) qty=1.00 sl=514.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 515.95 | 513.50 | 513.28 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 511.45 | 513.09 | 513.11 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 513.75 | 513.01 | 513.00 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 512.15 | 513.00 | 513.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 511.20 | 512.33 | 512.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 514.40 | 511.86 | 512.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 514.40 | 511.86 | 512.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 514.40 | 511.86 | 512.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 514.40 | 511.86 | 512.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 514.00 | 512.29 | 512.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 514.60 | 512.29 | 512.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 510.05 | 511.99 | 512.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 510.05 | 511.99 | 512.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 508.85 | 510.33 | 511.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 505.95 | 507.47 | 509.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 496.10 | 495.59 | 495.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 496.10 | 495.59 | 495.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 498.40 | 496.15 | 495.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 494.70 | 497.15 | 496.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 494.70 | 497.15 | 496.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 494.70 | 497.15 | 496.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 492.90 | 497.15 | 496.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 492.45 | 496.21 | 496.29 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 15:15:00 | 498.00 | 495.33 | 495.32 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 491.75 | 494.61 | 494.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 12:15:00 | 488.75 | 492.91 | 494.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 15:15:00 | 485.80 | 485.79 | 488.42 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:15:00 | 482.50 | 485.79 | 488.42 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 485.00 | 483.60 | 485.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 484.50 | 483.60 | 485.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 486.25 | 484.13 | 485.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 486.25 | 484.13 | 485.76 | SL hit (close>ema400) qty=1.00 sl=485.76 alert=retest1 |

### Cycle 144 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 495.00 | 486.83 | 486.73 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 484.35 | 486.67 | 486.91 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 489.70 | 487.27 | 487.16 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 482.60 | 486.57 | 486.97 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 488.80 | 487.39 | 487.30 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 15:15:00 | 486.55 | 487.25 | 487.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 484.65 | 486.73 | 487.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 482.95 | 482.11 | 483.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 482.95 | 482.11 | 483.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 482.95 | 482.11 | 483.42 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 485.50 | 484.03 | 483.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 12:15:00 | 486.80 | 484.58 | 484.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 481.90 | 484.64 | 484.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 481.90 | 484.64 | 484.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 481.90 | 484.64 | 484.37 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 483.40 | 484.13 | 484.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 483.00 | 483.90 | 484.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 482.95 | 481.71 | 482.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 482.95 | 481.71 | 482.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 482.95 | 481.71 | 482.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:15:00 | 483.00 | 481.71 | 482.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 483.35 | 482.04 | 482.39 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 13:15:00 | 485.30 | 483.05 | 482.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 14:15:00 | 485.65 | 483.57 | 483.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 15:15:00 | 483.00 | 483.45 | 483.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 15:15:00 | 483.00 | 483.45 | 483.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 483.00 | 483.45 | 483.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 481.55 | 483.45 | 483.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 483.25 | 483.41 | 483.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 483.25 | 483.41 | 483.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 484.90 | 483.71 | 483.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 483.85 | 483.71 | 483.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 488.85 | 485.78 | 484.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:45:00 | 484.85 | 485.78 | 484.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 484.65 | 485.77 | 484.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 484.65 | 485.77 | 484.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 484.60 | 485.53 | 484.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 484.60 | 485.53 | 484.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 485.05 | 485.44 | 484.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 483.60 | 485.44 | 484.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 485.35 | 485.42 | 484.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 485.00 | 485.42 | 484.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 483.00 | 484.94 | 484.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:45:00 | 483.50 | 484.94 | 484.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 483.15 | 484.58 | 484.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:30:00 | 483.00 | 484.58 | 484.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 483.00 | 484.26 | 484.36 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 485.05 | 484.39 | 484.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 486.95 | 484.90 | 484.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 492.65 | 494.14 | 491.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 490.60 | 493.04 | 491.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 490.60 | 493.04 | 491.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 490.60 | 493.04 | 491.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 490.00 | 492.43 | 491.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 490.00 | 492.43 | 491.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 491.00 | 492.06 | 491.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 492.00 | 492.06 | 491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 491.05 | 491.86 | 491.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 486.20 | 491.86 | 491.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 483.75 | 490.24 | 490.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 482.40 | 488.67 | 489.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 481.00 | 480.82 | 484.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 481.65 | 481.00 | 482.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 481.65 | 481.00 | 482.65 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 488.35 | 484.36 | 483.88 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 483.45 | 484.93 | 484.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 13:15:00 | 481.30 | 484.20 | 484.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 486.25 | 483.68 | 484.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 486.25 | 483.68 | 484.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 486.25 | 483.68 | 484.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 486.00 | 483.68 | 484.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 486.00 | 484.14 | 484.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 486.20 | 484.14 | 484.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 486.30 | 484.58 | 484.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 490.00 | 486.63 | 485.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 485.70 | 487.11 | 486.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 12:15:00 | 485.70 | 487.11 | 486.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 485.70 | 487.11 | 486.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 485.70 | 487.11 | 486.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 485.95 | 486.88 | 486.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:15:00 | 487.00 | 486.46 | 486.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 477.30 | 484.72 | 485.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 477.30 | 484.72 | 485.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 476.15 | 478.90 | 481.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 479.90 | 478.25 | 480.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:00:00 | 479.90 | 478.25 | 480.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 480.60 | 478.84 | 480.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 480.30 | 478.84 | 480.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 482.50 | 479.57 | 480.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 478.05 | 479.57 | 480.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 477.15 | 479.09 | 480.26 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 486.00 | 481.50 | 481.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 499.80 | 485.81 | 483.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 499.00 | 499.86 | 494.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 499.00 | 499.86 | 494.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 496.80 | 499.75 | 496.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 496.80 | 499.75 | 496.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 496.05 | 499.01 | 496.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:30:00 | 500.00 | 497.70 | 496.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 493.85 | 497.31 | 496.53 | SL hit (close<static) qty=1.00 sl=495.00 alert=retest2 |

### Cycle 161 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 491.25 | 495.53 | 495.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 490.05 | 494.19 | 495.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 478.80 | 478.49 | 482.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 479.30 | 478.49 | 482.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 482.85 | 479.84 | 482.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:15:00 | 482.80 | 479.84 | 482.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 482.05 | 480.29 | 482.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 479.00 | 482.18 | 482.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 473.00 | 470.16 | 470.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 10:15:00 | 473.00 | 470.16 | 470.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 13:15:00 | 474.65 | 471.96 | 470.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 11:15:00 | 472.85 | 472.97 | 471.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 11:30:00 | 473.40 | 472.97 | 471.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 473.25 | 473.84 | 472.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 473.25 | 473.84 | 472.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 473.80 | 473.83 | 472.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:15:00 | 474.90 | 473.95 | 473.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 474.55 | 474.07 | 473.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 474.50 | 474.07 | 473.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 475.05 | 474.72 | 473.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 482.10 | 484.56 | 480.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 475.05 | 478.64 | 478.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 475.05 | 478.64 | 478.91 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 477.60 | 474.57 | 474.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 478.90 | 475.85 | 475.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 476.10 | 477.29 | 476.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 15:15:00 | 476.10 | 477.29 | 476.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 476.10 | 477.29 | 476.10 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 472.80 | 475.48 | 475.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 472.05 | 474.03 | 474.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 471.75 | 471.54 | 472.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 471.75 | 471.54 | 472.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 466.70 | 470.18 | 471.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 11:45:00 | 463.10 | 466.71 | 468.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 471.65 | 465.18 | 464.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 471.65 | 465.18 | 464.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 474.70 | 468.30 | 466.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 474.00 | 474.05 | 470.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:45:00 | 474.65 | 473.84 | 470.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 474.70 | 476.79 | 474.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 474.45 | 476.79 | 474.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 473.40 | 476.11 | 474.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 473.70 | 476.11 | 474.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 468.60 | 474.61 | 473.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 468.60 | 474.61 | 473.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 469.80 | 473.65 | 473.34 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 469.25 | 472.77 | 472.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 460.75 | 469.54 | 471.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 13:15:00 | 456.40 | 453.55 | 458.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:45:00 | 456.10 | 453.55 | 458.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 458.00 | 454.44 | 458.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:45:00 | 458.45 | 454.44 | 458.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 455.00 | 454.55 | 458.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 455.95 | 454.55 | 458.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 458.50 | 455.34 | 458.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 459.35 | 455.34 | 458.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 460.40 | 456.35 | 458.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 460.40 | 456.35 | 458.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 468.00 | 458.68 | 459.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 468.00 | 458.68 | 459.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 468.05 | 460.56 | 460.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 476.20 | 464.94 | 462.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 466.55 | 467.25 | 464.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:30:00 | 467.10 | 467.25 | 464.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 467.30 | 467.26 | 464.58 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 452.45 | 462.41 | 463.13 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 14:15:00 | 454.25 | 451.38 | 451.18 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 446.45 | 450.66 | 450.90 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 12:15:00 | 461.00 | 452.66 | 451.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 467.30 | 455.59 | 453.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 475.25 | 475.49 | 468.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 09:45:00 | 474.10 | 475.49 | 468.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 476.00 | 479.25 | 474.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 479.20 | 478.63 | 474.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 462.65 | 474.39 | 474.27 | SL hit (close<static) qty=1.00 sl=470.10 alert=retest2 |

### Cycle 173 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 460.00 | 471.51 | 472.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 453.50 | 466.41 | 470.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 10:15:00 | 463.70 | 462.52 | 466.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:00:00 | 463.70 | 462.52 | 466.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 463.80 | 462.77 | 466.27 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 479.25 | 468.56 | 468.25 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 463.95 | 471.91 | 472.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 462.70 | 470.07 | 471.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 469.95 | 466.10 | 468.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 15:15:00 | 469.95 | 466.10 | 468.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 469.95 | 466.10 | 468.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 457.30 | 466.10 | 468.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:30:00 | 462.50 | 453.72 | 454.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 463.40 | 455.66 | 455.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 463.40 | 455.66 | 455.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 468.10 | 459.06 | 456.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 466.25 | 466.43 | 462.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:00:00 | 466.25 | 466.43 | 462.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 457.30 | 465.39 | 463.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 457.30 | 465.39 | 463.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 457.00 | 463.71 | 462.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 448.15 | 463.71 | 462.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 448.80 | 460.73 | 461.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 440.40 | 450.54 | 455.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 453.40 | 449.77 | 454.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 453.40 | 449.77 | 454.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 453.40 | 449.77 | 454.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 448.75 | 449.77 | 454.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 448.95 | 450.31 | 453.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 448.50 | 450.63 | 452.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 453.50 | 446.78 | 446.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 453.50 | 446.78 | 446.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 462.30 | 453.33 | 450.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 452.15 | 453.35 | 450.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 468.00 | 453.35 | 450.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 468.35 | 468.40 | 461.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 472.00 | 469.18 | 462.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 470.65 | 475.43 | 471.93 | SL hit (close<ema400) qty=1.00 sl=471.93 alert=retest1 |

### Cycle 179 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 467.20 | 470.68 | 470.71 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 473.70 | 471.29 | 470.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 480.20 | 473.07 | 471.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 485.50 | 492.79 | 488.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 485.50 | 492.79 | 488.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 485.50 | 492.79 | 488.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 484.85 | 492.79 | 488.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 486.05 | 491.45 | 488.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:45:00 | 485.65 | 491.45 | 488.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 481.75 | 486.46 | 486.62 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 497.50 | 487.17 | 486.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 502.90 | 490.32 | 488.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 494.25 | 495.21 | 491.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 494.25 | 495.21 | 491.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 494.40 | 494.87 | 492.11 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 490.30 | 490.97 | 491.00 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 492.30 | 491.24 | 491.11 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 488.40 | 490.72 | 490.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 486.80 | 489.66 | 490.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 488.20 | 488.01 | 489.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:45:00 | 488.40 | 488.01 | 489.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 487.55 | 487.92 | 488.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 489.30 | 487.92 | 488.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 486.35 | 487.61 | 488.75 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 491.55 | 489.36 | 489.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 495.00 | 491.16 | 490.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 491.15 | 491.76 | 490.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 491.15 | 491.76 | 490.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 491.40 | 493.20 | 492.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 491.10 | 493.20 | 492.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 489.80 | 492.52 | 491.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 489.60 | 492.52 | 491.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 490.00 | 491.30 | 491.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 487.90 | 490.28 | 490.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 488.80 | 487.72 | 489.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 488.80 | 487.72 | 489.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 488.80 | 487.72 | 489.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 490.20 | 487.72 | 489.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 488.40 | 487.85 | 489.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 490.00 | 487.85 | 489.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 490.00 | 488.28 | 489.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 493.90 | 488.28 | 489.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 497.90 | 490.21 | 489.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 502.35 | 496.81 | 493.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 508.00 | 508.45 | 502.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 508.00 | 508.45 | 502.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 508.00 | 508.45 | 502.11 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 499.00 | 501.76 | 502.12 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 504.80 | 502.19 | 502.14 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 501.00 | 501.96 | 502.03 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 15:15:00 | 502.75 | 502.11 | 502.10 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 10:30:00 | 338.25 | 2024-05-24 15:15:00 | 340.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-05-23 13:15:00 | 336.75 | 2024-05-24 15:15:00 | 340.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-06-03 09:15:00 | 347.50 | 2024-06-04 09:15:00 | 335.05 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-06-03 12:30:00 | 346.00 | 2024-06-04 09:15:00 | 335.05 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-06-03 13:45:00 | 344.10 | 2024-06-04 09:15:00 | 335.05 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-06-12 12:00:00 | 391.00 | 2024-06-14 10:15:00 | 430.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-01 09:30:00 | 415.10 | 2024-07-09 09:15:00 | 430.20 | STOP_HIT | 1.00 | 3.64% |
| BUY | retest2 | 2024-07-15 12:45:00 | 437.70 | 2024-07-18 09:15:00 | 481.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-15 14:00:00 | 446.00 | 2024-07-19 14:15:00 | 445.55 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-08-01 12:00:00 | 441.55 | 2024-08-05 09:15:00 | 419.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 13:15:00 | 441.50 | 2024-08-05 09:15:00 | 419.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 12:00:00 | 441.55 | 2024-08-06 09:15:00 | 416.35 | STOP_HIT | 0.50 | 5.71% |
| SELL | retest2 | 2024-08-01 13:15:00 | 441.50 | 2024-08-06 09:15:00 | 416.35 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2024-08-16 15:15:00 | 393.25 | 2024-08-21 10:15:00 | 399.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-08-19 09:30:00 | 393.50 | 2024-08-21 10:15:00 | 399.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-08-19 10:45:00 | 393.75 | 2024-08-21 11:15:00 | 403.10 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-08-19 13:30:00 | 393.40 | 2024-08-21 11:15:00 | 403.10 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-08-20 12:45:00 | 391.60 | 2024-08-21 11:15:00 | 403.10 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-08-21 09:45:00 | 391.25 | 2024-08-21 11:15:00 | 403.10 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-09-09 15:15:00 | 429.00 | 2024-09-13 09:15:00 | 471.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-21 10:45:00 | 431.70 | 2024-10-21 15:15:00 | 447.00 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2024-10-22 09:15:00 | 430.55 | 2024-10-28 09:15:00 | 409.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 430.55 | 2024-10-28 09:15:00 | 423.65 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest2 | 2024-11-06 13:00:00 | 456.90 | 2024-11-11 11:15:00 | 455.30 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-11-11 09:45:00 | 455.95 | 2024-11-11 11:15:00 | 455.30 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-11-14 11:30:00 | 435.05 | 2024-11-21 09:15:00 | 413.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 13:15:00 | 435.50 | 2024-11-21 09:15:00 | 413.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 14:45:00 | 435.90 | 2024-11-21 09:15:00 | 414.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 11:30:00 | 435.05 | 2024-11-21 11:15:00 | 424.90 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest2 | 2024-11-14 13:15:00 | 435.50 | 2024-11-21 11:15:00 | 424.90 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2024-11-14 14:45:00 | 435.90 | 2024-11-21 11:15:00 | 424.90 | STOP_HIT | 0.50 | 2.52% |
| BUY | retest2 | 2024-11-29 09:15:00 | 421.05 | 2024-12-06 09:15:00 | 431.45 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2024-12-10 11:30:00 | 427.40 | 2024-12-12 11:15:00 | 430.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-12-16 13:15:00 | 433.75 | 2024-12-17 13:15:00 | 430.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-12-16 15:00:00 | 437.00 | 2024-12-17 13:15:00 | 430.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-01-17 10:45:00 | 399.35 | 2025-01-21 11:15:00 | 404.85 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-01-21 10:15:00 | 398.60 | 2025-01-21 11:15:00 | 404.85 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-01-21 11:00:00 | 399.55 | 2025-01-21 11:15:00 | 404.85 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-01-29 09:15:00 | 381.75 | 2025-01-29 10:15:00 | 391.25 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-01-31 09:45:00 | 392.30 | 2025-02-04 09:15:00 | 392.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-01-31 10:15:00 | 393.10 | 2025-02-04 09:15:00 | 392.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-02-03 10:15:00 | 393.60 | 2025-02-04 09:15:00 | 392.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-02-24 09:30:00 | 376.95 | 2025-02-27 10:15:00 | 386.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-02-24 10:15:00 | 376.45 | 2025-02-27 10:15:00 | 386.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-02-24 11:15:00 | 376.20 | 2025-02-27 10:15:00 | 386.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-02-24 11:45:00 | 376.55 | 2025-02-27 10:15:00 | 386.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-03-17 14:30:00 | 418.15 | 2025-03-18 09:15:00 | 423.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-04-21 11:15:00 | 478.20 | 2025-04-21 11:15:00 | 473.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-04-21 14:45:00 | 478.10 | 2025-04-22 13:15:00 | 472.10 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-22 09:30:00 | 479.40 | 2025-04-22 13:15:00 | 472.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-07 09:15:00 | 453.80 | 2025-05-13 11:15:00 | 457.05 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-05-15 10:15:00 | 452.85 | 2025-05-15 10:15:00 | 449.95 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-06-02 09:15:00 | 442.85 | 2025-06-02 09:15:00 | 444.60 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-10 12:30:00 | 448.70 | 2025-07-14 09:15:00 | 453.95 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-10 13:00:00 | 448.75 | 2025-07-14 09:15:00 | 453.95 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-23 15:00:00 | 517.20 | 2025-07-24 09:15:00 | 511.85 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-04 13:15:00 | 499.65 | 2025-08-07 11:15:00 | 505.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-05 09:30:00 | 500.75 | 2025-08-07 11:15:00 | 505.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-08-11 09:15:00 | 509.90 | 2025-08-13 11:15:00 | 503.95 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-11 14:00:00 | 506.75 | 2025-08-13 11:15:00 | 503.95 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-08-12 14:00:00 | 506.40 | 2025-08-13 11:15:00 | 503.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-13 09:15:00 | 506.00 | 2025-08-13 11:15:00 | 503.95 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-08-22 09:15:00 | 519.35 | 2025-08-25 14:15:00 | 516.55 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-08-22 11:15:00 | 518.00 | 2025-08-25 14:15:00 | 516.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-22 12:15:00 | 519.90 | 2025-08-25 14:15:00 | 516.55 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-08-29 11:45:00 | 499.40 | 2025-09-02 09:15:00 | 508.60 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-01 09:15:00 | 498.10 | 2025-09-02 09:15:00 | 508.60 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-09-10 09:15:00 | 518.30 | 2025-09-11 12:15:00 | 508.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-19 15:00:00 | 539.00 | 2025-09-22 14:15:00 | 527.20 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-09-29 11:15:00 | 515.85 | 2025-10-06 09:15:00 | 526.15 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-30 11:00:00 | 516.00 | 2025-10-06 09:15:00 | 526.15 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-10-29 13:15:00 | 511.40 | 2025-10-29 14:15:00 | 514.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-06 09:30:00 | 505.95 | 2025-11-12 15:15:00 | 496.10 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest1 | 2025-11-21 09:15:00 | 482.50 | 2025-11-24 09:15:00 | 486.25 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-26 15:15:00 | 487.00 | 2025-12-29 09:15:00 | 477.30 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-01-06 14:30:00 | 500.00 | 2026-01-07 09:15:00 | 493.85 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-14 09:15:00 | 479.00 | 2026-01-27 10:15:00 | 473.00 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2026-01-29 13:15:00 | 474.90 | 2026-02-02 09:15:00 | 475.05 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2026-01-29 14:30:00 | 474.55 | 2026-02-02 09:15:00 | 475.05 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-01-29 15:00:00 | 474.50 | 2026-02-02 09:15:00 | 475.05 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2026-01-30 09:30:00 | 475.05 | 2026-02-02 09:15:00 | 475.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2026-02-16 11:45:00 | 463.10 | 2026-02-18 14:15:00 | 471.65 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-03-12 11:30:00 | 479.20 | 2026-03-13 09:15:00 | 462.65 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-03-20 09:15:00 | 457.30 | 2026-03-24 14:15:00 | 463.40 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-03-24 13:30:00 | 462.50 | 2026-03-24 14:15:00 | 463.40 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-04-01 10:15:00 | 448.75 | 2026-04-06 14:15:00 | 453.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-04-01 13:45:00 | 448.95 | 2026-04-06 14:15:00 | 453.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-04-01 15:15:00 | 448.50 | 2026-04-06 14:15:00 | 453.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2026-04-08 09:15:00 | 468.00 | 2026-04-13 09:15:00 | 470.65 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2026-04-09 10:30:00 | 472.00 | 2026-04-13 15:15:00 | 467.20 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-04-13 10:00:00 | 470.65 | 2026-04-13 15:15:00 | 467.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-13 13:30:00 | 470.20 | 2026-04-13 15:15:00 | 467.20 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-13 14:30:00 | 470.45 | 2026-04-13 15:15:00 | 467.20 | STOP_HIT | 1.00 | -0.69% |

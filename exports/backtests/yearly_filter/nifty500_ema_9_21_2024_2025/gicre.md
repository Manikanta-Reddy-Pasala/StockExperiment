# General Insurance Corporation of India (GICRE)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 394.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 170 |
| ALERT1 | 116 |
| ALERT2 | 112 |
| ALERT2_SKIP | 56 |
| ALERT3 | 298 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 117 |
| PARTIAL | 6 |
| TARGET_HIT | 10 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 96
- **Target hits / Stop hits / Partials:** 10 / 111 / 6
- **Avg / median % per leg:** 0.17% / -0.68%
- **Sum % (uncompounded):** 21.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 13 | 19.4% | 10 | 57 | 0 | 0.41% | 27.1% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.50% | -1.5% |
| BUY @ 3rd Alert (retest2) | 66 | 13 | 19.7% | 10 | 56 | 0 | 0.43% | 28.6% |
| SELL (all) | 60 | 18 | 30.0% | 0 | 54 | 6 | -0.09% | -5.5% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 3 | 2 | 1.73% | 8.7% |
| SELL @ 3rd Alert (retest2) | 55 | 14 | 25.5% | 0 | 51 | 4 | -0.26% | -14.1% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.19% | 7.2% |
| retest2 (combined) | 121 | 27 | 22.3% | 10 | 107 | 4 | 0.12% | 14.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 329.45 | 324.27 | 323.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 330.25 | 326.93 | 325.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 337.75 | 338.55 | 334.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 337.75 | 338.55 | 334.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 342.60 | 342.88 | 340.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 358.45 | 343.73 | 341.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 365.95 | 369.43 | 369.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 365.95 | 369.43 | 369.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 364.65 | 367.99 | 369.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 10:15:00 | 359.30 | 359.02 | 362.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 10:15:00 | 359.30 | 359.02 | 362.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 359.30 | 359.02 | 362.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 359.20 | 359.02 | 362.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 361.40 | 359.50 | 362.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 361.45 | 359.50 | 362.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 361.30 | 360.02 | 362.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 361.90 | 360.02 | 362.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 356.00 | 357.09 | 359.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 355.70 | 356.57 | 358.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 369.20 | 359.16 | 358.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 369.20 | 359.16 | 358.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 373.70 | 364.97 | 361.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 355.00 | 365.21 | 363.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 355.00 | 365.21 | 363.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 355.00 | 365.21 | 363.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 350.45 | 365.21 | 363.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 336.55 | 359.48 | 360.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 329.75 | 353.53 | 357.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 346.00 | 345.77 | 349.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 365.15 | 345.77 | 349.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 365.25 | 349.67 | 351.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 366.05 | 349.67 | 351.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 368.40 | 353.41 | 352.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 376.75 | 365.48 | 361.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 14:15:00 | 390.00 | 390.78 | 387.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 14:45:00 | 389.55 | 390.78 | 387.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 392.50 | 395.49 | 392.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 388.55 | 393.84 | 392.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 388.85 | 392.84 | 391.74 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 386.55 | 390.60 | 390.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 380.50 | 387.75 | 389.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 390.00 | 383.92 | 385.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 390.00 | 383.92 | 385.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 390.00 | 383.92 | 385.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 391.45 | 383.92 | 385.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 387.30 | 384.60 | 386.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:30:00 | 389.55 | 384.60 | 386.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 384.50 | 385.08 | 386.01 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 407.95 | 390.80 | 388.46 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 390.90 | 396.26 | 396.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 387.75 | 394.56 | 395.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 399.55 | 389.86 | 391.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 399.55 | 389.86 | 391.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 399.55 | 389.86 | 391.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 399.55 | 389.86 | 391.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 394.45 | 390.78 | 391.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:15:00 | 392.00 | 391.29 | 391.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 14:15:00 | 385.90 | 383.96 | 383.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 385.90 | 383.96 | 383.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 10:15:00 | 387.00 | 384.98 | 384.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 385.35 | 385.71 | 384.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 14:15:00 | 385.35 | 385.71 | 384.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 385.35 | 385.71 | 384.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 385.35 | 385.71 | 384.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 387.15 | 386.00 | 385.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 397.80 | 386.00 | 385.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 405.15 | 409.75 | 410.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 405.15 | 409.75 | 410.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 14:15:00 | 402.05 | 407.48 | 409.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 408.65 | 405.56 | 407.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 13:15:00 | 408.65 | 405.56 | 407.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 408.65 | 405.56 | 407.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 407.40 | 405.56 | 407.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 409.55 | 406.36 | 407.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 15:15:00 | 407.50 | 406.36 | 407.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 417.20 | 409.65 | 408.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 417.20 | 409.65 | 408.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 422.70 | 412.92 | 410.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 410.30 | 414.56 | 412.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 410.30 | 414.56 | 412.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 410.30 | 414.56 | 412.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 410.30 | 414.56 | 412.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 412.50 | 414.14 | 412.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 410.50 | 414.14 | 412.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 410.95 | 413.51 | 412.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 410.95 | 413.51 | 412.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 408.50 | 412.50 | 411.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 408.50 | 412.50 | 411.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 406.60 | 411.32 | 411.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 405.70 | 410.20 | 410.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 381.95 | 380.52 | 386.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 10:00:00 | 381.95 | 380.52 | 386.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 371.60 | 376.49 | 381.35 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 413.30 | 386.06 | 382.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 433.80 | 395.61 | 387.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 403.35 | 404.30 | 396.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 403.35 | 404.30 | 396.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 418.95 | 424.63 | 419.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 418.95 | 424.63 | 419.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 416.45 | 422.99 | 418.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 415.30 | 422.99 | 418.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 411.85 | 420.76 | 418.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:45:00 | 411.35 | 420.76 | 418.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 413.40 | 416.31 | 416.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 395.95 | 409.19 | 412.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 396.10 | 393.16 | 400.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 396.10 | 393.16 | 400.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 391.00 | 386.92 | 390.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 391.00 | 386.92 | 390.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 390.00 | 387.53 | 390.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 390.25 | 387.53 | 390.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 387.50 | 387.53 | 390.32 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 400.00 | 391.86 | 391.73 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 388.40 | 391.87 | 392.06 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 396.45 | 392.79 | 392.46 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 390.20 | 393.71 | 393.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 388.60 | 392.03 | 393.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 383.40 | 382.97 | 385.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 15:15:00 | 385.95 | 383.76 | 385.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 385.95 | 383.76 | 385.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 391.60 | 383.76 | 385.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 401.90 | 387.39 | 386.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 407.50 | 397.96 | 393.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 15:15:00 | 411.75 | 412.00 | 406.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:15:00 | 411.35 | 412.00 | 406.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 407.75 | 410.72 | 407.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 407.75 | 410.72 | 407.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 408.50 | 410.28 | 407.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 407.50 | 410.28 | 407.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 407.45 | 409.71 | 407.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 407.45 | 409.71 | 407.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 408.15 | 409.40 | 407.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 413.40 | 409.40 | 407.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:30:00 | 409.25 | 409.48 | 408.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 14:00:00 | 409.25 | 409.48 | 408.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 408.50 | 408.62 | 408.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 408.35 | 408.57 | 408.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-26 15:15:00 | 407.00 | 407.89 | 407.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 407.00 | 407.89 | 407.99 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 410.50 | 408.41 | 408.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 421.00 | 410.93 | 409.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 14:15:00 | 414.10 | 414.34 | 411.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 14:45:00 | 415.00 | 414.34 | 411.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 413.20 | 416.78 | 415.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 413.20 | 416.78 | 415.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 410.20 | 415.46 | 414.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 409.65 | 415.46 | 414.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 405.10 | 413.39 | 413.79 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 12:15:00 | 422.65 | 412.53 | 411.39 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 400.25 | 412.85 | 414.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 398.00 | 405.07 | 409.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 389.30 | 388.54 | 394.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 389.30 | 388.54 | 394.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 389.30 | 388.54 | 394.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:45:00 | 392.20 | 388.54 | 394.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 395.60 | 389.96 | 394.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 395.60 | 389.96 | 394.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 393.00 | 390.56 | 394.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:15:00 | 391.50 | 390.56 | 394.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 405.55 | 394.82 | 395.38 | SL hit (close>static) qty=1.00 sl=395.60 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 406.00 | 397.06 | 396.34 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 388.00 | 395.92 | 396.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 386.30 | 391.79 | 394.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 390.00 | 389.47 | 392.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 390.00 | 389.47 | 392.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 390.00 | 389.47 | 392.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 390.00 | 389.47 | 392.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 391.35 | 389.83 | 391.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:45:00 | 391.50 | 389.83 | 391.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 395.00 | 390.86 | 392.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 395.00 | 390.86 | 392.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 391.75 | 391.04 | 392.05 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 14:15:00 | 393.10 | 392.45 | 392.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 395.10 | 393.18 | 392.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 393.00 | 394.00 | 393.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 393.00 | 394.00 | 393.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 393.00 | 394.00 | 393.45 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 393.10 | 393.79 | 393.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 380.95 | 391.22 | 392.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 387.90 | 387.24 | 389.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 387.90 | 387.24 | 389.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 387.90 | 387.24 | 389.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 390.00 | 387.24 | 389.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 389.90 | 387.77 | 389.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 389.90 | 387.77 | 389.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 390.00 | 388.22 | 389.81 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 396.00 | 391.13 | 390.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 398.65 | 392.63 | 391.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 396.65 | 397.06 | 395.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 14:30:00 | 396.40 | 397.06 | 395.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 395.75 | 396.89 | 395.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:45:00 | 395.85 | 396.89 | 395.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 396.00 | 396.71 | 395.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 395.50 | 396.71 | 395.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 398.00 | 396.97 | 395.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 398.00 | 396.97 | 395.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 392.50 | 396.51 | 396.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:45:00 | 396.10 | 396.41 | 396.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 396.45 | 396.07 | 396.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 395.20 | 395.90 | 395.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 395.20 | 395.90 | 395.96 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 398.55 | 396.43 | 396.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 14:15:00 | 402.50 | 397.97 | 397.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 396.20 | 397.93 | 397.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 396.20 | 397.93 | 397.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 396.20 | 397.93 | 397.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 395.00 | 397.93 | 397.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 396.90 | 397.72 | 397.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 396.65 | 397.72 | 397.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 397.00 | 397.58 | 397.27 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 395.00 | 397.06 | 397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 393.95 | 396.44 | 396.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 15:15:00 | 396.30 | 396.20 | 396.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 395.05 | 396.20 | 396.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 396.10 | 396.18 | 396.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 396.10 | 396.18 | 396.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 398.10 | 396.56 | 396.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 398.10 | 396.56 | 396.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 396.50 | 396.55 | 396.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:15:00 | 397.05 | 396.55 | 396.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 397.15 | 396.67 | 396.72 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 397.90 | 396.92 | 396.83 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 395.15 | 396.56 | 396.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 15:15:00 | 393.70 | 395.99 | 396.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 368.30 | 367.69 | 373.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 368.30 | 367.69 | 373.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 385.30 | 371.88 | 373.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 385.30 | 371.88 | 373.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 389.00 | 375.30 | 375.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 389.90 | 383.09 | 379.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 386.80 | 387.27 | 384.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 387.90 | 387.27 | 384.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 391.60 | 388.13 | 384.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 401.40 | 390.23 | 387.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 13:15:00 | 394.20 | 395.30 | 395.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 394.20 | 395.30 | 395.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 390.00 | 394.09 | 394.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 15:15:00 | 389.70 | 389.18 | 391.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 09:15:00 | 389.00 | 389.18 | 391.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 387.45 | 388.83 | 391.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 384.00 | 386.86 | 389.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 364.80 | 372.66 | 379.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 356.85 | 353.33 | 357.69 | SL hit (close>ema200) qty=0.50 sl=353.33 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 361.60 | 358.91 | 358.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 364.30 | 359.99 | 359.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 369.55 | 369.93 | 366.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 09:30:00 | 369.75 | 369.93 | 366.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 370.30 | 369.35 | 367.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:30:00 | 366.95 | 369.35 | 367.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 368.50 | 369.21 | 367.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 373.85 | 370.13 | 368.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 362.95 | 368.72 | 367.77 | SL hit (close<static) qty=1.00 sl=367.35 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 363.20 | 366.62 | 366.92 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 369.90 | 366.13 | 365.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 376.25 | 368.94 | 367.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 378.10 | 378.32 | 374.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 378.10 | 378.32 | 374.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 376.35 | 377.50 | 374.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 375.20 | 377.50 | 374.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 374.90 | 376.98 | 374.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 371.65 | 376.98 | 374.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 371.20 | 375.82 | 374.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 370.25 | 375.82 | 374.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 369.40 | 374.54 | 374.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 369.40 | 374.54 | 374.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 366.70 | 372.97 | 373.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 365.30 | 370.51 | 372.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 374.20 | 369.10 | 370.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 374.20 | 369.10 | 370.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 374.20 | 369.10 | 370.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 374.20 | 369.10 | 370.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 371.45 | 369.57 | 370.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 370.50 | 369.57 | 370.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:30:00 | 370.55 | 369.82 | 370.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 372.10 | 371.13 | 371.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 372.10 | 371.13 | 371.02 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 368.95 | 370.70 | 370.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 368.15 | 370.19 | 370.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 361.05 | 356.01 | 360.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 361.05 | 356.01 | 360.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 361.05 | 356.01 | 360.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 361.05 | 356.01 | 360.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 352.80 | 355.37 | 360.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 361.15 | 355.37 | 360.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 354.10 | 355.76 | 359.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 353.30 | 355.76 | 359.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 360.05 | 356.22 | 358.53 | SL hit (close>static) qty=1.00 sl=359.60 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 371.70 | 361.16 | 360.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 378.75 | 368.00 | 365.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 11:15:00 | 396.35 | 397.66 | 392.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 12:00:00 | 396.35 | 397.66 | 392.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 401.20 | 397.64 | 394.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 403.40 | 399.96 | 398.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 11:00:00 | 403.45 | 400.66 | 398.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:30:00 | 403.40 | 402.16 | 399.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 407.00 | 402.37 | 400.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 412.95 | 411.38 | 408.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 410.55 | 411.38 | 408.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 414.00 | 414.37 | 412.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 418.75 | 414.37 | 412.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:45:00 | 414.90 | 414.87 | 413.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 415.05 | 414.26 | 413.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:00:00 | 414.90 | 414.40 | 413.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 415.25 | 415.16 | 414.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:30:00 | 419.70 | 417.83 | 415.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-16 09:15:00 | 443.74 | 430.58 | 427.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 463.35 | 471.20 | 471.58 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 473.05 | 471.11 | 470.97 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 463.00 | 469.60 | 470.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 10:15:00 | 461.50 | 467.98 | 469.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 09:15:00 | 448.55 | 446.07 | 452.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 448.55 | 446.07 | 452.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 448.55 | 446.07 | 452.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 451.15 | 446.07 | 452.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 461.55 | 449.57 | 452.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:00:00 | 461.55 | 449.57 | 452.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 461.55 | 451.97 | 453.14 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 457.35 | 454.22 | 454.03 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 10:15:00 | 453.00 | 453.76 | 453.84 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 454.80 | 453.96 | 453.93 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 451.85 | 453.62 | 453.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 15:15:00 | 451.00 | 453.10 | 453.54 | Break + close below crossover candle low |

### Cycle 51 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 457.40 | 453.96 | 453.89 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 442.70 | 452.49 | 453.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 438.25 | 446.81 | 450.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 449.45 | 444.19 | 448.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 449.45 | 444.19 | 448.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 449.45 | 444.19 | 448.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 449.45 | 444.19 | 448.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 454.15 | 446.18 | 448.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 454.15 | 446.18 | 448.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 449.05 | 446.76 | 448.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 450.80 | 446.76 | 448.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 451.45 | 447.70 | 448.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 451.45 | 447.70 | 448.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 453.65 | 448.89 | 449.38 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 453.95 | 449.90 | 449.80 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 447.50 | 449.65 | 449.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 445.05 | 447.34 | 448.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 12:15:00 | 452.30 | 448.26 | 448.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 12:15:00 | 452.30 | 448.26 | 448.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 452.30 | 448.26 | 448.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:00:00 | 452.30 | 448.26 | 448.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 13:15:00 | 459.80 | 450.57 | 449.62 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 445.00 | 450.62 | 450.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 437.40 | 447.97 | 449.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 416.75 | 415.96 | 424.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:00:00 | 416.75 | 415.96 | 424.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 423.45 | 417.46 | 424.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 423.45 | 417.46 | 424.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 420.90 | 418.14 | 423.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 418.35 | 418.14 | 423.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 416.00 | 417.72 | 423.17 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 436.60 | 423.20 | 422.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 14:15:00 | 438.90 | 433.10 | 429.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 446.00 | 446.53 | 440.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 446.00 | 446.53 | 440.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 442.15 | 445.65 | 440.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 442.15 | 445.65 | 440.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 441.50 | 444.78 | 440.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 439.80 | 444.78 | 440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 438.50 | 443.52 | 440.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:15:00 | 438.40 | 443.52 | 440.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 434.40 | 441.70 | 440.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 435.95 | 441.70 | 440.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 435.70 | 440.50 | 439.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 430.85 | 440.50 | 439.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 422.95 | 436.99 | 438.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 418.35 | 430.81 | 434.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 434.05 | 427.51 | 431.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 434.05 | 427.51 | 431.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 434.05 | 427.51 | 431.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 433.40 | 427.51 | 431.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 436.30 | 429.27 | 431.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 436.30 | 429.27 | 431.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 439.20 | 434.13 | 433.48 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 427.95 | 432.20 | 432.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 423.30 | 428.73 | 430.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 400.30 | 398.09 | 407.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 400.30 | 398.09 | 407.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 404.50 | 397.28 | 404.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 404.50 | 397.28 | 404.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 404.05 | 398.64 | 404.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 402.30 | 400.69 | 403.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 412.30 | 404.38 | 404.85 | SL hit (close>static) qty=1.00 sl=405.75 alert=retest2 |

### Cycle 61 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 410.00 | 405.51 | 405.32 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 398.95 | 404.48 | 404.94 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 412.95 | 405.00 | 404.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 414.65 | 406.93 | 405.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 408.90 | 409.06 | 407.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:45:00 | 408.30 | 409.06 | 407.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 412.35 | 409.72 | 407.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 414.20 | 410.10 | 408.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:30:00 | 416.30 | 411.29 | 409.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:30:00 | 420.15 | 411.10 | 409.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 404.40 | 409.76 | 408.70 | SL hit (close<static) qty=1.00 sl=407.50 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 403.00 | 407.92 | 408.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 382.50 | 402.83 | 405.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 389.05 | 388.89 | 396.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 389.05 | 388.89 | 396.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 388.00 | 388.72 | 395.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 402.40 | 388.72 | 395.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 403.80 | 391.73 | 396.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 403.80 | 391.73 | 396.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 402.80 | 393.95 | 397.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 13:30:00 | 401.25 | 398.16 | 398.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 15:00:00 | 400.50 | 398.63 | 398.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 15:15:00 | 401.00 | 399.10 | 398.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 401.00 | 399.10 | 398.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 404.10 | 400.10 | 399.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 415.50 | 416.59 | 412.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 14:00:00 | 415.50 | 416.59 | 412.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 402.85 | 413.83 | 412.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 402.85 | 413.83 | 412.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 401.85 | 411.43 | 411.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 399.45 | 409.04 | 410.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 383.25 | 382.62 | 390.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 382.00 | 382.62 | 390.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 393.55 | 382.84 | 387.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 392.60 | 382.84 | 387.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 393.30 | 384.93 | 388.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 393.45 | 384.93 | 388.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 387.70 | 388.27 | 389.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:45:00 | 388.95 | 388.27 | 389.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 378.35 | 386.09 | 387.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 376.40 | 386.09 | 387.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 10:15:00 | 357.58 | 367.87 | 375.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 11:15:00 | 366.40 | 362.28 | 367.76 | SL hit (close>ema200) qty=0.50 sl=362.28 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 382.70 | 372.01 | 370.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 385.00 | 381.22 | 376.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 384.30 | 386.24 | 383.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 13:15:00 | 384.30 | 386.24 | 383.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 384.30 | 386.24 | 383.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 384.30 | 386.24 | 383.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 386.50 | 386.29 | 383.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:15:00 | 384.30 | 386.29 | 383.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 384.30 | 385.89 | 383.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 380.10 | 385.89 | 383.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 382.75 | 385.26 | 383.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 387.40 | 385.99 | 384.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 386.65 | 389.42 | 386.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:45:00 | 387.25 | 390.25 | 388.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 377.15 | 386.73 | 387.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 377.15 | 386.73 | 387.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 372.55 | 381.08 | 384.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 364.60 | 364.36 | 371.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 364.60 | 364.36 | 371.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 372.40 | 365.70 | 370.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 372.40 | 365.70 | 370.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 372.25 | 367.01 | 370.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:30:00 | 372.00 | 367.01 | 370.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 15:15:00 | 380.40 | 373.66 | 372.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 385.90 | 377.25 | 374.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 387.15 | 388.55 | 385.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 387.15 | 388.55 | 385.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 387.15 | 388.55 | 385.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 386.60 | 388.55 | 385.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 382.85 | 387.21 | 385.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 382.85 | 387.21 | 385.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 383.00 | 386.37 | 384.98 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 381.95 | 383.88 | 384.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 377.80 | 382.06 | 383.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 377.90 | 377.86 | 380.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 11:00:00 | 377.90 | 377.86 | 380.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 377.85 | 376.53 | 378.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 380.75 | 376.53 | 378.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 378.70 | 376.96 | 378.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 378.70 | 376.96 | 378.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 375.40 | 376.65 | 378.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 373.45 | 376.65 | 377.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 15:15:00 | 384.95 | 378.07 | 377.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 15:15:00 | 384.95 | 378.07 | 377.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 392.20 | 385.55 | 382.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 410.05 | 410.09 | 404.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:15:00 | 408.00 | 410.09 | 404.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 426.00 | 435.20 | 428.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 426.00 | 435.20 | 428.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 421.60 | 432.48 | 427.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 421.60 | 432.48 | 427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 428.45 | 427.79 | 426.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 427.85 | 427.79 | 426.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 426.15 | 427.47 | 426.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 426.45 | 427.47 | 426.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 426.20 | 427.21 | 426.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 426.20 | 427.21 | 426.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 424.20 | 426.10 | 426.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 416.25 | 423.39 | 425.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 424.70 | 419.32 | 421.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 424.70 | 419.32 | 421.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 424.70 | 419.32 | 421.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 424.55 | 419.32 | 421.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 424.75 | 420.41 | 421.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 425.50 | 420.41 | 421.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 422.65 | 422.24 | 422.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:45:00 | 425.15 | 422.24 | 422.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 419.70 | 421.73 | 422.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:00:00 | 417.05 | 420.80 | 421.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 417.60 | 415.44 | 417.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 417.45 | 415.63 | 417.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 423.60 | 418.56 | 418.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 423.60 | 418.56 | 418.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 429.40 | 423.74 | 421.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 411.65 | 421.97 | 420.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 411.65 | 421.97 | 420.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 411.65 | 421.97 | 420.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 411.65 | 421.97 | 420.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 412.20 | 420.01 | 420.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 408.25 | 415.95 | 418.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 390.65 | 389.13 | 398.41 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:30:00 | 388.15 | 389.20 | 397.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 399.15 | 392.81 | 396.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-08 14:15:00 | 399.15 | 392.81 | 396.76 | SL hit (close>ema400) qty=1.00 sl=396.76 alert=retest1 |

### Cycle 75 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 399.65 | 396.47 | 396.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 404.00 | 398.24 | 397.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 424.30 | 425.81 | 419.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 423.40 | 426.25 | 424.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 423.40 | 426.25 | 424.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 423.40 | 426.25 | 424.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 418.75 | 424.75 | 424.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 418.15 | 424.75 | 424.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 420.25 | 423.85 | 423.93 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 424.95 | 424.04 | 423.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 428.30 | 424.93 | 424.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 425.05 | 425.08 | 424.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 13:00:00 | 425.05 | 425.08 | 424.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 427.65 | 425.59 | 424.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 424.95 | 425.59 | 424.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 424.00 | 425.51 | 424.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 442.60 | 425.51 | 424.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 11:45:00 | 428.10 | 428.84 | 426.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 420.50 | 426.59 | 426.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 09:15:00 | 420.50 | 426.59 | 426.70 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 430.35 | 426.99 | 426.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 430.80 | 428.35 | 427.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 427.65 | 428.52 | 427.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 427.65 | 428.52 | 427.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 427.65 | 428.52 | 427.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 427.65 | 428.52 | 427.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 428.60 | 428.54 | 427.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 13:15:00 | 428.65 | 428.54 | 427.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 13:45:00 | 429.20 | 429.03 | 428.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 14:15:00 | 417.40 | 430.08 | 429.90 | SL hit (close<static) qty=1.00 sl=423.95 alert=retest2 |

### Cycle 80 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 415.05 | 427.07 | 428.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 413.05 | 420.87 | 423.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 414.95 | 414.68 | 419.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 13:45:00 | 413.75 | 414.26 | 418.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 14:45:00 | 411.20 | 413.93 | 417.61 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 412.25 | 413.18 | 416.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:45:00 | 409.35 | 411.91 | 415.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 393.06 | 404.35 | 410.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 390.64 | 404.35 | 410.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 13:15:00 | 388.88 | 398.14 | 405.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 409.40 | 399.62 | 403.95 | SL hit (close>ema200) qty=0.50 sl=399.62 alert=retest1 |

### Cycle 81 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 412.65 | 407.15 | 406.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 413.40 | 408.40 | 407.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 415.30 | 416.25 | 413.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 415.30 | 416.25 | 413.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 424.50 | 419.96 | 417.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:45:00 | 429.40 | 423.20 | 419.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 431.85 | 434.09 | 432.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 429.15 | 432.16 | 431.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 429.00 | 432.54 | 432.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 428.55 | 431.74 | 431.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 428.55 | 431.74 | 431.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 426.30 | 430.66 | 431.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 433.90 | 427.23 | 428.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 433.90 | 427.23 | 428.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 433.90 | 427.23 | 428.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 433.90 | 427.23 | 428.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 426.45 | 427.08 | 428.53 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 430.60 | 429.04 | 428.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 432.70 | 430.02 | 429.39 | Break + close above crossover candle high |

### Cycle 84 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 420.40 | 428.52 | 428.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 10:15:00 | 418.00 | 426.41 | 427.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 413.55 | 406.14 | 409.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 413.55 | 406.14 | 409.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 413.55 | 406.14 | 409.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 413.55 | 406.14 | 409.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 407.45 | 406.40 | 409.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 401.10 | 404.61 | 406.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:15:00 | 401.30 | 401.68 | 403.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 10:15:00 | 406.75 | 402.11 | 401.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 406.75 | 402.11 | 401.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 409.15 | 405.28 | 403.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 406.65 | 409.44 | 407.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 406.65 | 409.44 | 407.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 406.65 | 409.44 | 407.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 407.80 | 409.44 | 407.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 408.00 | 409.16 | 407.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:15:00 | 404.00 | 409.16 | 407.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 404.00 | 408.12 | 407.31 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 401.65 | 406.13 | 406.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 398.60 | 404.62 | 405.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 391.85 | 391.84 | 396.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 386.60 | 391.84 | 396.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 386.20 | 387.07 | 390.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 384.70 | 386.45 | 390.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 384.20 | 377.07 | 376.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 384.20 | 377.07 | 376.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 390.30 | 383.02 | 380.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 384.00 | 384.51 | 382.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:15:00 | 385.80 | 384.51 | 382.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 380.00 | 383.48 | 382.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 380.00 | 383.48 | 382.34 | SL hit (close<ema400) qty=1.00 sl=382.34 alert=retest1 |

### Cycle 88 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 381.75 | 381.77 | 381.77 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 385.25 | 382.46 | 382.08 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 382.35 | 383.62 | 383.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 379.75 | 381.92 | 382.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 379.80 | 379.40 | 380.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:00:00 | 379.80 | 379.40 | 380.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 379.25 | 378.54 | 379.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 379.50 | 378.54 | 379.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 380.50 | 378.93 | 379.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 380.50 | 378.93 | 379.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 377.70 | 378.69 | 379.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 376.65 | 378.11 | 378.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 376.75 | 373.37 | 374.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 376.70 | 373.99 | 374.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 377.65 | 375.69 | 375.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 377.65 | 375.69 | 375.43 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 372.55 | 374.93 | 375.14 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 375.25 | 374.86 | 374.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 379.75 | 376.32 | 375.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 389.35 | 391.02 | 387.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 389.60 | 391.02 | 387.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 387.00 | 389.73 | 387.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 387.00 | 389.73 | 387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 386.25 | 389.04 | 387.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 386.25 | 389.04 | 387.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 385.95 | 388.42 | 387.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 385.95 | 388.42 | 387.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 386.70 | 388.07 | 387.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:30:00 | 386.00 | 388.07 | 387.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 386.70 | 387.80 | 387.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 385.20 | 386.89 | 386.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 382.20 | 385.95 | 386.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 378.80 | 382.51 | 384.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 382.35 | 382.19 | 383.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 382.35 | 382.19 | 383.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 380.05 | 381.38 | 382.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 379.35 | 381.38 | 382.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 388.00 | 382.77 | 383.16 | SL hit (close>static) qty=1.00 sl=383.25 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 385.80 | 383.79 | 383.58 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 381.05 | 383.20 | 383.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 12:15:00 | 379.50 | 381.71 | 382.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 382.35 | 381.31 | 382.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 382.35 | 381.31 | 382.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 382.35 | 381.31 | 382.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:45:00 | 380.75 | 381.33 | 381.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 380.95 | 381.28 | 381.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 380.50 | 381.25 | 381.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 380.80 | 381.35 | 381.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 378.70 | 380.82 | 381.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 376.85 | 380.82 | 381.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 381.10 | 374.68 | 374.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 381.10 | 374.68 | 374.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 386.40 | 377.55 | 375.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 386.35 | 386.59 | 382.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 10:00:00 | 386.35 | 386.59 | 382.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 383.85 | 385.50 | 383.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:15:00 | 382.00 | 385.50 | 383.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 382.00 | 384.80 | 383.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 384.10 | 384.80 | 383.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 381.15 | 384.07 | 383.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 382.00 | 384.07 | 383.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 382.45 | 383.75 | 383.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 380.00 | 383.75 | 383.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 383.65 | 383.59 | 383.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:15:00 | 382.65 | 383.59 | 383.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 383.70 | 383.61 | 383.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 389.00 | 383.52 | 383.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 13:45:00 | 384.00 | 385.02 | 384.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 383.85 | 384.70 | 384.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 386.25 | 384.50 | 384.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 381.60 | 383.92 | 383.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 381.60 | 383.92 | 383.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 380.10 | 383.16 | 383.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 387.65 | 380.36 | 380.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 387.65 | 380.36 | 380.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 387.65 | 380.36 | 380.95 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 390.25 | 382.34 | 381.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 12:15:00 | 393.95 | 385.45 | 383.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 395.35 | 395.49 | 392.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 395.35 | 395.49 | 392.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 394.95 | 395.29 | 392.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 392.00 | 394.74 | 392.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 391.85 | 394.16 | 392.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 392.00 | 394.16 | 392.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 393.00 | 393.93 | 392.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 390.40 | 393.93 | 392.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 395.50 | 394.24 | 393.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 395.35 | 394.24 | 393.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 393.80 | 394.18 | 393.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 393.80 | 394.18 | 393.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 392.10 | 393.74 | 393.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 392.15 | 393.74 | 393.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 392.50 | 393.49 | 393.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 392.50 | 393.49 | 393.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 389.65 | 392.72 | 392.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 386.20 | 390.53 | 391.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 398.55 | 391.41 | 391.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 398.55 | 391.41 | 391.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 398.55 | 391.41 | 391.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 400.95 | 391.41 | 391.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 396.65 | 392.46 | 392.29 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 390.35 | 392.14 | 392.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 386.90 | 388.92 | 390.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 389.85 | 388.80 | 389.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 389.85 | 388.80 | 389.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 389.85 | 388.80 | 389.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 385.00 | 388.24 | 389.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 15:15:00 | 365.75 | 370.12 | 374.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 367.55 | 365.90 | 369.37 | SL hit (close>ema200) qty=0.50 sl=365.90 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 371.55 | 370.30 | 370.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 373.45 | 371.84 | 371.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 369.25 | 371.32 | 371.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 369.25 | 371.32 | 371.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 369.25 | 371.32 | 371.01 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 368.30 | 370.72 | 370.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 367.30 | 369.93 | 370.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 367.10 | 366.19 | 367.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 367.10 | 366.19 | 367.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 367.65 | 366.48 | 367.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 368.15 | 366.48 | 367.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 366.40 | 366.47 | 367.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 364.00 | 366.47 | 367.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 365.70 | 366.31 | 367.53 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 367.40 | 366.62 | 366.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 367.65 | 366.88 | 366.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 367.80 | 368.64 | 367.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 367.80 | 368.64 | 367.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 367.80 | 368.64 | 367.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 367.45 | 368.64 | 367.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 369.25 | 368.76 | 368.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 369.25 | 368.76 | 368.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 369.60 | 369.11 | 368.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 368.90 | 369.11 | 368.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 373.60 | 370.10 | 369.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 374.45 | 371.00 | 369.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 374.50 | 372.38 | 370.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 369.15 | 370.34 | 370.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 369.15 | 370.34 | 370.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 368.50 | 369.76 | 370.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 364.25 | 363.84 | 365.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 15:00:00 | 364.25 | 363.84 | 365.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 367.75 | 364.73 | 365.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:00:00 | 365.00 | 365.61 | 365.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 366.05 | 364.92 | 365.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 363.85 | 361.84 | 361.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 363.85 | 361.84 | 361.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 366.60 | 363.80 | 362.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 10:15:00 | 363.25 | 363.69 | 362.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 10:15:00 | 363.25 | 363.69 | 362.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 363.25 | 363.69 | 362.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 363.25 | 363.69 | 362.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 366.10 | 364.17 | 363.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 366.80 | 364.17 | 363.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:00:00 | 366.55 | 366.85 | 365.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 366.40 | 366.76 | 365.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 364.55 | 365.40 | 365.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 364.55 | 365.40 | 365.43 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 366.40 | 365.60 | 365.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 366.60 | 365.80 | 365.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 377.55 | 380.61 | 377.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 377.55 | 380.61 | 377.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 377.55 | 380.61 | 377.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 377.55 | 380.61 | 377.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 375.70 | 379.63 | 376.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 375.70 | 379.63 | 376.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 375.80 | 378.86 | 376.85 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 370.55 | 375.19 | 375.67 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 378.25 | 375.52 | 375.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 380.35 | 376.84 | 376.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 378.05 | 380.22 | 378.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 378.05 | 380.22 | 378.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 378.05 | 380.22 | 378.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 378.15 | 380.22 | 378.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 382.70 | 380.71 | 378.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:45:00 | 385.05 | 382.08 | 380.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 376.40 | 379.76 | 380.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 376.40 | 379.76 | 380.06 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 381.50 | 380.29 | 380.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 383.00 | 380.83 | 380.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 379.70 | 381.03 | 380.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 379.70 | 381.03 | 380.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 379.70 | 381.03 | 380.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 379.70 | 381.03 | 380.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 381.80 | 381.18 | 380.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 381.80 | 381.18 | 380.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 381.00 | 381.15 | 380.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 381.00 | 381.15 | 380.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 381.00 | 381.12 | 380.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 381.00 | 381.12 | 380.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 380.50 | 380.99 | 380.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 380.50 | 380.99 | 380.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 385.05 | 381.80 | 381.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 387.55 | 381.80 | 381.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 386.60 | 383.75 | 382.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 386.95 | 383.75 | 382.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 13:00:00 | 387.05 | 384.77 | 383.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 387.00 | 387.72 | 386.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:15:00 | 385.95 | 387.72 | 386.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 387.45 | 387.66 | 386.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 387.45 | 387.66 | 386.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 385.90 | 387.56 | 386.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 385.90 | 387.56 | 386.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 384.45 | 386.94 | 386.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 383.20 | 386.94 | 386.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 385.70 | 386.32 | 386.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 385.70 | 386.32 | 386.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 383.20 | 385.69 | 386.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 384.85 | 384.65 | 385.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 384.85 | 384.65 | 385.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 384.85 | 384.65 | 385.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 384.85 | 384.65 | 385.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 384.80 | 384.68 | 385.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 387.25 | 384.68 | 385.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 387.05 | 385.16 | 385.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 383.95 | 385.10 | 385.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 384.10 | 384.43 | 384.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 385.65 | 385.16 | 385.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 385.65 | 385.16 | 385.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 386.20 | 385.37 | 385.24 | Break + close above crossover candle high |

### Cycle 116 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 383.60 | 385.02 | 385.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 10:15:00 | 383.00 | 384.61 | 384.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 377.65 | 377.13 | 379.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:00:00 | 377.65 | 377.13 | 379.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 379.20 | 377.82 | 379.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 377.30 | 378.05 | 379.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 380.75 | 378.59 | 379.30 | SL hit (close>static) qty=1.00 sl=379.80 alert=retest2 |

### Cycle 117 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 380.00 | 376.23 | 376.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 383.00 | 380.17 | 378.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 392.05 | 393.78 | 392.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 392.05 | 393.78 | 392.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 392.05 | 393.78 | 392.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:45:00 | 395.30 | 393.83 | 392.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 394.85 | 393.72 | 392.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:30:00 | 394.65 | 393.85 | 392.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 389.25 | 392.43 | 392.33 | SL hit (close<static) qty=1.00 sl=391.35 alert=retest2 |

### Cycle 118 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 390.10 | 391.96 | 392.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 388.15 | 390.78 | 391.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 381.00 | 379.39 | 381.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 381.00 | 379.39 | 381.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 380.50 | 379.62 | 381.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:45:00 | 380.25 | 379.62 | 381.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 380.90 | 379.87 | 381.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 382.40 | 379.87 | 381.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 381.70 | 380.34 | 381.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 381.70 | 380.34 | 381.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 382.15 | 380.70 | 381.58 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 390.00 | 382.82 | 382.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 391.10 | 385.63 | 383.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 387.10 | 389.13 | 387.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 387.10 | 389.13 | 387.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 387.10 | 389.13 | 387.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 387.10 | 389.13 | 387.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 386.90 | 388.69 | 387.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 389.35 | 386.73 | 386.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 389.65 | 386.98 | 386.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 391.20 | 387.78 | 387.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 389.10 | 388.28 | 387.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 387.60 | 388.25 | 387.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 386.75 | 388.25 | 387.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 384.50 | 387.50 | 387.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 384.50 | 387.50 | 387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 383.90 | 386.78 | 387.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 383.90 | 386.78 | 387.05 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 390.60 | 387.54 | 387.37 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 383.95 | 387.12 | 387.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 383.50 | 385.87 | 386.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 386.30 | 385.96 | 386.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 386.30 | 385.96 | 386.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 386.30 | 385.96 | 386.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 386.30 | 385.96 | 386.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 388.20 | 386.41 | 386.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 388.20 | 386.41 | 386.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 386.50 | 386.43 | 386.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 384.25 | 386.38 | 386.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:45:00 | 383.95 | 385.61 | 386.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 13:15:00 | 380.40 | 378.94 | 378.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 380.40 | 378.94 | 378.83 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 372.20 | 377.99 | 378.47 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 380.35 | 377.69 | 377.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 381.50 | 378.45 | 377.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 378.45 | 379.15 | 378.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 378.45 | 379.15 | 378.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 378.45 | 379.15 | 378.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 378.35 | 379.15 | 378.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 380.65 | 379.45 | 378.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:45:00 | 383.00 | 380.67 | 379.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 376.50 | 378.91 | 379.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 376.50 | 378.91 | 379.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 375.40 | 378.20 | 378.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 365.05 | 364.88 | 368.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 10:00:00 | 365.05 | 364.88 | 368.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 368.90 | 365.69 | 368.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 368.90 | 365.69 | 368.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 369.30 | 366.41 | 368.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 369.30 | 366.41 | 368.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 368.60 | 366.85 | 368.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 369.40 | 366.85 | 368.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 370.45 | 367.57 | 368.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 370.45 | 367.57 | 368.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 365.50 | 367.15 | 368.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 364.80 | 367.15 | 368.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 369.40 | 367.42 | 368.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:00:00 | 365.60 | 367.05 | 368.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 366.30 | 366.90 | 368.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 370.90 | 368.39 | 368.52 | SL hit (close>static) qty=1.00 sl=370.45 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 372.95 | 369.31 | 368.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 373.35 | 370.11 | 369.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 368.65 | 370.49 | 369.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 368.65 | 370.49 | 369.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 368.65 | 370.49 | 369.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 368.65 | 370.49 | 369.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 369.30 | 370.25 | 369.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 371.65 | 370.25 | 369.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 370.30 | 370.26 | 369.90 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 368.20 | 369.60 | 369.66 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 13:15:00 | 370.45 | 369.77 | 369.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 14:15:00 | 372.90 | 370.40 | 370.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 373.10 | 373.71 | 372.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:30:00 | 373.40 | 373.71 | 372.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 371.60 | 373.28 | 372.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 371.60 | 373.28 | 372.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 372.00 | 373.03 | 372.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 371.05 | 373.03 | 372.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 371.65 | 372.75 | 372.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:15:00 | 372.00 | 372.75 | 372.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 367.45 | 371.69 | 371.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 367.65 | 371.69 | 371.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 369.90 | 371.33 | 371.50 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 378.05 | 372.31 | 371.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 386.25 | 375.09 | 372.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 380.40 | 381.75 | 378.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 380.40 | 381.75 | 378.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 380.40 | 381.75 | 378.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 383.10 | 381.86 | 378.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 383.05 | 381.92 | 379.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 377.25 | 379.16 | 379.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 377.25 | 379.16 | 379.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 375.80 | 378.21 | 378.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 11:15:00 | 373.85 | 372.91 | 375.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 373.85 | 372.91 | 375.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 368.80 | 367.95 | 369.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 361.50 | 367.95 | 369.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:45:00 | 363.80 | 366.76 | 368.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 372.80 | 368.61 | 368.75 | SL hit (close>static) qty=1.00 sl=370.15 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 372.70 | 369.43 | 369.11 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 14:15:00 | 366.95 | 368.83 | 368.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 15:15:00 | 366.00 | 368.26 | 368.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 361.45 | 361.42 | 363.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:45:00 | 360.75 | 361.42 | 363.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 364.00 | 362.18 | 363.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 364.10 | 362.18 | 363.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 365.00 | 362.74 | 363.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:00:00 | 365.00 | 362.74 | 363.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 364.40 | 363.59 | 363.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 364.45 | 363.59 | 363.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 09:15:00 | 366.85 | 364.24 | 364.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 11:15:00 | 369.10 | 366.30 | 365.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 10:15:00 | 368.50 | 368.82 | 367.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:45:00 | 368.70 | 368.82 | 367.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 370.80 | 373.12 | 370.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 370.15 | 373.12 | 370.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 374.95 | 373.49 | 370.98 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 363.15 | 370.35 | 370.56 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 370.60 | 369.99 | 369.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 375.70 | 372.31 | 371.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 372.15 | 372.92 | 371.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 372.15 | 372.92 | 371.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 372.15 | 372.92 | 371.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 372.15 | 372.92 | 371.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 373.75 | 373.09 | 372.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 375.15 | 373.57 | 372.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 370.10 | 376.49 | 376.16 | SL hit (close<static) qty=1.00 sl=371.75 alert=retest2 |

### Cycle 138 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 367.95 | 374.78 | 375.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 367.10 | 373.24 | 374.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 371.65 | 371.24 | 372.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 10:30:00 | 371.40 | 371.24 | 372.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 371.90 | 371.37 | 372.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 372.75 | 371.37 | 372.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 372.60 | 371.52 | 372.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:00:00 | 372.60 | 371.52 | 372.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 372.50 | 371.72 | 372.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 372.50 | 371.72 | 372.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 371.55 | 371.69 | 372.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 371.60 | 371.85 | 372.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 371.30 | 371.74 | 372.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:45:00 | 370.45 | 371.82 | 372.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 13:15:00 | 373.95 | 372.45 | 372.55 | SL hit (close>static) qty=1.00 sl=373.30 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 374.50 | 372.86 | 372.72 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 370.55 | 372.65 | 372.67 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 373.90 | 372.73 | 372.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 376.60 | 373.79 | 373.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 393.70 | 393.97 | 389.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:00:00 | 393.70 | 393.97 | 389.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 387.60 | 392.22 | 390.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 386.95 | 392.22 | 390.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 387.05 | 391.19 | 389.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 387.40 | 391.19 | 389.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 393.05 | 390.90 | 390.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 391.05 | 390.90 | 390.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 385.60 | 390.03 | 389.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 385.60 | 390.03 | 389.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 383.50 | 388.72 | 389.26 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 14:15:00 | 390.65 | 389.47 | 389.40 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 387.20 | 388.94 | 389.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 385.30 | 388.21 | 388.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 387.15 | 387.00 | 387.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 386.70 | 387.00 | 387.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 387.65 | 387.13 | 387.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:15:00 | 388.85 | 387.13 | 387.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 389.35 | 387.57 | 388.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 390.00 | 387.57 | 388.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 389.10 | 387.88 | 388.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:15:00 | 390.00 | 387.88 | 388.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 389.20 | 388.14 | 388.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:30:00 | 390.85 | 388.14 | 388.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 388.95 | 388.30 | 388.28 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 387.00 | 388.21 | 388.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 385.40 | 387.32 | 387.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 381.65 | 381.64 | 383.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 381.65 | 381.64 | 383.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 386.20 | 381.90 | 382.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 387.65 | 381.90 | 382.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 387.50 | 383.02 | 383.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 386.85 | 383.02 | 383.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 386.70 | 383.76 | 383.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 13:15:00 | 388.70 | 385.26 | 384.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 388.00 | 388.24 | 386.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 12:45:00 | 388.35 | 388.24 | 386.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 387.45 | 388.08 | 386.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 387.50 | 388.08 | 386.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 387.25 | 387.91 | 386.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 387.25 | 387.91 | 386.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 388.60 | 388.05 | 386.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:45:00 | 390.35 | 388.63 | 387.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 11:00:00 | 389.00 | 390.22 | 389.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 389.20 | 388.81 | 388.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 386.30 | 388.34 | 388.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 386.30 | 388.34 | 388.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 383.75 | 387.43 | 388.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 366.30 | 364.62 | 368.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 366.30 | 364.62 | 368.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 369.00 | 365.49 | 368.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 369.45 | 365.49 | 368.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 368.50 | 366.09 | 368.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 365.95 | 366.20 | 368.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 365.00 | 364.77 | 365.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 366.95 | 365.95 | 365.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 366.95 | 365.95 | 365.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 370.30 | 367.12 | 366.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 366.45 | 367.94 | 367.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 366.45 | 367.94 | 367.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 366.45 | 367.94 | 367.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 366.45 | 367.94 | 367.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 365.50 | 367.45 | 367.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 362.85 | 367.45 | 367.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 363.95 | 366.75 | 366.78 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 368.80 | 367.16 | 366.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 369.75 | 367.68 | 367.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 365.75 | 368.19 | 367.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 365.75 | 368.19 | 367.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 365.75 | 368.19 | 367.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 365.75 | 368.19 | 367.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 365.85 | 367.72 | 367.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 365.85 | 367.72 | 367.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 365.00 | 367.18 | 367.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 362.50 | 366.24 | 366.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 362.55 | 360.39 | 362.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 362.55 | 360.39 | 362.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 362.55 | 360.39 | 362.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 363.30 | 360.39 | 362.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 359.30 | 360.17 | 362.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 363.05 | 360.17 | 362.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 363.80 | 360.89 | 362.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 363.65 | 360.89 | 362.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 369.80 | 362.68 | 363.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 369.80 | 362.68 | 363.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 371.55 | 364.45 | 363.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 372.65 | 366.09 | 364.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 15:15:00 | 367.40 | 367.42 | 365.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:15:00 | 370.00 | 367.42 | 365.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 369.25 | 367.79 | 366.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:30:00 | 373.80 | 369.25 | 366.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 363.00 | 367.94 | 368.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 363.00 | 367.94 | 368.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 360.45 | 366.44 | 367.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 368.35 | 365.79 | 366.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 368.35 | 365.79 | 366.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 368.35 | 365.79 | 366.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 368.90 | 365.79 | 366.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 366.55 | 365.94 | 366.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 367.05 | 365.94 | 366.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 365.45 | 365.84 | 366.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 364.40 | 365.84 | 366.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 363.90 | 365.21 | 366.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 14:15:00 | 373.25 | 366.82 | 366.95 | SL hit (close>static) qty=1.00 sl=367.50 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 371.05 | 367.66 | 367.32 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 360.00 | 366.13 | 366.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 358.80 | 364.66 | 365.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 359.65 | 358.74 | 361.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 359.15 | 358.74 | 361.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 363.40 | 359.16 | 360.63 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 363.25 | 361.50 | 361.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 365.70 | 362.34 | 361.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 358.50 | 362.84 | 362.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 358.50 | 362.84 | 362.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 358.50 | 362.84 | 362.33 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 357.70 | 361.81 | 361.91 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 15:15:00 | 363.00 | 361.86 | 361.78 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 353.05 | 360.10 | 360.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 351.75 | 358.43 | 360.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 363.60 | 359.46 | 360.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 11:15:00 | 363.60 | 359.46 | 360.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 363.60 | 359.46 | 360.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 363.60 | 359.46 | 360.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 363.15 | 360.20 | 360.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:00:00 | 363.15 | 360.20 | 360.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 362.85 | 360.73 | 360.90 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 14:15:00 | 363.10 | 361.20 | 361.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 371.35 | 363.64 | 362.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 370.00 | 374.71 | 369.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 370.00 | 374.71 | 369.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 370.00 | 374.71 | 369.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 374.30 | 374.17 | 370.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 411.73 | 401.07 | 396.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 11:15:00 | 396.05 | 399.63 | 399.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 10:15:00 | 394.50 | 396.74 | 398.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 13:15:00 | 397.00 | 396.38 | 397.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 13:45:00 | 397.00 | 396.38 | 397.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 397.45 | 396.59 | 397.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 397.45 | 396.59 | 397.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 396.60 | 396.59 | 397.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 396.00 | 396.59 | 397.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 399.50 | 397.17 | 397.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 399.50 | 397.17 | 397.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 401.00 | 397.94 | 397.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 401.00 | 397.94 | 397.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 399.85 | 398.32 | 398.16 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 397.25 | 398.74 | 398.75 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 399.90 | 398.66 | 398.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 402.80 | 400.04 | 399.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 399.80 | 400.34 | 399.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 14:15:00 | 399.80 | 400.34 | 399.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 399.80 | 400.34 | 399.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:45:00 | 398.70 | 400.34 | 399.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 398.65 | 400.00 | 399.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 401.35 | 400.00 | 399.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 393.85 | 398.69 | 399.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 393.85 | 398.69 | 399.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 392.40 | 397.43 | 398.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 396.90 | 395.43 | 396.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 396.90 | 395.43 | 396.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 396.90 | 395.43 | 396.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 396.90 | 395.43 | 396.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 399.15 | 396.17 | 397.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 399.15 | 396.17 | 397.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 399.95 | 396.93 | 397.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 400.95 | 396.93 | 397.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 402.60 | 398.63 | 398.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 405.70 | 400.04 | 398.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 13:15:00 | 400.60 | 402.78 | 401.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 13:15:00 | 400.60 | 402.78 | 401.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 400.60 | 402.78 | 401.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 400.60 | 402.78 | 401.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 398.25 | 401.87 | 400.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 398.30 | 401.87 | 400.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 397.00 | 400.90 | 400.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 401.90 | 400.90 | 400.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 397.80 | 400.28 | 400.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 397.80 | 400.28 | 400.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 11:15:00 | 397.45 | 399.38 | 399.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 12:15:00 | 399.65 | 399.43 | 399.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 13:00:00 | 399.65 | 399.43 | 399.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 398.50 | 399.25 | 399.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 400.15 | 399.25 | 399.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 399.55 | 399.31 | 399.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:15:00 | 399.95 | 399.31 | 399.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 399.95 | 399.44 | 399.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 402.15 | 399.44 | 399.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 400.00 | 399.55 | 399.76 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 400.95 | 399.73 | 399.71 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 398.70 | 399.52 | 399.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 397.15 | 399.05 | 399.39 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 13:15:00 | 321.35 | 2024-05-14 09:15:00 | 325.20 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-05-22 09:15:00 | 358.45 | 2024-05-27 12:15:00 | 365.95 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2024-05-31 10:45:00 | 355.70 | 2024-06-03 09:15:00 | 369.20 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2024-06-27 12:15:00 | 392.00 | 2024-07-03 14:15:00 | 385.90 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2024-07-05 09:15:00 | 397.80 | 2024-07-12 12:15:00 | 405.15 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2024-07-15 15:15:00 | 407.50 | 2024-07-16 10:15:00 | 417.20 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-08-23 09:15:00 | 413.40 | 2024-08-26 15:15:00 | 407.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-08-23 13:30:00 | 409.25 | 2024-08-26 15:15:00 | 407.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-08-23 14:00:00 | 409.25 | 2024-08-26 15:15:00 | 407.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-08-26 09:15:00 | 408.50 | 2024-08-26 15:15:00 | 407.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-09-09 12:15:00 | 391.50 | 2024-09-09 14:15:00 | 405.55 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2024-09-26 10:45:00 | 396.10 | 2024-09-26 12:15:00 | 395.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-09-26 12:15:00 | 396.45 | 2024-09-26 12:15:00 | 395.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-10-14 09:15:00 | 401.40 | 2024-10-17 13:15:00 | 394.20 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-10-21 11:30:00 | 384.00 | 2024-10-22 14:15:00 | 364.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 384.00 | 2024-10-28 10:15:00 | 356.85 | STOP_HIT | 0.50 | 7.07% |
| BUY | retest2 | 2024-11-01 18:00:00 | 373.85 | 2024-11-04 09:15:00 | 362.95 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-11-11 12:15:00 | 370.50 | 2024-11-12 11:15:00 | 372.10 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-11-11 13:30:00 | 370.55 | 2024-11-12 11:15:00 | 372.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-11-14 14:15:00 | 353.30 | 2024-11-18 09:15:00 | 360.05 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-12-02 09:30:00 | 403.40 | 2024-12-16 09:15:00 | 443.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 11:00:00 | 403.45 | 2024-12-16 09:15:00 | 443.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 13:30:00 | 403.40 | 2024-12-16 09:15:00 | 443.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 09:15:00 | 407.00 | 2024-12-17 09:15:00 | 447.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 09:15:00 | 418.75 | 2024-12-18 09:15:00 | 460.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 12:45:00 | 414.90 | 2024-12-18 09:15:00 | 456.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-10 09:15:00 | 415.05 | 2024-12-18 09:15:00 | 456.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-10 11:00:00 | 414.90 | 2024-12-18 09:15:00 | 456.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-10 14:30:00 | 419.70 | 2024-12-20 09:15:00 | 461.67 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-29 13:30:00 | 402.30 | 2025-01-30 09:15:00 | 412.30 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-02-01 09:15:00 | 414.20 | 2025-02-01 12:15:00 | 404.40 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-02-01 10:30:00 | 416.30 | 2025-02-01 12:15:00 | 404.40 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-02-01 11:30:00 | 420.15 | 2025-02-01 12:15:00 | 404.40 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-02-04 13:30:00 | 401.25 | 2025-02-04 15:15:00 | 401.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-02-04 15:00:00 | 400.50 | 2025-02-04 15:15:00 | 401.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-02-14 10:15:00 | 376.40 | 2025-02-17 10:15:00 | 357.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 376.40 | 2025-02-18 11:15:00 | 366.40 | STOP_HIT | 0.50 | 2.66% |
| BUY | retest2 | 2025-02-24 10:30:00 | 387.40 | 2025-02-28 09:15:00 | 377.15 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-02-25 09:30:00 | 386.65 | 2025-02-28 09:15:00 | 377.15 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-02-27 09:45:00 | 387.25 | 2025-02-28 09:15:00 | 377.15 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-03-13 11:30:00 | 373.45 | 2025-03-13 15:15:00 | 384.95 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-04-01 10:00:00 | 417.05 | 2025-04-03 09:15:00 | 423.60 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-04-02 12:45:00 | 417.60 | 2025-04-03 09:15:00 | 423.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-04-02 13:30:00 | 417.45 | 2025-04-03 09:15:00 | 423.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest1 | 2025-04-08 10:30:00 | 388.15 | 2025-04-08 14:15:00 | 399.15 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-04-09 09:15:00 | 389.50 | 2025-04-11 11:15:00 | 399.65 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-04-25 09:15:00 | 442.60 | 2025-04-28 09:15:00 | 420.50 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2025-04-25 11:45:00 | 428.10 | 2025-04-28 09:15:00 | 420.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-04-29 13:15:00 | 428.65 | 2025-04-30 14:15:00 | 417.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-04-29 13:45:00 | 429.20 | 2025-04-30 14:15:00 | 417.40 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest1 | 2025-05-07 13:45:00 | 413.75 | 2025-05-09 09:15:00 | 393.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-07 14:45:00 | 411.20 | 2025-05-09 09:15:00 | 390.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 12:45:00 | 409.35 | 2025-05-09 13:15:00 | 388.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-07 13:45:00 | 413.75 | 2025-05-12 09:15:00 | 409.40 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-05-07 14:45:00 | 411.20 | 2025-05-12 09:15:00 | 409.40 | STOP_HIT | 0.50 | 0.44% |
| SELL | retest2 | 2025-05-08 12:45:00 | 409.35 | 2025-05-12 09:15:00 | 409.40 | STOP_HIT | 0.50 | -0.01% |
| SELL | retest2 | 2025-05-12 09:30:00 | 409.65 | 2025-05-12 13:15:00 | 412.65 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-05-16 11:45:00 | 429.40 | 2025-05-22 10:15:00 | 428.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-05-21 09:30:00 | 431.85 | 2025-05-22 10:15:00 | 428.55 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-21 12:15:00 | 429.15 | 2025-05-22 10:15:00 | 428.55 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-05-22 10:15:00 | 429.00 | 2025-05-22 10:15:00 | 428.55 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-06-04 09:30:00 | 401.10 | 2025-06-09 10:15:00 | 406.75 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-06-05 10:15:00 | 401.30 | 2025-06-09 10:15:00 | 406.75 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-17 10:30:00 | 384.70 | 2025-06-24 09:15:00 | 384.20 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2025-06-26 09:15:00 | 385.80 | 2025-06-26 10:15:00 | 380.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-26 14:45:00 | 381.00 | 2025-06-26 15:15:00 | 381.75 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-07-08 09:30:00 | 376.65 | 2025-07-10 14:15:00 | 377.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-07-10 09:30:00 | 376.75 | 2025-07-10 14:15:00 | 377.65 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-10 10:45:00 | 376.70 | 2025-07-10 14:15:00 | 377.65 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-22 10:15:00 | 379.35 | 2025-07-22 11:15:00 | 388.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-07-24 11:45:00 | 380.75 | 2025-07-30 10:15:00 | 381.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-07-24 14:15:00 | 380.95 | 2025-07-30 10:15:00 | 381.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-07-24 14:45:00 | 380.50 | 2025-07-30 10:15:00 | 381.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-07-25 09:15:00 | 380.80 | 2025-07-30 10:15:00 | 381.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-07-25 10:15:00 | 376.85 | 2025-07-30 10:15:00 | 381.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-05 09:15:00 | 389.00 | 2025-08-06 09:15:00 | 381.60 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-08-05 13:45:00 | 384.00 | 2025-08-06 09:15:00 | 381.60 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-08-05 14:30:00 | 383.85 | 2025-08-06 09:15:00 | 381.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-08-06 09:15:00 | 386.25 | 2025-08-06 09:15:00 | 381.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-08-21 11:30:00 | 385.00 | 2025-08-28 15:15:00 | 365.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:30:00 | 385.00 | 2025-09-01 09:15:00 | 367.55 | STOP_HIT | 0.50 | 4.53% |
| BUY | retest2 | 2025-09-15 12:15:00 | 374.45 | 2025-09-17 09:15:00 | 369.15 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-15 14:30:00 | 374.50 | 2025-09-17 09:15:00 | 369.15 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-09-22 13:00:00 | 365.00 | 2025-09-29 11:15:00 | 363.85 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-09-23 10:30:00 | 366.05 | 2025-09-29 11:15:00 | 363.85 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-09-30 12:15:00 | 366.80 | 2025-10-03 13:15:00 | 364.55 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-01 14:00:00 | 366.55 | 2025-10-03 13:15:00 | 364.55 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-01 15:00:00 | 366.40 | 2025-10-03 13:15:00 | 364.55 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-13 14:45:00 | 385.05 | 2025-10-14 15:15:00 | 376.40 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-16 15:15:00 | 387.55 | 2025-10-24 15:15:00 | 385.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-17 09:30:00 | 386.60 | 2025-10-24 15:15:00 | 385.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-10-17 10:00:00 | 386.95 | 2025-10-24 15:15:00 | 385.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-10-20 13:00:00 | 387.05 | 2025-10-24 15:15:00 | 385.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-10-28 14:15:00 | 383.95 | 2025-10-29 14:15:00 | 385.65 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-29 09:45:00 | 384.10 | 2025-10-29 14:15:00 | 385.65 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-04 09:30:00 | 377.30 | 2025-11-04 10:15:00 | 380.75 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-11-04 14:30:00 | 376.50 | 2025-11-07 15:15:00 | 380.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-07 13:45:00 | 377.45 | 2025-11-07 15:15:00 | 380.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-11-18 10:45:00 | 395.30 | 2025-11-19 10:15:00 | 389.25 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-11-18 11:30:00 | 394.85 | 2025-11-19 10:15:00 | 389.25 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-11-18 12:30:00 | 394.65 | 2025-11-19 10:15:00 | 389.25 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-01 09:15:00 | 389.35 | 2025-12-02 10:15:00 | 383.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-01 10:15:00 | 389.65 | 2025-12-02 10:15:00 | 383.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-12-01 10:45:00 | 391.20 | 2025-12-02 10:15:00 | 383.90 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-01 14:00:00 | 389.10 | 2025-12-02 10:15:00 | 383.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-04 09:15:00 | 384.25 | 2025-12-10 13:15:00 | 380.40 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-12-04 10:45:00 | 383.95 | 2025-12-10 13:15:00 | 380.40 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2025-12-15 13:45:00 | 383.00 | 2025-12-16 15:15:00 | 376.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-22 11:00:00 | 365.60 | 2025-12-22 14:15:00 | 370.90 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-12-22 12:00:00 | 366.30 | 2025-12-22 14:15:00 | 370.90 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-01 09:45:00 | 383.10 | 2026-01-05 13:15:00 | 377.25 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-01-01 11:30:00 | 383.05 | 2026-01-05 13:15:00 | 377.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-12 09:15:00 | 361.50 | 2026-01-13 09:15:00 | 372.80 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-01-12 10:45:00 | 363.80 | 2026-01-13 09:15:00 | 372.80 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-01-30 10:00:00 | 375.15 | 2026-02-02 09:15:00 | 370.10 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-02-04 11:45:00 | 370.45 | 2026-02-04 13:15:00 | 373.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-25 09:45:00 | 390.35 | 2026-02-27 09:15:00 | 386.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-26 11:00:00 | 389.00 | 2026-02-27 09:15:00 | 386.30 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-02-26 15:00:00 | 389.20 | 2026-02-27 09:15:00 | 386.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-03-06 10:45:00 | 365.95 | 2026-03-10 14:15:00 | 366.95 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-03-10 09:15:00 | 365.00 | 2026-03-10 14:15:00 | 366.95 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-03-18 10:30:00 | 373.80 | 2026-03-19 13:15:00 | 363.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-03-20 12:15:00 | 364.40 | 2026-03-20 14:15:00 | 373.25 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-03-20 13:30:00 | 363.90 | 2026-03-20 14:15:00 | 373.25 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-04-02 11:45:00 | 374.30 | 2026-04-16 09:15:00 | 411.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 401.35 | 2026-04-30 09:15:00 | 393.85 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-05-06 09:15:00 | 401.90 | 2026-05-06 09:15:00 | 397.80 | STOP_HIT | 1.00 | -1.02% |

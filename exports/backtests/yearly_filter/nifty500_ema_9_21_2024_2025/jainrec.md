# Jain Resource Recycling Ltd. (JAINREC)

## Backtest Summary

- **Window:** 2025-10-01 09:15:00 → 2026-05-11 15:15:00 (1031 bars)
- **Last close:** 561.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 42 |
| ALERT1 | 30 |
| ALERT2 | 29 |
| ALERT2_SKIP | 15 |
| ALERT3 | 69 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 25
- **Target hits / Stop hits / Partials:** 0 / 31 / 6
- **Avg / median % per leg:** -0.54% / -1.02%
- **Sum % (uncompounded):** -20.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 1 | 6.7% | 0 | 15 | 0 | -2.38% | -35.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 0 | 15 | 0 | -2.38% | -35.6% |
| SELL (all) | 22 | 11 | 50.0% | 0 | 16 | 6 | 0.71% | 15.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 11 | 50.0% | 0 | 16 | 6 | 0.71% | 15.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 12 | 32.4% | 0 | 31 | 6 | -0.54% | -20.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 326.03 | 338.39 | 338.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 322.06 | 329.52 | 333.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 330.00 | 329.62 | 333.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 330.00 | 329.62 | 333.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 330.00 | 329.62 | 333.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 329.68 | 329.62 | 333.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 326.45 | 323.27 | 325.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 326.00 | 323.27 | 325.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 326.24 | 323.87 | 325.95 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 339.60 | 327.45 | 327.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 342.90 | 332.30 | 329.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 385.30 | 387.73 | 374.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:00:00 | 385.30 | 387.73 | 374.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 395.88 | 389.29 | 381.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 397.30 | 391.57 | 383.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:30:00 | 396.80 | 393.39 | 386.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:45:00 | 407.91 | 391.64 | 388.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 399.74 | 394.78 | 391.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 395.60 | 394.95 | 391.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 394.21 | 394.95 | 391.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 391.50 | 394.26 | 391.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 391.50 | 394.26 | 391.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 391.99 | 393.80 | 391.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 391.99 | 393.80 | 391.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 387.00 | 392.44 | 391.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 387.00 | 392.44 | 391.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 386.19 | 391.19 | 391.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 386.78 | 391.19 | 391.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 386.59 | 390.27 | 390.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 386.59 | 390.27 | 390.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 384.40 | 388.68 | 389.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 375.35 | 374.94 | 380.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 375.35 | 374.94 | 380.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 375.35 | 374.94 | 380.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:45:00 | 378.55 | 374.94 | 380.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 378.40 | 374.65 | 377.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 378.40 | 374.65 | 377.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 374.25 | 374.57 | 377.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 371.10 | 374.57 | 377.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 363.65 | 372.38 | 376.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 362.75 | 370.67 | 375.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 362.75 | 369.08 | 374.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 390.60 | 376.57 | 375.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 390.60 | 376.57 | 375.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 409.50 | 386.65 | 380.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 430.45 | 432.04 | 420.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:45:00 | 430.55 | 432.04 | 420.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 422.90 | 428.59 | 421.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 437.35 | 428.59 | 421.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 431.60 | 444.32 | 445.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 431.60 | 444.32 | 445.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 428.20 | 439.34 | 443.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 14:15:00 | 428.00 | 427.82 | 433.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:30:00 | 429.65 | 427.82 | 433.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 397.50 | 395.72 | 400.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 399.10 | 395.72 | 400.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 399.55 | 397.00 | 399.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 399.45 | 397.00 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 395.50 | 396.70 | 399.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 394.20 | 396.35 | 399.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 394.60 | 396.35 | 399.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 374.49 | 379.30 | 384.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 374.87 | 379.30 | 384.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 384.30 | 379.75 | 383.54 | SL hit (close>ema200) qty=0.50 sl=379.75 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 383.00 | 376.11 | 375.35 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 371.65 | 374.63 | 374.77 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 394.20 | 378.38 | 376.38 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 377.05 | 382.06 | 382.33 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 384.75 | 382.60 | 382.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 388.00 | 384.71 | 383.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 400.10 | 402.56 | 396.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 10:15:00 | 384.85 | 399.02 | 395.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 384.85 | 399.02 | 395.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 384.85 | 399.02 | 395.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 384.60 | 396.13 | 394.81 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 387.00 | 392.81 | 393.44 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 14:15:00 | 400.95 | 394.44 | 394.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 414.55 | 403.01 | 400.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 417.90 | 420.43 | 415.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:30:00 | 418.50 | 420.43 | 415.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 419.50 | 419.87 | 416.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 422.65 | 416.06 | 415.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 412.70 | 416.13 | 415.87 | SL hit (close<static) qty=1.00 sl=416.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 410.90 | 415.08 | 415.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 387.50 | 409.56 | 412.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 398.25 | 395.67 | 403.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 398.25 | 395.67 | 403.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 399.55 | 395.54 | 402.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 403.50 | 395.54 | 402.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 400.35 | 396.22 | 399.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 400.35 | 396.22 | 399.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 398.00 | 396.58 | 399.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 409.15 | 396.58 | 399.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 412.55 | 399.77 | 400.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 411.70 | 399.77 | 400.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 411.45 | 402.11 | 401.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 416.80 | 408.93 | 405.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 409.15 | 411.16 | 407.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:00:00 | 409.15 | 411.16 | 407.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 406.85 | 410.03 | 408.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 406.85 | 410.03 | 408.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 405.50 | 409.13 | 407.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 412.75 | 409.13 | 407.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 403.85 | 412.58 | 413.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 403.85 | 412.58 | 413.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 397.80 | 409.63 | 412.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 405.00 | 402.46 | 407.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 15:00:00 | 405.00 | 402.46 | 407.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 404.05 | 403.34 | 406.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:30:00 | 400.90 | 402.49 | 406.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 401.75 | 399.38 | 401.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 398.25 | 399.90 | 401.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:30:00 | 401.95 | 402.09 | 402.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 397.00 | 401.07 | 401.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:30:00 | 396.85 | 401.07 | 401.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 397.60 | 398.87 | 400.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 397.85 | 398.87 | 400.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 381.85 | 394.78 | 398.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 405.00 | 396.27 | 397.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 405.00 | 396.27 | 397.99 | SL hit (close>ema200) qty=0.50 sl=396.27 alert=retest2 |

### Cycle 16 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 409.30 | 400.12 | 399.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 414.35 | 407.29 | 403.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 10:15:00 | 407.25 | 408.49 | 405.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 11:00:00 | 407.25 | 408.49 | 405.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 408.00 | 410.43 | 408.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 408.30 | 410.43 | 408.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 409.65 | 410.27 | 408.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 410.15 | 410.27 | 408.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 15:15:00 | 407.25 | 409.51 | 408.44 | SL hit (close<static) qty=1.00 sl=407.55 alert=retest2 |

### Cycle 17 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 398.25 | 407.26 | 407.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 391.10 | 395.74 | 399.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 14:15:00 | 396.05 | 394.43 | 398.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-20 15:00:00 | 396.05 | 394.43 | 398.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 389.50 | 387.74 | 391.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 393.50 | 387.74 | 391.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 388.35 | 387.86 | 391.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 387.00 | 387.86 | 391.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 385.25 | 384.20 | 387.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 10:15:00 | 367.65 | 376.20 | 381.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 381.30 | 375.35 | 379.07 | SL hit (close>ema200) qty=0.50 sl=375.35 alert=retest2 |

### Cycle 18 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 388.00 | 381.46 | 380.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 389.45 | 383.06 | 381.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 402.00 | 403.17 | 397.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 402.00 | 403.17 | 397.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 401.80 | 402.14 | 397.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 407.95 | 404.11 | 398.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:30:00 | 406.25 | 405.95 | 400.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 394.80 | 400.41 | 399.90 | SL hit (close<static) qty=1.00 sl=397.00 alert=retest2 |

### Cycle 19 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 382.55 | 396.83 | 398.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 379.35 | 393.34 | 396.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 386.95 | 386.55 | 391.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 386.95 | 386.55 | 391.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 399.00 | 388.95 | 391.97 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 402.50 | 394.92 | 394.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 411.70 | 400.96 | 397.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 402.45 | 406.20 | 402.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 402.45 | 406.20 | 402.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 402.45 | 406.20 | 402.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 402.50 | 406.20 | 402.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 402.90 | 405.54 | 402.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 402.90 | 405.54 | 402.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 402.60 | 404.95 | 402.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:15:00 | 401.50 | 404.95 | 402.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 401.90 | 404.34 | 402.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 401.15 | 404.34 | 402.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 403.45 | 403.86 | 402.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:15:00 | 401.00 | 403.86 | 402.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 401.00 | 403.29 | 402.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 391.75 | 403.29 | 402.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 389.35 | 400.50 | 401.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 14:15:00 | 387.50 | 392.91 | 396.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 398.90 | 393.36 | 396.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 398.90 | 393.36 | 396.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 398.90 | 393.36 | 396.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 402.50 | 393.36 | 396.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 408.00 | 396.29 | 397.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 408.00 | 396.29 | 397.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 410.00 | 399.03 | 398.49 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 393.90 | 399.98 | 400.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 10:15:00 | 387.00 | 397.38 | 398.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 393.35 | 391.58 | 395.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 393.35 | 391.58 | 395.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 395.20 | 392.27 | 394.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 395.45 | 392.27 | 394.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 391.00 | 392.01 | 394.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:30:00 | 388.55 | 391.32 | 393.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 395.90 | 391.65 | 393.40 | SL hit (close>static) qty=1.00 sl=395.20 alert=retest2 |

### Cycle 24 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 380.25 | 378.78 | 378.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 381.95 | 379.54 | 379.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 12:15:00 | 378.40 | 379.75 | 379.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 12:15:00 | 378.40 | 379.75 | 379.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 378.40 | 379.75 | 379.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 378.40 | 379.75 | 379.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 378.75 | 379.55 | 379.29 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 372.60 | 378.05 | 378.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 369.65 | 375.38 | 377.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 379.00 | 371.75 | 373.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 14:15:00 | 379.00 | 371.75 | 373.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 379.00 | 371.75 | 373.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 379.00 | 371.75 | 373.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 379.00 | 373.20 | 373.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 382.60 | 373.20 | 373.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 391.10 | 376.78 | 375.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 14:15:00 | 396.05 | 384.52 | 382.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 382.90 | 386.03 | 383.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 382.90 | 386.03 | 383.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 382.90 | 386.03 | 383.27 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 375.30 | 381.71 | 381.79 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 400.15 | 384.61 | 383.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 14:15:00 | 403.05 | 391.32 | 387.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 404.00 | 409.90 | 404.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 404.00 | 409.90 | 404.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 404.00 | 409.90 | 404.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 417.50 | 411.29 | 407.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 417.55 | 421.85 | 422.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 417.55 | 421.85 | 422.35 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 427.90 | 422.79 | 422.68 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 419.65 | 422.04 | 422.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 418.50 | 421.33 | 422.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 426.05 | 421.31 | 421.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 426.05 | 421.31 | 421.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 426.05 | 421.31 | 421.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 426.05 | 421.31 | 421.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 422.95 | 421.64 | 421.79 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 426.60 | 422.48 | 421.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 445.30 | 427.74 | 424.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 11:15:00 | 433.65 | 436.27 | 432.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 11:15:00 | 433.65 | 436.27 | 432.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 433.65 | 436.27 | 432.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 433.65 | 436.27 | 432.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 433.15 | 435.64 | 432.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:30:00 | 435.20 | 435.56 | 432.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 417.85 | 433.60 | 433.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 417.85 | 433.60 | 433.94 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 15:15:00 | 439.00 | 432.45 | 432.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 09:15:00 | 448.80 | 435.72 | 433.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 449.00 | 454.70 | 449.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 449.00 | 454.70 | 449.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 449.00 | 454.70 | 449.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 450.05 | 454.70 | 449.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 453.20 | 454.40 | 449.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 460.70 | 455.19 | 451.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 471.30 | 460.03 | 457.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 11:15:00 | 456.55 | 460.43 | 460.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 456.55 | 460.43 | 460.44 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 463.00 | 460.95 | 460.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 465.60 | 461.88 | 461.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 456.50 | 461.23 | 461.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 456.50 | 461.23 | 461.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 456.50 | 461.23 | 461.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 456.50 | 461.23 | 461.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 453.05 | 459.59 | 460.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 15:15:00 | 449.00 | 454.53 | 457.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 14:15:00 | 445.20 | 445.09 | 448.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:45:00 | 446.65 | 445.09 | 448.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 441.20 | 444.24 | 447.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 436.90 | 443.47 | 446.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:15:00 | 415.05 | 420.07 | 423.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 15:15:00 | 419.95 | 418.93 | 421.46 | SL hit (close>ema200) qty=0.50 sl=418.93 alert=retest2 |

### Cycle 38 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 422.55 | 419.02 | 418.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 425.00 | 420.31 | 419.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 10:15:00 | 418.50 | 419.95 | 419.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 10:15:00 | 418.50 | 419.95 | 419.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 418.50 | 419.95 | 419.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 418.50 | 419.95 | 419.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 419.45 | 419.85 | 419.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:45:00 | 418.25 | 419.85 | 419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 417.70 | 419.42 | 419.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:45:00 | 417.40 | 419.42 | 419.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 417.35 | 419.00 | 418.98 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 413.15 | 417.83 | 418.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 411.20 | 416.01 | 417.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 10:15:00 | 418.20 | 416.44 | 417.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 418.20 | 416.44 | 417.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 418.20 | 416.44 | 417.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 418.20 | 416.44 | 417.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 414.05 | 415.97 | 417.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:15:00 | 413.70 | 415.97 | 417.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 413.40 | 414.73 | 416.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 416.70 | 413.09 | 413.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 416.70 | 413.09 | 413.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 419.80 | 414.44 | 413.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 454.70 | 456.88 | 446.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 454.70 | 456.88 | 446.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 460.40 | 464.79 | 460.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 459.00 | 464.79 | 460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 462.55 | 464.34 | 461.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 464.60 | 462.12 | 460.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 458.70 | 461.18 | 460.65 | SL hit (close<static) qty=1.00 sl=460.30 alert=retest2 |

### Cycle 41 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 458.00 | 460.15 | 460.25 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 464.00 | 461.03 | 460.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 486.00 | 466.02 | 462.94 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-28 11:45:00 | 397.30 | 2025-10-31 13:15:00 | 386.59 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-10-28 14:30:00 | 396.80 | 2025-10-31 13:15:00 | 386.59 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-10-30 09:45:00 | 407.91 | 2025-10-31 13:15:00 | 386.59 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest2 | 2025-10-30 14:45:00 | 399.74 | 2025-10-31 13:15:00 | 386.59 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-11-06 11:15:00 | 362.75 | 2025-11-07 12:15:00 | 390.60 | STOP_HIT | 1.00 | -7.68% |
| SELL | retest2 | 2025-11-06 12:00:00 | 362.75 | 2025-11-07 12:15:00 | 390.60 | STOP_HIT | 1.00 | -7.68% |
| BUY | retest2 | 2025-11-13 09:15:00 | 437.35 | 2025-11-20 10:15:00 | 431.60 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-01 09:30:00 | 394.20 | 2025-12-03 12:15:00 | 374.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:15:00 | 394.60 | 2025-12-03 12:15:00 | 374.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 394.20 | 2025-12-03 14:15:00 | 384.30 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-12-01 10:15:00 | 394.60 | 2025-12-03 14:15:00 | 384.30 | STOP_HIT | 0.50 | 2.61% |
| BUY | retest2 | 2025-12-26 09:15:00 | 422.65 | 2025-12-26 14:15:00 | 412.70 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-01-02 09:15:00 | 412.75 | 2026-01-06 09:15:00 | 403.85 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-01-07 10:30:00 | 400.90 | 2026-01-12 11:15:00 | 381.85 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-07 10:30:00 | 400.90 | 2026-01-12 14:15:00 | 405.00 | STOP_HIT | 0.50 | -1.02% |
| SELL | retest2 | 2026-01-08 15:00:00 | 401.75 | 2026-01-13 09:15:00 | 409.30 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-01-09 09:15:00 | 398.25 | 2026-01-13 09:15:00 | 409.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-01-09 12:30:00 | 401.95 | 2026-01-13 09:15:00 | 409.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-01-16 14:15:00 | 410.15 | 2026-01-16 15:15:00 | 407.25 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-22 10:15:00 | 387.00 | 2026-01-27 10:15:00 | 367.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 387.00 | 2026-01-27 14:15:00 | 381.30 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2026-01-23 10:45:00 | 385.25 | 2026-01-28 12:15:00 | 388.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-28 12:15:00 | 386.50 | 2026-01-28 12:15:00 | 388.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2026-01-30 14:45:00 | 407.95 | 2026-02-01 15:15:00 | 394.80 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2026-02-01 09:30:00 | 406.25 | 2026-02-01 15:15:00 | 394.80 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-02-12 11:30:00 | 388.55 | 2026-02-12 14:15:00 | 395.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-02-13 09:15:00 | 390.60 | 2026-02-16 11:15:00 | 371.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 390.60 | 2026-02-17 12:15:00 | 379.70 | STOP_HIT | 0.50 | 2.79% |
| BUY | retest2 | 2026-03-09 14:30:00 | 417.50 | 2026-03-13 12:15:00 | 417.55 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-03-19 14:30:00 | 435.20 | 2026-03-23 09:15:00 | 417.85 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2026-03-27 14:45:00 | 460.70 | 2026-04-02 11:15:00 | 456.55 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-04-01 09:15:00 | 471.30 | 2026-04-02 11:15:00 | 456.55 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-04-09 12:15:00 | 436.90 | 2026-04-16 11:15:00 | 415.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-09 12:15:00 | 436.90 | 2026-04-16 15:15:00 | 419.95 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-04-23 12:15:00 | 413.70 | 2026-04-27 11:15:00 | 416.70 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-04-23 13:30:00 | 413.40 | 2026-04-27 11:15:00 | 416.70 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-05-06 09:30:00 | 464.60 | 2026-05-06 11:15:00 | 458.70 | STOP_HIT | 1.00 | -1.27% |

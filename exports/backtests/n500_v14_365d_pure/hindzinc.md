# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 634.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 25
- **Target hits / Stop hits / Partials:** 0 / 25 / 0
- **Avg / median % per leg:** -2.31% / -2.06%
- **Sum % (uncompounded):** -57.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.85% | -39.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.04% | -3.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.84% | -36.9% |
| SELL (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.62% | -17.8% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.65% | -6.6% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.60% | -11.2% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.93% | -9.6% |
| retest2 (combined) | 20 | 0 | 0.0% | 0 | 20 | 0 | -2.40% | -48.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 451.20 | 438.84 | 438.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 455.35 | 439.00 | 438.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 422.55 | 453.45 | 453.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 419.70 | 442.02 | 446.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 431.70 | 431.35 | 438.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 12:15:00 | 430.80 | 431.37 | 437.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 14:30:00 | 431.10 | 431.38 | 437.86 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 15:15:00 | 430.80 | 431.38 | 437.86 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 430.85 | 431.30 | 437.31 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | SL hit (close>ema400) qty=1.00 sl=436.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | SL hit (close>ema400) qty=1.00 sl=436.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | SL hit (close>ema400) qty=1.00 sl=436.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | SL hit (close>ema400) qty=1.00 sl=436.20 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 438.00 | 430.31 | 436.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 437.40 | 430.38 | 436.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 437.40 | 430.38 | 436.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 439.05 | 430.46 | 436.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 440.25 | 430.46 | 436.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 437.25 | 430.53 | 436.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:15:00 | 435.30 | 430.53 | 436.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 441.10 | 430.79 | 436.24 | SL hit (close>static) qty=1.00 sl=439.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 436.00 | 432.30 | 436.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 440.40 | 432.38 | 436.69 | SL hit (close>static) qty=1.00 sl=439.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 435.95 | 432.87 | 436.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 436.30 | 432.87 | 436.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 434.65 | 432.89 | 436.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 436.45 | 432.89 | 436.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 438.95 | 432.98 | 436.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 438.95 | 432.98 | 436.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 440.05 | 433.05 | 436.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 440.05 | 433.05 | 436.79 | SL hit (close>static) qty=1.00 sl=439.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 440.05 | 433.05 | 436.79 | SL hit (close>static) qty=1.00 sl=439.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 440.05 | 433.05 | 436.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 434.90 | 433.45 | 436.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 438.00 | 433.45 | 436.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 436.40 | 433.48 | 436.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 433.65 | 433.48 | 436.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 435.85 | 433.50 | 436.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:15:00 | 431.80 | 433.50 | 436.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:30:00 | 432.00 | 433.50 | 436.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 431.10 | 433.50 | 436.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 441.80 | 433.68 | 436.69 | SL hit (close>static) qty=1.00 sl=440.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 441.80 | 433.68 | 436.69 | SL hit (close>static) qty=1.00 sl=440.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 441.80 | 433.68 | 436.69 | SL hit (close>static) qty=1.00 sl=440.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 459.90 | 439.28 | 439.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 465.95 | 442.46 | 440.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 476.55 | 480.39 | 467.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 482.65 | 479.88 | 467.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | SL hit (close<ema400) qty=1.00 sl=468.95 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 468.00 | 479.26 | 468.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 467.75 | 479.15 | 468.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 466.25 | 479.15 | 468.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 467.25 | 478.80 | 468.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 467.35 | 478.80 | 468.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 469.00 | 481.03 | 472.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 469.00 | 481.03 | 472.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 473.15 | 480.95 | 472.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 474.00 | 480.87 | 472.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 475.05 | 480.87 | 472.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 473.80 | 480.16 | 472.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:00:00 | 473.95 | 480.10 | 472.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 472.15 | 480.02 | 472.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 472.15 | 480.02 | 472.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 472.00 | 479.94 | 472.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 469.60 | 479.94 | 472.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 465.25 | 479.79 | 472.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 465.25 | 479.79 | 472.60 | SL hit (close<static) qty=1.00 sl=468.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 465.25 | 479.79 | 472.60 | SL hit (close<static) qty=1.00 sl=468.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 465.25 | 479.79 | 472.60 | SL hit (close<static) qty=1.00 sl=468.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 465.25 | 479.79 | 472.60 | SL hit (close<static) qty=1.00 sl=468.20 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 465.25 | 479.79 | 472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 578.55 | 639.21 | 593.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 564.80 | 639.21 | 593.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 582.00 | 634.19 | 592.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 583.90 | 634.19 | 592.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 609.10 | 632.44 | 592.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:15:00 | 613.50 | 632.44 | 592.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:30:00 | 613.45 | 631.63 | 592.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:15:00 | 612.40 | 630.75 | 595.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 612.20 | 630.56 | 595.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 596.70 | 630.02 | 595.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:45:00 | 604.95 | 629.46 | 595.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 605.65 | 628.72 | 595.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 593.20 | 625.39 | 599.19 | SL hit (close<static) qty=1.00 sl=594.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 593.20 | 625.39 | 599.19 | SL hit (close<static) qty=1.00 sl=594.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 588.50 | 624.71 | 599.11 | SL hit (close<static) qty=1.00 sl=592.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 588.50 | 624.71 | 599.11 | SL hit (close<static) qty=1.00 sl=592.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 588.50 | 624.71 | 599.11 | SL hit (close<static) qty=1.00 sl=592.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 588.50 | 624.71 | 599.11 | SL hit (close<static) qty=1.00 sl=592.25 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 609.70 | 610.94 | 596.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 617.05 | 611.33 | 598.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 598.15 | 611.55 | 599.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 598.15 | 611.55 | 599.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 590.15 | 611.34 | 599.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 590.15 | 611.34 | 599.02 | SL hit (close<static) qty=1.00 sl=594.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 590.15 | 611.34 | 599.02 | SL hit (close<static) qty=1.00 sl=594.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 590.15 | 611.34 | 599.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 595.00 | 610.07 | 598.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 595.00 | 610.07 | 598.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 598.65 | 609.95 | 598.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 600.30 | 604.56 | 597.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 12:15:00 | 589.35 | 604.10 | 597.17 | SL hit (close<static) qty=1.00 sl=593.20 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 547.35 | 591.31 | 591.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 09:15:00 | 531.10 | 590.27 | 590.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 621.05 | 573.70 | 573.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 630.75 | 584.55 | 579.52 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-21 12:15:00 | 430.80 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest1 | 2025-08-21 14:30:00 | 431.10 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest1 | 2025-08-21 15:15:00 | 430.80 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest1 | 2025-08-26 09:30:00 | 430.85 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-01 13:15:00 | 435.30 | 2025-09-02 09:15:00 | 441.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-04 09:15:00 | 436.00 | 2025-09-04 09:15:00 | 440.40 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-05 10:45:00 | 435.95 | 2025-09-05 14:15:00 | 440.05 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-05 11:15:00 | 436.30 | 2025-09-05 14:15:00 | 440.05 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-09 12:15:00 | 431.80 | 2025-09-11 11:15:00 | 441.80 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-09-10 12:30:00 | 432.00 | 2025-09-11 11:15:00 | 441.80 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-10 13:00:00 | 431.10 | 2025-09-11 11:15:00 | 441.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2025-10-29 09:30:00 | 482.65 | 2025-11-06 09:15:00 | 468.00 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-11-18 11:30:00 | 474.00 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-11-18 12:15:00 | 475.05 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-20 13:15:00 | 473.80 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-11-20 14:00:00 | 473.95 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-02-02 15:15:00 | 613.50 | 2026-02-13 14:15:00 | 593.20 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2026-02-03 11:30:00 | 613.45 | 2026-02-13 14:15:00 | 593.20 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-02-05 14:15:00 | 612.40 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2026-02-05 14:45:00 | 612.20 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-02-06 11:45:00 | 604.95 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-02-06 14:30:00 | 605.65 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-25 09:15:00 | 609.70 | 2026-03-04 10:15:00 | 590.15 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-03-02 09:15:00 | 617.05 | 2026-03-04 10:15:00 | 590.15 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2026-03-11 09:15:00 | 600.30 | 2026-03-11 12:15:00 | 589.35 | STOP_HIT | 1.00 | -1.82% |

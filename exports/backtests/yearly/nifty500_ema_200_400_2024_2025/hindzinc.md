# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 634.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 27 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 31
- **Target hits / Stop hits / Partials:** 0 / 32 / 1
- **Avg / median % per leg:** -1.97% / -2.06%
- **Sum % (uncompounded):** -64.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 0 | 0.0% | 0 | 18 | 0 | -2.70% | -48.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.04% | -3.0% |
| BUY @ 3rd Alert (retest2) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.68% | -45.5% |
| SELL (all) | 15 | 2 | 13.3% | 0 | 14 | 1 | -1.10% | -16.4% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.65% | -6.6% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 10 | 1 | -0.89% | -9.8% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.93% | -9.6% |
| retest2 (combined) | 28 | 2 | 7.1% | 0 | 27 | 1 | -1.98% | -55.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 513.40 | 603.08 | 603.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 497.15 | 574.15 | 587.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 516.45 | 514.37 | 539.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:00:00 | 516.45 | 514.37 | 539.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 529.60 | 512.54 | 527.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:15:00 | 528.55 | 512.54 | 527.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 528.60 | 512.70 | 527.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:30:00 | 535.00 | 512.70 | 527.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 525.15 | 512.82 | 527.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 523.55 | 513.08 | 527.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 534.70 | 513.49 | 527.12 | SL hit (close>static) qty=1.00 sl=530.25 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 451.20 | 438.84 | 438.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 455.35 | 439.00 | 438.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-07-10 10:15:00)

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

### Cycle 4 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 459.90 | 439.28 | 439.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 465.95 | 442.46 | 440.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 476.55 | 480.39 | 467.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 482.65 | 479.88 | 467.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | SL hit (close<ema400) qty=1.00 sl=468.95 alert=retest1 |

### Cycle 5 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 547.35 | 591.31 | 591.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 09:15:00 | 531.10 | 590.27 | 590.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 621.05 | 573.70 | 573.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 630.75 | 584.55 | 579.52 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-26 09:15:00 | 619.50 | 2024-07-26 13:15:00 | 607.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-07-29 09:45:00 | 618.75 | 2024-08-06 15:15:00 | 606.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-07-29 11:00:00 | 619.40 | 2024-08-06 15:15:00 | 606.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-07-29 13:15:00 | 620.80 | 2024-08-06 15:15:00 | 606.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-10-23 14:15:00 | 523.55 | 2024-10-24 09:15:00 | 534.70 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-25 09:30:00 | 519.35 | 2024-10-29 09:15:00 | 538.95 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2024-11-06 09:15:00 | 518.70 | 2024-11-13 09:15:00 | 492.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 09:15:00 | 518.70 | 2024-12-03 14:15:00 | 506.85 | STOP_HIT | 0.50 | 2.28% |
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

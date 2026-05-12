# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 561.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 55 |
| ALERT1 | 40 |
| ALERT2 | 39 |
| ALERT2_SKIP | 16 |
| ALERT3 | 101 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 47 |
| PARTIAL | 13 |
| TARGET_HIT | 5 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 29
- **Target hits / Stop hits / Partials:** 5 / 45 / 13
- **Avg / median % per leg:** 1.35% / 0.96%
- **Sum % (uncompounded):** 84.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 7 | 36.8% | 3 | 16 | 0 | 0.30% | 5.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 7 | 36.8% | 3 | 16 | 0 | 0.30% | 5.7% |
| SELL (all) | 44 | 27 | 61.4% | 2 | 29 | 13 | 1.80% | 79.1% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.43% | 26.6% |
| SELL @ 3rd Alert (retest2) | 38 | 21 | 55.3% | 2 | 26 | 10 | 1.38% | 52.5% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.43% | 26.6% |
| retest2 (combined) | 57 | 28 | 49.1% | 5 | 42 | 10 | 1.02% | 58.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 454.35 | 440.06 | 439.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 471.60 | 452.62 | 446.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 459.65 | 460.22 | 455.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 459.65 | 460.22 | 455.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 506.70 | 513.65 | 506.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 506.70 | 513.65 | 506.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 507.95 | 512.51 | 506.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 511.35 | 511.40 | 507.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 510.25 | 511.28 | 508.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 514.70 | 511.93 | 509.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 516.25 | 518.35 | 518.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 516.25 | 518.35 | 518.64 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 520.85 | 519.09 | 518.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 538.50 | 524.14 | 521.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 580.00 | 581.37 | 571.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 575.55 | 581.37 | 571.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 570.25 | 579.15 | 571.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 569.70 | 579.15 | 571.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 574.20 | 578.16 | 571.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 581.40 | 574.19 | 572.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 576.35 | 575.65 | 573.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 581.90 | 573.48 | 573.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 579.50 | 574.84 | 574.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 592.55 | 578.38 | 575.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 15:15:00 | 578.00 | 579.31 | 579.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 578.00 | 579.31 | 579.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 575.40 | 578.15 | 578.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 579.45 | 577.59 | 578.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 571.60 | 576.39 | 577.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 567.00 | 572.41 | 575.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-13 09:15:00 | 510.30 | 558.73 | 566.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 536.50 | 525.14 | 523.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 546.45 | 531.65 | 526.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 564.00 | 564.21 | 557.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:00:00 | 564.00 | 564.21 | 557.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 553.30 | 562.03 | 557.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 553.30 | 562.03 | 557.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 556.95 | 561.01 | 557.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 562.55 | 561.01 | 557.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 554.40 | 561.61 | 561.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 554.40 | 561.61 | 561.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 552.10 | 555.50 | 558.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 547.00 | 545.57 | 549.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:45:00 | 546.70 | 545.57 | 549.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 547.10 | 546.03 | 549.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 547.50 | 546.03 | 549.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 540.90 | 543.80 | 546.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 539.50 | 542.94 | 545.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 550.45 | 543.63 | 544.97 | SL hit (close>static) qty=1.00 sl=548.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 549.75 | 546.03 | 545.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 560.15 | 550.29 | 548.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 605.00 | 607.11 | 597.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 605.00 | 607.11 | 597.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 602.50 | 604.80 | 599.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 596.60 | 604.80 | 599.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 592.55 | 602.35 | 598.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 592.55 | 602.35 | 598.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 594.50 | 600.78 | 598.54 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 590.85 | 596.36 | 596.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 588.30 | 593.57 | 595.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 565.05 | 564.84 | 570.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 565.85 | 564.84 | 570.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 565.80 | 565.55 | 569.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 567.10 | 565.55 | 569.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 591.20 | 569.63 | 570.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 597.40 | 569.63 | 570.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 10:15:00 | 582.65 | 572.24 | 571.57 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 565.85 | 572.96 | 573.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 561.65 | 570.70 | 571.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 567.50 | 567.26 | 569.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 567.50 | 567.26 | 569.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 571.00 | 568.00 | 569.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 571.85 | 568.00 | 569.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 572.80 | 568.96 | 570.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 571.80 | 568.96 | 570.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 573.05 | 569.78 | 570.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 573.45 | 569.78 | 570.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 579.00 | 571.62 | 571.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 584.90 | 575.83 | 573.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 571.40 | 574.21 | 574.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 565.40 | 572.45 | 573.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 565.05 | 564.02 | 566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 565.30 | 564.14 | 565.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 568.10 | 564.14 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 564.80 | 564.27 | 565.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 564.95 | 564.27 | 565.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 565.30 | 564.10 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 566.00 | 564.10 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 560.05 | 563.29 | 564.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:45:00 | 558.80 | 562.93 | 564.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 559.40 | 562.93 | 564.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 559.30 | 560.19 | 563.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 531.43 | 539.31 | 544.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 530.86 | 537.27 | 542.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 531.33 | 537.27 | 542.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 537.65 | 537.34 | 542.50 | SL hit (close>ema200) qty=0.50 sl=537.34 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 547.90 | 539.21 | 538.86 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 538.60 | 541.17 | 541.42 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 553.30 | 543.29 | 542.33 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 544.00 | 546.63 | 546.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 543.35 | 545.48 | 546.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 526.00 | 519.51 | 524.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 522.70 | 520.13 | 522.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 525.55 | 520.13 | 522.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 529.70 | 522.04 | 523.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 531.35 | 522.04 | 523.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 534.30 | 524.49 | 524.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 536.00 | 526.79 | 525.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 539.25 | 540.62 | 536.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:15:00 | 539.00 | 540.62 | 536.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 536.40 | 539.77 | 536.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 536.40 | 539.77 | 536.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 536.70 | 539.16 | 536.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 541.50 | 537.12 | 536.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 533.30 | 536.25 | 535.81 | SL hit (close<static) qty=1.00 sl=535.10 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 529.20 | 534.84 | 535.21 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 536.85 | 535.43 | 535.33 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 532.65 | 534.81 | 535.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 531.75 | 534.19 | 534.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 533.45 | 524.35 | 527.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 533.00 | 526.08 | 527.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 535.35 | 526.08 | 527.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 531.65 | 529.25 | 529.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 535.00 | 530.40 | 529.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 531.55 | 530.34 | 529.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 530.45 | 530.36 | 529.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 529.65 | 530.36 | 529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 529.80 | 530.25 | 529.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 529.55 | 530.25 | 529.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 528.65 | 529.93 | 529.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 528.65 | 529.93 | 529.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 525.90 | 529.12 | 529.27 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 534.45 | 529.89 | 529.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 566.65 | 539.10 | 534.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 585.65 | 587.15 | 578.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 585.65 | 587.15 | 578.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 588.00 | 587.23 | 584.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 600.10 | 587.23 | 584.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 09:15:00 | 660.11 | 627.33 | 608.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 701.65 | 704.21 | 704.47 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 734.25 | 710.05 | 707.07 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 701.40 | 706.43 | 707.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 696.90 | 702.30 | 704.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 698.00 | 697.72 | 701.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 698.00 | 697.72 | 701.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 695.45 | 696.67 | 700.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 691.80 | 696.14 | 699.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 692.55 | 693.15 | 696.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 692.50 | 693.04 | 696.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.21 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.92 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.88 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 673.80 | 667.48 | 676.56 | SL hit (close>ema200) qty=0.50 sl=667.48 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 636.90 | 628.05 | 627.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 646.70 | 636.03 | 632.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 642.70 | 644.93 | 641.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 642.70 | 644.93 | 641.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 637.45 | 643.43 | 641.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 637.75 | 643.43 | 641.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 641.70 | 643.08 | 641.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:15:00 | 645.85 | 642.39 | 641.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 627.80 | 643.10 | 643.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 627.80 | 643.10 | 643.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 623.70 | 636.61 | 640.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 622.30 | 616.83 | 624.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 15:00:00 | 622.30 | 616.83 | 624.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 629.80 | 620.25 | 625.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 636.35 | 620.25 | 625.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 625.95 | 623.44 | 625.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 626.30 | 623.44 | 625.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 626.75 | 624.10 | 625.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 626.75 | 624.10 | 625.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 627.25 | 624.73 | 625.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 627.25 | 624.73 | 625.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 626.00 | 624.98 | 625.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 618.70 | 624.98 | 625.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 634.75 | 620.35 | 619.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 634.75 | 620.35 | 619.13 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 621.55 | 623.70 | 623.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 617.95 | 622.55 | 623.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 627.55 | 619.93 | 620.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 624.85 | 620.91 | 621.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 626.00 | 620.91 | 621.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 603.00 | 611.67 | 615.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 605.20 | 611.67 | 615.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 610.45 | 594.90 | 597.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 610.60 | 594.90 | 597.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 610.00 | 597.92 | 598.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 611.90 | 597.92 | 598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 607.75 | 600.74 | 600.06 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 594.85 | 599.63 | 600.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 589.00 | 597.51 | 599.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 582.50 | 580.80 | 586.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:15:00 | 578.60 | 580.80 | 586.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:30:00 | 578.95 | 580.22 | 585.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 12:30:00 | 578.00 | 579.93 | 584.69 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 549.67 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 550.00 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 549.10 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-03 15:15:00 | 556.20 | 554.69 | 561.92 | SL hit (close>ema200) qty=0.50 sl=554.69 alert=retest1 |

### Cycle 33 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 533.20 | 514.45 | 512.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 543.75 | 520.31 | 515.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 528.30 | 535.88 | 536.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 525.20 | 533.75 | 535.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 532.20 | 529.78 | 532.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 532.20 | 529.78 | 532.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 530.55 | 529.94 | 532.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 532.00 | 529.94 | 532.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 531.35 | 530.22 | 532.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 533.40 | 530.22 | 532.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 532.60 | 530.70 | 532.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 532.60 | 530.70 | 532.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 530.60 | 530.68 | 532.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 535.15 | 530.68 | 532.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 540.00 | 532.54 | 532.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 541.80 | 532.54 | 532.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 544.60 | 534.95 | 533.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 551.75 | 541.54 | 537.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 557.30 | 557.97 | 553.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 557.30 | 557.97 | 553.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 561.00 | 563.57 | 559.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 561.00 | 563.57 | 559.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 559.45 | 562.23 | 559.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 559.45 | 562.23 | 559.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 559.35 | 561.65 | 559.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 557.05 | 561.65 | 559.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 553.20 | 559.96 | 559.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 553.20 | 559.96 | 559.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 552.65 | 558.50 | 558.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 545.60 | 555.20 | 557.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 537.25 | 533.59 | 541.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 537.25 | 533.59 | 541.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 545.15 | 535.90 | 541.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 545.15 | 535.90 | 541.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 548.00 | 538.32 | 542.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 541.70 | 539.23 | 542.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 554.90 | 542.36 | 543.24 | SL hit (close>static) qty=1.00 sl=548.40 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 556.35 | 545.16 | 544.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 560.65 | 554.01 | 550.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 585.35 | 586.76 | 577.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 585.35 | 586.76 | 577.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 591.90 | 586.58 | 581.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 592.95 | 586.58 | 581.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 592.55 | 587.58 | 582.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 592.05 | 587.58 | 582.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:00:00 | 592.45 | 588.56 | 583.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 597.00 | 591.41 | 586.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 606.60 | 593.26 | 587.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 578.25 | 589.57 | 587.17 | SL hit (close<static) qty=1.00 sl=586.00 alert=retest2 |

### Cycle 38 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 572.80 | 584.58 | 585.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 569.65 | 581.59 | 583.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 558.25 | 552.74 | 561.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 566.15 | 555.42 | 562.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 566.15 | 555.42 | 562.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 571.70 | 558.67 | 563.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 566.45 | 558.67 | 563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 565.65 | 561.34 | 563.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 563.80 | 561.34 | 563.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 563.70 | 563.25 | 563.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 560.60 | 562.72 | 563.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 535.61 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 535.51 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.57 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 527.05 | 526.98 | 535.93 | SL hit (close>ema200) qty=0.50 sl=526.98 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 535.50 | 531.82 | 531.60 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 530.00 | 531.45 | 531.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 524.40 | 530.04 | 530.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 489.10 | 487.26 | 497.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 489.10 | 487.26 | 497.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 492.45 | 489.38 | 496.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 490.20 | 489.38 | 496.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 489.50 | 491.32 | 494.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 512.65 | 497.02 | 496.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 512.65 | 497.02 | 496.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 514.20 | 505.90 | 501.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 562.85 | 564.10 | 551.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 554.50 | 564.10 | 551.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 564.00 | 564.08 | 552.26 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 545.85 | 551.53 | 551.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 539.20 | 547.91 | 550.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 561.05 | 546.17 | 547.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 565.85 | 550.11 | 549.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 574.75 | 563.45 | 557.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 565.50 | 568.25 | 562.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 565.50 | 568.25 | 562.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 565.00 | 567.60 | 562.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 564.85 | 567.60 | 562.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 555.80 | 565.09 | 562.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 555.80 | 565.09 | 562.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 558.85 | 563.84 | 562.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 561.45 | 563.77 | 562.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 551.50 | 560.17 | 561.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 551.50 | 560.17 | 561.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 548.65 | 554.29 | 557.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 542.45 | 539.23 | 543.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 543.50 | 540.09 | 543.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 544.70 | 540.09 | 543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 543.05 | 540.68 | 543.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 541.00 | 540.68 | 543.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 544.20 | 541.38 | 543.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 543.90 | 541.38 | 543.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 538.35 | 540.78 | 543.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:45:00 | 535.20 | 539.75 | 542.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 554.50 | 544.04 | 543.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 554.50 | 544.04 | 543.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 560.15 | 547.26 | 545.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 557.20 | 558.35 | 553.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 546.40 | 555.96 | 553.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 546.40 | 555.96 | 553.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 544.20 | 553.61 | 552.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 545.00 | 553.61 | 552.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 551.70 | 552.09 | 551.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 551.70 | 552.09 | 551.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 549.15 | 551.50 | 551.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 543.75 | 548.15 | 549.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 548.20 | 546.82 | 548.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 548.85 | 547.22 | 548.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 542.45 | 547.22 | 548.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 515.33 | 528.33 | 533.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 13:15:00 | 488.21 | 499.67 | 511.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 464.85 | 452.00 | 451.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 466.35 | 454.87 | 452.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 446.80 | 454.74 | 455.61 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 462.50 | 456.88 | 456.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 14:15:00 | 466.50 | 460.54 | 458.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 446.80 | 459.15 | 458.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 445.40 | 456.40 | 457.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 443.40 | 453.80 | 455.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 446.30 | 446.14 | 450.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 447.50 | 446.14 | 450.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 454.20 | 447.75 | 450.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 454.20 | 447.75 | 450.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 454.60 | 449.12 | 450.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 454.75 | 449.12 | 450.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 449.20 | 449.10 | 450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 465.00 | 449.10 | 450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 467.85 | 452.85 | 452.05 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 439.95 | 453.21 | 453.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 437.30 | 450.03 | 452.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 447.25 | 436.69 | 435.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 454.60 | 442.43 | 439.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 443.60 | 444.01 | 441.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 443.60 | 444.01 | 441.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 454.65 | 451.81 | 446.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:45:00 | 461.40 | 455.06 | 450.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 475.05 | 456.42 | 451.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 507.54 | 491.01 | 485.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 512.10 | 514.31 | 514.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 510.30 | 513.51 | 514.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 493.25 | 489.03 | 498.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 493.25 | 489.03 | 498.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 498.20 | 491.53 | 498.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 498.20 | 491.53 | 498.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 504.30 | 494.09 | 498.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 501.40 | 494.09 | 498.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 501.00 | 495.47 | 498.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 502.40 | 495.47 | 498.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 500.90 | 496.55 | 499.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 498.60 | 496.55 | 499.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 497.50 | 497.17 | 498.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 493.55 | 497.17 | 498.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:15:00 | 494.35 | 496.75 | 498.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 494.30 | 495.62 | 497.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 493.30 | 495.62 | 497.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 493.20 | 494.51 | 496.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 491.50 | 494.55 | 496.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 487.40 | 493.42 | 495.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 507.70 | 491.98 | 492.43 | SL hit (close>static) qty=1.00 sl=499.75 alert=retest2 |

### Cycle 55 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 508.80 | 495.35 | 493.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 515.30 | 506.80 | 501.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:45:00 | 447.50 | 2025-05-12 11:15:00 | 454.35 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-21 10:00:00 | 511.35 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-05-21 13:00:00 | 510.25 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-05-21 14:30:00 | 514.70 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-06-05 09:15:00 | 581.40 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-05 12:30:00 | 576.35 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-06 10:15:00 | 581.90 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-06-09 09:15:00 | 579.50 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-12 11:45:00 | 567.00 | 2025-06-13 09:15:00 | 510.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:15:00 | 562.55 | 2025-07-02 10:15:00 | 554.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-08 11:00:00 | 539.50 | 2025-07-08 14:15:00 | 550.45 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-05 14:45:00 | 558.80 | 2025-08-08 15:15:00 | 531.43 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-08-05 15:15:00 | 559.40 | 2025-08-11 09:15:00 | 530.86 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-08-06 09:30:00 | 559.30 | 2025-08-11 09:15:00 | 531.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 14:45:00 | 558.80 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-08-05 15:15:00 | 559.40 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-08-06 09:30:00 | 559.30 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2025-09-05 09:45:00 | 541.50 | 2025-09-05 11:15:00 | 533.30 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-09-19 10:15:00 | 600.10 | 2025-09-22 09:15:00 | 660.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-10 11:15:00 | 691.80 | 2025-10-14 14:15:00 | 657.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 15:00:00 | 692.55 | 2025-10-14 14:15:00 | 657.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 10:00:00 | 692.50 | 2025-10-14 14:15:00 | 657.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:15:00 | 691.80 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-10-10 15:00:00 | 692.55 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-10-13 10:00:00 | 692.50 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest2 | 2025-11-03 12:15:00 | 645.85 | 2025-11-06 09:15:00 | 627.80 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-11-11 09:15:00 | 618.70 | 2025-11-17 09:15:00 | 634.75 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest1 | 2025-12-01 10:15:00 | 578.60 | 2025-12-03 11:15:00 | 549.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 11:30:00 | 578.95 | 2025-12-03 11:15:00 | 550.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 12:30:00 | 578.00 | 2025-12-03 11:15:00 | 549.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 10:15:00 | 578.60 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest1 | 2025-12-01 11:30:00 | 578.95 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest1 | 2025-12-01 12:30:00 | 578.00 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-12-10 11:30:00 | 504.15 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-12-11 10:45:00 | 505.50 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-12-11 11:45:00 | 505.00 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-12-11 12:45:00 | 505.65 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-31 09:30:00 | 541.70 | 2025-12-31 10:15:00 | 554.90 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-07 10:15:00 | 592.95 | 2026-01-08 12:15:00 | 578.25 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-01-07 10:45:00 | 592.55 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-01-07 11:15:00 | 592.05 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-01-07 12:00:00 | 592.45 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-01-08 10:30:00 | 606.60 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2026-01-13 11:15:00 | 563.80 | 2026-01-20 09:15:00 | 535.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 563.70 | 2026-01-20 09:15:00 | 535.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 560.60 | 2026-01-20 09:15:00 | 532.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:15:00 | 563.80 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2026-01-14 10:00:00 | 563.70 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-01-14 11:00:00 | 560.60 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2026-01-29 10:15:00 | 490.20 | 2026-01-30 10:15:00 | 512.65 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-01-29 15:00:00 | 489.50 | 2026-01-30 10:15:00 | 512.65 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2026-02-11 11:45:00 | 561.45 | 2026-02-12 10:15:00 | 551.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-02-17 11:45:00 | 535.20 | 2026-02-18 10:15:00 | 554.50 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-02 09:15:00 | 515.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-04 13:15:00 | 488.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-07 13:45:00 | 461.40 | 2026-04-15 10:15:00 | 507.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 475.05 | 2026-04-21 10:15:00 | 522.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 493.55 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-28 12:15:00 | 494.35 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-28 13:30:00 | 494.30 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-28 14:00:00 | 493.30 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-29 14:00:00 | 491.50 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-30 09:15:00 | 487.40 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -4.16% |

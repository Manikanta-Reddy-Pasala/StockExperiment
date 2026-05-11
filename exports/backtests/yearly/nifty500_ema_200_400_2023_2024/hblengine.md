# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 850.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 16 |
| PARTIAL | 5 |
| TARGET_HIT | 9 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 9
- **Target hits / Stop hits / Partials:** 9 / 9 / 5
- **Avg / median % per leg:** 4.20% / 5.00%
- **Sum % (uncompounded):** 96.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 8 | 53.3% | 6 | 7 | 2 | 3.84% | 57.6% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 11 | 4 | 36.4% | 4 | 7 | 0 | 2.50% | 27.6% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 4.89% | 39.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 3 | 2 | 3 | 4.89% | 39.1% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 19 | 10 | 52.6% | 7 | 9 | 3 | 3.51% | 66.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 546.55 | 602.52 | 602.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 13:15:00 | 544.75 | 601.95 | 602.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 579.95 | 579.79 | 589.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:45:00 | 577.15 | 579.79 | 589.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 585.00 | 579.76 | 589.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:45:00 | 586.95 | 579.76 | 589.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 577.65 | 564.70 | 577.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:00:00 | 577.65 | 564.70 | 577.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 574.05 | 564.79 | 577.15 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 637.00 | 586.50 | 586.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 660.95 | 589.97 | 588.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 638.15 | 639.02 | 618.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 12:45:00 | 638.00 | 639.02 | 618.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 613.90 | 638.07 | 620.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 613.90 | 638.07 | 620.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 617.95 | 637.87 | 620.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 619.55 | 637.87 | 620.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:30:00 | 622.95 | 637.48 | 620.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 618.25 | 635.10 | 621.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 603.00 | 634.78 | 621.41 | SL hit (close<static) qty=1.00 sl=611.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 541.20 | 611.62 | 611.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 13:15:00 | 532.65 | 608.86 | 610.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 585.05 | 572.27 | 588.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 585.05 | 572.27 | 588.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 585.05 | 572.27 | 588.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:30:00 | 587.70 | 572.27 | 588.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 589.00 | 572.59 | 587.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:30:00 | 596.60 | 572.59 | 587.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 591.80 | 572.78 | 587.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 591.80 | 572.78 | 587.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 597.05 | 573.02 | 588.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 597.05 | 573.02 | 588.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 569.80 | 574.24 | 588.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 563.90 | 575.30 | 586.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 564.55 | 575.24 | 586.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 565.85 | 575.06 | 586.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 535.70 | 573.77 | 585.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 536.32 | 573.77 | 585.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 537.56 | 573.77 | 585.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 507.51 | 570.18 | 583.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 570.95 | 505.75 | 505.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 578.35 | 508.36 | 506.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 577.35 | 577.44 | 554.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:15:00 | 582.50 | 577.44 | 554.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:15:00 | 581.75 | 576.45 | 556.15 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 611.62 | 581.24 | 562.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 610.84 | 581.24 | 562.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-07-01 10:15:00 | 640.75 | 581.77 | 562.45 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 814.25 | 873.97 | 874.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 804.80 | 872.16 | 873.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 845.90 | 855.33 | 863.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 883.80 | 857.02 | 864.13 | SL hit (close>static) qty=1.00 sl=881.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 925.55 | 870.40 | 870.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 932.05 | 876.69 | 873.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 896.65 | 896.85 | 885.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:45:00 | 894.00 | 896.85 | 885.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 882.90 | 896.66 | 885.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 882.90 | 896.66 | 885.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 877.00 | 896.47 | 885.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 877.00 | 896.47 | 885.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 879.45 | 896.30 | 885.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 858.10 | 896.30 | 885.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 880.45 | 888.77 | 882.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 772.00 | 888.77 | 882.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 757.50 | 875.97 | 876.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 752.95 | 874.75 | 875.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 09:15:00 | 804.60 | 800.02 | 825.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:45:00 | 810.85 | 800.02 | 825.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 704.55 | 679.82 | 718.83 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 809.90 | 741.14 | 740.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 825.60 | 747.20 | 743.93 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-31 09:15:00 | 619.55 | 2025-01-06 10:15:00 | 603.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-12-31 10:30:00 | 622.95 | 2025-01-06 10:15:00 | 603.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-01-06 09:30:00 | 618.25 | 2025-01-06 10:15:00 | 603.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-02-07 09:15:00 | 563.90 | 2025-02-10 09:15:00 | 535.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 564.55 | 2025-02-10 09:15:00 | 536.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 11:30:00 | 565.85 | 2025-02-10 09:15:00 | 537.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 563.90 | 2025-02-11 09:15:00 | 507.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 564.55 | 2025-02-11 09:15:00 | 508.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 11:30:00 | 565.85 | 2025-02-11 09:15:00 | 509.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-14 15:15:00 | 563.00 | 2025-05-16 12:15:00 | 570.95 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest1 | 2025-06-19 09:15:00 | 582.50 | 2025-07-01 09:15:00 | 611.62 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-24 09:15:00 | 581.75 | 2025-07-01 09:15:00 | 610.84 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-19 09:15:00 | 582.50 | 2025-07-01 10:15:00 | 640.75 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-24 09:15:00 | 581.75 | 2025-07-01 10:15:00 | 639.93 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-24 09:15:00 | 588.50 | 2025-07-25 09:15:00 | 582.85 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-24 11:30:00 | 587.30 | 2025-07-25 09:15:00 | 582.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-24 12:15:00 | 588.85 | 2025-07-25 09:15:00 | 582.85 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-30 10:15:00 | 588.30 | 2025-07-30 14:15:00 | 580.25 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-31 13:00:00 | 588.70 | 2025-08-11 09:15:00 | 647.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 15:15:00 | 589.25 | 2025-08-11 09:15:00 | 648.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-01 09:30:00 | 591.05 | 2025-08-11 09:15:00 | 650.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 13:15:00 | 589.00 | 2025-08-11 09:15:00 | 647.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-22 15:15:00 | 845.90 | 2025-12-24 11:15:00 | 883.80 | STOP_HIT | 1.00 | -4.48% |

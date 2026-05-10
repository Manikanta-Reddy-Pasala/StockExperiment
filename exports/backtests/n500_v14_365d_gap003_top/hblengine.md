# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 850.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 8 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / Stop hits / Partials:** 6 / 4 / 2
- **Avg / median % per leg:** 5.49% / 10.00%
- **Sum % (uncompounded):** 65.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 6 | 4 | 2 | 5.49% | 65.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 4 | 4 | 0 | 4.49% | 35.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 8 | 4 | 50.0% | 4 | 4 | 0 | 4.49% | 35.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 570.95 | 505.75 | 505.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 578.35 | 508.36 | 506.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 577.35 | 577.44 | 554.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:15:00 | 582.50 | 577.44 | 554.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:15:00 | 581.75 | 576.45 | 556.15 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 611.62 | 581.24 | 562.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 610.84 | 581.24 | 562.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-07-01 10:15:00 | 640.75 | 581.77 | 562.45 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-07-01 10:15:00 | 639.93 | 581.77 | 562.45 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 583.75 | 601.49 | 584.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 583.75 | 601.49 | 584.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 585.20 | 601.33 | 584.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 588.50 | 601.19 | 584.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 11:30:00 | 587.30 | 600.80 | 584.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 588.85 | 600.80 | 584.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 582.85 | 600.15 | 584.56 | SL hit (close<static) qty=1.00 sl=583.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 582.85 | 600.15 | 584.56 | SL hit (close<static) qty=1.00 sl=583.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 582.85 | 600.15 | 584.56 | SL hit (close<static) qty=1.00 sl=583.15 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 588.30 | 594.22 | 583.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 584.00 | 594.03 | 583.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 584.00 | 594.03 | 583.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 583.65 | 593.92 | 583.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 583.65 | 593.92 | 583.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 583.60 | 593.82 | 583.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 583.85 | 593.82 | 583.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 580.25 | 593.69 | 583.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 580.25 | 593.69 | 583.00 | SL hit (close<static) qty=1.00 sl=583.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 580.25 | 593.69 | 583.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 582.45 | 593.57 | 583.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 579.90 | 593.57 | 583.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 586.00 | 593.50 | 583.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 588.70 | 593.31 | 583.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 589.25 | 593.20 | 583.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:30:00 | 591.05 | 593.12 | 583.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 589.00 | 595.63 | 585.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-11 09:15:00 | 647.57 | 596.63 | 587.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-11 09:15:00 | 648.18 | 596.63 | 587.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-11 09:15:00 | 650.15 | 596.63 | 587.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-11 09:15:00 | 647.90 | 596.63 | 587.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 899.60 | 958.50 | 901.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 899.60 | 958.50 | 901.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 896.50 | 957.88 | 901.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 892.80 | 957.88 | 901.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 886.30 | 957.17 | 901.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 882.70 | 957.17 | 901.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 884.00 | 956.44 | 901.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 884.00 | 956.44 | 901.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 890.00 | 951.68 | 900.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 890.00 | 951.68 | 900.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 882.90 | 942.83 | 899.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 882.90 | 942.83 | 899.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 882.60 | 935.17 | 898.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 882.50 | 935.17 | 898.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| CROSSOVER_SKIP | 2025-12-15 13:15:00 | 814.25 | 873.97 | 874.07 | min_gap filter: gap=0.012% < 0.030% |
| TREND_RESET | 2025-12-15 13:15:00 | 814.25 | 873.97 | 874.07 | EMA inversion without crossover edge (EMA200=873.97 EMA400=874.07) — end cycle |
| CROSSOVER_SKIP | 2025-12-31 10:15:00 | 925.55 | 870.40 | 870.30 | min_gap filter: gap=0.011% < 0.030% |
| CROSSOVER_SKIP | 2026-01-19 14:15:00 | 757.50 | 875.97 | 876.12 | min_gap filter: gap=0.020% < 0.030% |

### Cycle 2 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 809.90 | 741.14 | 740.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 825.60 | 747.20 | 743.93 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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

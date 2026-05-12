# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 850.05
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
| ALERT2_SKIP | 1 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 10 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / Stop hits / Partials:** 6 / 6 / 2
- **Avg / median % per leg:** 4.29% / 5.00%
- **Sum % (uncompounded):** 60.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 6 | 4 | 2 | 5.49% | 65.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 4 | 4 | 0 | 4.49% | 35.9% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.95% | -5.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.95% | -5.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 10 | 4 | 40.0% | 4 | 6 | 0 | 3.00% | 30.0% |

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

### Cycle 2 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 814.25 | 873.97 | 874.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 804.80 | 872.16 | 873.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 845.90 | 855.33 | 863.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 883.80 | 857.02 | 864.13 | SL hit (close>static) qty=1.00 sl=881.90 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-31 10:15:00)

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

### Cycle 4 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 757.50 | 875.97 | 876.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 752.95 | 874.75 | 875.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 09:15:00 | 804.60 | 800.02 | 825.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:45:00 | 810.85 | 800.02 | 825.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 704.55 | 679.82 | 718.83 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 809.90 | 741.14 | 740.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 825.60 | 747.20 | 743.93 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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

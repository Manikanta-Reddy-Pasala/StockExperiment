# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 565.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 211 |
| ALERT1 | 140 |
| ALERT2 | 140 |
| ALERT2_SKIP | 61 |
| ALERT3 | 378 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 161 |
| PARTIAL | 25 |
| TARGET_HIT | 9 |
| STOP_HIT | 153 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 187 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 92 / 95
- **Target hits / Stop hits / Partials:** 9 / 153 / 25
- **Avg / median % per leg:** 1.31% / -0.15%
- **Sum % (uncompounded):** 244.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 28 | 32.9% | 7 | 77 | 1 | 0.63% | 53.9% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.52% | 9.0% |
| BUY @ 3rd Alert (retest2) | 83 | 26 | 31.3% | 7 | 76 | 0 | 0.54% | 44.9% |
| SELL (all) | 102 | 64 | 62.7% | 2 | 76 | 24 | 1.87% | 190.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 102 | 64 | 62.7% | 2 | 76 | 24 | 1.87% | 190.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.52% | 9.0% |
| retest2 (combined) | 185 | 90 | 48.6% | 9 | 152 | 24 | 1.27% | 235.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 435.90 | 429.72 | 428.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 442.20 | 434.98 | 432.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 14:15:00 | 440.90 | 441.19 | 438.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 14:45:00 | 440.65 | 441.19 | 438.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 435.35 | 440.02 | 438.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:45:00 | 435.00 | 440.02 | 438.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 434.80 | 438.98 | 438.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:45:00 | 434.90 | 438.98 | 438.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 438.20 | 437.89 | 437.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 438.20 | 437.89 | 437.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 438.50 | 438.01 | 437.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 440.75 | 438.01 | 437.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 440.50 | 438.51 | 438.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 11:45:00 | 442.85 | 439.70 | 438.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 12:30:00 | 442.50 | 440.27 | 439.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 13:00:00 | 442.55 | 440.27 | 439.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-05 15:15:00 | 486.75 | 481.55 | 477.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 14:15:00 | 494.55 | 501.70 | 502.37 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 10:15:00 | 508.15 | 502.71 | 502.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 11:15:00 | 509.50 | 504.07 | 503.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 14:15:00 | 502.10 | 504.90 | 503.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 14:15:00 | 502.10 | 504.90 | 503.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 502.10 | 504.90 | 503.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 15:00:00 | 502.10 | 504.90 | 503.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 15:15:00 | 502.90 | 504.50 | 503.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:15:00 | 509.35 | 504.50 | 503.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-20 09:15:00 | 560.29 | 546.44 | 535.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 13:15:00 | 555.80 | 560.18 | 560.30 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 09:15:00 | 568.10 | 560.64 | 560.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 10:15:00 | 575.35 | 563.58 | 561.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-26 13:15:00 | 564.10 | 564.21 | 562.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-26 13:30:00 | 564.70 | 564.21 | 562.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 563.50 | 564.07 | 562.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:45:00 | 563.30 | 564.07 | 562.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 563.40 | 563.93 | 562.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 570.20 | 563.93 | 562.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 565.20 | 564.19 | 562.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 12:45:00 | 571.75 | 566.04 | 564.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 12:30:00 | 570.90 | 571.47 | 570.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 12:15:00 | 572.95 | 572.91 | 571.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 13:15:00 | 571.65 | 573.52 | 573.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-04 13:15:00 | 564.05 | 571.63 | 572.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 564.05 | 571.63 | 572.31 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 577.55 | 572.69 | 572.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 09:15:00 | 588.30 | 577.32 | 575.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 12:15:00 | 592.35 | 593.52 | 587.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 13:00:00 | 592.35 | 593.52 | 587.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 594.85 | 593.79 | 587.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 13:30:00 | 589.65 | 593.79 | 587.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 600.00 | 604.11 | 599.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 600.00 | 604.11 | 599.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 598.30 | 602.95 | 599.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:45:00 | 596.80 | 602.95 | 599.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 15:15:00 | 601.75 | 602.71 | 599.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:30:00 | 597.90 | 600.48 | 598.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 587.30 | 597.84 | 597.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:00:00 | 587.30 | 597.84 | 597.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 586.05 | 595.48 | 596.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 09:15:00 | 581.45 | 588.51 | 592.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 14:15:00 | 583.00 | 582.30 | 587.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-17 15:00:00 | 583.00 | 582.30 | 587.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 579.65 | 581.91 | 586.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 584.25 | 581.91 | 586.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 13:15:00 | 583.50 | 579.31 | 583.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 14:00:00 | 583.50 | 579.31 | 583.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 14:15:00 | 576.75 | 578.80 | 582.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 15:15:00 | 573.90 | 578.80 | 582.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:15:00 | 545.20 | 551.84 | 559.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-07-21 10:15:00 | 553.25 | 552.12 | 558.80 | SL hit (close>ema200) qty=0.50 sl=552.12 alert=retest2 |

### Cycle 9 — BUY (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 11:15:00 | 565.15 | 558.77 | 558.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 12:15:00 | 570.35 | 561.09 | 559.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 09:15:00 | 572.50 | 574.30 | 569.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 572.50 | 574.30 | 569.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 572.50 | 574.30 | 569.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:30:00 | 569.25 | 574.30 | 569.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 583.95 | 577.33 | 573.37 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 11:15:00 | 572.00 | 576.19 | 576.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 11:15:00 | 570.30 | 573.47 | 574.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 571.70 | 569.85 | 571.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 10:15:00 | 571.70 | 569.85 | 571.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 571.70 | 569.85 | 571.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:45:00 | 572.50 | 569.85 | 571.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 575.20 | 570.92 | 572.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 12:00:00 | 575.20 | 570.92 | 572.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 572.40 | 571.21 | 572.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 14:30:00 | 570.90 | 571.26 | 572.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 11:45:00 | 571.15 | 571.47 | 571.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 13:15:00 | 577.25 | 573.03 | 572.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 13:15:00 | 577.25 | 573.03 | 572.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 14:15:00 | 578.75 | 574.18 | 573.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 13:15:00 | 575.50 | 576.26 | 574.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 13:15:00 | 575.50 | 576.26 | 574.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 575.50 | 576.26 | 574.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:00:00 | 575.50 | 576.26 | 574.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 579.95 | 577.00 | 575.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:30:00 | 571.55 | 577.00 | 575.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 583.05 | 578.16 | 576.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 585.30 | 579.25 | 577.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 12:15:00 | 569.90 | 576.04 | 576.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 12:15:00 | 569.90 | 576.04 | 576.53 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 578.50 | 576.80 | 576.77 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 575.70 | 576.75 | 576.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 570.70 | 574.87 | 575.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 541.70 | 539.49 | 543.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 11:00:00 | 541.70 | 539.49 | 543.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 550.40 | 540.88 | 542.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:00:00 | 550.40 | 540.88 | 542.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 558.65 | 544.44 | 543.75 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 542.50 | 548.23 | 548.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 541.55 | 545.78 | 547.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 14:15:00 | 548.30 | 545.23 | 546.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 14:15:00 | 548.30 | 545.23 | 546.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 548.30 | 545.23 | 546.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:00:00 | 548.30 | 545.23 | 546.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 547.90 | 545.76 | 546.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:15:00 | 548.40 | 545.76 | 546.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 553.00 | 547.21 | 547.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:00:00 | 553.00 | 547.21 | 547.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 551.20 | 548.01 | 547.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 558.05 | 553.10 | 550.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 14:15:00 | 562.05 | 563.95 | 561.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 14:15:00 | 562.05 | 563.95 | 561.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 562.05 | 563.95 | 561.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:45:00 | 562.45 | 563.95 | 561.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 560.00 | 563.16 | 561.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 560.55 | 563.16 | 561.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 556.55 | 561.84 | 560.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:00:00 | 556.55 | 561.84 | 560.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 10:15:00 | 554.35 | 560.34 | 560.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 10:15:00 | 550.10 | 555.01 | 557.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 09:15:00 | 543.90 | 540.85 | 544.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 543.90 | 540.85 | 544.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 543.90 | 540.85 | 544.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:45:00 | 543.50 | 540.85 | 544.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 545.35 | 541.75 | 544.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:30:00 | 544.00 | 541.75 | 544.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 543.80 | 542.16 | 544.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 11:45:00 | 545.10 | 542.16 | 544.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 13:15:00 | 527.50 | 539.36 | 543.08 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 09:15:00 | 561.50 | 545.79 | 545.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 571.95 | 559.39 | 553.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 566.75 | 570.77 | 563.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 10:00:00 | 566.75 | 570.77 | 563.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 560.50 | 568.71 | 563.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 564.50 | 568.71 | 563.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 555.05 | 565.98 | 562.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:00:00 | 555.05 | 565.98 | 562.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 554.70 | 560.65 | 560.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 551.00 | 557.74 | 559.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 559.70 | 557.35 | 558.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 559.70 | 557.35 | 558.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 559.70 | 557.35 | 558.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 559.70 | 557.35 | 558.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 559.00 | 557.68 | 558.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:15:00 | 558.65 | 557.68 | 558.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 564.85 | 559.12 | 559.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:00:00 | 564.85 | 559.12 | 559.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 566.20 | 560.53 | 560.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 569.75 | 565.79 | 563.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 594.25 | 597.07 | 588.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 594.25 | 597.07 | 588.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 594.25 | 597.07 | 588.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:30:00 | 587.35 | 597.07 | 588.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 590.70 | 593.99 | 589.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:45:00 | 589.75 | 593.99 | 589.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 589.15 | 593.02 | 589.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 14:45:00 | 587.55 | 593.02 | 589.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 586.95 | 591.80 | 589.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:15:00 | 581.85 | 591.80 | 589.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 10:15:00 | 577.55 | 587.01 | 587.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 11:15:00 | 574.40 | 584.49 | 586.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 574.80 | 573.05 | 578.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 10:45:00 | 576.15 | 573.05 | 578.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 577.55 | 574.30 | 578.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:30:00 | 579.15 | 574.30 | 578.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 578.95 | 575.35 | 578.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 578.95 | 575.35 | 578.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 576.25 | 575.53 | 577.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 574.45 | 575.53 | 577.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 569.85 | 574.39 | 577.11 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 582.75 | 578.17 | 577.67 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 10:15:00 | 574.75 | 577.68 | 577.83 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 580.00 | 577.89 | 577.80 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 576.10 | 577.57 | 577.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 571.05 | 575.25 | 576.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 15:15:00 | 558.90 | 558.22 | 561.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 09:15:00 | 557.10 | 558.22 | 561.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 555.95 | 557.77 | 560.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 553.50 | 557.77 | 560.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:00:00 | 554.40 | 557.09 | 560.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:30:00 | 553.60 | 556.23 | 559.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 11:45:00 | 554.05 | 556.29 | 556.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 532.15 | 538.72 | 544.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 10:15:00 | 530.75 | 538.72 | 544.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:15:00 | 526.68 | 536.28 | 542.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 10:45:00 | 529.10 | 536.28 | 542.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 11:15:00 | 526.35 | 534.46 | 541.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 12:15:00 | 525.82 | 532.59 | 539.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 12:15:00 | 525.92 | 532.59 | 539.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-16 09:15:00 | 526.90 | 523.74 | 526.57 | SL hit (close>ema200) qty=0.50 sl=523.74 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 13:15:00 | 531.95 | 528.42 | 528.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 10:15:00 | 533.90 | 530.58 | 529.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 09:15:00 | 519.30 | 529.86 | 529.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 519.30 | 529.86 | 529.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 519.30 | 529.86 | 529.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:15:00 | 516.95 | 529.86 | 529.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 514.85 | 526.86 | 528.45 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 12:15:00 | 525.55 | 524.52 | 524.39 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 15:15:00 | 519.60 | 523.84 | 524.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 515.70 | 521.80 | 523.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 520.70 | 514.83 | 517.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 520.70 | 514.83 | 517.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 520.70 | 514.83 | 517.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 520.70 | 514.83 | 517.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 520.95 | 516.06 | 517.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:15:00 | 521.15 | 516.06 | 517.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 15:15:00 | 518.65 | 517.85 | 518.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 09:15:00 | 515.40 | 517.85 | 518.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-30 13:15:00 | 520.10 | 518.14 | 518.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 13:15:00 | 520.10 | 518.14 | 518.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 14:15:00 | 521.05 | 518.73 | 518.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 15:15:00 | 518.50 | 518.68 | 518.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 15:15:00 | 518.50 | 518.68 | 518.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 15:15:00 | 518.50 | 518.68 | 518.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 09:15:00 | 525.70 | 518.68 | 518.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 11:30:00 | 521.25 | 522.54 | 521.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 12:15:00 | 517.90 | 521.61 | 521.32 | SL hit (close<static) qty=1.00 sl=518.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 516.55 | 520.60 | 520.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 514.90 | 519.46 | 520.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 520.15 | 519.02 | 519.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 520.15 | 519.02 | 519.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 520.15 | 519.02 | 519.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:00:00 | 520.15 | 519.02 | 519.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 521.50 | 519.52 | 520.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:30:00 | 521.95 | 519.52 | 520.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 522.15 | 520.04 | 520.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 523.45 | 520.04 | 520.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 12:15:00 | 523.95 | 520.82 | 520.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 13:15:00 | 525.00 | 521.66 | 521.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 11:15:00 | 520.70 | 522.40 | 521.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 11:15:00 | 520.70 | 522.40 | 521.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 11:15:00 | 520.70 | 522.40 | 521.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:00:00 | 520.70 | 522.40 | 521.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 521.45 | 522.21 | 521.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:00:00 | 521.45 | 522.21 | 521.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 522.00 | 522.17 | 521.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 524.60 | 521.57 | 521.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 15:15:00 | 522.95 | 522.78 | 522.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 09:15:00 | 517.55 | 521.76 | 521.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 09:15:00 | 517.55 | 521.76 | 521.95 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 13:15:00 | 536.05 | 523.77 | 522.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 14:15:00 | 538.40 | 526.69 | 524.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 528.80 | 529.73 | 526.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 11:00:00 | 528.80 | 529.73 | 526.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 526.80 | 529.62 | 527.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:00:00 | 526.80 | 529.62 | 527.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 527.90 | 529.28 | 527.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 11:15:00 | 528.80 | 529.28 | 527.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 11:45:00 | 528.15 | 529.32 | 528.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 10:45:00 | 528.50 | 529.58 | 528.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 13:15:00 | 528.60 | 529.23 | 528.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 528.40 | 529.06 | 528.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:15:00 | 527.50 | 529.06 | 528.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 529.30 | 529.11 | 528.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:30:00 | 527.85 | 529.11 | 528.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 527.20 | 528.87 | 528.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-12 19:00:00 | 527.20 | 528.87 | 528.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-13 09:15:00 | 527.10 | 528.52 | 528.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 09:15:00 | 527.10 | 528.52 | 528.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 10:15:00 | 524.85 | 527.78 | 528.25 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 538.75 | 529.03 | 528.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 10:15:00 | 544.35 | 532.10 | 529.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 14:15:00 | 559.35 | 560.34 | 552.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 15:00:00 | 559.35 | 560.34 | 552.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 555.00 | 559.38 | 553.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:00:00 | 555.00 | 559.38 | 553.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 551.45 | 557.79 | 553.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 551.45 | 557.79 | 553.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 551.70 | 556.57 | 552.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:15:00 | 549.20 | 556.57 | 552.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 550.65 | 555.39 | 552.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:30:00 | 548.55 | 555.39 | 552.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 549.80 | 554.27 | 552.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 10:15:00 | 557.20 | 552.34 | 551.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 09:45:00 | 554.80 | 556.10 | 554.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 552.00 | 556.05 | 556.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 552.00 | 556.05 | 556.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 14:15:00 | 551.20 | 553.67 | 555.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 11:15:00 | 553.40 | 552.81 | 554.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-29 12:00:00 | 553.40 | 552.81 | 554.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 552.45 | 552.74 | 553.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 12:30:00 | 554.00 | 552.74 | 553.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 553.45 | 552.88 | 553.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:00:00 | 553.45 | 552.88 | 553.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 551.60 | 552.63 | 553.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:30:00 | 554.20 | 552.63 | 553.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 561.20 | 554.03 | 554.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 560.85 | 554.03 | 554.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 559.90 | 555.21 | 554.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 12:15:00 | 565.35 | 558.14 | 556.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 09:15:00 | 557.50 | 560.17 | 558.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 557.50 | 560.17 | 558.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 557.50 | 560.17 | 558.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 10:00:00 | 557.50 | 560.17 | 558.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 559.05 | 559.95 | 558.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 561.15 | 558.46 | 558.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 10:30:00 | 560.70 | 559.06 | 558.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 14:15:00 | 560.75 | 559.72 | 558.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 09:15:00 | 546.70 | 558.15 | 558.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 546.70 | 558.15 | 558.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 541.25 | 545.61 | 548.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 11:15:00 | 544.15 | 544.05 | 546.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-11 11:30:00 | 544.10 | 544.05 | 546.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 546.65 | 541.75 | 543.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:30:00 | 546.40 | 541.75 | 543.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 541.90 | 541.78 | 543.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 11:45:00 | 540.05 | 541.69 | 543.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 13:15:00 | 549.65 | 543.32 | 543.94 | SL hit (close>static) qty=1.00 sl=547.30 alert=retest2 |

### Cycle 41 — BUY (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 14:15:00 | 550.80 | 544.81 | 544.56 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 542.30 | 544.33 | 544.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 10:15:00 | 534.70 | 541.53 | 542.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 11:15:00 | 523.00 | 522.50 | 527.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-18 12:00:00 | 523.00 | 522.50 | 527.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 523.80 | 521.60 | 523.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 13:00:00 | 520.95 | 522.06 | 523.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 09:45:00 | 520.90 | 516.80 | 517.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:00:00 | 520.95 | 518.49 | 518.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 519.95 | 518.63 | 518.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 519.95 | 518.63 | 518.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 524.25 | 521.34 | 520.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 12:15:00 | 531.15 | 532.67 | 530.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 13:00:00 | 531.15 | 532.67 | 530.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 531.20 | 532.08 | 530.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 09:15:00 | 539.00 | 532.95 | 531.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 539.50 | 536.19 | 534.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 11:00:00 | 536.70 | 536.68 | 535.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 15:15:00 | 540.60 | 541.85 | 541.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-01-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 15:15:00 | 540.60 | 541.85 | 541.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 09:15:00 | 538.50 | 541.18 | 541.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 497.80 | 493.41 | 503.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 10:00:00 | 497.80 | 493.41 | 503.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 490.45 | 493.14 | 498.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 11:15:00 | 485.30 | 490.66 | 494.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 09:45:00 | 484.00 | 483.22 | 488.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 11:15:00 | 485.05 | 483.91 | 487.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:00:00 | 485.45 | 486.53 | 487.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 487.40 | 486.39 | 487.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:45:00 | 488.20 | 486.39 | 487.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 484.95 | 486.10 | 487.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 14:30:00 | 483.75 | 486.02 | 486.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 488.80 | 486.69 | 487.08 | SL hit (close>static) qty=1.00 sl=488.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 490.60 | 487.47 | 487.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 12:15:00 | 492.15 | 488.98 | 488.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 11:15:00 | 510.50 | 511.65 | 506.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-02 12:00:00 | 510.50 | 511.65 | 506.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 508.00 | 510.92 | 506.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:00:00 | 508.00 | 510.92 | 506.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 505.85 | 509.91 | 506.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:00:00 | 505.85 | 509.91 | 506.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 507.55 | 509.43 | 506.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:45:00 | 506.20 | 509.43 | 506.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 506.70 | 508.89 | 506.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 512.80 | 506.53 | 506.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 10:15:00 | 513.40 | 522.01 | 522.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 513.40 | 522.01 | 522.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 10:15:00 | 509.85 | 517.38 | 519.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 512.05 | 509.76 | 514.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 09:30:00 | 511.80 | 509.76 | 514.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 511.25 | 510.21 | 512.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:30:00 | 511.05 | 510.21 | 512.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 512.15 | 510.60 | 512.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:15:00 | 511.00 | 510.60 | 512.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 511.00 | 510.68 | 512.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 506.35 | 510.68 | 512.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 12:15:00 | 514.80 | 509.55 | 509.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 514.80 | 509.55 | 509.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 518.70 | 514.79 | 512.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 520.55 | 521.64 | 518.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-21 14:00:00 | 520.55 | 521.64 | 518.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 520.25 | 521.37 | 518.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 520.10 | 521.37 | 518.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 520.00 | 521.09 | 519.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 517.05 | 521.09 | 519.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 517.20 | 520.31 | 518.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:30:00 | 518.00 | 520.31 | 518.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 518.35 | 519.92 | 518.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 518.35 | 519.92 | 518.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 517.00 | 518.91 | 518.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:00:00 | 517.00 | 518.91 | 518.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 518.00 | 518.55 | 518.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:45:00 | 518.00 | 518.55 | 518.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 518.20 | 518.48 | 518.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 520.85 | 518.48 | 518.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 522.40 | 519.26 | 518.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 12:30:00 | 523.40 | 520.91 | 519.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 13:30:00 | 523.50 | 521.24 | 519.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 09:30:00 | 524.50 | 522.76 | 521.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 15:15:00 | 524.35 | 523.58 | 522.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 534.40 | 525.87 | 523.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 10:15:00 | 537.00 | 525.87 | 523.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 11:45:00 | 535.80 | 529.85 | 525.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:30:00 | 536.20 | 529.53 | 526.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 522.70 | 528.17 | 525.74 | SL hit (close<static) qty=1.00 sl=523.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 521.25 | 524.36 | 524.51 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 10:15:00 | 527.05 | 524.96 | 524.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 12:15:00 | 528.90 | 526.21 | 525.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 11:15:00 | 538.00 | 538.04 | 534.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-02 12:00:00 | 538.00 | 538.04 | 534.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 539.90 | 538.69 | 535.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 10:15:00 | 540.90 | 538.69 | 535.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 11:00:00 | 541.65 | 539.28 | 535.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 12:30:00 | 548.45 | 542.98 | 538.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-07 15:15:00 | 594.99 | 577.53 | 568.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 13:15:00 | 570.80 | 584.76 | 586.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 09:15:00 | 567.65 | 573.97 | 578.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 11:15:00 | 555.50 | 555.11 | 559.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 11:30:00 | 556.35 | 555.11 | 559.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 561.00 | 556.67 | 559.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:45:00 | 558.95 | 556.67 | 559.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 560.15 | 557.37 | 559.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:30:00 | 561.20 | 557.37 | 559.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 561.80 | 558.25 | 559.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:15:00 | 566.95 | 558.25 | 559.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 571.60 | 560.92 | 560.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 571.60 | 560.92 | 560.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 578.80 | 564.50 | 562.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 580.60 | 571.96 | 567.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 15:15:00 | 578.20 | 579.59 | 574.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 09:15:00 | 596.60 | 579.59 | 574.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 09:15:00 | 626.43 | 609.43 | 602.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-03 10:15:00 | 620.65 | 620.70 | 616.12 | SL hit (close<ema200) qty=0.50 sl=620.70 alert=retest1 |

### Cycle 52 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 607.85 | 615.78 | 616.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 14:15:00 | 606.25 | 613.87 | 615.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 625.25 | 615.30 | 615.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 625.25 | 615.30 | 615.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 625.25 | 615.30 | 615.77 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 623.05 | 616.85 | 616.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 628.15 | 621.43 | 619.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 09:15:00 | 624.70 | 625.09 | 622.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 10:15:00 | 627.80 | 625.63 | 622.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 627.80 | 625.63 | 622.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 10:30:00 | 624.00 | 625.63 | 622.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 623.25 | 625.21 | 623.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:00:00 | 623.25 | 625.21 | 623.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 627.75 | 625.71 | 623.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:30:00 | 626.65 | 625.71 | 623.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 623.10 | 625.23 | 623.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:45:00 | 623.40 | 625.34 | 623.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 624.90 | 625.24 | 624.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:15:00 | 623.35 | 625.24 | 624.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 625.70 | 625.33 | 624.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 14:00:00 | 626.80 | 625.63 | 624.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 614.65 | 627.94 | 627.82 | SL hit (close<static) qty=1.00 sl=622.60 alert=retest2 |

### Cycle 54 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 615.05 | 625.36 | 626.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 10:15:00 | 613.45 | 618.51 | 622.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 11:15:00 | 587.05 | 582.87 | 588.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-23 12:00:00 | 587.05 | 582.87 | 588.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 596.40 | 586.40 | 589.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:45:00 | 597.70 | 586.40 | 589.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 595.50 | 588.22 | 590.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 575.60 | 590.15 | 590.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 15:15:00 | 571.50 | 568.72 | 568.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 15:15:00 | 571.50 | 568.72 | 568.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 578.50 | 570.68 | 569.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 11:15:00 | 573.05 | 576.14 | 573.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 11:15:00 | 573.05 | 576.14 | 573.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 573.05 | 576.14 | 573.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:00:00 | 573.05 | 576.14 | 573.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 575.75 | 576.06 | 573.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:45:00 | 573.70 | 576.06 | 573.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 575.35 | 575.92 | 574.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:45:00 | 575.30 | 575.92 | 574.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 576.85 | 576.10 | 574.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:15:00 | 575.90 | 576.10 | 574.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 575.90 | 576.06 | 574.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 09:15:00 | 580.95 | 576.06 | 574.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 12:15:00 | 573.35 | 575.31 | 574.59 | SL hit (close<static) qty=1.00 sl=573.65 alert=retest2 |

### Cycle 56 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 568.50 | 573.42 | 573.87 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 14:15:00 | 577.30 | 574.31 | 574.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 11:15:00 | 587.00 | 578.38 | 576.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 574.35 | 581.30 | 578.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 574.35 | 581.30 | 578.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 574.35 | 581.30 | 578.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:45:00 | 574.75 | 581.30 | 578.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 572.05 | 579.45 | 578.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:00:00 | 572.05 | 579.45 | 578.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 570.20 | 577.03 | 577.35 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 13:15:00 | 581.45 | 577.92 | 577.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 10:15:00 | 585.15 | 579.65 | 578.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 12:15:00 | 596.10 | 596.13 | 591.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 13:00:00 | 596.10 | 596.13 | 591.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 589.85 | 595.10 | 592.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 589.85 | 595.10 | 592.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 586.75 | 593.43 | 592.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 587.40 | 593.43 | 592.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 12:15:00 | 583.75 | 590.09 | 590.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 13:15:00 | 581.60 | 588.39 | 589.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 590.55 | 585.84 | 587.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 590.55 | 585.84 | 587.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 590.55 | 585.84 | 587.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 590.55 | 585.84 | 587.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 588.55 | 586.38 | 587.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 14:00:00 | 585.25 | 586.15 | 587.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 09:15:00 | 592.00 | 588.53 | 588.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 592.00 | 588.53 | 588.37 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 09:15:00 | 587.00 | 588.53 | 588.59 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 590.95 | 589.02 | 588.80 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 585.45 | 588.45 | 588.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 11:15:00 | 578.55 | 585.48 | 587.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 580.60 | 578.19 | 580.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 580.60 | 578.19 | 580.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 580.60 | 578.19 | 580.66 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 15:15:00 | 583.35 | 581.76 | 581.57 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 579.65 | 581.87 | 581.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 578.25 | 581.14 | 581.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 582.00 | 577.46 | 579.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 582.00 | 577.46 | 579.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 582.00 | 577.46 | 579.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 582.00 | 577.46 | 579.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 581.50 | 578.27 | 579.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:45:00 | 576.55 | 577.94 | 579.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 13:00:00 | 575.95 | 577.54 | 578.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 574.65 | 578.82 | 579.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:15:00 | 547.72 | 556.09 | 564.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:15:00 | 547.15 | 556.09 | 564.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:15:00 | 545.92 | 556.09 | 564.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 13:15:00 | 550.85 | 550.17 | 555.49 | SL hit (close>ema200) qty=0.50 sl=550.17 alert=retest2 |

### Cycle 67 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 563.15 | 550.28 | 550.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 566.00 | 559.73 | 555.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 578.25 | 578.49 | 572.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 577.35 | 578.49 | 572.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 586.10 | 581.45 | 578.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:45:00 | 592.30 | 585.43 | 581.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 599.35 | 601.62 | 601.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 599.35 | 601.62 | 601.67 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 603.15 | 601.93 | 601.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 14:15:00 | 604.85 | 602.83 | 602.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 605.40 | 605.45 | 603.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 605.40 | 605.45 | 603.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 602.80 | 605.12 | 604.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 602.80 | 605.12 | 604.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 604.00 | 604.90 | 604.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 599.00 | 604.90 | 604.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 605.25 | 604.97 | 604.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 596.85 | 604.97 | 604.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 604.65 | 605.84 | 604.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 604.65 | 605.84 | 604.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 605.70 | 605.81 | 604.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 608.30 | 605.99 | 605.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 10:15:00 | 603.00 | 605.33 | 605.01 | SL hit (close<static) qty=1.00 sl=603.80 alert=retest2 |

### Cycle 70 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 598.60 | 603.98 | 604.43 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 611.65 | 604.50 | 604.38 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 601.80 | 603.96 | 604.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 597.00 | 602.57 | 603.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 602.80 | 600.46 | 601.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 602.80 | 600.46 | 601.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 602.80 | 600.46 | 601.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 602.80 | 600.46 | 601.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 600.75 | 600.52 | 601.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:45:00 | 597.85 | 599.24 | 600.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 604.95 | 600.53 | 601.08 | SL hit (close>static) qty=1.00 sl=603.30 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 606.45 | 601.71 | 601.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 613.30 | 606.94 | 604.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 603.45 | 612.25 | 609.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 603.45 | 612.25 | 609.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 603.45 | 612.25 | 609.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 603.05 | 612.25 | 609.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 615.25 | 612.85 | 609.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 11:15:00 | 616.00 | 612.85 | 609.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:30:00 | 617.05 | 615.03 | 611.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 642.30 | 652.18 | 652.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 642.30 | 652.18 | 652.82 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 14:15:00 | 652.90 | 652.72 | 652.71 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 646.00 | 651.38 | 652.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 11:15:00 | 644.40 | 649.98 | 651.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 650.15 | 648.77 | 650.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 650.15 | 648.77 | 650.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 650.15 | 648.77 | 650.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 650.15 | 648.77 | 650.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 655.75 | 650.17 | 651.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 655.75 | 650.17 | 651.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 653.90 | 650.91 | 651.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 646.20 | 650.91 | 651.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 613.89 | 630.13 | 636.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 13:15:00 | 632.45 | 630.60 | 636.22 | SL hit (close>ema200) qty=0.50 sl=630.60 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 683.65 | 643.30 | 640.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 687.05 | 652.05 | 645.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 10:15:00 | 717.50 | 718.60 | 704.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 11:00:00 | 717.50 | 718.60 | 704.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 713.05 | 717.95 | 710.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:00:00 | 724.20 | 719.20 | 711.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 724.10 | 723.49 | 717.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 721.35 | 729.86 | 730.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 721.35 | 729.86 | 730.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 699.65 | 711.35 | 717.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 715.40 | 710.66 | 715.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 715.40 | 710.66 | 715.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 715.40 | 710.66 | 715.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 715.40 | 710.66 | 715.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 709.95 | 710.52 | 715.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 716.10 | 710.52 | 715.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 718.75 | 712.08 | 714.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 718.75 | 712.08 | 714.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 725.75 | 714.81 | 715.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 725.75 | 714.81 | 715.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 719.50 | 717.03 | 716.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 725.90 | 719.52 | 718.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 729.50 | 736.12 | 731.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 729.50 | 736.12 | 731.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 729.50 | 736.12 | 731.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 729.50 | 736.12 | 731.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 737.90 | 736.48 | 732.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:45:00 | 739.35 | 736.29 | 732.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 14:15:00 | 729.70 | 734.90 | 732.89 | SL hit (close<static) qty=1.00 sl=729.80 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 728.40 | 731.27 | 731.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 723.70 | 727.82 | 729.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 716.40 | 714.95 | 719.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:30:00 | 716.35 | 714.95 | 719.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 719.40 | 715.84 | 719.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 719.40 | 715.84 | 719.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 718.50 | 716.37 | 719.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 716.50 | 716.37 | 719.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:00:00 | 717.05 | 717.58 | 719.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:45:00 | 717.85 | 717.79 | 718.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 15:15:00 | 717.40 | 717.79 | 718.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 717.40 | 717.71 | 718.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 727.95 | 717.71 | 718.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 731.05 | 720.38 | 719.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 731.05 | 720.38 | 719.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 735.20 | 725.37 | 722.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 15:15:00 | 740.55 | 741.25 | 735.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:15:00 | 735.10 | 741.25 | 735.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 735.70 | 740.14 | 735.56 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 732.55 | 734.36 | 734.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 724.80 | 732.45 | 733.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 729.05 | 727.15 | 729.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 729.05 | 727.15 | 729.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 729.05 | 727.15 | 729.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 729.05 | 727.15 | 729.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 731.10 | 727.94 | 729.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 731.10 | 727.94 | 729.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 739.50 | 730.25 | 730.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 739.50 | 730.25 | 730.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 737.40 | 731.68 | 731.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 14:15:00 | 744.55 | 735.80 | 733.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 11:15:00 | 741.80 | 743.45 | 740.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:00:00 | 741.80 | 743.45 | 740.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 745.35 | 743.83 | 740.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:45:00 | 739.40 | 743.83 | 740.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 740.60 | 743.18 | 740.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 740.60 | 743.18 | 740.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 741.45 | 742.84 | 740.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 744.75 | 742.66 | 741.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 752.90 | 759.04 | 759.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 752.90 | 759.04 | 759.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 749.45 | 756.24 | 757.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 757.20 | 754.65 | 756.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 757.20 | 754.65 | 756.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 757.20 | 754.65 | 756.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:45:00 | 757.70 | 754.65 | 756.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 761.95 | 756.11 | 756.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 761.95 | 756.11 | 756.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 764.10 | 757.71 | 757.53 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 754.75 | 757.12 | 757.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 752.80 | 756.25 | 756.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 758.80 | 756.76 | 757.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 758.80 | 756.76 | 757.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 758.80 | 756.76 | 757.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 758.80 | 756.76 | 757.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 759.25 | 757.26 | 757.25 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 744.55 | 754.72 | 756.09 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 756.55 | 752.59 | 752.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 759.00 | 754.97 | 753.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 754.55 | 755.55 | 754.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 12:15:00 | 754.55 | 755.55 | 754.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 754.55 | 755.55 | 754.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 754.55 | 755.55 | 754.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 756.05 | 755.65 | 754.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 14:30:00 | 759.45 | 755.45 | 754.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 750.30 | 754.49 | 754.18 | SL hit (close<static) qty=1.00 sl=752.85 alert=retest2 |

### Cycle 90 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 750.30 | 753.65 | 753.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 746.10 | 751.10 | 752.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 749.90 | 749.83 | 751.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 12:00:00 | 749.90 | 749.83 | 751.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 752.20 | 749.48 | 750.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 752.20 | 749.48 | 750.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 753.00 | 750.19 | 751.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 755.95 | 750.19 | 751.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 758.35 | 751.82 | 751.69 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 746.85 | 751.08 | 751.52 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 755.80 | 751.70 | 751.67 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 748.50 | 751.39 | 751.59 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 754.95 | 752.17 | 751.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 764.80 | 755.14 | 753.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 781.95 | 782.62 | 773.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 781.95 | 782.62 | 773.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 777.00 | 779.32 | 773.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 15:00:00 | 778.05 | 778.40 | 774.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 15:15:00 | 767.65 | 776.25 | 773.83 | SL hit (close<static) qty=1.00 sl=772.85 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 765.05 | 771.25 | 771.92 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 773.75 | 772.52 | 772.40 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 767.70 | 772.15 | 772.33 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 12:15:00 | 778.40 | 773.40 | 772.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 784.65 | 776.31 | 774.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 783.95 | 786.02 | 782.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 11:00:00 | 783.95 | 786.02 | 782.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 782.45 | 785.31 | 782.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 782.35 | 785.31 | 782.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 786.95 | 785.64 | 782.67 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 773.45 | 780.61 | 781.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 769.10 | 778.31 | 779.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 760.50 | 759.94 | 765.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 770.40 | 762.41 | 765.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 770.40 | 762.41 | 765.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 770.40 | 762.41 | 765.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 763.25 | 762.58 | 765.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 758.55 | 762.58 | 765.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 762.70 | 749.53 | 748.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 762.70 | 749.53 | 748.74 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 745.30 | 748.78 | 749.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 740.90 | 746.29 | 747.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 741.10 | 741.09 | 744.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 15:00:00 | 741.10 | 741.09 | 744.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 740.20 | 740.92 | 743.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 736.30 | 740.92 | 743.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:00:00 | 735.90 | 739.91 | 743.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:45:00 | 736.50 | 739.50 | 742.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 745.35 | 739.78 | 741.25 | SL hit (close>static) qty=1.00 sl=744.60 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 756.20 | 743.06 | 742.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 13:15:00 | 758.75 | 748.57 | 745.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 14:15:00 | 733.95 | 745.64 | 744.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 733.95 | 745.64 | 744.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 733.95 | 745.64 | 744.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 732.05 | 745.64 | 744.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 734.80 | 743.48 | 743.51 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 746.30 | 744.04 | 743.77 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 739.65 | 743.54 | 743.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 736.15 | 741.37 | 742.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 738.60 | 738.49 | 740.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 738.60 | 738.49 | 740.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 738.60 | 738.49 | 740.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 739.50 | 738.49 | 740.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 742.00 | 739.19 | 740.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 742.00 | 739.19 | 740.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 744.30 | 740.21 | 741.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 743.55 | 740.21 | 741.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 745.90 | 742.24 | 741.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 752.30 | 744.94 | 743.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 746.25 | 746.27 | 744.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 12:00:00 | 746.25 | 746.27 | 744.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 744.85 | 745.98 | 744.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 744.85 | 745.98 | 744.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 748.15 | 746.42 | 744.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 751.15 | 747.25 | 745.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 758.25 | 747.40 | 745.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 12:15:00 | 742.85 | 748.49 | 746.93 | SL hit (close<static) qty=1.00 sl=743.60 alert=retest2 |

### Cycle 108 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 732.10 | 743.76 | 744.96 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 13:15:00 | 755.80 | 746.49 | 745.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 10:15:00 | 764.65 | 751.81 | 748.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 756.30 | 760.76 | 755.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 10:00:00 | 756.30 | 760.76 | 755.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 751.95 | 759.00 | 754.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 753.00 | 759.00 | 754.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 739.75 | 755.15 | 753.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:45:00 | 740.05 | 755.15 | 753.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 12:15:00 | 732.85 | 750.69 | 751.54 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 754.05 | 749.89 | 749.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 757.10 | 751.34 | 750.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 11:15:00 | 761.55 | 761.81 | 757.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 11:30:00 | 763.10 | 761.81 | 757.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 754.35 | 760.32 | 757.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:30:00 | 755.45 | 760.32 | 757.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 746.70 | 757.59 | 756.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 746.70 | 757.59 | 756.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 748.90 | 754.17 | 754.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 741.35 | 751.60 | 753.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 749.35 | 745.05 | 748.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 749.35 | 745.05 | 748.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 749.35 | 745.05 | 748.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 749.35 | 745.05 | 748.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 744.30 | 744.90 | 747.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 744.30 | 744.90 | 747.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 735.45 | 726.02 | 733.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 735.45 | 726.02 | 733.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 730.85 | 726.99 | 733.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 726.00 | 726.99 | 733.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:15:00 | 689.70 | 699.33 | 704.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 696.45 | 693.36 | 698.68 | SL hit (close>ema200) qty=0.50 sl=693.36 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 687.95 | 685.16 | 685.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 694.50 | 687.03 | 685.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 687.60 | 688.63 | 687.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 687.60 | 688.63 | 687.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 687.60 | 688.63 | 687.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 687.60 | 688.63 | 687.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 688.05 | 688.52 | 687.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 693.10 | 688.52 | 687.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 694.75 | 689.76 | 688.09 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 680.10 | 687.67 | 688.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 677.35 | 681.52 | 684.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 13:15:00 | 683.85 | 680.57 | 683.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 13:15:00 | 683.85 | 680.57 | 683.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 683.85 | 680.57 | 683.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:30:00 | 688.00 | 680.57 | 683.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 692.55 | 682.96 | 683.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:45:00 | 693.25 | 682.96 | 683.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 15:15:00 | 693.00 | 684.97 | 684.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 699.10 | 691.63 | 688.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 11:15:00 | 695.30 | 695.47 | 691.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 12:00:00 | 695.30 | 695.47 | 691.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 690.50 | 694.08 | 691.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:45:00 | 690.05 | 694.08 | 691.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 691.80 | 693.62 | 691.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:15:00 | 690.50 | 693.62 | 691.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 690.50 | 693.00 | 691.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:00:00 | 694.95 | 692.83 | 691.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 14:15:00 | 683.55 | 690.79 | 691.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 683.55 | 690.79 | 691.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 10:15:00 | 682.05 | 687.29 | 689.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 14:15:00 | 674.60 | 673.51 | 678.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 15:00:00 | 674.60 | 673.51 | 678.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 679.00 | 674.61 | 678.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 683.00 | 674.61 | 678.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 682.20 | 676.13 | 678.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 14:15:00 | 673.95 | 676.66 | 678.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 674.60 | 676.02 | 677.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 674.90 | 676.51 | 677.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 687.45 | 678.62 | 678.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 687.45 | 678.62 | 678.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 695.15 | 688.71 | 684.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 10:15:00 | 690.00 | 690.36 | 686.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 11:00:00 | 690.00 | 690.36 | 686.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 692.45 | 690.86 | 687.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 692.45 | 690.86 | 687.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 683.25 | 689.49 | 687.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 683.25 | 689.49 | 687.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 682.35 | 688.06 | 687.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 680.00 | 688.06 | 687.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 686.60 | 687.11 | 687.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 686.60 | 687.11 | 687.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 14:15:00 | 685.45 | 686.78 | 686.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 15:15:00 | 684.20 | 686.26 | 686.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 661.70 | 660.72 | 665.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 09:15:00 | 654.10 | 660.72 | 665.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 653.50 | 659.28 | 664.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 648.00 | 655.65 | 660.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:45:00 | 649.00 | 652.69 | 657.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 648.00 | 651.26 | 655.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:30:00 | 649.65 | 650.20 | 654.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 659.20 | 652.00 | 654.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:45:00 | 659.95 | 652.00 | 654.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 662.80 | 654.16 | 655.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:30:00 | 662.75 | 654.16 | 655.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 661.55 | 656.86 | 656.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 661.55 | 656.86 | 656.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 09:15:00 | 666.15 | 659.22 | 657.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 664.30 | 665.01 | 662.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 11:15:00 | 664.30 | 665.01 | 662.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 664.30 | 665.01 | 662.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 664.30 | 665.01 | 662.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 660.20 | 664.04 | 662.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 660.20 | 664.04 | 662.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 659.20 | 663.08 | 661.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 659.20 | 663.08 | 661.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 655.65 | 661.59 | 661.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 655.65 | 661.59 | 661.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 654.70 | 660.21 | 660.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 652.30 | 658.63 | 660.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 11:15:00 | 649.80 | 649.47 | 653.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 12:00:00 | 649.80 | 649.47 | 653.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 654.75 | 650.48 | 652.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 657.15 | 650.48 | 652.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 655.40 | 651.46 | 653.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 652.75 | 651.46 | 653.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 645.55 | 650.28 | 652.33 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 658.60 | 652.83 | 652.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 13:15:00 | 676.15 | 668.41 | 662.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 670.55 | 670.73 | 665.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:30:00 | 669.40 | 670.73 | 665.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 660.80 | 668.75 | 665.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 660.80 | 668.75 | 665.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 659.10 | 666.82 | 664.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 654.75 | 666.82 | 664.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 661.20 | 665.11 | 664.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 661.80 | 665.11 | 664.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 660.95 | 663.65 | 663.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 665.00 | 663.65 | 663.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 09:15:00 | 662.60 | 663.44 | 663.52 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 666.60 | 664.07 | 663.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 669.70 | 665.36 | 664.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 14:15:00 | 662.75 | 664.97 | 664.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 14:15:00 | 662.75 | 664.97 | 664.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 662.75 | 664.97 | 664.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 662.75 | 664.97 | 664.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 662.10 | 664.39 | 664.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 660.00 | 664.39 | 664.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 660.95 | 663.70 | 663.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 657.65 | 662.49 | 663.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 10:15:00 | 648.95 | 647.56 | 652.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 11:00:00 | 648.95 | 647.56 | 652.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 650.95 | 647.03 | 649.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 650.95 | 647.03 | 649.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 646.65 | 646.95 | 649.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:30:00 | 652.00 | 646.95 | 649.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 642.10 | 640.31 | 642.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 642.10 | 640.31 | 642.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 641.10 | 640.47 | 642.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 638.70 | 640.47 | 642.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 660.00 | 639.88 | 640.13 | SL hit (close>static) qty=1.00 sl=643.75 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 658.20 | 643.54 | 641.77 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 638.00 | 643.73 | 643.77 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 647.25 | 644.10 | 643.86 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 636.70 | 642.62 | 643.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 10:15:00 | 632.45 | 640.58 | 642.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 13:15:00 | 642.40 | 640.38 | 641.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 13:15:00 | 642.40 | 640.38 | 641.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 642.40 | 640.38 | 641.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 642.40 | 640.38 | 641.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 640.95 | 640.50 | 641.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:15:00 | 644.60 | 640.50 | 641.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 644.60 | 641.32 | 641.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 646.70 | 641.32 | 641.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 641.90 | 641.43 | 641.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 638.75 | 641.43 | 641.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 13:45:00 | 640.25 | 640.64 | 641.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-22 09:15:00 | 574.88 | 627.28 | 634.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 14:15:00 | 601.25 | 595.41 | 595.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 609.30 | 598.77 | 596.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 611.45 | 612.89 | 609.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 611.45 | 612.89 | 609.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 573.45 | 611.26 | 610.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 573.45 | 611.26 | 610.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 605.90 | 610.19 | 610.35 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 611.70 | 608.79 | 608.50 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 606.40 | 608.42 | 608.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 602.00 | 605.09 | 606.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 600.25 | 599.91 | 602.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 13:00:00 | 600.25 | 599.91 | 602.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 599.00 | 599.73 | 602.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 600.45 | 599.73 | 602.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 602.30 | 600.24 | 602.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 602.30 | 600.24 | 602.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 599.85 | 600.17 | 602.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 600.10 | 600.17 | 602.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 597.40 | 599.61 | 601.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:30:00 | 594.95 | 598.17 | 600.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 565.20 | 575.71 | 584.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 10:15:00 | 577.40 | 576.05 | 583.43 | SL hit (close>ema200) qty=0.50 sl=576.05 alert=retest2 |

### Cycle 133 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 591.50 | 584.69 | 584.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 12:15:00 | 593.45 | 586.44 | 585.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 588.25 | 589.63 | 587.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 588.25 | 589.63 | 587.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 588.25 | 589.63 | 587.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 588.25 | 589.63 | 587.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 588.10 | 589.33 | 587.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:30:00 | 586.90 | 589.33 | 587.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 585.00 | 588.46 | 587.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 585.00 | 588.46 | 587.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 580.35 | 586.84 | 586.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:00:00 | 580.35 | 586.84 | 586.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 14:15:00 | 582.30 | 585.93 | 586.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 15:15:00 | 580.00 | 584.75 | 585.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 13:15:00 | 572.95 | 572.72 | 576.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 14:00:00 | 572.95 | 572.72 | 576.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 577.85 | 573.73 | 575.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 577.85 | 573.73 | 575.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 575.85 | 574.15 | 575.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 572.10 | 574.56 | 575.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 09:45:00 | 571.30 | 574.59 | 575.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 13:15:00 | 572.25 | 574.20 | 574.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 568.50 | 574.05 | 574.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 567.00 | 566.44 | 568.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:30:00 | 567.90 | 566.44 | 568.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 569.45 | 567.18 | 568.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:30:00 | 568.55 | 567.18 | 568.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 571.00 | 567.95 | 569.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 12:00:00 | 565.70 | 567.50 | 568.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:15:00 | 543.50 | 550.27 | 554.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:15:00 | 543.64 | 550.27 | 554.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 551.35 | 549.52 | 552.13 | SL hit (close>ema200) qty=0.50 sl=549.52 alert=retest2 |

### Cycle 135 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 546.85 | 543.53 | 543.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 554.00 | 547.70 | 545.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 592.75 | 596.10 | 590.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 15:00:00 | 592.75 | 596.10 | 590.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 591.90 | 595.26 | 590.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 598.90 | 595.26 | 590.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 587.40 | 592.89 | 590.40 | SL hit (close<static) qty=1.00 sl=587.60 alert=retest2 |

### Cycle 136 — SELL (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 10:15:00 | 585.20 | 588.71 | 589.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 581.10 | 586.46 | 587.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 592.55 | 587.46 | 588.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 592.55 | 587.46 | 588.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 592.55 | 587.46 | 588.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 592.55 | 587.46 | 588.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 590.00 | 587.97 | 588.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:45:00 | 588.00 | 588.05 | 588.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 558.60 | 569.07 | 574.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 565.60 | 564.37 | 568.96 | SL hit (close>ema200) qty=0.50 sl=564.37 alert=retest2 |

### Cycle 137 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 556.45 | 551.99 | 551.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 565.55 | 554.70 | 553.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 10:15:00 | 564.65 | 565.02 | 560.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 10:45:00 | 562.25 | 565.02 | 560.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 555.00 | 562.95 | 560.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:00:00 | 555.00 | 562.95 | 560.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 552.55 | 560.87 | 559.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:00:00 | 552.55 | 560.87 | 559.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 15:15:00 | 550.95 | 557.69 | 558.30 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 13:15:00 | 566.05 | 559.46 | 558.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 572.50 | 563.38 | 560.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 12:15:00 | 602.80 | 602.89 | 596.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 13:00:00 | 602.80 | 602.89 | 596.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 594.30 | 600.46 | 597.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 594.30 | 600.46 | 597.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 592.90 | 598.95 | 596.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 595.60 | 598.95 | 596.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 589.55 | 598.54 | 599.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 589.55 | 598.54 | 599.33 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 599.30 | 598.10 | 598.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 612.15 | 604.38 | 601.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 608.90 | 612.74 | 609.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 11:15:00 | 608.90 | 612.74 | 609.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 608.90 | 612.74 | 609.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 608.90 | 612.74 | 609.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 609.45 | 612.08 | 609.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:45:00 | 612.75 | 610.70 | 609.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 14:45:00 | 611.20 | 609.84 | 609.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 606.20 | 609.23 | 609.09 | SL hit (close<static) qty=1.00 sl=606.25 alert=retest2 |

### Cycle 142 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 603.90 | 608.16 | 608.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 601.50 | 606.83 | 607.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 596.65 | 596.22 | 600.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:00:00 | 596.65 | 596.22 | 600.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 596.00 | 583.55 | 586.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 597.45 | 583.55 | 586.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 601.20 | 589.86 | 589.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 603.20 | 595.61 | 592.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 602.05 | 603.79 | 598.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 602.05 | 603.79 | 598.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 600.20 | 604.98 | 602.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 600.20 | 604.98 | 602.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 605.35 | 605.06 | 602.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:45:00 | 607.15 | 606.03 | 603.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 09:15:00 | 613.80 | 616.16 | 616.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 613.80 | 616.16 | 616.26 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 618.15 | 616.41 | 616.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 620.40 | 617.80 | 616.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 14:15:00 | 619.40 | 620.10 | 618.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 619.40 | 620.10 | 618.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 661.40 | 663.39 | 661.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 672.10 | 663.39 | 661.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 670.35 | 664.78 | 661.90 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 653.75 | 661.57 | 661.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 644.50 | 658.16 | 660.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 640.95 | 640.02 | 643.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:00:00 | 640.95 | 640.02 | 643.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 636.15 | 634.50 | 637.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:30:00 | 638.95 | 634.50 | 637.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 636.50 | 634.90 | 637.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 630.30 | 634.90 | 637.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 638.80 | 635.68 | 637.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:15:00 | 640.40 | 635.68 | 637.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 640.10 | 636.57 | 637.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 640.10 | 636.57 | 637.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 637.80 | 637.33 | 637.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 637.65 | 637.33 | 637.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 637.90 | 637.45 | 637.89 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 641.35 | 638.64 | 638.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 11:15:00 | 643.00 | 639.51 | 638.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 638.30 | 640.73 | 639.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 638.30 | 640.73 | 639.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 638.30 | 640.73 | 639.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 638.30 | 640.73 | 639.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 636.55 | 639.89 | 639.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 635.90 | 639.89 | 639.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 634.85 | 638.89 | 639.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 630.20 | 636.70 | 638.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 629.65 | 628.89 | 632.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 629.65 | 628.89 | 632.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 628.95 | 628.92 | 631.78 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 640.50 | 634.01 | 633.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 642.00 | 637.63 | 635.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 638.50 | 638.79 | 636.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 636.90 | 638.79 | 636.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 635.85 | 638.20 | 636.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 632.60 | 638.20 | 636.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 632.70 | 637.10 | 636.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 631.45 | 637.10 | 636.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 629.00 | 635.48 | 635.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 626.95 | 633.77 | 634.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 629.45 | 626.76 | 629.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 629.45 | 626.76 | 629.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 629.45 | 626.76 | 629.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:15:00 | 634.70 | 626.76 | 629.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 638.95 | 629.20 | 630.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 638.95 | 629.20 | 630.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 635.90 | 630.54 | 630.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 639.55 | 630.54 | 630.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 636.90 | 631.81 | 631.53 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 628.85 | 631.17 | 631.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 14:15:00 | 627.55 | 630.09 | 630.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 636.55 | 631.12 | 631.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 636.55 | 631.12 | 631.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 636.55 | 631.12 | 631.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 636.55 | 631.12 | 631.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 637.00 | 632.29 | 631.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 645.50 | 638.54 | 636.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 659.20 | 659.27 | 655.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 11:00:00 | 659.20 | 659.27 | 655.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 659.35 | 659.65 | 656.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 658.85 | 659.65 | 656.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 658.00 | 659.37 | 657.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 653.80 | 659.37 | 657.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 649.90 | 657.47 | 656.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 650.70 | 657.47 | 656.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 647.90 | 655.56 | 655.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 644.45 | 652.34 | 654.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 647.20 | 646.68 | 650.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 647.20 | 646.68 | 650.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 647.20 | 646.68 | 650.47 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 659.00 | 652.07 | 651.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 11:15:00 | 660.60 | 654.71 | 653.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 15:15:00 | 662.35 | 663.46 | 660.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 662.95 | 663.46 | 660.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 677.50 | 666.27 | 662.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 678.85 | 666.27 | 662.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 677.80 | 675.81 | 669.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 662.10 | 670.01 | 670.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 662.10 | 670.01 | 670.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 659.80 | 666.89 | 668.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 10:15:00 | 664.65 | 664.27 | 666.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 10:15:00 | 664.65 | 664.27 | 666.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 664.65 | 664.27 | 666.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 666.15 | 664.27 | 666.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 668.40 | 664.61 | 666.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 668.40 | 664.61 | 666.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 672.90 | 666.27 | 666.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 672.90 | 666.27 | 666.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 671.15 | 667.24 | 667.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 674.05 | 669.35 | 668.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 666.30 | 669.14 | 668.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 12:15:00 | 666.30 | 669.14 | 668.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 666.30 | 669.14 | 668.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 666.30 | 669.14 | 668.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 678.50 | 671.01 | 669.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 677.95 | 671.01 | 669.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 668.50 | 670.51 | 669.17 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 655.95 | 667.84 | 668.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 12:15:00 | 642.25 | 650.52 | 656.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 640.90 | 638.25 | 644.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:45:00 | 641.20 | 638.25 | 644.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 628.40 | 628.57 | 631.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 627.05 | 628.57 | 630.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 626.40 | 628.08 | 630.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 611.85 | 609.83 | 609.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 611.85 | 609.83 | 609.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 615.35 | 611.28 | 610.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 609.20 | 611.35 | 610.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 609.20 | 611.35 | 610.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 609.20 | 611.35 | 610.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 609.20 | 611.35 | 610.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 610.85 | 611.25 | 610.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 616.45 | 611.56 | 610.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 613.85 | 614.72 | 613.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 609.75 | 614.29 | 614.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 609.75 | 614.29 | 614.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 608.45 | 613.12 | 613.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 610.00 | 608.45 | 610.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 15:00:00 | 610.00 | 608.45 | 610.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 610.00 | 608.76 | 610.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 615.05 | 608.76 | 610.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 616.65 | 610.34 | 611.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:45:00 | 619.30 | 610.34 | 611.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 612.90 | 610.85 | 611.29 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 615.45 | 611.77 | 611.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 15:15:00 | 617.05 | 614.59 | 613.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 15:15:00 | 620.60 | 621.04 | 618.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:15:00 | 624.70 | 621.04 | 618.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 633.50 | 636.66 | 632.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 632.90 | 636.66 | 632.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 632.65 | 635.54 | 632.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 632.65 | 635.54 | 632.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 632.25 | 634.88 | 632.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 632.25 | 634.88 | 632.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 629.90 | 633.89 | 632.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 629.90 | 633.89 | 632.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 633.05 | 633.72 | 632.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:30:00 | 635.10 | 633.93 | 632.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:15:00 | 634.70 | 634.16 | 632.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:00:00 | 634.85 | 634.30 | 633.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:15:00 | 635.00 | 633.55 | 632.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 635.00 | 633.84 | 633.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 630.85 | 632.81 | 632.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 630.85 | 632.81 | 632.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 629.50 | 632.15 | 632.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 628.20 | 626.99 | 628.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 628.20 | 626.99 | 628.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 628.20 | 626.99 | 628.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 628.20 | 626.99 | 628.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 628.40 | 627.27 | 628.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:30:00 | 628.75 | 627.27 | 628.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 629.55 | 627.73 | 628.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 629.55 | 627.73 | 628.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 629.90 | 628.16 | 628.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 630.05 | 628.16 | 628.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 630.05 | 628.54 | 628.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 630.05 | 628.54 | 628.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 628.00 | 628.43 | 628.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 622.30 | 628.43 | 628.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 618.40 | 610.57 | 609.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 618.40 | 610.57 | 609.55 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 603.25 | 608.41 | 608.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 599.30 | 606.59 | 608.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 596.40 | 594.20 | 596.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 12:15:00 | 596.40 | 594.20 | 596.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 596.40 | 594.20 | 596.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 596.40 | 594.20 | 596.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 595.90 | 594.54 | 596.82 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 605.15 | 598.88 | 598.28 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 597.65 | 599.49 | 599.55 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 600.45 | 599.57 | 599.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 15:15:00 | 601.70 | 600.12 | 599.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 15:15:00 | 603.00 | 603.27 | 601.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:15:00 | 602.45 | 603.27 | 601.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 603.20 | 603.26 | 601.95 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 601.10 | 602.25 | 602.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 599.45 | 601.69 | 602.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 602.75 | 601.52 | 601.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 602.75 | 601.52 | 601.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 602.75 | 601.52 | 601.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 602.90 | 601.52 | 601.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 608.35 | 602.89 | 602.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 612.80 | 606.18 | 604.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 609.80 | 610.16 | 607.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 609.45 | 610.16 | 607.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 607.40 | 609.58 | 607.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 605.45 | 609.58 | 607.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 603.35 | 608.34 | 607.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 603.35 | 608.34 | 607.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 602.50 | 607.17 | 607.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 602.00 | 607.17 | 607.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 601.90 | 606.12 | 606.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 600.95 | 604.54 | 605.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 600.60 | 599.83 | 602.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 600.60 | 599.83 | 602.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 585.30 | 588.19 | 590.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 591.85 | 588.19 | 590.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 596.95 | 589.94 | 591.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 600.40 | 589.94 | 591.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 597.00 | 591.35 | 591.90 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 592.95 | 592.27 | 592.26 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 590.05 | 592.27 | 592.43 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 593.75 | 592.67 | 592.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 595.20 | 593.17 | 592.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 595.80 | 595.82 | 594.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:00:00 | 595.80 | 595.82 | 594.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 599.20 | 596.50 | 594.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:30:00 | 600.45 | 597.40 | 595.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 600.50 | 598.65 | 596.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 599.55 | 599.25 | 597.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:45:00 | 601.00 | 599.41 | 597.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 603.65 | 602.78 | 600.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 594.50 | 598.84 | 599.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 594.50 | 598.84 | 599.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 591.75 | 597.42 | 598.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 600.35 | 595.53 | 596.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 600.35 | 595.53 | 596.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 600.35 | 595.53 | 596.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 601.25 | 595.53 | 596.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 600.60 | 596.54 | 596.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 600.15 | 596.54 | 596.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 602.55 | 597.74 | 597.29 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 594.00 | 597.51 | 597.60 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 604.20 | 598.66 | 597.92 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 585.55 | 596.24 | 597.31 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 596.70 | 590.82 | 590.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 15:15:00 | 598.00 | 593.04 | 591.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 595.55 | 595.71 | 594.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:30:00 | 595.55 | 595.71 | 594.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 609.75 | 598.51 | 595.52 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 600.05 | 601.27 | 601.27 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 601.50 | 601.31 | 601.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 601.65 | 601.36 | 601.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 601.00 | 601.29 | 601.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 601.00 | 601.29 | 601.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 601.00 | 601.29 | 601.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 601.00 | 601.29 | 601.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 601.80 | 601.39 | 601.34 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 600.00 | 601.11 | 601.21 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 603.65 | 601.43 | 601.31 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 595.20 | 600.50 | 601.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 593.50 | 598.12 | 599.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 595.05 | 593.56 | 596.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:00:00 | 595.05 | 593.56 | 596.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 597.60 | 594.37 | 596.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 598.50 | 594.37 | 596.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 599.55 | 595.41 | 596.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 599.55 | 595.41 | 596.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 601.05 | 597.91 | 597.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 10:15:00 | 605.50 | 599.43 | 598.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 603.00 | 604.68 | 602.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:30:00 | 603.40 | 604.68 | 602.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 600.95 | 603.63 | 602.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:00:00 | 608.35 | 604.15 | 603.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 623.20 | 628.01 | 628.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 623.20 | 628.01 | 628.05 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 630.20 | 628.23 | 628.09 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 626.40 | 627.86 | 627.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 625.00 | 627.29 | 627.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 616.75 | 616.65 | 620.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:45:00 | 619.45 | 616.65 | 620.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 618.25 | 617.49 | 619.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:30:00 | 619.15 | 617.49 | 619.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 613.75 | 617.07 | 619.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:30:00 | 611.50 | 614.87 | 617.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 609.25 | 613.46 | 616.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 610.45 | 611.56 | 614.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:30:00 | 610.95 | 610.18 | 612.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 612.55 | 610.03 | 611.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:15:00 | 614.90 | 610.03 | 611.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 614.90 | 611.00 | 612.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 616.30 | 611.00 | 612.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 620.20 | 612.84 | 612.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 620.20 | 612.84 | 612.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 622.80 | 619.17 | 616.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 620.00 | 623.51 | 620.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 620.00 | 623.51 | 620.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 620.00 | 623.51 | 620.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 620.00 | 623.51 | 620.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 618.40 | 622.49 | 620.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 618.50 | 622.49 | 620.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 619.70 | 621.93 | 620.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:15:00 | 616.95 | 621.93 | 620.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 617.10 | 620.96 | 620.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 621.00 | 619.94 | 619.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 621.95 | 620.49 | 620.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 618.50 | 619.89 | 620.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 618.50 | 619.89 | 620.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 616.55 | 618.65 | 619.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 612.55 | 612.14 | 614.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 612.55 | 612.14 | 614.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 614.00 | 612.65 | 614.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 614.00 | 612.65 | 614.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 614.35 | 612.99 | 614.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 614.35 | 612.99 | 614.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 617.00 | 613.79 | 614.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 615.05 | 613.79 | 614.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 616.00 | 614.23 | 614.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:15:00 | 618.05 | 614.23 | 614.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 620.45 | 615.48 | 615.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 623.00 | 616.98 | 615.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 618.85 | 624.09 | 621.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 13:15:00 | 618.85 | 624.09 | 621.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 618.85 | 624.09 | 621.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:45:00 | 620.00 | 624.09 | 621.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 615.75 | 622.42 | 620.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 615.75 | 622.42 | 620.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 620.25 | 621.23 | 620.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 617.80 | 621.23 | 620.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 627.10 | 622.41 | 621.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 619.35 | 622.41 | 621.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 624.30 | 623.55 | 622.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:30:00 | 623.40 | 623.55 | 622.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 629.05 | 624.60 | 622.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 10:30:00 | 631.80 | 626.89 | 624.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 635.30 | 642.39 | 642.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 635.30 | 642.39 | 642.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 633.50 | 639.48 | 641.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 634.50 | 632.68 | 635.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 634.35 | 632.68 | 635.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 642.15 | 634.57 | 636.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 642.15 | 634.57 | 636.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 643.35 | 636.33 | 636.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 643.35 | 636.33 | 636.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 646.40 | 638.34 | 637.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 648.80 | 641.79 | 639.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 647.40 | 647.84 | 644.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 647.40 | 647.84 | 644.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 646.10 | 647.49 | 645.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:45:00 | 646.25 | 647.49 | 645.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 651.50 | 649.22 | 646.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 653.50 | 649.22 | 646.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 653.95 | 651.66 | 650.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 653.00 | 650.52 | 650.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 652.65 | 650.61 | 650.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 650.30 | 650.55 | 650.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 650.30 | 650.55 | 650.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 649.10 | 650.26 | 650.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:30:00 | 648.65 | 650.26 | 650.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 650.80 | 650.37 | 650.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 649.40 | 650.37 | 650.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 653.25 | 650.94 | 650.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 653.85 | 651.26 | 650.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 653.30 | 651.16 | 650.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 654.25 | 651.04 | 650.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:45:00 | 656.30 | 652.03 | 651.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 653.85 | 652.40 | 651.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 653.20 | 652.40 | 651.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 684.15 | 687.56 | 685.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 683.85 | 687.56 | 685.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 684.65 | 686.97 | 685.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 686.10 | 686.97 | 685.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 686.85 | 686.95 | 685.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 678.95 | 686.95 | 685.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 680.60 | 685.68 | 684.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 679.55 | 685.68 | 684.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 675.70 | 683.68 | 684.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 675.70 | 683.68 | 684.02 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 689.00 | 683.64 | 683.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 09:15:00 | 691.00 | 687.26 | 685.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 11:15:00 | 681.40 | 686.31 | 685.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 681.40 | 686.31 | 685.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 681.40 | 686.31 | 685.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 681.40 | 686.31 | 685.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 687.25 | 686.50 | 685.80 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 14:15:00 | 680.55 | 684.44 | 684.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 15:15:00 | 678.35 | 683.22 | 684.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 682.60 | 680.34 | 682.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 12:15:00 | 682.60 | 680.34 | 682.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 682.60 | 680.34 | 682.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 683.25 | 680.34 | 682.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 694.40 | 683.15 | 683.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 698.90 | 683.15 | 683.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 679.85 | 682.49 | 683.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 674.75 | 681.86 | 682.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 673.10 | 680.97 | 682.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 676.05 | 673.80 | 677.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 666.15 | 677.00 | 677.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 670.05 | 675.61 | 676.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:30:00 | 662.85 | 673.96 | 676.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:30:00 | 663.55 | 671.82 | 674.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 641.01 | 651.80 | 659.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 642.25 | 651.80 | 659.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 639.44 | 648.17 | 656.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 649.95 | 648.50 | 654.81 | SL hit (close>ema200) qty=0.50 sl=648.50 alert=retest2 |

### Cycle 197 — BUY (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 14:15:00 | 645.10 | 637.66 | 636.66 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 633.55 | 636.21 | 636.45 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 641.25 | 637.35 | 636.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 660.85 | 642.21 | 639.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 653.50 | 653.76 | 648.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 653.50 | 653.76 | 648.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 654.95 | 656.49 | 654.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:15:00 | 653.75 | 656.49 | 654.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 653.75 | 655.95 | 654.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 653.35 | 655.95 | 654.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 653.15 | 655.39 | 654.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 650.85 | 655.39 | 654.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 650.90 | 654.49 | 653.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 645.90 | 654.49 | 653.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 647.00 | 652.66 | 653.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 645.50 | 651.23 | 652.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 652.75 | 651.53 | 652.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 652.75 | 651.53 | 652.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 652.75 | 651.53 | 652.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 652.75 | 651.53 | 652.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 650.35 | 651.29 | 652.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 651.15 | 651.30 | 652.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 652.85 | 651.61 | 652.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 654.65 | 651.61 | 652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 649.35 | 651.16 | 651.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 647.10 | 650.48 | 651.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 645.25 | 648.22 | 650.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:15:00 | 647.00 | 648.18 | 649.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:00:00 | 643.95 | 647.33 | 649.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 640.00 | 637.65 | 639.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:30:00 | 634.95 | 638.46 | 639.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 640.80 | 639.61 | 639.76 | SL hit (close>static) qty=1.00 sl=640.70 alert=retest2 |

### Cycle 201 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 639.90 | 639.21 | 639.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 650.25 | 641.42 | 640.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 14:15:00 | 651.60 | 654.27 | 650.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 651.60 | 654.27 | 650.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 651.35 | 653.68 | 650.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 659.10 | 653.68 | 650.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 654.80 | 665.01 | 665.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 654.80 | 665.01 | 665.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 652.35 | 662.48 | 664.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 626.45 | 625.77 | 634.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 628.40 | 625.77 | 634.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 606.15 | 605.16 | 612.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 604.35 | 605.13 | 612.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 604.40 | 605.02 | 611.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:15:00 | 601.75 | 605.10 | 610.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 599.55 | 590.12 | 589.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 599.55 | 590.12 | 589.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 600.30 | 592.16 | 590.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 15:15:00 | 592.70 | 592.83 | 591.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:15:00 | 595.20 | 592.83 | 591.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 592.35 | 592.73 | 591.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 592.35 | 592.73 | 591.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 587.80 | 591.64 | 591.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 587.80 | 591.64 | 591.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 590.65 | 591.44 | 591.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 13:30:00 | 591.85 | 591.13 | 591.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 14:15:00 | 588.80 | 590.66 | 590.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 14:15:00 | 588.80 | 590.66 | 590.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 15:15:00 | 586.25 | 589.78 | 590.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 541.00 | 535.21 | 545.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 540.60 | 535.21 | 545.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 548.30 | 539.13 | 543.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 529.95 | 541.76 | 543.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 503.45 | 512.12 | 518.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 504.85 | 503.97 | 511.14 | SL hit (close>ema200) qty=0.50 sl=503.97 alert=retest2 |

### Cycle 205 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 512.90 | 510.56 | 510.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 516.40 | 512.55 | 511.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 541.75 | 541.79 | 533.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 542.40 | 541.79 | 533.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 546.10 | 546.18 | 541.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 566.95 | 546.31 | 543.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 554.80 | 556.49 | 556.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 554.80 | 556.49 | 556.50 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 558.90 | 556.97 | 556.72 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 542.20 | 553.91 | 555.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 540.05 | 544.44 | 548.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 521.05 | 520.51 | 528.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:45:00 | 521.20 | 520.51 | 528.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 519.40 | 518.72 | 522.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 519.40 | 518.72 | 522.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 522.00 | 519.38 | 522.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 524.50 | 519.38 | 522.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 528.35 | 521.17 | 522.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 528.35 | 521.17 | 522.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 527.95 | 522.53 | 523.08 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 529.70 | 523.96 | 523.68 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 513.40 | 522.43 | 523.25 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 531.25 | 521.93 | 521.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 535.00 | 525.83 | 523.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 565.25 | 565.71 | 558.83 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 12:15:00 | 427.50 | 2023-05-22 10:15:00 | 434.20 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-05-26 11:45:00 | 442.85 | 2023-06-05 15:15:00 | 486.75 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2023-05-26 12:30:00 | 442.50 | 2023-06-05 15:15:00 | 486.81 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2023-05-26 13:00:00 | 442.55 | 2023-06-06 09:15:00 | 487.14 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2023-06-13 09:15:00 | 509.35 | 2023-06-20 09:15:00 | 560.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-27 12:45:00 | 571.75 | 2023-07-04 13:15:00 | 564.05 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-06-30 12:30:00 | 570.90 | 2023-07-04 13:15:00 | 564.05 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2023-07-03 12:15:00 | 572.95 | 2023-07-04 13:15:00 | 564.05 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-07-04 13:15:00 | 571.65 | 2023-07-04 13:15:00 | 564.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-07-18 15:15:00 | 573.90 | 2023-07-21 09:15:00 | 545.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-18 15:15:00 | 573.90 | 2023-07-21 10:15:00 | 553.25 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2023-08-03 14:30:00 | 570.90 | 2023-08-04 13:15:00 | 577.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-08-04 11:45:00 | 571.15 | 2023-08-04 13:15:00 | 577.25 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-08-09 09:15:00 | 585.30 | 2023-08-09 12:15:00 | 569.90 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2023-10-05 10:15:00 | 553.50 | 2023-10-11 10:15:00 | 526.68 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2023-10-05 11:00:00 | 554.40 | 2023-10-11 11:15:00 | 526.35 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2023-10-05 11:30:00 | 553.60 | 2023-10-11 12:15:00 | 525.82 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2023-10-09 11:45:00 | 554.05 | 2023-10-11 12:15:00 | 525.92 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2023-10-05 10:15:00 | 553.50 | 2023-10-16 09:15:00 | 526.90 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2023-10-05 11:00:00 | 554.40 | 2023-10-16 09:15:00 | 526.90 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest2 | 2023-10-05 11:30:00 | 553.60 | 2023-10-16 09:15:00 | 526.90 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2023-10-09 11:45:00 | 554.05 | 2023-10-16 09:15:00 | 526.90 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2023-10-11 10:15:00 | 530.75 | 2023-10-16 13:15:00 | 531.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-10-11 10:45:00 | 529.10 | 2023-10-16 13:15:00 | 531.95 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-10-16 12:30:00 | 530.20 | 2023-10-16 13:15:00 | 531.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2023-10-30 09:15:00 | 515.40 | 2023-10-30 13:15:00 | 520.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-10-31 09:15:00 | 525.70 | 2023-11-01 12:15:00 | 517.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-11-01 11:30:00 | 521.25 | 2023-11-01 12:15:00 | 517.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-11-06 09:15:00 | 524.60 | 2023-11-07 09:15:00 | 517.55 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-11-06 15:15:00 | 522.95 | 2023-11-07 09:15:00 | 517.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-11-09 11:15:00 | 528.80 | 2023-11-13 09:15:00 | 527.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-11-09 11:45:00 | 528.15 | 2023-11-13 09:15:00 | 527.10 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-11-10 10:45:00 | 528.50 | 2023-11-13 09:15:00 | 527.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-11-10 13:15:00 | 528.60 | 2023-11-13 09:15:00 | 527.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2023-11-21 10:15:00 | 557.20 | 2023-11-28 09:15:00 | 552.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-11-22 09:45:00 | 554.80 | 2023-11-28 09:15:00 | 552.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-12-04 09:15:00 | 561.15 | 2023-12-05 09:15:00 | 546.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-12-04 10:30:00 | 560.70 | 2023-12-05 09:15:00 | 546.70 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2023-12-04 14:15:00 | 560.75 | 2023-12-05 09:15:00 | 546.70 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2023-12-12 11:45:00 | 540.05 | 2023-12-12 13:15:00 | 549.65 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2023-12-20 13:00:00 | 520.95 | 2023-12-26 09:15:00 | 519.95 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2023-12-22 09:45:00 | 520.90 | 2023-12-26 09:15:00 | 519.95 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2023-12-22 12:00:00 | 520.95 | 2023-12-26 09:15:00 | 519.95 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-01-03 09:15:00 | 539.00 | 2024-01-11 15:15:00 | 540.60 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2024-01-04 09:15:00 | 539.50 | 2024-01-11 15:15:00 | 540.60 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-01-04 11:00:00 | 536.70 | 2024-01-11 15:15:00 | 540.60 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-01-23 11:15:00 | 485.30 | 2024-01-29 09:15:00 | 488.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-01-24 09:45:00 | 484.00 | 2024-01-29 10:15:00 | 490.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-01-24 11:15:00 | 485.05 | 2024-01-29 10:15:00 | 490.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-01-25 11:00:00 | 485.45 | 2024-01-29 10:15:00 | 490.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-01-25 14:30:00 | 483.75 | 2024-01-29 10:15:00 | 490.60 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-02-06 09:15:00 | 512.80 | 2024-02-09 10:15:00 | 513.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-02-14 09:15:00 | 506.35 | 2024-02-16 12:15:00 | 514.80 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-02-23 12:30:00 | 523.40 | 2024-02-27 13:15:00 | 522.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-02-23 13:30:00 | 523.50 | 2024-02-27 13:15:00 | 522.70 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-02-26 09:30:00 | 524.50 | 2024-02-27 13:15:00 | 522.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-02-26 15:15:00 | 524.35 | 2024-02-28 12:15:00 | 521.25 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-02-27 10:15:00 | 537.00 | 2024-02-28 12:15:00 | 521.25 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-02-27 11:45:00 | 535.80 | 2024-02-28 12:15:00 | 521.25 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-02-27 12:30:00 | 536.20 | 2024-02-28 12:15:00 | 521.25 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-03-04 10:15:00 | 540.90 | 2024-03-07 15:15:00 | 594.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-04 11:00:00 | 541.65 | 2024-03-11 09:15:00 | 595.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-04 12:30:00 | 548.45 | 2024-03-11 09:15:00 | 603.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-03-26 09:15:00 | 596.60 | 2024-04-01 09:15:00 | 626.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-26 09:15:00 | 596.60 | 2024-04-03 10:15:00 | 620.65 | STOP_HIT | 0.50 | 4.03% |
| BUY | retest2 | 2024-04-10 14:00:00 | 626.80 | 2024-04-15 09:15:00 | 614.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-04-24 09:15:00 | 575.60 | 2024-04-30 15:15:00 | 571.50 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-05-06 09:15:00 | 580.95 | 2024-05-06 12:15:00 | 573.35 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-05-16 14:00:00 | 585.25 | 2024-05-17 09:15:00 | 592.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-05-28 11:45:00 | 576.55 | 2024-05-30 11:15:00 | 547.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 13:00:00 | 575.95 | 2024-05-30 11:15:00 | 547.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 574.65 | 2024-05-30 11:15:00 | 545.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 11:45:00 | 576.55 | 2024-05-31 13:15:00 | 550.85 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2024-05-28 13:00:00 | 575.95 | 2024-05-31 13:15:00 | 550.85 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2024-05-29 09:15:00 | 574.65 | 2024-05-31 13:15:00 | 550.85 | STOP_HIT | 0.50 | 4.14% |
| BUY | retest2 | 2024-06-13 13:45:00 | 592.30 | 2024-06-20 11:15:00 | 599.35 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2024-06-25 09:15:00 | 608.30 | 2024-06-25 10:15:00 | 603.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-27 13:45:00 | 597.85 | 2024-06-28 09:15:00 | 604.95 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-07-02 11:15:00 | 616.00 | 2024-07-16 09:15:00 | 642.30 | STOP_HIT | 1.00 | 4.27% |
| BUY | retest2 | 2024-07-02 13:30:00 | 617.05 | 2024-07-16 09:15:00 | 642.30 | STOP_HIT | 1.00 | 4.09% |
| SELL | retest2 | 2024-07-19 09:15:00 | 646.20 | 2024-07-23 12:15:00 | 613.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 646.20 | 2024-07-23 13:15:00 | 632.45 | STOP_HIT | 0.50 | 2.13% |
| BUY | retest2 | 2024-07-30 11:00:00 | 724.20 | 2024-08-05 09:15:00 | 721.35 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-07-31 09:45:00 | 724.10 | 2024-08-05 09:15:00 | 721.35 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-08-12 11:45:00 | 739.35 | 2024-08-12 14:15:00 | 729.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-19 09:15:00 | 716.50 | 2024-08-20 09:15:00 | 731.05 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-08-19 12:00:00 | 717.05 | 2024-08-20 09:15:00 | 731.05 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-08-19 14:45:00 | 717.85 | 2024-08-20 09:15:00 | 731.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-08-19 15:15:00 | 717.40 | 2024-08-20 09:15:00 | 731.05 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-08-30 10:15:00 | 744.75 | 2024-09-06 10:15:00 | 752.90 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2024-09-13 14:30:00 | 759.45 | 2024-09-16 09:15:00 | 750.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-24 15:00:00 | 778.05 | 2024-09-24 15:15:00 | 767.65 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-10-04 13:15:00 | 758.55 | 2024-10-09 13:15:00 | 762.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-10-14 09:15:00 | 736.30 | 2024-10-15 09:15:00 | 745.35 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-14 10:00:00 | 735.90 | 2024-10-15 09:15:00 | 745.35 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-14 11:45:00 | 736.50 | 2024-10-15 09:15:00 | 745.35 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-10-21 14:30:00 | 751.15 | 2024-10-22 12:15:00 | 742.85 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-10-22 09:15:00 | 758.25 | 2024-10-22 12:15:00 | 742.85 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-11-05 15:15:00 | 726.00 | 2024-11-13 10:15:00 | 689.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-05 15:15:00 | 726.00 | 2024-11-14 09:15:00 | 696.45 | STOP_HIT | 0.50 | 4.07% |
| BUY | retest2 | 2024-12-03 13:00:00 | 694.95 | 2024-12-03 14:15:00 | 683.55 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-12-06 14:15:00 | 673.95 | 2024-12-10 09:15:00 | 687.45 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-12-09 09:15:00 | 674.60 | 2024-12-10 09:15:00 | 687.45 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-12-09 14:15:00 | 674.90 | 2024-12-10 09:15:00 | 687.45 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-12-23 09:15:00 | 648.00 | 2024-12-24 14:15:00 | 661.55 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-12-23 12:45:00 | 649.00 | 2024-12-24 14:15:00 | 661.55 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-12-23 15:15:00 | 648.00 | 2024-12-24 14:15:00 | 661.55 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-12-24 10:30:00 | 649.65 | 2024-12-24 14:15:00 | 661.55 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-01-15 09:15:00 | 638.70 | 2025-01-16 09:15:00 | 660.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-01-21 10:15:00 | 638.75 | 2025-01-22 09:15:00 | 574.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-21 13:45:00 | 640.25 | 2025-01-22 09:15:00 | 576.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 10:30:00 | 594.95 | 2025-02-12 09:15:00 | 565.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:30:00 | 594.95 | 2025-02-12 10:15:00 | 577.40 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2025-02-19 14:15:00 | 572.10 | 2025-03-04 11:15:00 | 543.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 09:45:00 | 571.30 | 2025-03-04 11:15:00 | 543.64 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-02-19 14:15:00 | 572.10 | 2025-03-05 09:15:00 | 551.35 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-02-21 09:45:00 | 571.30 | 2025-03-05 09:15:00 | 551.35 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2025-02-21 13:15:00 | 572.25 | 2025-03-10 15:15:00 | 542.73 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-02-24 09:15:00 | 568.50 | 2025-03-11 09:15:00 | 540.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 12:00:00 | 565.70 | 2025-03-11 09:15:00 | 537.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 13:15:00 | 572.25 | 2025-03-11 14:15:00 | 546.85 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2025-02-24 09:15:00 | 568.50 | 2025-03-11 14:15:00 | 546.85 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2025-02-27 12:00:00 | 565.70 | 2025-03-11 14:15:00 | 546.85 | STOP_HIT | 0.50 | 3.33% |
| BUY | retest2 | 2025-03-26 09:15:00 | 598.90 | 2025-03-26 11:15:00 | 587.40 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-03-28 10:45:00 | 588.00 | 2025-04-02 09:15:00 | 558.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 10:45:00 | 588.00 | 2025-04-03 09:15:00 | 565.60 | STOP_HIT | 0.50 | 3.81% |
| BUY | retest2 | 2025-04-23 11:15:00 | 595.60 | 2025-04-25 10:15:00 | 589.55 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-05 09:45:00 | 612.75 | 2025-05-06 09:15:00 | 606.20 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-05 14:45:00 | 611.20 | 2025-05-06 09:15:00 | 606.20 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-05-15 11:45:00 | 607.15 | 2025-05-21 09:15:00 | 613.80 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-07-09 10:15:00 | 678.85 | 2025-07-11 11:15:00 | 662.10 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-07-09 14:30:00 | 677.80 | 2025-07-11 11:15:00 | 662.10 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-07-24 12:15:00 | 627.05 | 2025-08-05 12:15:00 | 611.85 | STOP_HIT | 1.00 | 2.42% |
| SELL | retest2 | 2025-07-24 13:45:00 | 626.40 | 2025-08-05 12:15:00 | 611.85 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-08-06 12:15:00 | 616.45 | 2025-08-08 14:15:00 | 609.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-07 14:15:00 | 613.85 | 2025-08-08 14:15:00 | 609.75 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-20 09:30:00 | 635.10 | 2025-08-21 12:15:00 | 630.85 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-20 12:15:00 | 634.70 | 2025-08-21 12:15:00 | 630.85 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-08-20 13:00:00 | 634.85 | 2025-08-21 12:15:00 | 630.85 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-20 15:15:00 | 635.00 | 2025-08-21 12:15:00 | 630.85 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-26 09:15:00 | 622.30 | 2025-09-04 09:15:00 | 618.40 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-10-03 14:30:00 | 600.45 | 2025-10-08 13:15:00 | 594.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-06 09:30:00 | 600.50 | 2025-10-08 13:15:00 | 594.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-06 12:30:00 | 599.55 | 2025-10-08 13:15:00 | 594.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-06 13:45:00 | 601.00 | 2025-10-08 13:15:00 | 594.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-07 12:00:00 | 608.35 | 2025-11-18 09:15:00 | 623.20 | STOP_HIT | 1.00 | 2.44% |
| SELL | retest2 | 2025-11-21 14:30:00 | 611.50 | 2025-11-26 09:15:00 | 620.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-24 10:15:00 | 609.25 | 2025-11-26 09:15:00 | 620.20 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-24 14:45:00 | 610.45 | 2025-11-26 09:15:00 | 620.20 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-25 10:30:00 | 610.95 | 2025-11-26 09:15:00 | 620.20 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-28 15:15:00 | 621.00 | 2025-12-02 10:15:00 | 618.50 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-12-01 09:30:00 | 621.95 | 2025-12-02 10:15:00 | 618.50 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-10 10:30:00 | 631.80 | 2025-12-16 15:15:00 | 635.30 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-12-23 10:15:00 | 653.50 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.40% |
| BUY | retest2 | 2025-12-26 09:45:00 | 653.95 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.33% |
| BUY | retest2 | 2025-12-29 10:00:00 | 653.00 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2025-12-29 12:15:00 | 652.65 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2025-12-30 09:30:00 | 653.85 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest2 | 2025-12-30 10:30:00 | 653.30 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2025-12-30 13:15:00 | 654.25 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 3.28% |
| BUY | retest2 | 2025-12-30 13:45:00 | 656.30 | 2026-01-08 10:15:00 | 675.70 | STOP_HIT | 1.00 | 2.96% |
| SELL | retest2 | 2026-01-14 10:15:00 | 674.75 | 2026-01-21 11:15:00 | 641.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:15:00 | 673.10 | 2026-01-21 11:15:00 | 642.25 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2026-01-16 10:00:00 | 676.05 | 2026-01-21 13:15:00 | 639.44 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2026-01-14 10:15:00 | 674.75 | 2026-01-21 15:15:00 | 649.95 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2026-01-14 11:15:00 | 673.10 | 2026-01-21 15:15:00 | 649.95 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2026-01-16 10:00:00 | 676.05 | 2026-01-21 15:15:00 | 649.95 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2026-01-19 09:15:00 | 666.15 | 2026-01-27 09:15:00 | 632.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:30:00 | 662.85 | 2026-01-27 09:15:00 | 629.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:30:00 | 663.55 | 2026-01-27 09:15:00 | 630.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 666.15 | 2026-01-27 10:15:00 | 644.65 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2026-01-19 10:30:00 | 662.85 | 2026-01-27 10:15:00 | 644.65 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2026-01-19 11:30:00 | 663.55 | 2026-01-27 10:15:00 | 644.65 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-02-09 15:15:00 | 647.10 | 2026-02-16 14:15:00 | 640.80 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2026-02-10 11:15:00 | 645.25 | 2026-02-18 11:15:00 | 640.80 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-02-10 12:15:00 | 647.00 | 2026-02-18 13:15:00 | 639.90 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2026-02-10 13:00:00 | 643.95 | 2026-02-18 13:15:00 | 639.90 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2026-02-16 09:30:00 | 634.95 | 2026-02-18 13:15:00 | 639.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-02-17 10:45:00 | 637.25 | 2026-02-18 13:15:00 | 639.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-02-23 09:15:00 | 659.10 | 2026-02-27 11:15:00 | 654.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-03-10 11:15:00 | 604.35 | 2026-03-17 11:15:00 | 599.55 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2026-03-10 12:15:00 | 604.40 | 2026-03-17 11:15:00 | 599.55 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2026-03-10 13:15:00 | 601.75 | 2026-03-17 11:15:00 | 599.55 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2026-03-18 13:30:00 | 591.85 | 2026-03-18 14:15:00 | 588.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-03-27 09:15:00 | 529.95 | 2026-04-02 09:15:00 | 503.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 529.95 | 2026-04-02 14:15:00 | 504.85 | STOP_HIT | 0.50 | 4.74% |
| BUY | retest2 | 2026-04-15 09:15:00 | 566.95 | 2026-04-20 13:15:00 | 554.80 | STOP_HIT | 1.00 | -2.14% |

# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 519.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT2_SKIP | 5 |
| ALERT3 | 85 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 80 |
| PARTIAL | 8 |
| TARGET_HIT | 26 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 54
- **Target hits / Stop hits / Partials:** 26 / 56 / 8
- **Avg / median % per leg:** 2.09% / -1.18%
- **Sum % (uncompounded):** 188.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 21 | 44.7% | 21 | 26 | 0 | 3.37% | 158.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.47% | -2.9% |
| BUY @ 3rd Alert (retest2) | 45 | 21 | 46.7% | 21 | 24 | 0 | 3.59% | 161.4% |
| SELL (all) | 43 | 15 | 34.9% | 5 | 30 | 8 | 0.70% | 30.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 15 | 34.9% | 5 | 30 | 8 | 0.70% | 30.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.47% | -2.9% |
| retest2 (combined) | 88 | 36 | 40.9% | 26 | 54 | 8 | 2.18% | 191.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 11:15:00 | 628.60 | 580.59 | 580.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 14:15:00 | 634.90 | 582.13 | 581.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 12:15:00 | 625.25 | 625.46 | 611.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-17 13:00:00 | 625.25 | 625.46 | 611.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 612.95 | 624.94 | 612.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 612.95 | 624.94 | 612.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 608.35 | 624.78 | 612.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 616.65 | 624.78 | 612.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 619.55 | 624.73 | 612.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 625.50 | 624.36 | 612.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:30:00 | 622.35 | 624.09 | 613.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 10:15:00 | 622.60 | 624.09 | 613.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 11:00:00 | 622.20 | 624.07 | 613.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 612.45 | 623.32 | 613.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:00:00 | 612.45 | 623.32 | 613.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 612.30 | 623.21 | 613.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:15:00 | 610.40 | 623.21 | 613.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 606.80 | 623.05 | 613.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-28 09:15:00 | 606.80 | 623.05 | 613.33 | SL hit (close<static) qty=1.00 sl=607.05 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 587.90 | 618.03 | 618.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 581.75 | 611.87 | 614.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 12:15:00 | 566.05 | 565.25 | 580.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 13:00:00 | 566.05 | 565.25 | 580.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 583.25 | 565.93 | 579.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 12:45:00 | 583.00 | 565.93 | 579.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 577.10 | 566.04 | 579.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 14:15:00 | 574.40 | 566.04 | 579.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 15:15:00 | 576.00 | 566.41 | 579.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 13:15:00 | 545.68 | 561.62 | 572.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 13:15:00 | 547.20 | 561.62 | 572.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-12-21 09:15:00 | 518.40 | 556.35 | 568.42 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 12:15:00 | 571.05 | 561.90 | 561.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 13:15:00 | 574.20 | 562.02 | 561.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 14:15:00 | 562.20 | 562.57 | 562.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 14:15:00 | 562.20 | 562.57 | 562.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 562.20 | 562.57 | 562.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:45:00 | 557.60 | 562.57 | 562.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 562.70 | 562.58 | 562.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 559.25 | 562.58 | 562.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 554.50 | 562.50 | 562.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:30:00 | 554.50 | 562.50 | 562.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 556.80 | 562.44 | 562.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 556.80 | 562.44 | 562.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 555.05 | 561.97 | 561.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 555.05 | 561.97 | 561.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 554.80 | 561.90 | 561.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 15:15:00 | 551.90 | 561.73 | 561.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 09:15:00 | 561.50 | 559.12 | 560.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 561.50 | 559.12 | 560.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 561.50 | 559.12 | 560.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 09:30:00 | 558.30 | 560.30 | 560.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 09:15:00 | 557.90 | 560.13 | 560.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 12:15:00 | 579.75 | 560.83 | 561.11 | SL hit (close>static) qty=1.00 sl=579.60 alert=retest2 |

### Cycle 5 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 578.65 | 561.45 | 561.42 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 543.50 | 561.60 | 561.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 526.80 | 555.89 | 558.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 14:15:00 | 552.95 | 551.56 | 555.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 15:00:00 | 552.95 | 551.56 | 555.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 554.65 | 551.55 | 555.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 10:15:00 | 553.00 | 551.55 | 555.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-02 11:15:00 | 564.90 | 551.79 | 555.29 | SL hit (close>static) qty=1.00 sl=562.55 alert=retest2 |

### Cycle 7 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 575.35 | 557.56 | 557.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 577.75 | 557.76 | 557.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 15:15:00 | 559.00 | 559.77 | 558.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 09:15:00 | 563.10 | 559.77 | 558.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:45:00 | 563.05 | 559.81 | 558.73 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 554.80 | 560.79 | 559.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-03 10:15:00 | 554.80 | 560.79 | 559.36 | SL hit (close<ema400) qty=1.00 sl=559.36 alert=retest1 |

### Cycle 8 — SELL (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 13:15:00 | 533.60 | 557.93 | 558.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 12:15:00 | 530.00 | 555.27 | 556.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 12:15:00 | 551.60 | 550.30 | 553.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 12:30:00 | 553.05 | 550.30 | 553.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 553.95 | 550.35 | 553.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 553.95 | 550.35 | 553.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 554.00 | 550.38 | 553.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 556.85 | 550.38 | 553.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 554.80 | 550.43 | 553.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 10:30:00 | 549.35 | 550.51 | 553.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 09:45:00 | 551.25 | 549.46 | 552.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 549.90 | 549.49 | 552.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 15:15:00 | 523.69 | 546.97 | 551.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:15:00 | 521.88 | 546.48 | 550.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:15:00 | 522.40 | 546.48 | 550.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 494.42 | 538.27 | 545.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 580.00 | 540.89 | 540.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 585.15 | 542.09 | 541.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 571.45 | 583.98 | 568.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 571.45 | 583.98 | 568.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 571.45 | 583.98 | 568.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:00:00 | 581.90 | 583.82 | 568.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 581.00 | 582.51 | 569.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 10:00:00 | 582.00 | 582.94 | 570.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 581.20 | 582.70 | 570.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 571.55 | 582.43 | 571.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 568.60 | 582.43 | 571.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 569.70 | 582.31 | 571.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 569.70 | 582.31 | 571.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 570.75 | 582.19 | 571.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:45:00 | 569.95 | 582.19 | 571.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 572.00 | 582.09 | 571.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:00:00 | 573.55 | 582.01 | 571.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-30 14:15:00 | 640.09 | 593.80 | 581.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 551.00 | 592.38 | 592.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 548.70 | 585.74 | 588.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 495.05 | 490.92 | 520.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:45:00 | 494.30 | 490.92 | 520.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 389.70 | 363.51 | 386.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 09:30:00 | 390.40 | 363.51 | 386.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 386.35 | 363.74 | 386.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 383.20 | 364.19 | 386.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 391.00 | 365.09 | 386.87 | SL hit (close>static) qty=1.00 sl=389.70 alert=retest2 |

### Cycle 11 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 440.00 | 391.95 | 391.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 447.80 | 394.28 | 393.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 438.95 | 442.78 | 425.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:00:00 | 438.95 | 442.78 | 425.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 426.70 | 442.03 | 426.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 426.70 | 442.03 | 426.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 425.05 | 441.86 | 426.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 425.05 | 441.86 | 426.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 427.75 | 441.72 | 426.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 424.50 | 441.72 | 426.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 424.00 | 441.42 | 426.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:00:00 | 430.55 | 440.33 | 426.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 432.30 | 440.25 | 426.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 430.95 | 439.98 | 426.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 430.75 | 439.78 | 426.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 425.15 | 439.33 | 426.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 425.15 | 439.33 | 426.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 427.80 | 439.22 | 426.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:30:00 | 429.05 | 439.22 | 426.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 427.70 | 439.10 | 426.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 430.70 | 439.10 | 426.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 430.70 | 438.59 | 427.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 424.75 | 438.26 | 427.06 | SL hit (close<static) qty=1.00 sl=426.15 alert=retest2 |

### Cycle 12 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 458.95 | 473.27 | 473.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 455.30 | 472.43 | 472.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 469.30 | 465.28 | 468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 468.80 | 465.31 | 468.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 469.55 | 465.31 | 468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 467.00 | 465.33 | 468.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 462.25 | 465.33 | 468.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 439.14 | 457.40 | 463.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 455.10 | 448.39 | 456.93 | SL hit (close>ema200) qty=0.50 sl=448.39 alert=retest2 |

### Cycle 13 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 472.30 | 461.79 | 461.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 476.60 | 462.04 | 461.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 463.10 | 463.55 | 462.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 463.10 | 463.55 | 462.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 463.00 | 463.55 | 462.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 462.75 | 463.55 | 462.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 461.05 | 463.52 | 462.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 461.65 | 463.52 | 462.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 458.90 | 463.47 | 462.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 458.90 | 463.47 | 462.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 454.20 | 463.34 | 462.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 459.75 | 462.84 | 462.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 459.90 | 462.78 | 462.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 458.75 | 462.61 | 462.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 473.80 | 462.05 | 462.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 460.60 | 463.52 | 462.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 445.60 | 462.14 | 462.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 445.60 | 462.14 | 462.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 440.55 | 461.92 | 462.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:00:00 | 463.00 | 459.88 | 460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 461.60 | 459.89 | 460.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 462.80 | 459.89 | 460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 461.00 | 459.91 | 460.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 458.55 | 459.90 | 460.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 458.80 | 459.90 | 460.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 464.50 | 459.94 | 460.93 | SL hit (close>static) qty=1.00 sl=461.80 alert=retest2 |

### Cycle 15 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 470.30 | 460.54 | 460.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 15:15:00 | 475.00 | 460.93 | 460.73 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-23 09:15:00 | 625.50 | 2023-08-28 09:15:00 | 606.80 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2023-08-24 09:30:00 | 622.35 | 2023-08-28 09:15:00 | 606.80 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2023-08-24 10:15:00 | 622.60 | 2023-08-28 09:15:00 | 606.80 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2023-08-24 11:00:00 | 622.20 | 2023-08-28 09:15:00 | 606.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-08-28 11:15:00 | 611.15 | 2023-09-08 09:15:00 | 672.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 13:30:00 | 611.25 | 2023-09-08 09:15:00 | 672.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-27 15:15:00 | 612.50 | 2023-09-28 09:15:00 | 602.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-11-28 14:15:00 | 574.40 | 2023-12-15 13:15:00 | 545.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-30 15:15:00 | 576.00 | 2023-12-15 13:15:00 | 547.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-28 14:15:00 | 574.40 | 2023-12-21 09:15:00 | 518.40 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2023-11-30 15:15:00 | 576.00 | 2023-12-21 10:15:00 | 516.96 | TARGET_HIT | 0.50 | 10.25% |
| SELL | retest2 | 2024-01-31 13:45:00 | 575.55 | 2024-02-01 13:15:00 | 590.55 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-01-31 14:45:00 | 576.60 | 2024-02-01 13:15:00 | 590.55 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-02-22 09:30:00 | 558.30 | 2024-02-26 12:15:00 | 579.75 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-02-23 09:15:00 | 557.90 | 2024-02-26 12:15:00 | 579.75 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2024-03-28 10:15:00 | 553.00 | 2024-04-02 11:15:00 | 564.90 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-04-12 13:45:00 | 553.05 | 2024-04-18 13:15:00 | 565.05 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-04-12 15:00:00 | 549.80 | 2024-04-18 13:15:00 | 565.05 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-04-15 12:00:00 | 553.05 | 2024-04-18 13:15:00 | 565.05 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-04-18 12:30:00 | 556.25 | 2024-04-18 13:15:00 | 565.05 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest1 | 2024-04-29 09:15:00 | 563.10 | 2024-05-03 10:15:00 | 554.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest1 | 2024-04-29 10:45:00 | 563.05 | 2024-05-03 10:15:00 | 554.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-05-06 09:15:00 | 559.00 | 2024-05-06 09:15:00 | 554.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-06 11:00:00 | 559.90 | 2024-05-06 14:15:00 | 551.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-05-06 13:30:00 | 558.50 | 2024-05-06 14:15:00 | 551.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-06 14:00:00 | 558.40 | 2024-05-06 14:15:00 | 551.90 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-05-21 10:30:00 | 549.35 | 2024-05-28 15:15:00 | 523.69 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2024-05-23 09:45:00 | 551.25 | 2024-05-29 10:15:00 | 521.88 | PARTIAL | 0.50 | 5.33% |
| SELL | retest2 | 2024-05-23 11:15:00 | 549.90 | 2024-05-29 10:15:00 | 522.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-21 10:30:00 | 549.35 | 2024-06-04 12:15:00 | 494.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-23 09:45:00 | 551.25 | 2024-06-04 12:15:00 | 496.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-23 11:15:00 | 549.90 | 2024-06-04 12:15:00 | 494.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-05 12:00:00 | 581.90 | 2024-08-30 14:15:00 | 640.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-07 15:00:00 | 581.00 | 2024-08-30 14:15:00 | 639.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 10:00:00 | 582.00 | 2024-08-30 14:15:00 | 640.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 15:15:00 | 581.20 | 2024-08-30 14:15:00 | 639.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 14:00:00 | 573.55 | 2024-08-30 14:15:00 | 630.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-07 10:00:00 | 576.95 | 2024-10-07 10:15:00 | 568.65 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-09 09:30:00 | 573.45 | 2024-10-10 09:15:00 | 568.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-10-09 13:30:00 | 573.60 | 2024-10-10 09:15:00 | 568.40 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-04-16 13:15:00 | 383.20 | 2025-04-17 09:15:00 | 391.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-30 13:30:00 | 382.60 | 2025-05-06 11:15:00 | 363.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 09:15:00 | 369.70 | 2025-05-07 09:15:00 | 351.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 13:30:00 | 382.60 | 2025-05-13 09:15:00 | 377.40 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2025-05-02 09:15:00 | 369.70 | 2025-05-13 09:15:00 | 377.40 | STOP_HIT | 0.50 | -2.08% |
| SELL | retest2 | 2025-05-13 09:45:00 | 382.50 | 2025-05-13 11:15:00 | 394.30 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-06-23 10:00:00 | 430.55 | 2025-06-26 11:15:00 | 424.75 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-06-23 11:00:00 | 432.30 | 2025-06-26 11:15:00 | 424.75 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-23 14:15:00 | 430.95 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-24 09:15:00 | 430.75 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-06-25 09:15:00 | 430.70 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-26 09:15:00 | 430.70 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-09 11:15:00 | 432.55 | 2025-07-10 13:15:00 | 425.25 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-07-16 09:15:00 | 432.40 | 2025-07-25 09:15:00 | 425.15 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-30 09:45:00 | 435.25 | 2025-09-04 09:15:00 | 478.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 12:45:00 | 433.60 | 2025-09-04 09:15:00 | 476.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 10:00:00 | 432.50 | 2025-09-04 09:15:00 | 475.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 12:15:00 | 432.55 | 2025-09-04 09:15:00 | 475.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:00:00 | 435.40 | 2025-09-04 09:15:00 | 478.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:30:00 | 435.15 | 2025-09-04 09:15:00 | 478.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 09:30:00 | 435.50 | 2025-09-04 09:15:00 | 479.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 12:15:00 | 436.45 | 2025-09-04 09:15:00 | 480.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 09:15:00 | 443.00 | 2025-10-08 13:15:00 | 485.10 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2025-09-10 15:15:00 | 441.00 | 2025-10-08 13:15:00 | 485.43 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2025-09-11 13:30:00 | 443.05 | 2025-10-15 15:15:00 | 487.30 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2025-09-12 10:15:00 | 441.30 | 2025-10-15 15:15:00 | 487.36 | TARGET_HIT | 1.00 | 10.44% |
| BUY | retest2 | 2025-09-30 09:15:00 | 445.30 | 2025-10-15 15:15:00 | 489.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 15:00:00 | 446.50 | 2025-10-16 13:15:00 | 491.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 12:15:00 | 462.25 | 2026-01-16 14:15:00 | 439.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 12:15:00 | 462.25 | 2026-01-29 09:15:00 | 455.10 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-02-01 12:15:00 | 465.10 | 2026-02-01 12:15:00 | 470.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-02 14:00:00 | 465.50 | 2026-02-03 09:15:00 | 474.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-02 14:30:00 | 466.30 | 2026-02-03 09:15:00 | 474.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-06 12:30:00 | 457.15 | 2026-02-09 09:15:00 | 464.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-23 09:15:00 | 459.75 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-02-23 11:00:00 | 459.90 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-02-23 15:00:00 | 458.75 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-02-25 09:15:00 | 473.80 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -5.95% |
| SELL | retest2 | 2026-03-13 09:30:00 | 458.55 | 2026-03-13 14:15:00 | 464.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-13 10:15:00 | 458.80 | 2026-03-13 14:15:00 | 464.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-16 10:15:00 | 455.40 | 2026-03-17 12:15:00 | 463.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-03-16 15:00:00 | 458.75 | 2026-03-17 12:15:00 | 463.55 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-03-17 14:00:00 | 458.30 | 2026-03-18 09:15:00 | 463.85 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-03-17 14:30:00 | 458.70 | 2026-03-18 09:15:00 | 463.85 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-17 15:00:00 | 458.40 | 2026-03-18 09:15:00 | 463.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-19 11:45:00 | 458.00 | 2026-03-25 09:15:00 | 462.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-03-20 15:15:00 | 449.95 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2026-03-30 09:15:00 | 451.35 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-03-30 15:15:00 | 450.00 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2026-04-01 09:30:00 | 451.95 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -3.77% |

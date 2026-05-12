# Indian Hotels Co. Ltd. (INDHOTEL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 672.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 157 |
| ALERT1 | 108 |
| ALERT2 | 107 |
| ALERT2_SKIP | 48 |
| ALERT3 | 273 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 102 |
| PARTIAL | 8 |
| TARGET_HIT | 8 |
| STOP_HIT | 100 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 76
- **Target hits / Stop hits / Partials:** 8 / 99 / 8
- **Avg / median % per leg:** 0.49% / -0.74%
- **Sum % (uncompounded):** 56.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 16 | 24.6% | 7 | 58 | 0 | 0.19% | 12.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.23% | -0.2% |
| BUY @ 3rd Alert (retest2) | 64 | 16 | 25.0% | 7 | 57 | 0 | 0.19% | 12.5% |
| SELL (all) | 50 | 23 | 46.0% | 1 | 41 | 8 | 0.88% | 44.1% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.58% | -2.3% |
| SELL @ 3rd Alert (retest2) | 46 | 23 | 50.0% | 1 | 37 | 8 | 1.01% | 46.4% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.51% | -2.5% |
| retest2 (combined) | 110 | 39 | 35.5% | 8 | 94 | 8 | 0.54% | 58.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 559.85 | 552.22 | 552.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 564.15 | 556.30 | 554.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 562.60 | 562.66 | 559.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:00:00 | 562.60 | 562.66 | 559.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 561.00 | 562.51 | 559.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:00:00 | 561.00 | 562.51 | 559.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 561.50 | 562.31 | 559.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 568.90 | 562.15 | 560.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 565.70 | 569.20 | 569.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 565.70 | 569.20 | 569.46 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 13:15:00 | 572.75 | 569.31 | 569.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 580.20 | 571.32 | 570.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 575.90 | 579.57 | 576.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 11:15:00 | 575.90 | 579.57 | 576.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 575.90 | 579.57 | 576.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 575.90 | 579.57 | 576.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 579.00 | 579.45 | 576.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 575.40 | 579.45 | 576.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 575.55 | 578.67 | 576.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 575.55 | 578.67 | 576.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 572.65 | 577.47 | 576.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:30:00 | 573.35 | 577.47 | 576.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 571.40 | 576.26 | 575.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 571.45 | 576.26 | 575.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 569.50 | 574.90 | 575.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 565.85 | 570.87 | 572.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 557.40 | 554.20 | 559.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 15:00:00 | 557.40 | 554.20 | 559.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 559.80 | 555.32 | 559.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 570.05 | 555.32 | 559.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 573.65 | 558.99 | 560.78 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 571.35 | 563.78 | 562.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 582.00 | 571.11 | 566.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 563.70 | 569.63 | 566.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 563.70 | 569.63 | 566.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 563.70 | 569.63 | 566.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 554.90 | 569.63 | 566.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 539.90 | 563.68 | 564.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 523.80 | 555.71 | 560.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 552.65 | 541.74 | 549.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 552.65 | 541.74 | 549.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 552.65 | 541.74 | 549.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 552.65 | 541.74 | 549.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 561.80 | 545.75 | 550.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 564.80 | 545.75 | 550.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 560.35 | 548.67 | 551.58 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 571.25 | 556.42 | 554.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 578.05 | 567.95 | 561.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 579.45 | 582.56 | 577.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:00:00 | 579.45 | 582.56 | 577.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 576.95 | 581.03 | 577.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:00:00 | 576.95 | 581.03 | 577.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 579.50 | 580.72 | 577.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:45:00 | 581.40 | 581.17 | 577.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 580.55 | 580.91 | 578.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:45:00 | 581.35 | 580.98 | 578.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:30:00 | 581.05 | 580.97 | 578.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 580.80 | 582.88 | 581.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 580.80 | 582.88 | 581.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 585.95 | 583.50 | 581.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:30:00 | 582.70 | 583.50 | 581.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 615.75 | 618.26 | 611.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 622.10 | 617.56 | 613.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-20 15:15:00 | 639.54 | 630.23 | 622.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 629.10 | 646.05 | 647.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 11:15:00 | 625.25 | 641.89 | 645.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 12:15:00 | 626.20 | 625.97 | 633.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 13:00:00 | 626.20 | 625.97 | 633.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 623.00 | 624.65 | 630.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 619.30 | 624.65 | 630.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 612.50 | 609.92 | 609.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 612.50 | 609.92 | 609.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 620.00 | 612.41 | 611.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 13:15:00 | 612.80 | 613.85 | 612.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 13:15:00 | 612.80 | 613.85 | 612.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 612.80 | 613.85 | 612.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 612.80 | 613.85 | 612.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 614.80 | 614.04 | 612.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:30:00 | 612.60 | 614.04 | 612.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 612.25 | 613.82 | 612.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 606.75 | 613.82 | 612.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 608.50 | 612.76 | 612.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 608.50 | 612.76 | 612.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 606.50 | 611.50 | 611.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 603.10 | 608.22 | 609.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 609.80 | 607.48 | 608.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 609.80 | 607.48 | 608.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 609.80 | 607.48 | 608.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:45:00 | 609.00 | 607.48 | 608.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 609.10 | 607.81 | 608.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 609.95 | 607.81 | 608.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 605.80 | 607.64 | 608.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:15:00 | 605.30 | 607.64 | 608.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 575.03 | 582.39 | 586.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 579.00 | 578.43 | 582.98 | SL hit (close>ema200) qty=0.50 sl=578.43 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 610.10 | 588.27 | 586.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 616.50 | 593.92 | 588.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 610.15 | 613.06 | 604.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 610.15 | 613.06 | 604.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 610.15 | 613.06 | 604.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 615.70 | 613.06 | 604.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 622.50 | 625.47 | 621.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:45:00 | 622.25 | 625.47 | 621.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 621.15 | 624.60 | 621.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:45:00 | 619.70 | 624.60 | 621.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 625.20 | 624.72 | 621.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 631.20 | 624.72 | 621.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 632.70 | 639.39 | 640.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 632.70 | 639.39 | 640.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 630.55 | 634.88 | 637.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 609.80 | 604.71 | 611.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 609.80 | 604.71 | 611.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 609.80 | 604.71 | 611.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 611.10 | 604.71 | 611.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 611.90 | 606.15 | 611.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 611.70 | 606.15 | 611.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 616.55 | 608.23 | 611.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 616.90 | 608.23 | 611.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 616.80 | 609.94 | 612.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 616.80 | 609.94 | 612.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 615.35 | 613.99 | 613.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 621.65 | 615.52 | 614.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 613.55 | 615.82 | 614.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 14:15:00 | 613.55 | 615.82 | 614.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 613.55 | 615.82 | 614.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 613.55 | 615.82 | 614.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 615.00 | 615.65 | 614.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 621.10 | 615.65 | 614.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 15:00:00 | 618.45 | 619.49 | 617.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 607.50 | 617.60 | 617.22 | SL hit (close<static) qty=1.00 sl=613.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 609.60 | 616.00 | 616.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 604.40 | 611.30 | 613.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 610.00 | 609.24 | 611.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:45:00 | 610.05 | 609.24 | 611.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 611.30 | 609.83 | 611.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:15:00 | 612.95 | 609.83 | 611.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 612.95 | 610.46 | 611.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 619.05 | 610.46 | 611.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 614.55 | 611.27 | 612.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 613.00 | 611.27 | 612.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 617.65 | 612.75 | 612.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 617.65 | 612.75 | 612.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 619.15 | 614.86 | 613.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 616.85 | 617.62 | 615.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 10:00:00 | 616.85 | 617.62 | 615.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 620.95 | 618.29 | 615.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:30:00 | 625.25 | 619.22 | 616.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 13:15:00 | 610.80 | 616.92 | 615.95 | SL hit (close<static) qty=1.00 sl=615.05 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 611.75 | 615.49 | 615.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 10:15:00 | 608.30 | 614.05 | 614.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 13:15:00 | 613.05 | 612.50 | 613.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 13:15:00 | 613.05 | 612.50 | 613.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 613.05 | 612.50 | 613.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:30:00 | 613.65 | 612.50 | 613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 616.80 | 613.36 | 614.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 616.80 | 613.36 | 614.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 615.95 | 613.88 | 614.25 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 617.90 | 614.68 | 614.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 13:15:00 | 618.85 | 616.53 | 615.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 10:15:00 | 660.55 | 662.99 | 657.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:45:00 | 660.95 | 662.99 | 657.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 657.30 | 661.33 | 657.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 13:00:00 | 657.30 | 661.33 | 657.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 656.20 | 660.30 | 657.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:00:00 | 656.20 | 660.30 | 657.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 652.45 | 658.73 | 656.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 652.45 | 658.73 | 656.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 660.00 | 658.98 | 657.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 652.05 | 658.98 | 657.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 652.20 | 657.63 | 656.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:45:00 | 650.75 | 657.63 | 656.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 647.35 | 655.57 | 655.80 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 655.95 | 650.95 | 650.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 15:15:00 | 658.00 | 652.36 | 651.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 655.35 | 657.37 | 655.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 655.35 | 657.37 | 655.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 655.35 | 657.37 | 655.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 662.05 | 657.82 | 656.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 13:15:00 | 657.10 | 660.50 | 660.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 657.10 | 660.50 | 660.52 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 13:15:00 | 662.30 | 660.37 | 660.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 14:15:00 | 666.50 | 661.60 | 660.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 688.80 | 689.01 | 680.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 12:00:00 | 688.80 | 689.01 | 680.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 690.35 | 695.13 | 690.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 690.35 | 695.13 | 690.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 691.50 | 694.40 | 690.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:30:00 | 696.10 | 691.84 | 690.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:30:00 | 691.75 | 692.34 | 691.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:15:00 | 691.85 | 692.13 | 691.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 686.85 | 690.70 | 690.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 686.85 | 690.70 | 690.75 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 694.40 | 690.61 | 690.57 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 687.40 | 690.05 | 690.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 681.70 | 688.38 | 689.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 691.70 | 682.26 | 684.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 691.70 | 682.26 | 684.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 691.70 | 682.26 | 684.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 691.70 | 682.26 | 684.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 691.30 | 684.07 | 685.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 681.10 | 684.07 | 685.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 690.00 | 686.20 | 685.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 705.50 | 691.10 | 688.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 709.70 | 711.13 | 706.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 709.70 | 711.13 | 706.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 711.00 | 711.24 | 708.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 714.95 | 711.24 | 708.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:00:00 | 711.20 | 711.23 | 709.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 10:15:00 | 706.65 | 710.32 | 708.90 | SL hit (close<static) qty=1.00 sl=707.95 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 09:15:00 | 702.15 | 707.66 | 708.10 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 711.85 | 708.34 | 708.14 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 696.45 | 706.55 | 707.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 11:15:00 | 686.30 | 699.77 | 704.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 675.50 | 674.11 | 679.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 675.50 | 674.11 | 679.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 664.95 | 667.84 | 674.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 660.35 | 667.84 | 674.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:30:00 | 658.35 | 664.58 | 671.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 685.40 | 668.44 | 669.08 | SL hit (close>static) qty=1.00 sl=681.70 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 681.90 | 671.13 | 670.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 705.75 | 683.23 | 676.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 702.50 | 702.65 | 695.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 11:15:00 | 696.60 | 701.02 | 696.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 696.60 | 701.02 | 696.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:00:00 | 696.60 | 701.02 | 696.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 703.00 | 701.41 | 696.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:45:00 | 704.75 | 702.17 | 697.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:15:00 | 708.30 | 704.13 | 699.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:45:00 | 709.30 | 704.63 | 701.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 706.10 | 705.53 | 702.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 701.25 | 707.29 | 705.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 701.25 | 707.29 | 705.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 703.05 | 706.44 | 705.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 700.55 | 706.44 | 705.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 704.95 | 705.42 | 704.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 704.95 | 705.42 | 704.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 704.70 | 705.28 | 704.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 694.00 | 705.28 | 704.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 683.85 | 700.99 | 703.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 683.85 | 700.99 | 703.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 681.35 | 697.06 | 701.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 689.55 | 689.26 | 694.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:30:00 | 691.15 | 689.26 | 694.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 710.80 | 692.52 | 693.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 710.80 | 692.52 | 693.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 706.80 | 695.38 | 694.58 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 681.85 | 692.57 | 693.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 677.95 | 689.65 | 692.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 668.00 | 661.16 | 665.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 14:15:00 | 668.00 | 661.16 | 665.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 668.00 | 661.16 | 665.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 668.00 | 661.16 | 665.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 667.50 | 662.42 | 665.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 672.90 | 662.42 | 665.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 11:15:00 | 677.10 | 667.22 | 666.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 12:15:00 | 679.80 | 669.74 | 668.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 11:15:00 | 676.95 | 680.42 | 675.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 11:15:00 | 676.95 | 680.42 | 675.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 676.95 | 680.42 | 675.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 673.55 | 680.42 | 675.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 677.30 | 679.79 | 675.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:45:00 | 677.25 | 679.79 | 675.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 675.90 | 679.02 | 675.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 675.90 | 679.02 | 675.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 671.45 | 677.50 | 675.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 670.90 | 677.50 | 675.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 670.65 | 676.13 | 674.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 682.30 | 676.13 | 674.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 682.15 | 678.15 | 676.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:30:00 | 681.00 | 678.15 | 676.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 677.00 | 678.10 | 676.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:45:00 | 674.45 | 678.10 | 676.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 677.65 | 678.01 | 676.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:15:00 | 678.70 | 678.01 | 676.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 678.70 | 678.15 | 676.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 673.65 | 678.15 | 676.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 683.05 | 679.13 | 677.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:30:00 | 675.30 | 679.13 | 677.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 681.35 | 683.96 | 680.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 681.35 | 683.96 | 680.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 686.60 | 684.49 | 681.24 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 673.00 | 679.39 | 679.77 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 687.45 | 680.87 | 680.10 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 674.50 | 678.71 | 679.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 669.70 | 676.91 | 678.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 667.35 | 663.55 | 669.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 667.35 | 663.55 | 669.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 668.90 | 664.62 | 669.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 668.90 | 664.62 | 669.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 668.00 | 665.30 | 669.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 676.00 | 665.30 | 669.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 677.15 | 667.67 | 669.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 677.20 | 667.67 | 669.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 673.60 | 668.86 | 670.16 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 681.90 | 671.46 | 671.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 15:15:00 | 687.00 | 678.88 | 675.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 13:15:00 | 728.60 | 729.79 | 722.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 13:45:00 | 727.10 | 729.79 | 722.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 719.35 | 727.58 | 722.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 719.35 | 727.58 | 722.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 719.95 | 726.06 | 722.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 11:00:00 | 719.95 | 726.06 | 722.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 717.95 | 724.44 | 722.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:00:00 | 717.95 | 724.44 | 722.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 711.65 | 719.56 | 720.36 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 733.30 | 721.78 | 721.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 12:15:00 | 734.15 | 727.08 | 723.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 11:15:00 | 736.05 | 736.12 | 730.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 11:45:00 | 736.00 | 736.12 | 730.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 788.30 | 794.65 | 787.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 788.30 | 794.65 | 787.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 793.90 | 794.50 | 788.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 11:15:00 | 795.65 | 794.50 | 788.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 778.35 | 788.75 | 790.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 778.35 | 788.75 | 790.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 775.95 | 786.19 | 788.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 790.60 | 784.06 | 786.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 790.60 | 784.06 | 786.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 790.60 | 784.06 | 786.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 789.55 | 784.06 | 786.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 792.45 | 785.74 | 786.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 792.50 | 785.74 | 786.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 798.40 | 789.57 | 788.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 802.65 | 795.13 | 791.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 12:15:00 | 806.40 | 807.05 | 802.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 13:00:00 | 806.40 | 807.05 | 802.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 838.10 | 839.00 | 835.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 833.85 | 839.00 | 835.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 835.50 | 838.30 | 835.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 839.90 | 838.30 | 835.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 838.50 | 838.34 | 835.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 844.00 | 838.62 | 836.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 862.50 | 872.97 | 873.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 862.50 | 872.97 | 873.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 858.50 | 870.08 | 872.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 11:15:00 | 862.55 | 861.68 | 866.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:30:00 | 863.90 | 861.68 | 866.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 862.50 | 859.77 | 863.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:45:00 | 860.85 | 859.77 | 863.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 869.50 | 861.71 | 863.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 869.50 | 861.71 | 863.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 871.45 | 863.66 | 864.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:30:00 | 869.35 | 863.66 | 864.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 861.80 | 863.79 | 864.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 858.85 | 863.43 | 864.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 861.00 | 860.77 | 862.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 870.35 | 863.63 | 863.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 870.35 | 863.63 | 863.47 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 861.05 | 863.22 | 863.43 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 866.40 | 863.51 | 863.51 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 860.45 | 862.90 | 863.23 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 871.70 | 864.94 | 864.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 892.50 | 871.28 | 867.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 862.35 | 871.05 | 868.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 862.35 | 871.05 | 868.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 862.35 | 871.05 | 868.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 862.35 | 871.05 | 868.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 861.90 | 869.22 | 867.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 859.40 | 869.22 | 867.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 880.05 | 870.31 | 868.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 880.85 | 871.97 | 869.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:30:00 | 880.50 | 874.11 | 872.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:00:00 | 880.70 | 875.53 | 873.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 13:15:00 | 869.00 | 873.18 | 873.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 869.00 | 873.18 | 873.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 865.70 | 871.12 | 872.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 851.70 | 851.43 | 858.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:00:00 | 851.70 | 851.43 | 858.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 841.75 | 849.91 | 855.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 835.75 | 849.91 | 855.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 832.85 | 844.20 | 851.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 793.96 | 803.67 | 816.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 791.21 | 803.67 | 816.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-14 09:15:00 | 752.18 | 770.74 | 791.12 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 814.50 | 792.16 | 790.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 818.40 | 811.81 | 805.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 807.65 | 810.98 | 805.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 807.65 | 810.98 | 805.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 806.95 | 810.17 | 805.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:15:00 | 805.55 | 810.17 | 805.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 807.15 | 809.57 | 806.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 813.05 | 808.61 | 805.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 784.65 | 805.43 | 805.27 | SL hit (close<static) qty=1.00 sl=804.55 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 783.45 | 801.03 | 803.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 777.30 | 790.13 | 796.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 773.40 | 768.13 | 775.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 773.40 | 768.13 | 775.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 775.35 | 770.22 | 775.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 775.40 | 770.22 | 775.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 765.10 | 769.20 | 774.37 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 783.15 | 775.48 | 775.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 786.50 | 777.68 | 776.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 780.55 | 780.58 | 778.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 15:00:00 | 780.55 | 780.58 | 778.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 779.00 | 780.26 | 778.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 774.40 | 780.26 | 778.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 762.35 | 776.68 | 776.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 15:15:00 | 757.90 | 765.24 | 770.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 768.20 | 759.09 | 763.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 768.20 | 759.09 | 763.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 768.20 | 759.09 | 763.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 768.20 | 759.09 | 763.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 773.25 | 761.92 | 763.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 773.25 | 761.92 | 763.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 773.75 | 765.76 | 765.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 779.05 | 769.71 | 767.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 769.05 | 772.31 | 769.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 11:15:00 | 769.05 | 772.31 | 769.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 769.05 | 772.31 | 769.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 769.05 | 772.31 | 769.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 764.85 | 770.82 | 769.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 764.85 | 770.82 | 769.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 762.50 | 769.16 | 768.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 762.40 | 769.16 | 768.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 762.25 | 767.78 | 768.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 11:15:00 | 756.30 | 764.24 | 765.88 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 778.00 | 766.99 | 766.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 802.80 | 777.84 | 772.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 817.45 | 824.80 | 815.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 13:15:00 | 817.45 | 824.80 | 815.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 817.45 | 824.80 | 815.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:45:00 | 819.65 | 824.80 | 815.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 816.55 | 823.15 | 815.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 816.55 | 823.15 | 815.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 815.80 | 821.68 | 815.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 804.45 | 821.68 | 815.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 800.50 | 817.45 | 814.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 800.50 | 817.45 | 814.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 801.60 | 814.28 | 813.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 802.20 | 814.28 | 813.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 796.50 | 810.72 | 811.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 794.05 | 807.39 | 810.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 797.70 | 797.62 | 803.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 797.70 | 797.62 | 803.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 794.60 | 797.01 | 802.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 800.95 | 797.01 | 802.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 782.45 | 791.77 | 797.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:15:00 | 780.95 | 789.74 | 796.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 780.30 | 784.17 | 791.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 741.90 | 750.60 | 766.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 741.28 | 750.60 | 766.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 746.30 | 739.94 | 752.05 | SL hit (close>ema200) qty=0.50 sl=739.94 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 739.90 | 720.19 | 719.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 746.65 | 729.59 | 724.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 755.95 | 759.69 | 751.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:00:00 | 755.95 | 759.69 | 751.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 745.60 | 756.04 | 751.67 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 735.45 | 749.00 | 749.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 734.15 | 746.03 | 747.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 727.90 | 727.75 | 733.70 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:00:00 | 720.85 | 726.37 | 732.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:45:00 | 719.60 | 724.85 | 731.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 13:45:00 | 721.10 | 723.36 | 729.44 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 15:15:00 | 720.95 | 723.11 | 728.78 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 710.35 | 715.36 | 721.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:30:00 | 714.80 | 715.36 | 721.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 719.55 | 716.27 | 720.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 720.70 | 716.27 | 720.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 718.60 | 716.74 | 720.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 718.75 | 716.74 | 720.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 724.80 | 718.03 | 720.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 724.80 | 718.03 | 720.35 | SL hit (close>ema400) qty=1.00 sl=720.35 alert=retest1 |

### Cycle 59 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 726.15 | 722.44 | 722.07 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 14:15:00 | 716.20 | 721.19 | 721.80 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 738.15 | 724.15 | 723.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 744.00 | 728.12 | 724.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 747.40 | 748.73 | 740.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 11:30:00 | 747.80 | 748.73 | 740.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 743.25 | 748.56 | 744.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 743.25 | 748.56 | 744.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 746.75 | 748.20 | 744.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:30:00 | 747.80 | 748.14 | 744.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:00:00 | 747.90 | 748.14 | 744.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 748.55 | 746.86 | 744.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 739.00 | 744.96 | 745.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 739.00 | 744.96 | 745.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 737.70 | 743.51 | 744.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 748.55 | 744.43 | 744.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 11:15:00 | 748.55 | 744.43 | 744.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 748.55 | 744.43 | 744.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 748.55 | 744.43 | 744.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 742.95 | 744.14 | 744.61 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 749.20 | 745.71 | 745.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 09:15:00 | 751.15 | 747.65 | 746.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 747.65 | 748.31 | 746.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 747.65 | 748.31 | 746.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 747.65 | 748.31 | 746.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 746.00 | 748.31 | 746.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 746.35 | 747.92 | 746.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 745.15 | 747.92 | 746.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 747.05 | 747.75 | 746.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:45:00 | 745.90 | 747.75 | 746.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 749.50 | 748.10 | 747.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:45:00 | 746.30 | 748.10 | 747.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 750.90 | 754.50 | 751.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 750.90 | 754.50 | 751.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 750.00 | 753.60 | 751.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 744.30 | 753.60 | 751.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 751.95 | 751.28 | 750.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 756.55 | 751.92 | 751.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 770.55 | 751.81 | 751.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 09:15:00 | 832.21 | 825.87 | 815.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 818.40 | 826.05 | 826.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 812.45 | 821.97 | 824.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 12:15:00 | 795.55 | 792.57 | 798.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-01 13:00:00 | 795.55 | 792.57 | 798.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 798.20 | 793.70 | 798.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 798.20 | 793.70 | 798.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 804.85 | 795.93 | 799.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 804.85 | 795.93 | 799.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 807.70 | 798.28 | 799.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 817.40 | 798.28 | 799.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 820.00 | 802.63 | 801.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 826.70 | 807.44 | 804.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 823.45 | 823.57 | 815.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 11:00:00 | 823.45 | 823.57 | 815.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 813.60 | 825.44 | 820.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 813.60 | 825.44 | 820.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 814.05 | 823.16 | 820.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 810.15 | 823.16 | 820.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 806.45 | 817.32 | 817.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 799.65 | 813.79 | 816.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 766.70 | 762.11 | 780.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:30:00 | 773.50 | 762.11 | 780.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 771.50 | 768.05 | 777.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 762.40 | 769.74 | 776.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:30:00 | 767.70 | 768.28 | 772.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 790.65 | 772.86 | 773.74 | SL hit (close>static) qty=1.00 sl=778.45 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 791.70 | 776.63 | 775.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 820.00 | 792.14 | 784.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 839.40 | 840.10 | 829.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 13:30:00 | 841.40 | 840.10 | 829.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 836.50 | 839.64 | 832.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:45:00 | 850.30 | 834.19 | 833.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 11:15:00 | 828.10 | 832.72 | 832.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 828.10 | 832.72 | 832.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 12:15:00 | 826.80 | 831.54 | 832.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 14:15:00 | 819.80 | 817.41 | 822.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 15:00:00 | 819.80 | 817.41 | 822.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 793.00 | 813.08 | 819.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 10:15:00 | 783.15 | 813.08 | 819.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:00:00 | 787.85 | 796.67 | 808.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 15:00:00 | 789.30 | 797.97 | 801.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 786.55 | 796.38 | 800.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 805.75 | 791.74 | 794.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 805.75 | 791.74 | 794.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 802.70 | 793.93 | 795.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:45:00 | 804.90 | 793.93 | 795.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 805.70 | 796.29 | 796.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 805.70 | 796.29 | 796.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 814.35 | 801.82 | 798.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 14:15:00 | 799.35 | 802.55 | 800.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 14:15:00 | 799.35 | 802.55 | 800.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 799.35 | 802.55 | 800.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 799.35 | 802.55 | 800.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 798.80 | 801.80 | 800.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 779.65 | 801.80 | 800.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 775.20 | 796.48 | 798.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 10:15:00 | 768.40 | 790.86 | 795.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 763.45 | 760.04 | 773.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:45:00 | 761.25 | 760.04 | 773.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 767.00 | 763.96 | 770.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 756.35 | 763.54 | 768.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 718.53 | 749.00 | 760.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 756.95 | 730.84 | 742.01 | SL hit (close>ema200) qty=0.50 sl=730.84 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 766.65 | 748.07 | 747.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 771.25 | 752.71 | 750.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 761.80 | 762.30 | 757.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 761.80 | 762.30 | 757.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 767.15 | 767.94 | 764.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 765.30 | 767.94 | 764.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 762.95 | 766.32 | 764.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 761.80 | 766.32 | 764.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 767.80 | 766.61 | 764.66 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 10:15:00 | 760.50 | 763.86 | 763.87 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 764.90 | 764.01 | 763.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 776.55 | 767.07 | 765.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 771.55 | 771.62 | 768.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 771.55 | 771.62 | 768.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 767.00 | 771.03 | 769.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 767.00 | 771.03 | 769.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 764.25 | 769.67 | 768.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 763.20 | 769.67 | 768.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 755.45 | 765.76 | 767.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 752.85 | 763.18 | 765.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 765.45 | 760.68 | 763.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 12:15:00 | 765.45 | 760.68 | 763.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 765.45 | 760.68 | 763.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:00:00 | 765.45 | 760.68 | 763.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 768.00 | 762.14 | 763.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 768.00 | 762.14 | 763.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 771.05 | 765.66 | 765.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 774.65 | 768.89 | 766.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 13:15:00 | 769.15 | 769.37 | 767.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:00:00 | 769.15 | 769.37 | 767.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 770.35 | 769.57 | 767.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 768.90 | 769.57 | 767.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 770.80 | 769.88 | 768.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:15:00 | 768.00 | 769.88 | 768.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 770.05 | 769.92 | 768.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 771.45 | 769.92 | 768.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 772.40 | 770.41 | 768.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 770.70 | 770.41 | 768.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 769.95 | 770.31 | 769.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:45:00 | 768.75 | 770.31 | 769.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 774.75 | 771.07 | 769.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 778.70 | 772.83 | 770.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 765.25 | 772.46 | 771.62 | SL hit (close<static) qty=1.00 sl=767.15 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 768.65 | 770.76 | 770.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 767.15 | 770.05 | 770.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 762.40 | 762.01 | 764.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 15:00:00 | 762.40 | 762.01 | 764.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 766.40 | 762.94 | 764.44 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 772.65 | 766.00 | 765.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 776.55 | 769.65 | 767.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 774.30 | 776.66 | 773.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 774.30 | 776.66 | 773.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 774.30 | 776.66 | 773.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 774.30 | 776.66 | 773.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 771.00 | 775.52 | 773.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 770.60 | 775.52 | 773.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 760.90 | 772.60 | 772.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 760.90 | 772.60 | 772.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 762.35 | 770.55 | 771.20 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 775.85 | 770.55 | 770.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 777.00 | 772.71 | 771.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 772.70 | 772.71 | 771.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 13:45:00 | 773.75 | 772.71 | 771.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 772.90 | 772.75 | 771.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 773.50 | 772.80 | 771.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 775.35 | 772.74 | 771.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 774.40 | 772.65 | 771.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 774.25 | 774.83 | 773.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 773.40 | 774.54 | 773.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:00:00 | 778.45 | 775.33 | 773.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:15:00 | 775.80 | 775.07 | 773.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:45:00 | 775.80 | 779.44 | 778.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 765.05 | 776.56 | 777.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 765.05 | 776.56 | 777.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 754.35 | 767.24 | 772.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 740.40 | 739.42 | 748.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:15:00 | 747.00 | 739.42 | 748.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 744.45 | 740.43 | 748.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 743.70 | 740.43 | 748.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 748.70 | 742.08 | 748.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 748.70 | 742.08 | 748.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 752.70 | 744.21 | 748.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 752.70 | 744.21 | 748.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 755.35 | 746.44 | 749.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 755.35 | 746.44 | 749.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 760.90 | 751.95 | 751.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 766.75 | 760.83 | 757.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 757.85 | 761.39 | 758.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 757.85 | 761.39 | 758.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 757.85 | 761.39 | 758.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:15:00 | 756.50 | 761.39 | 758.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 755.35 | 760.19 | 758.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 753.75 | 760.19 | 758.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 747.20 | 756.35 | 757.08 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 764.05 | 756.96 | 756.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 766.55 | 758.87 | 757.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 760.75 | 761.43 | 759.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 760.75 | 761.43 | 759.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 760.75 | 761.43 | 759.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 758.95 | 761.43 | 759.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 759.60 | 761.07 | 759.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 768.05 | 761.07 | 759.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 14:15:00 | 766.60 | 773.07 | 773.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 766.60 | 773.07 | 773.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 762.30 | 770.51 | 772.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 758.50 | 758.27 | 762.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 13:30:00 | 758.00 | 758.27 | 762.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 760.60 | 759.24 | 762.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 763.55 | 759.24 | 762.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 759.15 | 759.22 | 761.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 755.60 | 758.71 | 761.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:00:00 | 755.10 | 757.78 | 760.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 751.05 | 744.94 | 744.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 751.05 | 744.94 | 744.38 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 741.05 | 745.05 | 745.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 737.90 | 742.66 | 743.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 738.65 | 737.75 | 740.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 738.65 | 737.75 | 740.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 738.65 | 737.75 | 740.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 738.85 | 737.75 | 740.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 735.80 | 737.36 | 739.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 734.80 | 737.36 | 739.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 740.25 | 735.66 | 736.67 | SL hit (close>static) qty=1.00 sl=739.90 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 743.70 | 738.02 | 737.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 745.00 | 739.42 | 738.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 738.05 | 739.15 | 738.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 738.05 | 739.15 | 738.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 738.05 | 739.15 | 738.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 738.05 | 739.15 | 738.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 737.60 | 738.84 | 738.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 736.65 | 738.84 | 738.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 743.35 | 739.74 | 738.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 745.60 | 739.74 | 738.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 756.60 | 761.78 | 762.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 756.60 | 761.78 | 762.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 752.60 | 759.95 | 761.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 754.50 | 753.85 | 756.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 755.80 | 753.85 | 756.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 752.30 | 753.54 | 756.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 748.40 | 752.60 | 754.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 751.30 | 747.69 | 749.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 752.65 | 748.62 | 748.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 752.65 | 748.62 | 748.40 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 747.00 | 748.24 | 748.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 14:15:00 | 744.70 | 747.53 | 747.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 742.80 | 742.36 | 744.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 742.80 | 742.36 | 744.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 742.80 | 742.36 | 744.23 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 749.25 | 745.41 | 745.29 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 744.35 | 745.20 | 745.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 740.35 | 744.23 | 744.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 747.25 | 742.48 | 743.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 747.25 | 742.48 | 743.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 747.25 | 742.48 | 743.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 747.25 | 742.48 | 743.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 752.55 | 744.50 | 744.24 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 743.00 | 745.93 | 746.08 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 13:15:00 | 746.90 | 746.23 | 746.20 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 741.50 | 745.39 | 745.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 740.90 | 744.49 | 745.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 741.70 | 741.56 | 743.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 741.70 | 741.56 | 743.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 743.80 | 742.01 | 743.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 743.50 | 742.01 | 743.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 740.30 | 741.67 | 743.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 735.90 | 740.60 | 742.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 746.65 | 742.17 | 741.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 746.65 | 742.17 | 741.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 752.00 | 745.12 | 743.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 747.30 | 748.26 | 745.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 747.30 | 748.26 | 745.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 747.95 | 748.20 | 746.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 752.25 | 748.20 | 746.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 787.85 | 793.71 | 794.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 787.85 | 793.71 | 794.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 784.90 | 791.31 | 792.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 774.05 | 771.40 | 776.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 774.05 | 771.40 | 776.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 770.90 | 771.30 | 775.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 768.15 | 771.30 | 775.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 772.80 | 765.91 | 765.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 772.80 | 765.91 | 765.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 775.00 | 768.79 | 767.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 774.35 | 776.35 | 772.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:00:00 | 774.35 | 776.35 | 772.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 776.60 | 776.40 | 773.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 777.00 | 775.97 | 773.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 777.45 | 775.97 | 773.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 770.35 | 774.72 | 773.83 | SL hit (close<static) qty=1.00 sl=773.05 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 771.75 | 775.12 | 775.31 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 779.40 | 775.76 | 775.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 780.15 | 776.63 | 775.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 776.70 | 776.92 | 776.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 776.70 | 776.92 | 776.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 774.05 | 776.35 | 775.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 774.05 | 776.35 | 775.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 775.35 | 776.15 | 775.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 774.30 | 776.15 | 775.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 772.55 | 775.41 | 775.57 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 779.95 | 776.17 | 775.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 780.35 | 777.01 | 776.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 777.65 | 778.24 | 777.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 777.65 | 778.24 | 777.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 777.65 | 778.24 | 777.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 777.05 | 778.24 | 777.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 778.20 | 778.23 | 777.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 784.40 | 778.23 | 777.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 776.60 | 781.64 | 781.33 | SL hit (close<static) qty=1.00 sl=777.10 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 780.00 | 780.92 | 781.03 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 785.15 | 781.77 | 781.41 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 774.55 | 781.14 | 781.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 767.25 | 777.17 | 779.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 778.10 | 776.54 | 779.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 778.10 | 776.54 | 779.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 774.60 | 776.15 | 778.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 778.75 | 776.15 | 778.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 773.00 | 774.60 | 776.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 770.10 | 773.48 | 775.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 731.60 | 740.75 | 748.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 721.00 | 717.89 | 726.37 | SL hit (close>ema200) qty=0.50 sl=717.89 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 724.40 | 722.97 | 722.83 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 720.70 | 722.53 | 722.70 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 723.40 | 722.79 | 722.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 728.65 | 723.96 | 723.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 729.50 | 729.97 | 727.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 729.50 | 729.97 | 727.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 728.25 | 730.01 | 728.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 730.75 | 730.01 | 728.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 730.90 | 730.34 | 728.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:00:00 | 731.25 | 730.34 | 728.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 725.25 | 731.30 | 731.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 725.25 | 731.30 | 731.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 722.60 | 729.56 | 731.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 725.05 | 723.47 | 726.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 725.05 | 723.47 | 726.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 725.50 | 723.88 | 726.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:45:00 | 723.50 | 723.38 | 725.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 727.40 | 723.70 | 725.20 | SL hit (close>static) qty=1.00 sl=726.80 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 734.30 | 726.95 | 726.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 739.00 | 734.62 | 731.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 735.50 | 736.82 | 733.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 735.50 | 736.82 | 733.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 739.50 | 742.11 | 739.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 739.50 | 742.11 | 739.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 741.60 | 742.01 | 739.92 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 737.20 | 738.74 | 738.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 733.55 | 737.60 | 738.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 742.30 | 737.70 | 738.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 10:15:00 | 742.30 | 737.70 | 738.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 742.30 | 737.70 | 738.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 742.30 | 737.70 | 738.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 743.90 | 738.94 | 738.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 745.50 | 741.07 | 739.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 739.85 | 742.79 | 741.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 739.85 | 742.79 | 741.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 739.85 | 742.79 | 741.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 739.85 | 742.79 | 741.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 740.55 | 742.34 | 741.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 744.80 | 742.43 | 741.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 15:15:00 | 745.00 | 746.04 | 745.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 741.05 | 744.87 | 745.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 741.05 | 744.87 | 745.32 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 747.05 | 745.56 | 745.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 747.90 | 746.03 | 745.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 745.10 | 746.16 | 745.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 745.10 | 746.16 | 745.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 745.10 | 746.16 | 745.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 745.40 | 746.16 | 745.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 744.00 | 745.73 | 745.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 744.40 | 745.73 | 745.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 744.55 | 745.49 | 745.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 740.75 | 744.54 | 745.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 698.45 | 694.49 | 706.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:45:00 | 698.00 | 694.49 | 706.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 705.85 | 699.89 | 705.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 705.85 | 699.89 | 705.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 703.25 | 700.56 | 705.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:30:00 | 706.55 | 700.56 | 705.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 702.15 | 699.34 | 701.67 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 706.20 | 703.05 | 702.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 714.00 | 706.57 | 704.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 713.35 | 714.73 | 710.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:45:00 | 717.70 | 715.92 | 711.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 716.05 | 720.13 | 717.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 716.05 | 720.13 | 717.97 | SL hit (close<ema400) qty=1.00 sl=717.97 alert=retest1 |

### Cycle 118 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 712.55 | 716.40 | 716.81 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 722.65 | 717.88 | 717.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 728.25 | 720.46 | 718.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 732.30 | 732.53 | 728.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 15:00:00 | 732.30 | 732.53 | 728.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 726.00 | 731.45 | 728.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 726.00 | 731.45 | 728.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 721.00 | 729.36 | 728.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 721.00 | 729.36 | 728.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 719.65 | 727.42 | 727.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 718.00 | 722.93 | 725.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 724.80 | 723.21 | 724.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 10:15:00 | 724.80 | 723.21 | 724.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 724.80 | 723.21 | 724.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:15:00 | 725.25 | 723.21 | 724.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 724.65 | 723.50 | 724.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:30:00 | 724.25 | 723.50 | 724.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 724.90 | 723.78 | 724.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 724.95 | 723.78 | 724.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 726.35 | 724.29 | 724.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 726.35 | 724.29 | 724.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 726.45 | 724.73 | 725.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 727.10 | 724.73 | 725.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 726.00 | 724.98 | 725.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 727.25 | 724.98 | 725.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 731.00 | 726.18 | 725.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 732.50 | 727.45 | 726.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 729.10 | 730.54 | 728.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 729.10 | 730.54 | 728.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 729.10 | 730.54 | 728.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 730.55 | 730.54 | 728.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 736.55 | 731.75 | 729.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 738.00 | 731.75 | 729.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:45:00 | 738.30 | 733.62 | 730.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 737.15 | 734.32 | 731.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 739.50 | 734.44 | 731.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 743.05 | 746.59 | 743.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:45:00 | 742.50 | 746.59 | 743.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 742.55 | 745.78 | 743.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 744.30 | 745.78 | 743.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 742.80 | 745.19 | 743.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 740.95 | 745.19 | 743.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 742.90 | 744.73 | 743.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 741.65 | 744.73 | 743.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 742.50 | 744.28 | 743.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 741.65 | 744.28 | 743.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 738.40 | 742.23 | 742.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 738.40 | 742.23 | 742.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 733.65 | 740.51 | 741.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 738.70 | 733.29 | 735.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 738.70 | 733.29 | 735.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 738.70 | 733.29 | 735.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 738.20 | 733.29 | 735.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 733.50 | 733.33 | 735.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 732.65 | 733.33 | 735.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 732.80 | 731.99 | 733.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 730.25 | 725.90 | 725.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 730.25 | 725.90 | 725.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 738.00 | 728.88 | 727.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 733.30 | 733.42 | 730.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 731.75 | 733.42 | 730.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 733.35 | 733.40 | 730.81 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 728.05 | 730.36 | 730.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 725.80 | 729.45 | 730.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 720.80 | 717.82 | 721.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 720.80 | 717.82 | 721.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 720.80 | 717.82 | 721.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 722.00 | 717.82 | 721.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 723.00 | 719.16 | 721.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 723.00 | 719.16 | 721.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 723.00 | 719.93 | 721.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 730.60 | 719.93 | 721.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 727.35 | 721.42 | 721.91 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 727.70 | 723.13 | 722.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 731.90 | 725.82 | 724.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 736.70 | 736.86 | 732.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:30:00 | 735.40 | 736.86 | 732.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 735.10 | 737.15 | 734.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 735.10 | 737.15 | 734.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 735.15 | 736.75 | 734.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 735.15 | 736.75 | 734.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 736.30 | 736.66 | 734.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 738.05 | 736.87 | 735.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 731.00 | 738.53 | 738.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 731.00 | 738.53 | 738.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 727.40 | 734.86 | 736.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 732.00 | 731.78 | 734.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 732.00 | 731.78 | 734.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 732.00 | 731.78 | 734.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 736.80 | 731.78 | 734.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 734.45 | 732.31 | 734.46 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 738.90 | 735.66 | 735.53 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 733.90 | 735.32 | 735.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 732.85 | 734.66 | 735.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 737.35 | 735.20 | 735.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 737.35 | 735.20 | 735.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 737.35 | 735.20 | 735.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 737.35 | 735.20 | 735.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 739.80 | 736.12 | 735.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 741.30 | 737.16 | 736.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 739.85 | 743.31 | 740.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 739.85 | 743.31 | 740.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 739.85 | 743.31 | 740.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 739.20 | 743.31 | 740.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 742.05 | 743.06 | 740.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:00:00 | 742.80 | 743.01 | 740.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 743.80 | 743.06 | 741.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 736.00 | 741.66 | 741.21 | SL hit (close<static) qty=1.00 sl=739.40 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 733.05 | 739.93 | 740.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 12:15:00 | 731.45 | 738.24 | 739.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 690.40 | 690.23 | 697.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 691.50 | 690.23 | 697.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 685.85 | 683.47 | 687.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 685.85 | 683.47 | 687.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 688.55 | 684.48 | 687.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 688.55 | 684.48 | 687.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 690.40 | 685.67 | 688.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 690.40 | 685.67 | 688.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 688.65 | 686.26 | 688.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 692.00 | 686.26 | 688.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 693.20 | 687.93 | 688.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 692.75 | 687.93 | 688.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 688.30 | 688.00 | 688.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 691.00 | 688.00 | 688.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 689.70 | 688.34 | 688.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 686.00 | 688.34 | 688.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 651.70 | 659.90 | 669.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 659.20 | 652.31 | 660.50 | SL hit (close>ema200) qty=0.50 sl=652.31 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 654.05 | 651.03 | 650.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 656.60 | 652.33 | 651.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 653.05 | 653.14 | 652.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 653.05 | 653.14 | 652.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 653.05 | 653.14 | 652.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 651.20 | 653.14 | 652.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 656.95 | 654.20 | 652.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:30:00 | 652.70 | 654.20 | 652.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 669.95 | 672.19 | 667.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 667.65 | 672.19 | 667.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 670.70 | 671.89 | 667.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 670.00 | 671.89 | 667.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 664.70 | 670.45 | 667.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 664.70 | 670.45 | 667.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 657.70 | 667.90 | 666.42 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 649.90 | 663.14 | 664.44 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 683.15 | 665.44 | 664.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 694.60 | 686.68 | 684.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 699.55 | 700.06 | 695.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 699.55 | 700.06 | 695.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 702.20 | 703.83 | 699.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 708.85 | 703.83 | 699.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 692.80 | 704.59 | 702.98 | SL hit (close<static) qty=1.00 sl=696.20 alert=retest2 |

### Cycle 134 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 697.95 | 701.68 | 701.85 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 703.60 | 702.07 | 702.01 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 700.25 | 701.76 | 701.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 693.25 | 699.52 | 700.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 694.80 | 692.86 | 695.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 694.80 | 692.86 | 695.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 694.80 | 692.86 | 695.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:15:00 | 689.45 | 693.10 | 695.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:00:00 | 690.20 | 689.59 | 691.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 689.90 | 691.31 | 692.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 679.40 | 676.70 | 676.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 679.40 | 676.70 | 676.41 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 671.90 | 675.74 | 676.00 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 680.00 | 676.75 | 676.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 681.50 | 678.19 | 677.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 672.60 | 677.07 | 676.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 672.60 | 677.07 | 676.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 672.60 | 677.07 | 676.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 672.60 | 677.07 | 676.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 674.25 | 676.51 | 676.51 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 11:15:00 | 677.05 | 676.61 | 676.56 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 675.15 | 676.32 | 676.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 13:15:00 | 674.50 | 675.96 | 676.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 627.60 | 627.12 | 638.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 627.60 | 627.12 | 638.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 620.10 | 613.52 | 619.57 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 629.20 | 623.55 | 622.88 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 613.25 | 622.23 | 623.08 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 627.85 | 623.46 | 623.40 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 612.10 | 621.78 | 622.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 608.40 | 616.61 | 619.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 613.90 | 608.79 | 612.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 613.90 | 608.79 | 612.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 613.90 | 608.79 | 612.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 615.45 | 608.79 | 612.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 613.00 | 609.63 | 612.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 611.00 | 609.63 | 612.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 616.35 | 610.98 | 613.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 615.30 | 610.98 | 613.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 612.15 | 611.21 | 612.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 610.15 | 611.21 | 612.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 619.15 | 613.00 | 613.35 | SL hit (close>static) qty=1.00 sl=618.75 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 622.15 | 614.83 | 614.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 624.80 | 616.83 | 615.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 621.30 | 630.79 | 625.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 621.30 | 630.79 | 625.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 621.30 | 630.79 | 625.61 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 612.35 | 621.37 | 622.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 595.00 | 614.50 | 618.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 595.05 | 592.40 | 600.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 595.05 | 592.40 | 600.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 604.40 | 594.80 | 601.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 605.45 | 594.80 | 601.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 605.00 | 596.84 | 601.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 605.65 | 596.84 | 601.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 605.00 | 599.72 | 602.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 618.75 | 599.72 | 602.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 621.60 | 604.10 | 603.94 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 597.75 | 606.46 | 607.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 590.50 | 603.27 | 605.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 593.95 | 583.40 | 591.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 593.95 | 583.40 | 591.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 593.95 | 583.40 | 591.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 584.05 | 583.11 | 590.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 583.65 | 584.57 | 588.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 569.05 | 585.14 | 588.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 591.10 | 586.04 | 585.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 591.10 | 586.04 | 585.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 595.15 | 588.94 | 587.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 590.10 | 591.12 | 588.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 590.10 | 591.12 | 588.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 590.10 | 591.12 | 588.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 592.35 | 591.12 | 588.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 651.59 | 638.99 | 634.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 644.70 | 657.40 | 658.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 639.85 | 653.89 | 656.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 637.05 | 636.90 | 642.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 639.30 | 636.90 | 642.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 642.50 | 638.02 | 642.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 642.50 | 638.02 | 642.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 649.95 | 640.41 | 643.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 649.95 | 640.41 | 643.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 648.50 | 642.03 | 643.94 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 647.55 | 645.55 | 645.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 654.00 | 649.31 | 647.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 12:15:00 | 650.05 | 650.51 | 648.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 650.05 | 650.51 | 648.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 645.60 | 649.53 | 648.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 645.60 | 649.53 | 648.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 645.10 | 648.64 | 648.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 643.20 | 648.64 | 648.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 627.80 | 644.13 | 646.13 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 645.15 | 641.99 | 641.96 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 640.65 | 641.80 | 641.89 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 644.60 | 642.36 | 642.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 646.45 | 643.18 | 642.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 666.10 | 667.47 | 661.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 667.15 | 667.47 | 661.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 568.90 | 2024-05-23 12:15:00 | 565.70 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-06-10 14:45:00 | 581.40 | 2024-06-20 15:15:00 | 639.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 09:15:00 | 580.55 | 2024-06-20 15:15:00 | 638.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 09:45:00 | 581.35 | 2024-06-20 15:15:00 | 639.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 10:30:00 | 581.05 | 2024-06-20 15:15:00 | 639.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-20 09:15:00 | 622.10 | 2024-06-27 10:15:00 | 629.10 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2024-07-01 09:15:00 | 619.30 | 2024-07-04 14:15:00 | 612.50 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2024-07-11 11:15:00 | 605.30 | 2024-07-19 09:15:00 | 575.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 11:15:00 | 605.30 | 2024-07-19 13:15:00 | 579.00 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2024-07-26 09:15:00 | 631.20 | 2024-08-01 11:15:00 | 632.70 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-08-09 09:15:00 | 621.10 | 2024-08-12 09:15:00 | 607.50 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-08-09 15:00:00 | 618.45 | 2024-08-12 09:15:00 | 607.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-08-16 10:15:00 | 613.00 | 2024-08-16 11:15:00 | 617.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-08-19 11:30:00 | 625.25 | 2024-08-19 13:15:00 | 610.80 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-09-04 15:00:00 | 662.05 | 2024-09-06 13:15:00 | 657.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-09-16 09:30:00 | 696.10 | 2024-09-17 09:15:00 | 686.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-16 12:30:00 | 691.75 | 2024-09-17 09:15:00 | 686.85 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-16 14:15:00 | 691.85 | 2024-09-17 09:15:00 | 686.85 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-26 09:15:00 | 714.95 | 2024-09-26 10:15:00 | 706.65 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-09-26 10:00:00 | 711.20 | 2024-09-26 10:15:00 | 706.65 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-09-26 11:30:00 | 713.05 | 2024-09-26 12:15:00 | 707.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-10-07 10:15:00 | 660.35 | 2024-10-08 11:15:00 | 685.40 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2024-10-07 11:30:00 | 658.35 | 2024-10-08 11:15:00 | 685.40 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2024-10-11 13:45:00 | 704.75 | 2024-10-17 09:15:00 | 683.85 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-10-14 10:15:00 | 708.30 | 2024-10-17 09:15:00 | 683.85 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-10-14 13:45:00 | 709.30 | 2024-10-17 09:15:00 | 683.85 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2024-10-15 10:30:00 | 706.10 | 2024-10-17 09:15:00 | 683.85 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2024-11-26 11:15:00 | 795.65 | 2024-11-28 10:15:00 | 778.35 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-12-13 12:45:00 | 844.00 | 2024-12-20 12:15:00 | 862.50 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2024-12-26 09:15:00 | 858.85 | 2024-12-26 14:15:00 | 870.35 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-12-26 13:15:00 | 861.00 | 2024-12-26 14:15:00 | 870.35 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-12-31 14:45:00 | 880.85 | 2025-01-03 13:15:00 | 869.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-02 11:30:00 | 880.50 | 2025-01-03 13:15:00 | 869.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-01-02 14:00:00 | 880.70 | 2025-01-03 13:15:00 | 869.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-01-08 10:15:00 | 835.75 | 2025-01-13 09:15:00 | 793.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 832.85 | 2025-01-13 09:15:00 | 791.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 835.75 | 2025-01-14 09:15:00 | 752.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 832.85 | 2025-01-14 13:15:00 | 772.55 | STOP_HIT | 0.50 | 7.24% |
| BUY | retest2 | 2025-01-17 14:15:00 | 813.05 | 2025-01-20 09:15:00 | 784.65 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-02-10 11:15:00 | 780.95 | 2025-02-12 09:15:00 | 741.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 780.30 | 2025-02-12 09:15:00 | 741.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:15:00 | 780.95 | 2025-02-13 09:15:00 | 746.30 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2025-02-10 14:00:00 | 780.30 | 2025-02-13 09:15:00 | 746.30 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest1 | 2025-02-27 11:00:00 | 720.85 | 2025-03-03 12:15:00 | 724.80 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-02-27 11:45:00 | 719.60 | 2025-03-03 12:15:00 | 724.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2025-02-27 13:45:00 | 721.10 | 2025-03-03 12:15:00 | 724.80 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-02-27 15:15:00 | 720.95 | 2025-03-03 12:15:00 | 724.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-03-07 13:30:00 | 747.80 | 2025-03-10 15:15:00 | 739.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-03-07 14:00:00 | 747.90 | 2025-03-10 15:15:00 | 739.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-10 09:15:00 | 748.55 | 2025-03-10 15:15:00 | 739.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-03-17 11:30:00 | 756.55 | 2025-03-24 09:15:00 | 832.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 09:15:00 | 770.55 | 2025-03-25 09:15:00 | 847.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 762.40 | 2025-04-11 09:15:00 | 790.65 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-04-09 14:30:00 | 767.70 | 2025-04-11 09:15:00 | 790.65 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-04-23 09:45:00 | 850.30 | 2025-04-23 11:15:00 | 828.10 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-04-25 10:15:00 | 783.15 | 2025-05-02 11:15:00 | 805.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-04-25 14:00:00 | 787.85 | 2025-05-02 11:15:00 | 805.70 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-04-29 15:00:00 | 789.30 | 2025-05-02 11:15:00 | 805.70 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-30 09:15:00 | 786.55 | 2025-05-02 11:15:00 | 805.70 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-05-08 14:45:00 | 756.35 | 2025-05-09 09:15:00 | 718.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 756.35 | 2025-05-12 09:15:00 | 756.95 | STOP_HIT | 0.50 | -0.08% |
| SELL | retest2 | 2025-05-12 09:30:00 | 749.60 | 2025-05-12 12:15:00 | 766.65 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-05-26 10:45:00 | 778.70 | 2025-05-27 09:15:00 | 765.25 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-06 09:15:00 | 773.50 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-06 10:15:00 | 775.35 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-06 11:15:00 | 774.40 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-09 10:15:00 | 774.25 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-09 12:00:00 | 778.45 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-06-09 13:15:00 | 775.80 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-06-11 12:45:00 | 775.80 | 2025-06-11 13:15:00 | 765.05 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-06-24 09:15:00 | 768.05 | 2025-06-27 14:15:00 | 766.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-07-02 10:30:00 | 755.60 | 2025-07-09 11:15:00 | 751.05 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-07-02 13:00:00 | 755.10 | 2025-07-09 11:15:00 | 751.05 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-07-14 11:15:00 | 734.80 | 2025-07-15 12:15:00 | 740.25 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-16 12:15:00 | 745.60 | 2025-07-22 15:15:00 | 756.60 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2025-07-25 09:30:00 | 748.40 | 2025-07-30 09:15:00 | 752.65 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-28 11:30:00 | 751.30 | 2025-07-30 09:15:00 | 752.65 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-08-08 15:00:00 | 735.90 | 2025-08-11 13:15:00 | 746.65 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-08-13 09:15:00 | 752.25 | 2025-08-22 14:15:00 | 787.85 | STOP_HIT | 1.00 | 4.73% |
| SELL | retest2 | 2025-08-29 12:15:00 | 768.15 | 2025-09-03 09:15:00 | 772.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-05 09:30:00 | 777.00 | 2025-09-05 12:15:00 | 770.35 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-05 10:15:00 | 777.45 | 2025-09-05 12:15:00 | 770.35 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-08 09:30:00 | 779.00 | 2025-09-09 09:15:00 | 766.85 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-09-08 10:30:00 | 777.30 | 2025-09-09 09:15:00 | 766.85 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-15 09:15:00 | 784.40 | 2025-09-16 13:15:00 | 776.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-16 15:00:00 | 779.20 | 2025-09-16 15:15:00 | 780.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-09-22 14:30:00 | 770.10 | 2025-09-25 14:15:00 | 731.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:30:00 | 770.10 | 2025-09-29 13:15:00 | 721.00 | STOP_HIT | 0.50 | 6.38% |
| BUY | retest2 | 2025-10-09 09:15:00 | 730.75 | 2025-10-13 12:15:00 | 725.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-09 10:30:00 | 730.90 | 2025-10-13 12:15:00 | 725.25 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-09 11:00:00 | 731.25 | 2025-10-13 12:15:00 | 725.25 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-10-15 11:45:00 | 723.50 | 2025-10-15 14:15:00 | 727.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-29 10:45:00 | 744.80 | 2025-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-31 15:15:00 | 745.00 | 2025-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-11-14 10:45:00 | 717.70 | 2025-11-18 09:15:00 | 716.05 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-11-27 11:15:00 | 738.00 | 2025-12-03 10:15:00 | 738.40 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-27 12:45:00 | 738.30 | 2025-12-03 10:15:00 | 738.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-11-27 13:45:00 | 737.15 | 2025-12-03 10:15:00 | 738.40 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-11-28 09:15:00 | 739.50 | 2025-12-03 10:15:00 | 738.40 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-05 12:15:00 | 732.65 | 2025-12-11 14:15:00 | 730.25 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-12-08 10:00:00 | 732.80 | 2025-12-11 14:15:00 | 730.25 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-12-24 12:45:00 | 738.05 | 2025-12-30 09:15:00 | 731.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-01-05 12:00:00 | 742.80 | 2026-01-06 10:15:00 | 736.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-01-05 12:45:00 | 743.80 | 2026-01-06 10:15:00 | 736.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-16 13:15:00 | 686.00 | 2026-01-20 13:15:00 | 651.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:15:00 | 686.00 | 2026-01-21 12:15:00 | 659.20 | STOP_HIT | 0.50 | 3.91% |
| BUY | retest2 | 2026-02-12 10:15:00 | 708.85 | 2026-02-13 09:15:00 | 692.80 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-02-17 14:15:00 | 689.45 | 2026-02-25 11:15:00 | 679.40 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2026-02-18 14:00:00 | 690.20 | 2026-02-25 11:15:00 | 679.40 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2026-02-19 09:15:00 | 689.90 | 2026-02-25 11:15:00 | 679.40 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2026-03-17 11:15:00 | 610.15 | 2026-03-17 13:15:00 | 619.15 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-01 10:30:00 | 584.05 | 2026-04-06 11:15:00 | 591.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-01 14:30:00 | 583.65 | 2026-04-06 11:15:00 | 591.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-04-02 09:15:00 | 569.05 | 2026-04-06 11:15:00 | 591.10 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-04-07 10:15:00 | 592.35 | 2026-04-15 09:15:00 | 651.59 | TARGET_HIT | 1.00 | 10.00% |

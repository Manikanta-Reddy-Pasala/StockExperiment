# UPL (UPL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 646.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 8 |
| PENDING | 33 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 18 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 20 / 8
- **Target hits / Stop hits / Partials:** 5 / 16 / 7
- **Avg / median % per leg:** 2.96% / 3.22%
- **Sum % (uncompounded):** 82.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 17 | 100.0% | 3 | 7 | 7 | 4.81% | 81.8% |
| BUY @ 2nd Alert (retest1) | 14 | 14 | 100.0% | 0 | 7 | 7 | 3.70% | 51.8% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.02% | 30.1% |
| SELL (all) | 11 | 3 | 27.3% | 2 | 9 | 0 | 0.10% | 1.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 3 | 27.3% | 2 | 9 | 0 | 0.10% | 1.1% |
| retest1 (combined) | 14 | 14 | 100.0% | 0 | 7 | 7 | 3.70% | 51.8% |
| retest2 (combined) | 14 | 6 | 42.9% | 5 | 9 | 0 | 2.23% | 31.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 512.35 | 498.25 | 498.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 519.95 | 499.06 | 498.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 494.90 | 506.49 | 502.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 494.90 | 506.49 | 502.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 494.90 | 506.49 | 502.92 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-05 10:15:00 | 522.20 | 506.31 | 502.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 524.20 | 506.49 | 503.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-06-20 12:15:00 | 576.62 | 529.12 | 517.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 532.25 | 576.66 | 576.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 522.90 | 573.28 | 575.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 562.75 | 561.57 | 567.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 567.35 | 561.69 | 567.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 567.35 | 561.69 | 567.88 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-08 10:15:00 | 562.00 | 562.22 | 567.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 558.15 | 562.18 | 567.80 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 573.85 | 552.54 | 560.55 | SL hit (close>static) qty=1.00 sl=568.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-26 09:15:00 | 558.35 | 553.73 | 560.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:15:00 | 556.10 | 553.75 | 560.85 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-03 14:15:00 | 563.00 | 552.79 | 559.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 563.10 | 552.89 | 559.06 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 570.35 | 553.07 | 559.11 | SL hit (close>static) qty=1.00 sl=568.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 570.35 | 553.07 | 559.11 | SL hit (close>static) qty=1.00 sl=568.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-05 09:15:00 | 555.50 | 553.93 | 559.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:15:00 | 555.00 | 553.94 | 559.32 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 558.15 | 554.04 | 559.29 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 552.35 | 554.66 | 559.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 554.15 | 554.66 | 559.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-09 13:15:00 | 555.35 | 554.69 | 559.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 556.00 | 554.71 | 559.26 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 563.90 | 554.65 | 559.03 | SL hit (close>static) qty=1.00 sl=559.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 563.90 | 554.65 | 559.03 | SL hit (close>static) qty=1.00 sl=559.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-11 13:15:00 | 554.45 | 554.81 | 559.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:15:00 | 551.95 | 554.78 | 558.99 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2024-12-23 09:15:00 | 499.50 | 545.94 | 553.29 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-12-23 10:15:00 | 496.76 | 545.51 | 553.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-23 13:15:00 | 555.35 | 538.32 | 542.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-23 14:15:00 | 558.45 | 538.52 | 542.19 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-24 09:15:00 | 554.70 | 538.87 | 542.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-24 10:15:00 | 556.45 | 539.05 | 542.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-24 12:15:00 | 554.45 | 539.38 | 542.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:15:00 | 553.15 | 539.51 | 542.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 541.80 | 539.77 | 542.60 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-28 09:15:00 | 539.00 | 539.79 | 542.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:15:00 | 539.45 | 539.78 | 542.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 545.80 | 539.84 | 542.58 | SL hit (close>static) qty=1.00 sl=545.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-28 14:15:00 | 539.00 | 539.93 | 542.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 539.35 | 539.92 | 542.56 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 547.90 | 540.00 | 542.59 | SL hit (close>static) qty=1.00 sl=545.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 572.60 | 540.94 | 542.98 | SL hit (close>static) qty=1.00 sl=559.80 alert=retest2 |

### Cycle 3 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 607.40 | 545.03 | 544.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 626.50 | 546.42 | 545.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 11:15:00 | 611.60 | 612.38 | 590.26 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-03-05 09:15:00 | 623.45 | 612.50 | 590.87 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:15:00 | 628.20 | 612.66 | 591.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-11 10:15:00 | 616.40 | 615.92 | 595.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:15:00 | 616.00 | 615.92 | 595.74 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-17 10:15:00 | 616.05 | 614.63 | 596.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 11:15:00 | 620.85 | 614.69 | 597.11 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:15:00 | 646.80 | 617.28 | 599.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:15:00 | 651.89 | 620.45 | 602.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 659.61 | 620.84 | 602.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 630.25 | 631.61 | 611.56 | SL hit (close<ema200) qty=0.50 sl=631.61 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 630.25 | 631.61 | 611.56 | SL hit (close<ema200) qty=0.50 sl=631.61 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 630.25 | 631.61 | 611.56 | SL hit (close<ema200) qty=0.50 sl=631.61 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 606.55 | 634.27 | 615.96 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 630.75 | 630.36 | 615.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 639.85 | 630.46 | 615.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-30 11:15:00 | 625.00 | 645.40 | 640.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 625.50 | 645.21 | 640.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-07-04 09:15:00 | 688.05 | 648.50 | 643.92 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-21 10:15:00 | 703.83 | 662.20 | 653.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 650.95 | 688.13 | 688.14 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 685.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 740.97 | 724.37 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 748.00 | 740.93 | 725.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 746.80 | 740.99 | 725.34 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 749.45 | 741.28 | 726.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 10:15:00 | 744.90 | 741.32 | 726.57 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 14:15:00 | 748.95 | 741.47 | 726.93 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 15:15:00 | 748.35 | 741.54 | 727.04 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 747.80 | 741.63 | 727.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 753.35 | 741.75 | 727.36 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 14:15:00 | 746.95 | 743.50 | 729.46 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-17 15:15:00 | 743.55 | 743.50 | 729.53 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-18 11:15:00 | 747.25 | 743.50 | 729.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 12:15:00 | 747.05 | 743.54 | 729.83 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 784.14 | 746.23 | 732.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 784.40 | 746.23 | 732.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 10:15:00 | 785.77 | 746.60 | 732.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 791.02 | 755.02 | 739.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.28 | 753.56 | SL hit (close<ema200) qty=0.50 sl=773.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.28 | 753.56 | SL hit (close<ema200) qty=0.50 sl=773.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.28 | 753.56 | SL hit (close<ema200) qty=0.50 sl=773.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.28 | 753.56 | SL hit (close<ema200) qty=0.50 sl=773.28 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.31 | 757.94 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 678.40 | 744.68 | 744.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 657.25 | 743.55 | 744.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 633.75 | 629.93 | 662.86 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 628.00 | 629.92 | 662.69 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-08 12:15:00 | 637.40 | 629.99 | 662.56 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 627.30 | 632.12 | 660.87 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 635.15 | 632.15 | 660.75 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 14:15:00 | 659.50 | 634.07 | 660.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 659.50 | 634.07 | 660.20 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-16 10:15:00 | 656.30 | 634.80 | 660.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 653.65 | 634.99 | 660.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 670.55 | 636.20 | 660.13 | SL hit (close>static) qty=1.00 sl=661.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-20 14:15:00 | 657.10 | 638.32 | 660.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 655.95 | 638.50 | 660.26 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 654.70 | 639.30 | 660.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 653.65 | 639.45 | 660.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-07 09:15:00 | 653.20 | 642.19 | 655.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 10:15:00 | 657.05 | 642.34 | 655.46 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 653.50 | 642.55 | 655.44 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-05-07 14:15:00 | 649.65 | 642.74 | 655.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-07 15:15:00 | 652.00 | 642.83 | 655.39 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 647.85 | 642.88 | 655.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-08 10:15:00 | 652.30 | 642.97 | 655.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 11:15:00 | 644.65 | 642.99 | 655.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 644.60 | 643.00 | 655.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-05 11:15:00 | 524.20 | 2024-06-20 12:15:00 | 576.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 558.15 | 2024-11-25 09:15:00 | 573.85 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-11-26 10:15:00 | 556.10 | 2024-12-04 09:15:00 | 570.35 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-12-03 15:15:00 | 563.10 | 2024-12-04 09:15:00 | 570.35 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-12-05 10:15:00 | 555.00 | 2024-12-11 09:15:00 | 563.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-12-09 10:15:00 | 554.15 | 2024-12-11 09:15:00 | 563.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-09 14:15:00 | 556.00 | 2024-12-23 09:15:00 | 499.50 | TARGET_HIT | 1.00 | 10.16% |
| SELL | retest2 | 2024-12-11 14:15:00 | 551.95 | 2024-12-23 10:15:00 | 496.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 553.15 | 2025-01-28 11:15:00 | 545.80 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2025-01-28 10:15:00 | 539.45 | 2025-01-29 09:15:00 | 547.90 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-01-28 15:15:00 | 539.35 | 2025-01-30 09:15:00 | 572.60 | STOP_HIT | 1.00 | -6.16% |
| BUY | retest1 | 2025-03-05 10:15:00 | 628.20 | 2025-03-19 12:15:00 | 646.80 | PARTIAL | 0.50 | 2.96% |
| BUY | retest1 | 2025-03-11 11:15:00 | 616.00 | 2025-03-21 09:15:00 | 651.89 | PARTIAL | 0.50 | 5.83% |
| BUY | retest1 | 2025-03-17 11:15:00 | 620.85 | 2025-03-21 10:15:00 | 659.61 | PARTIAL | 0.50 | 6.24% |
| BUY | retest1 | 2025-03-05 10:15:00 | 628.20 | 2025-03-28 12:15:00 | 630.25 | STOP_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-03-11 11:15:00 | 616.00 | 2025-03-28 12:15:00 | 630.25 | STOP_HIT | 0.50 | 2.31% |
| BUY | retest1 | 2025-03-17 11:15:00 | 620.85 | 2025-03-28 12:15:00 | 630.25 | STOP_HIT | 0.50 | 1.51% |
| BUY | retest2 | 2025-04-11 10:15:00 | 639.85 | 2025-07-04 09:15:00 | 688.05 | TARGET_HIT | 1.00 | 7.53% |
| BUY | retest2 | 2025-05-30 12:15:00 | 625.50 | 2025-07-21 10:15:00 | 703.83 | TARGET_HIT | 1.00 | 12.52% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2025-12-23 09:15:00 | 784.14 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-12 15:15:00 | 748.35 | 2025-12-23 09:15:00 | 784.40 | PARTIAL | 0.50 | 4.82% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2025-12-23 10:15:00 | 785.77 | PARTIAL | 0.50 | 4.30% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2025-12-30 14:15:00 | 791.02 | PARTIAL | 0.50 | 5.89% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest1 | 2025-12-12 15:15:00 | 748.35 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.22% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2026-04-16 11:15:00 | 653.65 | 2026-04-17 09:15:00 | 670.55 | STOP_HIT | 1.00 | -2.59% |

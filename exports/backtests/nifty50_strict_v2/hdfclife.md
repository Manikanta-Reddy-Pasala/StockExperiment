# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 621.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 6 |
| ALERT3 | 16 |
| PENDING | 57 |
| PENDING_CANCEL | 17 |
| ENTRY1 | 10 |
| ENTRY2 | 30 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 34
- **Target hits / Stop hits / Partials:** 4 / 36 / 3
- **Avg / median % per leg:** -0.60% / -1.41%
- **Sum % (uncompounded):** -25.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 1 | 21 | 2 | -0.50% | -11.9% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.64% | 11.5% |
| BUY @ 3rd Alert (retest2) | 17 | 0 | 0.0% | 0 | 17 | 0 | -1.37% | -23.4% |
| SELL (all) | 19 | 5 | 26.3% | 3 | 15 | 1 | -0.73% | -13.9% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.14% | -6.8% |
| SELL @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 3 | 10 | 0 | -0.54% | -7.1% |
| retest1 (combined) | 13 | 6 | 46.2% | 1 | 9 | 3 | 0.36% | 4.6% |
| retest2 (combined) | 30 | 3 | 10.0% | 3 | 27 | 0 | -1.01% | -30.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 14:15:00 | 668.90 | 641.89 | 641.85 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 625.15 | 643.02 | 643.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 09:15:00 | 622.65 | 642.48 | 642.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 640.20 | 634.79 | 638.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 640.20 | 634.79 | 638.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 640.20 | 634.79 | 638.30 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-10-23 14:15:00 | 624.90 | 636.13 | 638.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:15:00 | 625.00 | 636.02 | 638.44 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-15 10:15:00 | 630.25 | 625.51 | 630.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-15 11:15:00 | 631.65 | 625.58 | 630.70 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 657.20 | 627.02 | 631.15 | SL hit (close>static) qty=1.00 sl=640.20 alert=retest2 |

### Cycle 3 — BUY (started 2023-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 15:15:00 | 668.50 | 635.01 | 634.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 09:15:00 | 672.80 | 635.38 | 635.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 665.55 | 665.73 | 654.87 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-20 10:15:00 | 671.00 | 665.83 | 655.08 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-12-20 11:15:00 | 668.55 | 665.86 | 655.15 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 655.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 655.16 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 12:15:00 | 614.55 | 649.76 | 649.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 10:15:00 | 613.15 | 648.03 | 648.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 612.00 | 610.07 | 625.59 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-02-08 13:15:00 | 595.25 | 609.50 | 624.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 14:15:00 | 591.15 | 609.32 | 624.30 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 615.35 | 591.91 | 606.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-04 12:15:00 | 615.35 | 591.91 | 606.42 | SL hit (close>ema400) qty=1.00 sl=606.42 alert=retest1 |

### Cycle 5 — BUY (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 12:15:00 | 634.00 | 613.88 | 613.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 13:15:00 | 636.70 | 614.11 | 613.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.64 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 601.15 | 615.53 | 615.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 598.15 | 615.36 | 615.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 576.10 | 575.45 | 588.53 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-29 09:15:00 | 567.40 | 575.42 | 588.13 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:15:00 | 568.00 | 575.35 | 588.03 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 539.60 | 570.19 | 583.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-10 10:15:00 | 567.85 | 565.62 | 579.37 | SL hit (close>ema200) qty=0.50 sl=565.62 alert=retest1 |
| Cross detected — sustain check pending | 2024-06-10 14:15:00 | 569.55 | 565.83 | 579.20 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 15:15:00 | 567.90 | 565.86 | 579.14 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-11 13:15:00 | 571.35 | 566.23 | 579.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 14:15:00 | 572.10 | 566.29 | 578.97 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-12 14:15:00 | 572.25 | 566.83 | 578.81 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-12 15:15:00 | 572.45 | 566.88 | 578.78 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 588.55 | 567.10 | 578.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 588.55 | 567.10 | 578.83 | SL hit (close>ema400) qty=1.00 sl=578.83 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 588.55 | 567.10 | 578.83 | SL hit (close>ema400) qty=1.00 sl=578.83 alert=retest1 |

### Cycle 7 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 607.45 | 585.31 | 585.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 612.40 | 585.79 | 585.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 10:15:00 | 711.75 | 715.89 | 683.65 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-09-20 09:15:00 | 718.75 | 710.22 | 688.19 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-20 10:15:00 | 710.75 | 710.23 | 688.31 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-23 10:15:00 | 721.90 | 710.44 | 689.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:15:00 | 720.50 | 710.54 | 689.32 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-26 09:15:00 | 724.20 | 712.13 | 692.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:15:00 | 726.75 | 712.28 | 692.26 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-30 14:15:00 | 718.00 | 714.98 | 695.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 15:15:00 | 718.00 | 715.01 | 695.51 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-09 09:15:00 | 719.15 | 712.89 | 697.57 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:15:00 | 722.80 | 712.99 | 697.70 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:15:00 | 753.90 | 719.96 | 705.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 721.50 | 722.62 | 707.77 | SL hit (close<ema200) qty=0.50 sl=722.62 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 702.50 | 722.22 | 708.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 702.50 | 722.22 | 708.44 | SL hit (close<ema400) qty=1.00 sl=708.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 702.50 | 722.22 | 708.44 | SL hit (close<ema400) qty=1.00 sl=708.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 702.50 | 722.22 | 708.44 | SL hit (close<ema400) qty=1.00 sl=708.44 alert=retest1 |
| Cross detected — sustain check pending | 2024-10-28 10:15:00 | 720.75 | 721.20 | 708.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-28 11:15:00 | 716.25 | 721.15 | 708.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-28 14:15:00 | 718.15 | 721.02 | 708.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 718.25 | 721.00 | 708.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 701.40 | 721.39 | 710.59 | SL hit (close<static) qty=1.00 sl=701.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-05 13:15:00 | 724.15 | 720.66 | 710.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-05 14:15:00 | 716.75 | 720.63 | 710.47 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-05 15:15:00 | 717.25 | 720.59 | 710.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-06 09:15:00 | 710.75 | 720.49 | 710.51 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-11-08 09:15:00 | 717.10 | 719.16 | 710.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 10:15:00 | 716.90 | 719.14 | 710.52 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-12 12:15:00 | 699.75 | 717.45 | 710.31 | SL hit (close<static) qty=1.00 sl=701.65 alert=retest2 |

### Cycle 8 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 686.05 | 704.75 | 704.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 680.70 | 702.18 | 703.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 653.10 | 625.04 | 646.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 653.10 | 625.04 | 646.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 653.10 | 625.04 | 646.15 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-21 10:15:00 | 626.55 | 627.70 | 645.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 11:15:00 | 628.25 | 627.70 | 645.28 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-30 12:15:00 | 627.00 | 624.11 | 639.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-30 13:15:00 | 631.00 | 624.18 | 639.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 614.10 | 625.19 | 639.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 626.80 | 625.20 | 639.05 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-05 12:15:00 | 629.95 | 625.11 | 637.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 13:15:00 | 628.20 | 625.14 | 637.88 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-07 09:15:00 | 630.35 | 625.87 | 637.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-07 10:15:00 | 633.75 | 625.95 | 637.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-07 13:15:00 | 629.90 | 626.12 | 637.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-07 14:15:00 | 636.90 | 626.23 | 637.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-10 11:15:00 | 630.00 | 626.51 | 637.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-10 12:15:00 | 635.05 | 626.60 | 637.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 09:15:00 | 624.35 | 626.83 | 637.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:15:00 | 624.90 | 626.82 | 637.27 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 633.20 | 626.30 | 636.60 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-14 12:15:00 | 624.55 | 627.00 | 636.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 13:15:00 | 623.45 | 626.96 | 636.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-07 11:15:00 | 624.90 | 621.48 | 629.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 12:15:00 | 623.50 | 621.50 | 629.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-10 14:15:00 | 626.05 | 622.05 | 629.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 15:15:00 | 623.50 | 622.06 | 629.16 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 637.10 | 622.74 | 629.30 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 637.10 | 622.74 | 629.30 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 637.10 | 622.74 | 629.30 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-13 12:15:00 | 622.80 | 623.63 | 629.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:15:00 | 622.40 | 623.62 | 629.34 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 627.40 | 623.63 | 629.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 637.90 | 624.35 | 629.36 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 659.80 | 625.96 | 630.00 | SL hit (close>static) qty=1.00 sl=657.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 659.80 | 625.96 | 630.00 | SL hit (close>static) qty=1.00 sl=657.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 659.80 | 625.96 | 630.00 | SL hit (close>static) qty=1.00 sl=657.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 659.80 | 625.96 | 630.00 | SL hit (close>static) qty=1.00 sl=657.90 alert=retest2 |

### Cycle 9 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 681.50 | 634.11 | 633.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.35 | 646.97 | 640.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.18 | 647.84 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-07 13:15:00 | 663.15 | 658.23 | 647.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-07 14:15:00 | 666.35 | 658.31 | 648.01 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:15:00 | 699.67 | 664.34 | 652.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-04-25 09:15:00 | 732.98 | 682.35 | 664.97 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.15 | 776.57 | 756.61 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-14 10:15:00 | 760.65 | 776.41 | 756.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-14 11:15:00 | 760.15 | 776.25 | 756.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-14 12:15:00 | 760.45 | 776.09 | 756.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 763.25 | 775.96 | 756.70 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 752.85 | 775.19 | 756.88 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-16 09:15:00 | 763.95 | 774.54 | 756.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 760.75 | 774.41 | 756.94 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 752.25 | 773.23 | 756.94 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-22 12:15:00 | 764.50 | 768.42 | 756.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 765.65 | 768.40 | 756.21 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 10:15:00 | 760.60 | 768.15 | 756.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 762.75 | 768.09 | 756.35 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 757.30 | 767.40 | 756.63 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-25 11:15:00 | 760.70 | 767.06 | 756.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:15:00 | 764.75 | 767.04 | 756.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 756.00 | 766.37 | 757.11 | SL hit (close<static) qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 760.45 | 766.20 | 757.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-30 10:15:00 | 752.25 | 766.06 | 757.09 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 752.25 | 766.06 | 757.09 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 752.25 | 766.06 | 757.09 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-31 13:15:00 | 760.40 | 765.11 | 757.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-31 14:15:00 | 755.45 | 765.01 | 757.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 10:15:00 | 759.00 | 759.07 | 755.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-08 11:15:00 | 758.20 | 759.06 | 755.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 12:15:00 | 761.15 | 759.08 | 755.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 759.50 | 759.09 | 755.19 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 755.30 | 759.07 | 755.24 | SL hit (close<static) qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-11 12:15:00 | 760.90 | 759.06 | 755.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 764.25 | 759.11 | 755.34 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 13:15:00 | 752.05 | 774.65 | 767.07 | SL hit (close<static) qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-05 10:15:00 | 760.00 | 773.94 | 766.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 761.25 | 773.82 | 766.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 754.85 | 772.96 | 766.61 | SL hit (close<static) qty=1.00 sl=756.10 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 766.25 | 772.32 | 767.60 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-18 09:15:00 | 775.40 | 772.19 | 767.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 783.65 | 772.31 | 767.70 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 764.55 | 774.52 | 769.67 | SL hit (close<static) qty=1.00 sl=766.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-26 09:15:00 | 770.05 | 773.88 | 769.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 10:15:00 | 769.50 | 773.83 | 769.51 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 764.35 | 773.57 | 769.46 | SL hit (close<static) qty=1.00 sl=766.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 743.15 | 766.39 | 766.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 740.00 | 763.13 | 764.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 763.35 | 761.90 | 763.97 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 741.50 | 761.70 | 763.86 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:15:00 | 750.55 | 761.59 | 763.79 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 756.55 | 754.15 | 759.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 760.05 | 754.26 | 759.12 | SL hit (close>ema400) qty=1.00 sl=759.12 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-30 09:15:00 | 749.50 | 754.41 | 759.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-30 10:15:00 | 750.50 | 754.37 | 759.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-30 11:15:00 | 745.55 | 754.28 | 758.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 747.05 | 754.21 | 758.93 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-10 15:15:00 | 750.00 | 750.14 | 755.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-11 09:15:00 | 752.80 | 750.16 | 755.64 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 761.00 | 750.41 | 755.68 | SL hit (close>static) qty=1.00 sl=759.65 alert=retest2 |

### Cycle 11 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-01 14:15:00 | 767.55 | 761.98 | 760.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 766.75 | 762.03 | 760.88 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 757.30 | 761.98 | 760.86 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-05 11:15:00 | 767.55 | 760.60 | 760.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 770.00 | 760.69 | 760.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 754.75 | 761.53 | 760.75 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-09 14:15:00 | 764.15 | 761.44 | 760.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 762.90 | 761.45 | 760.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 756.50 | 764.37 | 762.45 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 12:15:00 | 766.90 | 762.96 | 761.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 764.75 | 762.98 | 761.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 765.20 | 763.00 | 761.89 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-19 15:15:00 | 766.45 | 763.04 | 761.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 766.45 | 763.07 | 761.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 759.85 | 762.98 | 761.94 | SL hit (close<static) qty=1.00 sl=761.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 758.20 | 762.93 | 761.96 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |

### Cycle 12 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 747.05 | 761.00 | 761.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 743.40 | 760.68 | 760.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 761.70 | 758.12 | 759.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 761.70 | 758.12 | 759.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 761.70 | 758.12 | 759.46 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-09 12:15:00 | 750.80 | 759.99 | 760.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 749.40 | 759.88 | 760.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 09:15:00 | 744.55 | 759.31 | 759.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 748.50 | 759.21 | 759.88 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 14:15:00 | 748.05 | 758.85 | 759.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 749.40 | 758.76 | 759.63 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-03-05 10:15:00 | 674.46 | 721.84 | 731.52 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-05 10:15:00 | 674.46 | 721.84 | 731.52 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-05 11:15:00 | 673.65 | 721.34 | 731.22 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-23 15:15:00 | 625.00 | 2023-11-17 09:15:00 | 657.20 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest1 | 2024-02-08 14:15:00 | 591.15 | 2024-03-04 12:15:00 | 615.35 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest1 | 2024-05-29 10:15:00 | 568.00 | 2024-06-04 09:15:00 | 539.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-05-29 10:15:00 | 568.00 | 2024-06-10 10:15:00 | 567.85 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2024-06-10 15:15:00 | 567.90 | 2024-06-13 09:15:00 | 588.55 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest1 | 2024-06-11 14:15:00 | 572.10 | 2024-06-13 09:15:00 | 588.55 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest1 | 2024-09-23 11:15:00 | 720.50 | 2024-10-21 09:15:00 | 753.90 | PARTIAL | 0.50 | 4.64% |
| BUY | retest1 | 2024-09-23 11:15:00 | 720.50 | 2024-10-23 11:15:00 | 721.50 | STOP_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-09-26 10:15:00 | 726.75 | 2024-10-25 09:15:00 | 702.50 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest1 | 2024-09-30 15:15:00 | 718.00 | 2024-10-25 09:15:00 | 702.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest1 | 2024-10-09 10:15:00 | 722.80 | 2024-10-25 09:15:00 | 702.50 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-10-28 15:15:00 | 718.25 | 2024-11-05 09:15:00 | 701.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-11-08 10:15:00 | 716.90 | 2024-11-12 12:15:00 | 699.75 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-01-21 11:15:00 | 628.25 | 2025-03-11 14:15:00 | 637.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-02-03 10:15:00 | 626.80 | 2025-03-11 14:15:00 | 637.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-02-05 13:15:00 | 628.20 | 2025-03-11 14:15:00 | 637.10 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-02-11 10:15:00 | 624.90 | 2025-03-18 12:15:00 | 637.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-02-14 13:15:00 | 623.45 | 2025-03-19 12:15:00 | 659.80 | STOP_HIT | 1.00 | -5.83% |
| SELL | retest2 | 2025-03-07 12:15:00 | 623.50 | 2025-03-19 12:15:00 | 659.80 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2025-03-10 15:15:00 | 623.50 | 2025-03-19 12:15:00 | 659.80 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2025-03-13 13:15:00 | 622.40 | 2025-03-19 12:15:00 | 659.80 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.35 | 2025-04-15 11:15:00 | 699.67 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.35 | 2025-04-25 09:15:00 | 732.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-14 13:15:00 | 763.25 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-16 10:15:00 | 760.75 | 2025-07-17 10:15:00 | 752.25 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-22 13:15:00 | 765.65 | 2025-07-29 14:15:00 | 756.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-23 11:15:00 | 762.75 | 2025-07-30 10:15:00 | 752.25 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-25 12:15:00 | 764.75 | 2025-07-30 10:15:00 | 752.25 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-08-08 13:15:00 | 759.50 | 2025-08-11 09:15:00 | 755.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-08-11 13:15:00 | 764.25 | 2025-09-04 13:15:00 | 752.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-05 11:15:00 | 761.25 | 2025-09-08 10:15:00 | 754.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-18 10:15:00 | 783.65 | 2025-09-25 09:15:00 | 764.55 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-09-26 10:15:00 | 769.50 | 2025-09-26 14:15:00 | 764.35 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2025-10-16 10:15:00 | 750.55 | 2025-10-29 12:15:00 | 760.05 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-30 12:15:00 | 747.05 | 2025-11-11 12:15:00 | 761.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-01 15:15:00 | 766.75 | 2025-12-02 09:15:00 | 757.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-05 12:15:00 | 770.00 | 2025-12-09 09:15:00 | 754.75 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-09 15:15:00 | 762.90 | 2025-12-17 09:15:00 | 756.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-19 13:15:00 | 764.75 | 2025-12-23 10:15:00 | 759.85 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-22 09:15:00 | 766.45 | 2025-12-24 13:15:00 | 758.20 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-09 13:15:00 | 749.40 | 2026-03-05 10:15:00 | 674.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-13 10:15:00 | 748.50 | 2026-03-05 10:15:00 | 674.46 | TARGET_HIT | 1.00 | 9.89% |
| SELL | retest2 | 2026-01-13 15:15:00 | 749.40 | 2026-03-05 11:15:00 | 673.65 | TARGET_HIT | 1.00 | 10.11% |

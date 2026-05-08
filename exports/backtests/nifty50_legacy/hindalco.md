# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1044.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 6 |
| ALERT3 | 9 |
| PENDING | 20 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 16 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 10
- **Target hits / Stop hits / Partials:** 4 / 13 / 6
- **Avg / median % per leg:** 8.86% / 7.20%
- **Sum % (uncompounded):** 203.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 10 | 62.5% | 4 | 7 | 5 | 11.76% | 188.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 10 | 62.5% | 4 | 7 | 5 | 11.76% | 188.1% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 6 | 1 | 2.24% | 15.7% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 9.99% | 20.0% |
| SELL @ 3rd Alert (retest2) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.86% | -4.3% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 9.99% | 20.0% |
| retest2 (combined) | 21 | 11 | 52.4% | 4 | 12 | 5 | 8.75% | 183.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 505.70 | 544.71 | 544.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 505.15 | 544.32 | 544.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 10:15:00 | 533.40 | 533.24 | 538.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 13:15:00 | 539.75 | 533.36 | 538.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 539.75 | 533.36 | 538.13 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-07 15:15:00 | 536.70 | 533.44 | 538.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-11 09:15:00 | 540.15 | 533.50 | 538.13 | ENTRY2 sustain failed after 5400m |
| Cross detected — sustain check pending | 2024-03-11 10:15:00 | 535.60 | 533.52 | 538.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 11:15:00 | 535.80 | 533.55 | 538.11 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 13:15:00 | 536.85 | 533.62 | 538.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 14:15:00 | 533.20 | 533.62 | 538.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 541.65 | 530.70 | 535.38 | SL hit (close>static) qty=1.00 sl=540.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 541.65 | 530.70 | 535.38 | SL hit (close>static) qty=1.00 sl=540.75 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 15:15:00 | 567.45 | 539.30 | 539.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 13:15:00 | 569.65 | 540.62 | 539.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 637.10 | 655.58 | 624.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 636.00 | 655.12 | 624.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 636.00 | 655.12 | 624.43 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-04 13:15:00 | 650.65 | 655.07 | 624.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 14:15:00 | 653.90 | 655.06 | 624.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-09 13:15:00 | 625.10 | 657.77 | 657.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 625.10 | 657.77 | 657.81 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 692.50 | 656.61 | 656.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 694.30 | 658.33 | 657.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-09-12 14:15:00 | 677.10 | 668.18 | 664.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 676.20 | 668.26 | 664.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-16 09:15:00 | 680.10 | 668.94 | 665.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:15:00 | 678.20 | 669.03 | 665.44 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-25 14:15:00 | 678.60 | 717.17 | 702.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 15:15:00 | 678.05 | 716.78 | 702.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-04 14:15:00 | 674.70 | 707.63 | 700.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 674.00 | 707.30 | 699.87 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 685.10 | 707.07 | 699.80 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-11-05 13:15:00 | 696.35 | 706.34 | 699.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 698.35 | 706.26 | 699.57 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 661.90 | 705.79 | 699.62 | SL hit (close<static) qty=1.00 sl=675.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 652.40 | 705.26 | 699.39 | SL hit (close<static) qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 652.40 | 705.26 | 699.39 | SL hit (close<static) qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 652.40 | 705.26 | 699.39 | SL hit (close<static) qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 652.40 | 705.26 | 699.39 | SL hit (close<static) qty=1.00 sl=661.40 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 654.10 | 694.16 | 694.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 651.50 | 693.74 | 693.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 672.10 | 669.52 | 678.19 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 660.55 | 669.57 | 677.80 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:15:00 | 655.60 | 669.43 | 677.69 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-10 12:15:00 | 666.05 | 669.27 | 677.24 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-10 13:15:00 | 667.20 | 669.25 | 677.19 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 677.00 | 669.33 | 677.11 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-12 10:15:00 | 666.60 | 669.50 | 676.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 663.45 | 669.44 | 676.83 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-13 09:15:00 | 651.25 | 669.17 | 676.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 654.40 | 669.02 | 676.40 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 563.93 | 616.87 | 640.05 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 616.30 | 612.08 | 634.97 | SL hit (close>ema200) qty=0.50 sl=612.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 615.65 | 601.70 | 615.11 | SL hit (close>ema400) qty=1.00 sl=615.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-06 12:15:00 | 684.45 | 621.27 | 622.33 | SL hit (close>static) qty=1.00 sl=679.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 691.25 | 623.74 | 623.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 14:15:00 | 691.90 | 626.81 | 625.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 668.00 | 668.49 | 651.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 652.85 | 667.42 | 652.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 652.85 | 667.42 | 652.32 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 09:15:00 | 612.20 | 640.19 | 640.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 10:15:00 | 606.50 | 638.08 | 639.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-02 11:15:00 | 627.55 | 630.88 | 634.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-02 12:15:00 | 629.60 | 630.87 | 634.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-08 11:15:00 | 626.80 | 631.49 | 634.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:15:00 | 624.60 | 631.42 | 634.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-12 14:15:00 | 651.10 | 631.49 | 634.09 | SL hit (close>static) qty=1.00 sl=650.05 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 656.90 | 636.36 | 636.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.20 | 636.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-09 09:15:00 | 651.95 | 642.78 | 640.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 651.50 | 642.87 | 640.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-10 09:15:00 | 656.85 | 643.40 | 641.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:15:00 | 658.25 | 643.55 | 641.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-12 14:15:00 | 651.65 | 645.71 | 642.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 15:15:00 | 651.05 | 645.77 | 642.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 653.85 | 645.27 | 642.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 652.25 | 645.34 | 643.00 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 644.20 | 645.40 | 643.09 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-23 11:15:00 | 657.35 | 645.51 | 643.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 664.70 | 645.70 | 643.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:15:00 | 749.22 | 705.01 | 690.43 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:15:00 | 748.71 | 705.01 | 690.43 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:15:00 | 750.09 | 714.32 | 697.31 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 13:15:00 | 756.99 | 715.77 | 698.38 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 09:15:00 | 764.41 | 733.76 | 714.95 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 846.95 | 770.79 | 746.12 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 855.73 | 770.79 | 746.12 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 846.36 | 770.79 | 746.12 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 847.93 | 770.79 | 746.12 | Target hit (30%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 777.70 | 796.96 | 765.33 | SL hit (close<ema200) qty=0.50 sl=796.96 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-11 11:15:00 | 535.80 | 2024-03-21 09:15:00 | 541.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-03-11 14:15:00 | 533.20 | 2024-03-21 09:15:00 | 541.65 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-06-04 14:15:00 | 653.90 | 2024-08-09 13:15:00 | 625.10 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-09-12 15:15:00 | 676.20 | 2024-11-07 09:15:00 | 661.90 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-09-16 10:15:00 | 678.20 | 2024-11-07 10:15:00 | 652.40 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2024-10-25 15:15:00 | 678.05 | 2024-11-07 10:15:00 | 652.40 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-11-04 15:15:00 | 674.00 | 2024-11-07 10:15:00 | 652.40 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2024-11-05 14:15:00 | 698.35 | 2024-11-07 10:15:00 | 652.40 | STOP_HIT | 1.00 | -6.58% |
| SELL | retest1 | 2024-12-09 10:15:00 | 655.60 | 2025-01-13 14:15:00 | 563.93 | PARTIAL | 0.50 | 13.98% |
| SELL | retest1 | 2024-12-09 10:15:00 | 655.60 | 2025-01-17 09:15:00 | 616.30 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2024-12-12 11:15:00 | 663.45 | 2025-02-19 09:15:00 | 615.65 | STOP_HIT | 1.00 | 7.20% |
| SELL | retest2 | 2024-12-13 10:15:00 | 654.40 | 2025-03-06 12:15:00 | 684.45 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-05-08 12:15:00 | 624.60 | 2025-05-12 14:15:00 | 651.10 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-06-09 10:15:00 | 651.50 | 2025-09-08 09:15:00 | 749.22 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-10 10:15:00 | 658.25 | 2025-09-08 09:15:00 | 748.71 | PARTIAL | 0.50 | 13.74% |
| BUY | retest2 | 2025-06-12 15:15:00 | 651.05 | 2025-09-12 09:15:00 | 750.09 | PARTIAL | 0.50 | 15.21% |
| BUY | retest2 | 2025-06-20 11:15:00 | 652.25 | 2025-09-12 13:15:00 | 756.99 | PARTIAL | 0.50 | 16.06% |
| BUY | retest2 | 2025-06-23 12:15:00 | 664.70 | 2025-09-30 09:15:00 | 764.41 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-09 10:15:00 | 651.50 | 2025-10-28 09:15:00 | 846.95 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-10 10:15:00 | 658.25 | 2025-10-28 09:15:00 | 855.73 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-12 15:15:00 | 651.05 | 2025-10-28 09:15:00 | 846.36 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-20 11:15:00 | 652.25 | 2025-10-28 09:15:00 | 847.93 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-23 12:15:00 | 664.70 | 2025-11-06 09:15:00 | 777.70 | STOP_HIT | 0.50 | 17.00% |

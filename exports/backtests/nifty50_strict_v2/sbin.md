# SBIN (SBIN)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1019.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 7 |
| PENDING | 35 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 6 |
| ENTRY2 | 16 |
| PARTIAL | 1 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 18
- **Target hits / Stop hits / Partials:** 4 / 18 / 1
- **Avg / median % per leg:** 0.41% / -1.24%
- **Sum % (uncompounded):** 9.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.89% | 20.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.13% | -3.1% |
| BUY @ 3rd Alert (retest2) | 6 | 3 | 50.0% | 3 | 3 | 0 | 3.89% | 23.3% |
| SELL (all) | 16 | 2 | 12.5% | 1 | 14 | 1 | -0.68% | -10.8% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.42% | 8.5% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.94% | -19.4% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.77% | 5.4% |
| retest2 (combined) | 16 | 3 | 18.8% | 3 | 13 | 0 | 0.25% | 3.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 09:15:00 | 604.05 | 584.77 | 584.72 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 13:15:00 | 573.05 | 586.26 | 586.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 570.45 | 585.84 | 586.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 576.90 | 574.40 | 579.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 15:15:00 | 579.20 | 574.61 | 579.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 579.20 | 574.61 | 579.17 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-11-06 09:15:00 | 576.20 | 574.63 | 579.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 10:15:00 | 573.75 | 574.62 | 579.13 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-07 15:15:00 | 580.35 | 574.84 | 578.98 | SL hit (close>static) qty=1.00 sl=579.50 alert=retest2 |
| Cross detected — sustain check pending | 2023-11-09 10:15:00 | 577.10 | 575.28 | 579.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-09 11:15:00 | 577.90 | 575.30 | 579.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-09 12:15:00 | 577.50 | 575.33 | 579.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-09 13:15:00 | 578.50 | 575.36 | 579.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-13 09:15:00 | 577.15 | 575.67 | 578.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:15:00 | 577.00 | 575.69 | 578.98 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-13 14:15:00 | 581.20 | 575.84 | 578.99 | SL hit (close>static) qty=1.00 sl=579.50 alert=retest2 |
| Cross detected — sustain check pending | 2023-11-17 09:15:00 | 566.85 | 576.99 | 579.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 10:15:00 | 566.50 | 576.89 | 579.29 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 588.90 | 571.02 | 575.06 | SL hit (close>static) qty=1.00 sl=579.50 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 13:15:00 | 610.65 | 578.58 | 578.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 614.05 | 581.05 | 579.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 624.90 | 625.81 | 610.37 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-12 13:15:00 | 632.15 | 625.51 | 611.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 14:15:00 | 632.90 | 625.58 | 611.78 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-18 11:15:00 | 632.00 | 627.55 | 614.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-18 12:15:00 | 628.10 | 627.56 | 614.53 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-19 09:15:00 | 632.70 | 627.64 | 614.83 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-19 10:15:00 | 628.85 | 627.65 | 614.90 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 613.10 | 627.48 | 615.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 613.10 | 627.48 | 615.32 | SL hit (close<ema400) qty=1.00 sl=615.32 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-29 13:15:00 | 627.75 | 624.92 | 615.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-29 14:15:00 | 623.50 | 624.91 | 615.31 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-30 10:15:00 | 627.90 | 624.93 | 615.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 11:15:00 | 632.75 | 625.01 | 615.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-31 09:15:00 | 628.90 | 625.18 | 615.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:15:00 | 632.00 | 625.25 | 615.96 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2024-02-08 09:15:00 | 695.20 | 634.45 | 622.64 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-08 10:15:00 | 696.03 | 635.12 | 623.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 815.30 | 829.61 | 829.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 810.35 | 828.06 | 828.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 804.65 | 803.65 | 813.32 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-09-23 15:15:00 | 800.60 | 803.62 | 813.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:15:00 | 801.20 | 803.59 | 813.09 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-09-24 14:15:00 | 798.75 | 803.48 | 812.81 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 15:15:00 | 797.90 | 803.43 | 812.73 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-27 12:15:00 | 800.95 | 802.44 | 811.41 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-27 13:15:00 | 801.55 | 802.43 | 811.36 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-30 09:15:00 | 793.20 | 802.34 | 811.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:15:00 | 790.55 | 802.22 | 811.07 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-03 11:15:00 | 794.00 | 801.22 | 809.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 12:15:00 | 788.05 | 801.09 | 809.80 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 797.50 | 800.86 | 809.38 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-04 13:15:00 | 796.80 | 800.82 | 809.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 796.95 | 800.78 | 809.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-09 15:15:00 | 795.80 | 798.14 | 806.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-10 09:15:00 | 803.40 | 798.19 | 806.95 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-10-11 10:15:00 | 796.90 | 798.21 | 806.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 795.80 | 798.18 | 806.56 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 807.20 | 799.11 | 806.28 | SL hit (close>ema400) qty=1.00 sl=806.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 807.20 | 799.11 | 806.28 | SL hit (close>ema400) qty=1.00 sl=806.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 807.20 | 799.11 | 806.28 | SL hit (close>ema400) qty=1.00 sl=806.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 807.20 | 799.11 | 806.28 | SL hit (close>ema400) qty=1.00 sl=806.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.59 | 806.28 | SL hit (close>static) qty=1.00 sl=809.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.59 | 806.28 | SL hit (close>static) qty=1.00 sl=809.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-22 12:15:00 | 795.45 | 802.27 | 806.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-22 13:15:00 | 800.50 | 802.25 | 806.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-22 14:15:00 | 788.70 | 802.12 | 806.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 15:15:00 | 790.40 | 802.00 | 806.75 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-28 11:15:00 | 795.75 | 799.02 | 804.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:15:00 | 794.50 | 798.98 | 804.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 811.20 | 798.88 | 804.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 811.20 | 798.88 | 804.36 | SL hit (close>static) qty=1.00 sl=809.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 811.20 | 798.88 | 804.36 | SL hit (close>static) qty=1.00 sl=809.70 alert=retest2 |

### Cycle 5 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 856.30 | 808.89 | 808.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 860.25 | 811.57 | 810.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 10:15:00 | 814.95 | 818.72 | 814.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 11:15:00 | 816.60 | 818.70 | 814.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 816.60 | 818.70 | 814.21 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-11-13 12:15:00 | 817.60 | 818.69 | 814.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-13 13:15:00 | 815.00 | 818.65 | 814.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-22 13:15:00 | 819.10 | 813.67 | 812.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-22 14:15:00 | 816.50 | 813.70 | 812.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-25 09:15:00 | 844.00 | 814.01 | 812.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 846.05 | 814.33 | 812.64 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 810.40 | 839.91 | 830.69 | SL hit (close<static) qty=1.00 sl=813.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-23 09:15:00 | 818.00 | 839.43 | 830.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 821.35 | 839.25 | 830.50 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-24 13:15:00 | 812.95 | 837.33 | 829.94 | SL hit (close<static) qty=1.00 sl=813.60 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 801.25 | 824.16 | 824.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 795.75 | 822.75 | 823.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 10:15:00 | 774.80 | 773.35 | 789.61 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-05 11:15:00 | 768.90 | 773.54 | 789.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-05 12:15:00 | 771.00 | 773.52 | 788.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-05 13:15:00 | 768.40 | 773.46 | 788.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 14:15:00 | 765.85 | 773.39 | 788.76 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:15:00 | 727.56 | 767.01 | 783.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-28 12:15:00 | 689.26 | 739.64 | 761.65 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 746.90 | 732.12 | 749.10 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-19 14:15:00 | 744.85 | 732.24 | 749.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 15:15:00 | 745.25 | 732.37 | 749.06 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 749.50 | 733.24 | 749.01 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 738.70 | 752.61 | 755.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-07 10:15:00 | 746.50 | 752.55 | 755.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 11:15:00 | 735.40 | 752.38 | 755.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 12:15:00 | 739.50 | 752.25 | 755.33 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 764.15 | 752.22 | 755.25 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-09 13:15:00 | 745.45 | 753.16 | 755.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:15:00 | 742.10 | 753.05 | 755.51 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 751.30 | 752.93 | 755.43 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.67 | 757.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.32 | 757.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.10 | 781.81 | 772.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 776.20 | 781.71 | 772.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 776.20 | 781.71 | 772.11 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-07 10:15:00 | 778.45 | 781.37 | 772.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 11:15:00 | 776.50 | 781.32 | 772.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-08 09:15:00 | 783.75 | 781.10 | 772.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:15:00 | 781.50 | 781.11 | 772.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 770.25 | 780.91 | 772.47 | SL hit (close<static) qty=1.00 sl=771.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-09 14:15:00 | 780.20 | 780.42 | 772.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 780.00 | 780.41 | 772.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-09-17 15:15:00 | 858.00 | 817.33 | 812.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 1020.45 | 1069.63 | 1069.75 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1111.00 | 1069.30 | 1069.16 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-06 10:15:00 | 573.75 | 2023-11-07 15:15:00 | 580.35 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2023-11-13 10:15:00 | 577.00 | 2023-11-13 14:15:00 | 581.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-11-17 10:15:00 | 566.50 | 2023-12-04 09:15:00 | 588.90 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest1 | 2024-01-12 14:15:00 | 632.90 | 2024-01-23 11:15:00 | 613.10 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-01-30 11:15:00 | 632.75 | 2024-02-08 09:15:00 | 695.20 | TARGET_HIT | 1.00 | 9.87% |
| BUY | retest2 | 2024-01-31 10:15:00 | 632.00 | 2024-02-08 10:15:00 | 696.03 | TARGET_HIT | 1.00 | 10.13% |
| SELL | retest1 | 2024-09-24 09:15:00 | 801.20 | 2024-10-16 09:15:00 | 807.20 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2024-09-24 15:15:00 | 797.90 | 2024-10-16 09:15:00 | 807.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2024-09-30 10:15:00 | 790.55 | 2024-10-16 09:15:00 | 807.20 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest1 | 2024-10-03 12:15:00 | 788.05 | 2024-10-16 09:15:00 | 807.20 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-10-04 14:15:00 | 796.95 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-10-11 11:15:00 | 795.80 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-10-22 15:15:00 | 790.40 | 2024-10-29 11:15:00 | 811.20 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-10-28 12:15:00 | 794.50 | 2024-10-29 11:15:00 | 811.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-11-25 10:15:00 | 846.05 | 2024-12-20 14:15:00 | 810.40 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2024-12-23 10:15:00 | 821.35 | 2024-12-24 13:15:00 | 812.95 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest1 | 2025-02-05 14:15:00 | 765.85 | 2025-02-11 10:15:00 | 727.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-05 14:15:00 | 765.85 | 2025-02-28 12:15:00 | 689.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-19 15:15:00 | 745.25 | 2025-03-20 14:15:00 | 749.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-04-07 12:15:00 | 739.50 | 2025-04-08 09:15:00 | 764.15 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-04-09 14:15:00 | 742.10 | 2025-04-11 09:15:00 | 751.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-05-08 10:15:00 | 781.50 | 2025-05-08 14:15:00 | 770.25 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-05-09 15:15:00 | 780.00 | 2025-09-17 15:15:00 | 858.00 | TARGET_HIT | 1.00 | 10.00% |

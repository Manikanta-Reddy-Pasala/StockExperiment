# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4997 bars)
- **Last close:** 1007.75
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
| ALERT2_SKIP | 4 |
| ALERT3 | 7 |
| PENDING | 22 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 10 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 8
- **Target hits / Stop hits / Partials:** 10 / 9 / 2
- **Avg / median % per leg:** 4.29% / 5.52%
- **Sum % (uncompounded):** 90.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 12 | 70.6% | 9 | 6 | 2 | 5.06% | 86.1% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.81% | 19.2% |
| BUY @ 3rd Alert (retest2) | 13 | 9 | 69.2% | 8 | 5 | 0 | 5.14% | 66.9% |
| SELL (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.01% | 4.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.01% | 4.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.81% | 19.2% |
| retest2 (combined) | 17 | 10 | 58.8% | 9 | 8 | 0 | 4.17% | 70.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 572.28 | 638.94 | 639.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 569.94 | 637.01 | 638.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 617.40 | 611.49 | 622.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 11:15:00 | 618.75 | 611.63 | 622.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 618.75 | 611.63 | 622.29 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-05 10:15:00 | 616.78 | 614.41 | 622.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-05 11:15:00 | 622.53 | 614.49 | 622.73 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-17 09:15:00 | 606.43 | 622.38 | 625.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 607.60 | 622.24 | 625.13 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-01-10 09:15:00 | 546.84 | 596.86 | 607.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 631.60 | 576.35 | 576.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 642.90 | 577.56 | 576.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 606.35 | 635.14 | 615.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 606.35 | 635.14 | 615.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 606.35 | 635.14 | 615.14 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 644.40 | 633.59 | 615.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 640.40 | 633.66 | 615.16 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 632.15 | 633.85 | 615.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 632.45 | 633.84 | 616.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 640.30 | 633.65 | 616.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 641.80 | 633.73 | 616.46 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2025-04-21 09:15:00 | 695.70 | 642.51 | 623.25 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-04-21 10:15:00 | 704.44 | 643.15 | 623.66 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-04-21 10:15:00 | 705.98 | 643.15 | 623.66 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-07 11:15:00 | 633.10 | 643.19 | 631.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 632.20 | 643.08 | 631.12 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 630.20 | 642.68 | 631.21 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-12 11:15:00 | 646.15 | 638.89 | 630.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 12:15:00 | 641.30 | 638.92 | 630.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-13 11:15:00 | 642.50 | 639.10 | 630.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 639.45 | 639.10 | 630.53 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-14 09:15:00 | 654.65 | 639.11 | 630.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 647.50 | 639.19 | 630.79 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-02 11:15:00 | 638.90 | 649.52 | 640.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 643.55 | 649.46 | 640.53 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 642.40 | 649.29 | 640.58 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-03 09:15:00 | 645.95 | 649.25 | 640.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-03 10:15:00 | 644.10 | 649.20 | 640.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-03 11:15:00 | 646.30 | 649.17 | 640.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:15:00 | 652.00 | 649.20 | 640.70 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 639.15 | 649.02 | 641.07 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 648.75 | 648.95 | 641.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 653.50 | 648.99 | 641.17 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-06-09 09:15:00 | 695.42 | 651.79 | 643.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-09 09:15:00 | 705.43 | 651.79 | 643.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-09 09:15:00 | 703.40 | 651.79 | 643.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-09 09:15:00 | 712.25 | 651.79 | 643.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-09 09:15:00 | 707.90 | 651.79 | 643.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-21 10:15:00 | 648.40 | 670.71 | 663.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 646.85 | 670.47 | 663.64 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 12:15:00 | 647.15 | 667.08 | 662.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 650.10 | 666.91 | 662.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 637.75 | 666.18 | 662.04 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 637.75 | 666.18 | 662.04 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 637.75 | 666.18 | 662.04 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 637.50 | 658.16 | 658.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 631.25 | 657.32 | 657.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 610.80 | 609.78 | 624.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 626.75 | 610.57 | 624.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 626.75 | 610.57 | 624.58 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-16 11:15:00 | 617.10 | 612.91 | 624.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 615.40 | 612.93 | 624.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 629.20 | 613.98 | 624.53 | SL hit (close>static) qty=1.00 sl=627.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-25 10:15:00 | 618.50 | 618.39 | 625.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 617.10 | 618.37 | 625.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-30 11:15:00 | 618.70 | 617.26 | 623.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 615.55 | 617.25 | 623.90 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 627.80 | 617.28 | 623.78 | SL hit (close>static) qty=1.00 sl=627.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 627.80 | 617.28 | 623.78 | SL hit (close>static) qty=1.00 sl=627.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 666.90 | 629.51 | 629.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 669.20 | 630.26 | 629.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.60 | 972.06 | 913.27 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1004.60 | 971.30 | 914.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1007.50 | 971.66 | 915.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 994.00 | 976.10 | 922.81 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 09:15:00 | 987.80 | 976.22 | 923.13 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-06 14:15:00 | 1004.40 | 976.88 | 924.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 15:15:00 | 1002.50 | 977.14 | 925.16 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 13:15:00 | 1052.62 | 980.03 | 927.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 14:15:00 | 1057.88 | 980.85 | 928.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-26 09:15:00 | 1102.75 | 1030.82 | 976.97 | Target hit (10%) qty=0.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1000.20 | 1038.49 | 986.44 | SL hit (close<ema200) qty=0.50 sl=1038.49 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 954.50 | 1035.12 | 989.85 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 971.50 | 1033.67 | 989.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:15:00 | 982.10 | 1033.16 | 989.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 931.40 | 1021.04 | 994.36 | SL hit (close<static) qty=1.00 sl=938.80 alert=retest2 |

### Cycle 5 — SELL (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 13:15:00 | 891.60 | 973.65 | 973.97 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 1025.80 | 973.92 | 973.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 14:15:00 | 1028.40 | 974.46 | 973.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 997.04 | 986.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1021.80 | 997.36 | 987.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.80 | 997.36 | 987.24 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-17 10:15:00 | 607.60 | 2025-01-10 09:15:00 | 546.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 640.40 | 2025-04-21 09:15:00 | 695.70 | TARGET_HIT | 1.00 | 8.63% |
| BUY | retest2 | 2025-04-09 12:15:00 | 632.45 | 2025-04-21 10:15:00 | 704.44 | TARGET_HIT | 1.00 | 11.38% |
| BUY | retest2 | 2025-04-11 10:15:00 | 641.80 | 2025-04-21 10:15:00 | 705.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 12:15:00 | 632.20 | 2025-06-05 09:15:00 | 639.15 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-05-12 12:15:00 | 641.30 | 2025-06-09 09:15:00 | 695.42 | TARGET_HIT | 1.00 | 8.44% |
| BUY | retest2 | 2025-05-13 12:15:00 | 639.45 | 2025-06-09 09:15:00 | 705.43 | TARGET_HIT | 1.00 | 10.32% |
| BUY | retest2 | 2025-05-14 10:15:00 | 647.50 | 2025-06-09 09:15:00 | 703.40 | TARGET_HIT | 1.00 | 8.63% |
| BUY | retest2 | 2025-06-02 12:15:00 | 643.55 | 2025-06-09 09:15:00 | 712.25 | TARGET_HIT | 1.00 | 10.68% |
| BUY | retest2 | 2025-06-03 12:15:00 | 652.00 | 2025-06-09 09:15:00 | 707.90 | TARGET_HIT | 1.00 | 8.57% |
| BUY | retest2 | 2025-06-05 12:15:00 | 653.50 | 2025-07-24 10:15:00 | 637.75 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-21 11:15:00 | 646.85 | 2025-07-24 10:15:00 | 637.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-23 13:15:00 | 650.10 | 2025-07-24 10:15:00 | 637.75 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-09-16 12:15:00 | 615.40 | 2025-09-18 10:15:00 | 629.20 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-25 11:15:00 | 617.10 | 2025-10-01 09:15:00 | 627.80 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-30 12:15:00 | 615.55 | 2025-10-01 09:15:00 | 627.80 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest1 | 2026-02-03 10:15:00 | 1007.50 | 2026-02-09 13:15:00 | 1052.62 | PARTIAL | 0.50 | 4.48% |
| BUY | retest1 | 2026-02-06 15:15:00 | 1002.50 | 2026-02-09 14:15:00 | 1057.88 | PARTIAL | 0.50 | 5.52% |
| BUY | retest1 | 2026-02-03 10:15:00 | 1007.50 | 2026-02-26 09:15:00 | 1102.75 | TARGET_HIT | 0.50 | 9.45% |
| BUY | retest1 | 2026-02-06 15:15:00 | 1002.50 | 2026-03-04 09:15:00 | 1000.20 | STOP_HIT | 0.50 | -0.23% |
| BUY | retest2 | 2026-03-09 12:15:00 | 982.10 | 2026-03-20 09:15:00 | 931.40 | STOP_HIT | 1.00 | -5.16% |

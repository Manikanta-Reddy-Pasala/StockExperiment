# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 948.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 18 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 7 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 4
- **Avg / median % per leg:** 0.93% / -2.27%
- **Sum % (uncompounded):** 16.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 4 | 4 | 4 | 2.71% | 32.6% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.86% | -27.4% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.65% | -15.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.86% | -8.6% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.44% | -7.3% |
| retest1 (combined) | 11 | 8 | 72.7% | 4 | 3 | 4 | 4.68% | 51.4% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -4.97% | -34.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 779.45 | 818.57 | 818.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 771.80 | 818.11 | 818.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.31 | 768.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 11:15:00 | 759.70 | 749.48 | 765.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 759.70 | 749.48 | 765.37 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-14 09:15:00 | 753.45 | 750.82 | 765.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 747.95 | 750.79 | 765.05 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-17 12:15:00 | 749.40 | 749.36 | 762.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:15:00 | 751.15 | 749.38 | 762.68 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 768.20 | 749.67 | 762.57 | SL hit (close>static) qty=1.00 sl=765.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 768.20 | 749.67 | 762.57 | SL hit (close>static) qty=1.00 sl=765.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-24 09:15:00 | 754.90 | 751.20 | 762.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 752.00 | 751.21 | 762.38 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 769.65 | 752.06 | 762.22 | SL hit (close>static) qty=1.00 sl=765.80 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 796.95 | 770.30 | 770.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 820.50 | 772.96 | 771.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 832.00 | 833.00 | 813.28 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-11 09:15:00 | 838.20 | 833.05 | 813.40 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 10:15:00 | 833.20 | 833.05 | 813.50 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 841.50 | 833.14 | 813.64 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 12:15:00 | 835.15 | 833.16 | 813.75 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 848.10 | 833.36 | 814.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:15:00 | 840.00 | 833.43 | 814.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-18 09:15:00 | 838.50 | 835.60 | 817.94 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:15:00 | 838.45 | 835.63 | 818.05 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-19 11:15:00 | 836.15 | 835.56 | 818.70 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 12:15:00 | 838.90 | 835.60 | 818.81 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-30 13:15:00 | 839.50 | 839.88 | 824.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 14:15:00 | 841.60 | 839.89 | 824.56 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:15:00 | 880.37 | 841.96 | 826.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:15:00 | 882.00 | 842.40 | 826.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:15:00 | 880.85 | 842.40 | 826.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:15:00 | 883.68 | 842.40 | 826.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-06 10:15:00 | 922.30 | 853.10 | 833.76 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-06 13:15:00 | 922.79 | 855.01 | 835.01 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-07 09:15:00 | 924.00 | 856.70 | 836.16 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-14 09:15:00 | 925.76 | 868.17 | 845.64 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.15 | 862.06 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-27 14:15:00 | 894.60 | 888.14 | 862.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 893.00 | 888.19 | 862.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-28 11:15:00 | 894.10 | 888.22 | 863.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 895.15 | 888.29 | 863.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 11:15:00 | 891.85 | 890.33 | 867.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-02 12:15:00 | 888.50 | 890.32 | 867.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 893.25 | 890.34 | 867.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:15:00 | 911.00 | 890.55 | 868.09 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 898.55 | 920.78 | 902.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 897.60 | 920.55 | 902.23 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.23 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.23 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.23 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.23 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 823.80 | 887.82 | 888.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 817.30 | 886.50 | 887.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.13 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 12:15:00 | 831.40 | 835.23 | 856.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 13:15:00 | 830.40 | 835.18 | 856.77 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 826.10 | 835.10 | 856.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 828.00 | 835.03 | 856.26 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 14:15:00 | 830.85 | 833.92 | 854.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 15:15:00 | 830.00 | 833.88 | 854.43 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | SL hit (close>ema400) qty=1.00 sl=852.62 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | SL hit (close>ema400) qty=1.00 sl=852.62 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | SL hit (close>ema400) qty=1.00 sl=852.62 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 840.50 | 841.62 | 853.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-24 14:15:00 | 848.85 | 841.69 | 853.22 | ENTRY2 sustain failed after 60m |
| CROSSOVER_SKIP | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.48 | min_gap filter: gap=0.004% < 0.010% |
| TREND_RESET | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.48 | EMA inversion without crossover edge (EMA200=862.52 EMA400=862.48) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-14 10:15:00 | 747.95 | 2025-10-20 10:15:00 | 768.20 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-10-17 13:15:00 | 751.15 | 2025-10-20 10:15:00 | 768.20 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-10-24 10:15:00 | 752.00 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-01-01 10:15:00 | 880.37 | PARTIAL | 0.50 | 4.81% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-01-01 11:15:00 | 882.00 | PARTIAL | 0.50 | 5.19% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-01-01 11:15:00 | 880.85 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-01-01 11:15:00 | 883.68 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-01-06 10:15:00 | 922.30 | TARGET_HIT | 0.50 | 9.80% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-01-06 13:15:00 | 922.79 | TARGET_HIT | 0.50 | 10.06% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-01-07 09:15:00 | 924.00 | TARGET_HIT | 0.50 | 10.14% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-01-14 09:15:00 | 925.76 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-27 15:15:00 | 893.00 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest2 | 2026-01-28 12:15:00 | 895.15 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest2 | 2026-02-02 14:15:00 | 911.00 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -8.07% |
| BUY | retest2 | 2026-03-10 11:15:00 | 897.60 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.70% |
| SELL | retest1 | 2026-04-08 13:15:00 | 830.40 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest1 | 2026-04-09 10:15:00 | 828.00 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest1 | 2026-04-10 15:15:00 | 830.00 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -2.79% |

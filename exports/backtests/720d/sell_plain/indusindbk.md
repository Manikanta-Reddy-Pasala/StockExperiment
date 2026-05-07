# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 950.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 15 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 2
- **Avg / median % per leg:** 6.75% / -2.01%
- **Sum % (uncompounded):** 67.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | 6.75% | 67.5% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.30% | -17.2% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 2 | 2 | 2 | 14.12% | 84.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.30% | -17.2% |
| retest2 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 14.12% | 84.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 14:15:00 | 1359.30 | 1424.25 | 1424.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 13:15:00 | 1349.00 | 1420.52 | 1422.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 996.00 | 992.56 | 1068.36 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 10:15:00 | 978.50 | 993.04 | 1065.27 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 12:15:00 | 971.90 | 992.65 | 1064.36 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-07 12:15:00 | 989.95 | 991.70 | 1061.41 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-07 13:15:00 | 991.80 | 991.70 | 1061.06 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-07 14:15:00 | 982.20 | 991.61 | 1060.67 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:15:00 | 982.45 | 991.37 | 1059.86 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-01-31 13:15:00 | 985.60 | 969.05 | 1014.25 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-31 15:15:00 | 995.20 | 969.52 | 1014.04 | ENTRY1 sustain failed after 120m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1018.75 | 970.01 | 1014.06 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 1018.75 | 970.01 | 1014.06 | SL hit (close>ema400) qty=1.00 sl=1014.06 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 1018.75 | 970.01 | 1014.06 | SL hit (close>ema400) qty=1.00 sl=1014.06 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-28 15:15:00 | 972.30 | 1020.25 | 1026.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 955.80 | 1019.61 | 1026.00 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-03-05 14:15:00 | 971.95 | 1012.98 | 1021.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-05 15:15:00 | 973.00 | 1012.59 | 1021.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-06 12:15:00 | 971.45 | 1011.06 | 1020.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-06 13:15:00 | 974.15 | 1010.69 | 1020.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-06 15:15:00 | 969.55 | 1009.91 | 1020.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 948.80 | 1009.30 | 1019.68 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-11 09:15:00 | 812.43 | 995.60 | 1011.94 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-11 09:15:00 | 806.48 | 995.60 | 1011.94 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2025-03-11 12:15:00 | 669.06 | 986.36 | 1007.04 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-03-11 13:15:00 | 664.16 | 983.17 | 1005.33 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 795.95 | 833.84 | 833.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.59 | 828.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.34 | 770.40 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 765.60 | 750.20 | 767.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 765.60 | 750.20 | 767.05 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-14 09:15:00 | 753.45 | 750.85 | 766.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 743.70 | 750.75 | 766.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 768.20 | 749.69 | 764.02 | SL hit (close>static) qty=1.00 sl=767.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-21 14:15:00 | 757.00 | 750.46 | 763.92 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-23 09:15:00 | 763.95 | 750.60 | 763.92 | ENTRY2 sustain failed after 2580m |
| Cross detected — sustain check pending | 2025-10-24 09:15:00 | 754.90 | 751.22 | 763.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 754.45 | 751.26 | 763.67 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 769.65 | 752.08 | 763.48 | SL hit (close>static) qty=1.00 sl=767.95 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 820.75 | 888.47 | 888.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 814.65 | 885.78 | 887.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.16 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 12:15:00 | 831.40 | 835.23 | 856.93 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-08 14:15:00 | 835.15 | 835.18 | 856.69 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 826.10 | 835.10 | 856.43 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 823.95 | 834.92 | 856.13 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-10 14:15:00 | 830.85 | 833.92 | 854.58 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 811.30 | 833.66 | 854.24 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 4020m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | SL hit (close>ema400) qty=1.00 sl=852.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | SL hit (close>ema400) qty=1.00 sl=852.64 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 840.50 | 841.62 | 853.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-24 14:15:00 | 848.85 | 841.69 | 853.24 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-06 12:15:00 | 971.90 | 2025-02-01 09:15:00 | 1018.75 | STOP_HIT | 1.00 | -4.82% |
| SELL | retest1 | 2025-01-08 09:15:00 | 982.45 | 2025-02-01 09:15:00 | 1018.75 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-03-03 09:15:00 | 955.80 | 2025-03-11 09:15:00 | 812.43 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-03-07 09:15:00 | 948.80 | 2025-03-11 09:15:00 | 806.48 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-03-03 09:15:00 | 955.80 | 2025-03-11 12:15:00 | 669.06 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-03-07 09:15:00 | 948.80 | 2025-03-11 13:15:00 | 664.16 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-10-14 11:15:00 | 743.70 | 2025-10-20 10:15:00 | 768.20 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-10-24 11:15:00 | 754.45 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest1 | 2026-04-09 11:15:00 | 823.95 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest1 | 2026-04-13 09:15:00 | 811.30 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -5.16% |

# HDFCBANK (HDFCBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 796.55
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 7 |
| PENDING | 15 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 14 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 12
- **Target hits / Stop hits / Partials:** 0 / 15 / 2
- **Avg / median % per leg:** 2.76% / -0.48%
- **Sum % (uncompounded):** 46.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 5 | 29.4% | 0 | 15 | 2 | 2.76% | 46.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.95% | -2.9% |
| BUY @ 3rd Alert (retest2) | 16 | 5 | 31.2% | 0 | 14 | 2 | 3.12% | 49.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.95% | -2.9% |
| retest2 (combined) | 16 | 5 | 31.2% | 0 | 14 | 2 | 3.12% | 49.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 10:15:00 | 826.20 | 773.27 | 773.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 826.83 | 775.33 | 774.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 822.83 | 824.57 | 808.45 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-15 09:15:00 | 832.50 | 824.54 | 809.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:15:00 | 835.45 | 824.65 | 809.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 790.62 | 825.64 | 810.83 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 810.83 | 825.64 | 810.83 | SL hit qty=1.00 sl=810.83 alert=retest1 |
| CROSSOVER_SKIP | 2024-01-25 11:15:00 | 713.60 | 798.01 | 798.41 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 767.10 | 743.84 | 743.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 767.65 | 748.82 | 746.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 09:15:00 | 745.85 | 752.24 | 748.82 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 745.85 | 752.24 | 748.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 745.85 | 752.24 | 748.82 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-05-14 13:15:00 | 730.47 | 745.93 | 745.95 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2024-05-24 09:15:00 | 748.65 | 740.32 | 742.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 753.50 | 740.45 | 742.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-29 12:15:00 | 754.05 | 744.93 | 744.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 754.05 | 744.93 | 744.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 755.47 | 745.31 | 745.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 767.95 | 750.08 | 747.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 776.05 | 750.34 | 747.88 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-07-03 09:15:00 | 892.46 | 807.33 | 783.97 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-01-14 10:15:00 | 826.45 | 876.38 | 876.60 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-03-24 09:15:00 | 897.05 | 857.20 | 857.08 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 897.05 | 857.20 | 857.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 898.95 | 857.62 | 857.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.77 | 869.69 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 12:15:00 | 872.08 | 878.65 | 869.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 872.08 | 878.65 | 869.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 14:15:00 | 879.30 | 878.60 | 869.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 878.85 | 878.61 | 869.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-26 14:15:00 | 1010.68 | 970.94 | 953.77 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-09-04 09:15:00 | 958.75 | 980.49 | 980.56 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 1002.50 | 974.02 | 973.99 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1002.50 | 974.02 | 973.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.36 | 974.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.97 | 981.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 979.75 | 986.89 | 981.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 979.75 | 986.89 | 981.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-06 11:15:00 | 987.60 | 986.84 | 981.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 989.10 | 986.86 | 981.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 15:15:00 | 984.25 | 986.40 | 981.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 986.35 | 986.40 | 981.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 984.10 | 986.30 | 982.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 986.85 | 986.30 | 982.06 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-14 11:15:00 | 984.90 | 986.72 | 982.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 985.85 | 986.71 | 982.71 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 983.35 | 986.68 | 982.71 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-14 14:15:00 | 990.50 | 986.72 | 982.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 989.00 | 986.74 | 982.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-19 11:15:00 | 987.30 | 987.62 | 983.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 988.30 | 987.63 | 983.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 13:15:00 | 988.50 | 995.84 | 990.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 989.40 | 995.77 | 990.51 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.65 | 995.68 | 991.29 | SL hit qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.65 | 995.68 | 991.29 | SL hit qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.65 | 995.68 | 991.29 | SL hit qty=1.00 sl=981.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 988.30 | 993.74 | 990.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 987.30 | 993.68 | 990.63 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 986.00 | 993.60 | 990.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-23 09:15:00 | 994.10 | 993.34 | 990.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 993.50 | 993.35 | 990.57 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-29 12:15:00 | 989.40 | 993.46 | 990.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 990.90 | 993.43 | 990.94 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 985.50 | 993.22 | 990.89 | SL hit qty=1.00 sl=985.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 985.50 | 993.22 | 990.89 | SL hit qty=1.00 sl=985.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-30 14:15:00 | 991.00 | 993.04 | 990.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 990.90 | 993.02 | 990.84 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 981.65 | 993.17 | 991.16 | SL hit qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 985.50 | 993.17 | 991.16 | SL hit qty=1.00 sl=985.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 974.20 | 992.32 | 990.78 | SL hit qty=1.00 sl=974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 974.20 | 992.32 | 990.78 | SL hit qty=1.00 sl=974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 974.20 | 992.32 | 990.78 | SL hit qty=1.00 sl=974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 974.20 | 992.32 | 990.78 | SL hit qty=1.00 sl=974.20 alert=retest2 |
| CROSSOVER_SKIP | 2026-01-07 12:15:00 | 948.60 | 989.03 | 989.17 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-15 10:15:00 | 835.45 | 2024-01-17 09:15:00 | 810.83 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-05-24 10:15:00 | 753.50 | 2024-05-29 12:15:00 | 754.05 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-06-05 14:15:00 | 776.05 | 2024-07-03 09:15:00 | 892.46 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 14:15:00 | 776.05 | 2025-03-24 09:15:00 | 897.05 | STOP_HIT | 0.50 | 15.59% |
| BUY | retest2 | 2025-04-07 15:15:00 | 878.85 | 2025-06-26 14:15:00 | 1010.68 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-07 15:15:00 | 878.85 | 2025-10-17 15:15:00 | 1002.50 | STOP_HIT | 0.50 | 14.07% |
| BUY | retest2 | 2025-11-06 12:15:00 | 989.10 | 2025-12-17 12:15:00 | 981.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-10 09:15:00 | 986.35 | 2025-12-17 12:15:00 | 981.65 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-11-11 12:15:00 | 986.85 | 2025-12-17 12:15:00 | 981.65 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-11-14 12:15:00 | 985.85 | 2025-12-30 11:15:00 | 985.50 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-11-14 15:15:00 | 989.00 | 2025-12-30 11:15:00 | 985.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-11-19 12:15:00 | 988.30 | 2026-01-05 11:15:00 | 981.65 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-12-10 14:15:00 | 989.40 | 2026-01-05 11:15:00 | 985.50 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-22 10:15:00 | 987.30 | 2026-01-06 09:15:00 | 974.20 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-12-23 10:15:00 | 993.50 | 2026-01-06 09:15:00 | 974.20 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-29 13:15:00 | 990.90 | 2026-01-06 09:15:00 | 974.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-30 15:15:00 | 990.90 | 2026-01-06 09:15:00 | 974.20 | STOP_HIT | 1.00 | -1.69% |

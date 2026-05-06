# INDUSINDBK (INDUSINDBK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 946.75
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 17 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 3 |
| ENTRY2 | 7 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 1
- **Avg / median % per leg:** 2.36% / -1.74%
- **Sum % (uncompounded):** 25.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.71% | -1.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.71% | -1.4% |
| SELL (all) | 9 | 2 | 22.2% | 1 | 7 | 1 | 3.04% | 27.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.53% | -10.6% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 1 | 4 | 1 | 6.32% | 37.9% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.53% | -10.6% |
| retest2 (combined) | 8 | 2 | 25.0% | 1 | 6 | 1 | 4.56% | 36.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1553.75 | 1518.07 | 1518.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1558.50 | 1518.81 | 1518.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.26 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.26 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-04-22 13:15:00 | 1471.05 | 1519.50 | 1519.66 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2024-04-30 13:15:00 | 1531.20 | 1506.66 | 1512.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-30 14:15:00 | 1518.00 | 1506.77 | 1512.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-03 13:15:00 | 1530.85 | 1458.51 | 1475.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-03 14:15:00 | 1528.05 | 1459.20 | 1475.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-19 09:15:00 | 1542.90 | 1473.39 | 1478.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 1541.50 | 1474.06 | 1478.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-19 15:15:00 | 1536.00 | 1476.98 | 1480.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-20 09:15:00 | 1515.50 | 1477.36 | 1480.48 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-06-20 11:15:00 | 1531.55 | 1478.37 | 1480.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 12:15:00 | 1534.65 | 1478.93 | 1481.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 10:15:00 | 1532.15 | 1481.43 | 1482.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-21 11:15:00 | 1529.95 | 1481.91 | 1482.67 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-06-21 15:15:00 | 1527.15 | 1483.64 | 1483.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-21 15:15:00 | 1527.15 | 1483.64 | 1483.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 1527.15 | 1483.64 | 1483.53 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2024-07-01 15:15:00 | 1456.00 | 1483.60 | 1483.66 | slope filter: EMA200 not falling 2.00% over 1400 bars |

### Cycle 3 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1491.00 | 1426.86 | 1426.54 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1349.20 | 1428.84 | 1429.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 1342.05 | 1419.65 | 1424.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 996.00 | 992.78 | 1069.27 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 10:15:00 | 978.50 | 993.26 | 1066.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 974.90 | 993.08 | 1065.70 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 12:15:00 | 989.95 | 991.91 | 1062.25 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-07 13:15:00 | 992.00 | 991.91 | 1061.90 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-07 14:15:00 | 982.55 | 991.82 | 1061.50 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 15:15:00 | 975.05 | 991.65 | 1061.07 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-31 13:15:00 | 985.20 | 969.12 | 1014.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 14:15:00 | 989.75 | 969.33 | 1014.58 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1019.55 | 970.46 | 1014.48 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1014.48 | 970.46 | 1014.48 | SL hit qty=1.00 sl=1014.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1014.48 | 970.46 | 1014.48 | SL hit qty=1.00 sl=1014.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1014.48 | 970.46 | 1014.48 | SL hit qty=1.00 sl=1014.48 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-28 09:15:00 | 997.65 | 1021.89 | 1028.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 10:15:00 | 991.00 | 1021.58 | 1027.83 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-11 09:15:00 | 842.35 | 995.14 | 1012.17 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2025-03-11 11:15:00 | 693.70 | 989.13 | 1008.98 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.22 | HTF filter: close below htf_sma |

### Cycle 5 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 796.10 | 833.76 | 833.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.54 | 828.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.31 | 770.36 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 765.60 | 750.16 | 767.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 765.60 | 750.16 | 767.01 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-13 15:15:00 | 758.20 | 750.78 | 766.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 753.45 | 750.81 | 766.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 767.95 | 749.67 | 763.98 | SL hit qty=1.00 sl=767.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-20 15:15:00 | 758.90 | 750.29 | 763.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 13:15:00 | 758.75 | 750.37 | 763.92 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1320m) |
| Cross detected — sustain check pending | 2025-10-23 15:15:00 | 758.45 | 751.17 | 763.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 754.80 | 751.21 | 763.75 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 767.95 | 751.89 | 763.43 | SL hit qty=1.00 sl=767.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 767.95 | 751.89 | 763.43 | SL hit qty=1.00 sl=767.95 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-10 13:15:00 | 799.00 | 771.99 | 771.96 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2026-03-19 11:15:00 | 823.90 | 887.83 | 887.88 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2026-03-30 13:15:00 | 758.90 | 855.93 | 870.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:15:00 | 751.50 | 854.89 | 869.72 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 767.95 | 853.15 | 868.70 | SL hit qty=1.00 sl=767.95 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.28 | 852.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-17 09:15:00 | 841.80 | 834.22 | 852.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-17 10:15:00 | 843.80 | 834.31 | 852.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 840.50 | 841.12 | 853.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-24 14:15:00 | 848.85 | 841.20 | 853.11 | ENTRY2 sustain failed after 60m |

### Cycle 6 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 913.70 | 862.72 | 862.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 925.35 | 864.77 | 863.69 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-19 10:15:00 | 1541.50 | 2024-06-21 15:15:00 | 1527.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-06-20 12:15:00 | 1534.65 | 2024-06-21 15:15:00 | 1527.15 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-06 11:15:00 | 974.90 | 2025-02-03 10:15:00 | 1014.48 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest1 | 2025-01-07 15:15:00 | 975.05 | 2025-02-03 10:15:00 | 1014.48 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest1 | 2025-01-31 14:15:00 | 989.75 | 2025-02-03 10:15:00 | 1014.48 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-28 10:15:00 | 991.00 | 2025-03-11 09:15:00 | 842.35 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-28 10:15:00 | 991.00 | 2025-03-11 11:15:00 | 693.70 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-10-14 09:15:00 | 753.45 | 2025-10-20 10:15:00 | 767.95 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-10-21 13:15:00 | 758.75 | 2025-10-27 13:15:00 | 767.95 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-24 09:15:00 | 754.80 | 2025-10-27 13:15:00 | 767.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-30 14:15:00 | 751.50 | 2026-04-01 09:15:00 | 767.95 | STOP_HIT | 1.00 | -2.19% |

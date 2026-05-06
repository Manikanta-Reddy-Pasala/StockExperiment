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
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 13 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 7 / 1
- **Avg / median % per leg:** 2.30% / -2.50%
- **Sum % (uncompounded):** 20.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 7 | 1 | 2.30% | 20.7% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 1 | 3 | 1 | 5.90% | 29.5% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| retest2 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 5.90% | 29.5% |

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

### Cycle 2 — SELL (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 13:15:00 | 1471.05 | 1519.50 | 1519.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 1469.00 | 1515.48 | 1517.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 12:15:00 | 1507.90 | 1506.41 | 1512.38 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 13:15:00 | 1531.20 | 1506.66 | 1512.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 1531.20 | 1506.66 | 1512.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-02 11:15:00 | 1502.50 | 1506.83 | 1512.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 12:15:00 | 1503.30 | 1506.80 | 1512.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-03 11:15:00 | 1487.10 | 1506.51 | 1512.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:15:00 | 1479.05 | 1506.24 | 1511.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-03 13:15:00 | 1531.20 | 1458.51 | 1475.71 | SL hit qty=1.00 sl=1531.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-03 13:15:00 | 1531.20 | 1458.51 | 1475.71 | SL hit qty=1.00 sl=1531.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-04 09:15:00 | 1471.00 | 1460.01 | 1476.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 1390.20 | 1459.32 | 1475.79 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-14 13:15:00 | 1499.15 | 1469.42 | 1476.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-14 14:15:00 | 1504.90 | 1469.77 | 1477.12 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-14 15:15:00 | 1502.35 | 1470.09 | 1477.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-18 09:15:00 | 1504.10 | 1470.43 | 1477.38 | ENTRY2 sustain failed after 5400m |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1531.20 | 1473.39 | 1478.64 | SL hit qty=1.00 sl=1531.20 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 1527.15 | 1483.64 | 1483.53 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 1456.00 | 1483.60 | 1483.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 1440.55 | 1483.17 | 1483.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 1458.80 | 1457.60 | 1467.89 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-07-18 15:15:00 | 1452.00 | 1457.55 | 1467.81 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1441.40 | 1457.39 | 1467.68 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 1419.00 | 1391.63 | 1415.39 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 1415.39 | 1391.63 | 1415.39 | SL hit qty=1.00 sl=1415.39 alert=retest1 |
| Cross detected — sustain check pending | 2024-09-06 09:15:00 | 1407.65 | 1406.06 | 1418.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-06 10:15:00 | 1413.40 | 1406.13 | 1418.39 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-09 09:15:00 | 1409.00 | 1406.40 | 1418.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-09 10:15:00 | 1418.40 | 1406.52 | 1418.17 | ENTRY2 sustain failed after 60m |

### Cycle 5 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1491.00 | 1426.86 | 1426.54 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-10-07 14:15:00)

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
| CROSSOVER_SKIP | 2025-08-04 09:15:00 | 796.10 | 833.76 | 833.83 | slope filter: EMA200 not falling 0.50% over 350 bars |
| CROSSOVER_SKIP | 2025-11-10 13:15:00 | 799.00 | 771.99 | 771.96 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2026-03-19 11:15:00 | 823.90 | 887.83 | 887.88 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 7 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 913.70 | 862.72 | 862.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 925.35 | 864.77 | 863.69 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-02 12:15:00 | 1503.30 | 2024-06-03 13:15:00 | 1531.20 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-05-03 12:15:00 | 1479.05 | 2024-06-03 13:15:00 | 1531.20 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-06-04 10:15:00 | 1390.20 | 2024-06-19 09:15:00 | 1531.20 | STOP_HIT | 1.00 | -10.14% |
| SELL | retest1 | 2024-07-19 09:15:00 | 1441.40 | 2024-08-28 11:15:00 | 1415.39 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest1 | 2025-01-06 11:15:00 | 974.90 | 2025-02-03 10:15:00 | 1014.48 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest1 | 2025-01-07 15:15:00 | 975.05 | 2025-02-03 10:15:00 | 1014.48 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest1 | 2025-01-31 14:15:00 | 989.75 | 2025-02-03 10:15:00 | 1014.48 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-28 10:15:00 | 991.00 | 2025-03-11 09:15:00 | 842.35 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-28 10:15:00 | 991.00 | 2025-03-11 11:15:00 | 693.70 | TARGET_HIT | 0.50 | 30.00% |

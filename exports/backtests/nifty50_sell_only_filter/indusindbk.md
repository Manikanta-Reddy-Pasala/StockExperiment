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
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 7 |
| PENDING | 31 |
| PENDING_CANCEL | 11 |
| ENTRY1 | 10 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 1 / 19 / 1
- **Avg / median % per leg:** 0.30% / -1.86%
- **Sum % (uncompounded):** 6.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 0 | 12 | 0 | -1.20% | -14.4% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 6 | 0 | 1.14% | 6.8% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 0 | 6 | 0 | -3.54% | -21.2% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 7 | 1 | 2.30% | 20.7% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 1 | 3 | 1 | 5.90% | 29.5% |
| retest1 (combined) | 10 | 5 | 50.0% | 0 | 10 | 0 | -0.20% | -2.0% |
| retest2 (combined) | 11 | 3 | 27.3% | 1 | 9 | 1 | 0.75% | 8.2% |

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

### Cycle 7 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.18 | 825.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.14 | 839.77 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 13:15:00 | 852.95 | 850.07 | 839.89 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 14:15:00 | 859.65 | 850.16 | 839.99 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 13:15:00 | 851.70 | 849.80 | 840.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-23 14:15:00 | 849.95 | 849.81 | 840.49 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-23 15:15:00 | 851.15 | 849.82 | 840.54 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 855.55 | 849.88 | 840.62 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.73 | 840.86 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 840.86 | 849.73 | 840.86 | SL hit qty=1.00 sl=840.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 840.86 | 849.73 | 840.86 | SL hit qty=1.00 sl=840.86 alert=retest1 |
| CROSSOVER_SKIP | 2025-08-04 09:15:00 | 796.10 | 833.76 | 833.83 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 8 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.99 | 771.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 819.85 | 773.01 | 772.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 830.35 | 833.02 | 813.71 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-11 09:15:00 | 838.25 | 833.07 | 813.84 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 10:15:00 | 833.20 | 833.07 | 813.93 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 841.50 | 833.16 | 814.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 12:15:00 | 835.15 | 833.17 | 814.17 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-11 14:15:00 | 835.90 | 833.22 | 814.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 15:15:00 | 833.95 | 833.23 | 814.49 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 847.80 | 833.37 | 814.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:15:00 | 840.00 | 833.44 | 814.78 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 12:15:00 | 838.20 | 835.67 | 817.99 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-17 13:15:00 | 830.40 | 835.62 | 818.05 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-18 09:15:00 | 838.50 | 835.63 | 818.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:15:00 | 838.45 | 835.65 | 818.42 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-19 11:15:00 | 836.15 | 835.61 | 819.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 12:15:00 | 838.90 | 835.64 | 819.17 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-30 13:15:00 | 839.50 | 839.94 | 824.79 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 14:15:00 | 841.60 | 839.95 | 824.87 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.23 | 862.27 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 862.27 | 888.23 | 862.27 | SL hit qty=1.00 sl=862.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 862.27 | 888.23 | 862.27 | SL hit qty=1.00 sl=862.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 862.27 | 888.23 | 862.27 | SL hit qty=1.00 sl=862.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 862.27 | 888.23 | 862.27 | SL hit qty=1.00 sl=862.27 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-27 14:15:00 | 894.60 | 888.21 | 862.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 894.75 | 888.28 | 863.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-28 11:15:00 | 894.00 | 888.30 | 863.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 895.15 | 888.37 | 863.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 11:15:00 | 891.85 | 889.93 | 866.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-02 12:15:00 | 888.45 | 889.92 | 866.89 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 893.25 | 889.95 | 867.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:15:00 | 910.95 | 890.16 | 867.24 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 09:15:00 | 890.05 | 920.96 | 901.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:15:00 | 898.55 | 920.74 | 901.90 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 897.60 | 920.51 | 901.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 851.75 | 916.49 | 900.91 | SL hit qty=1.00 sl=851.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 851.75 | 916.49 | 900.91 | SL hit qty=1.00 sl=851.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 851.75 | 916.49 | 900.91 | SL hit qty=1.00 sl=851.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 851.75 | 916.49 | 900.91 | SL hit qty=1.00 sl=851.75 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-19 11:15:00 | 823.90 | 887.83 | 887.88 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 918.10 | 848.19 | 855.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 916.10 | 848.87 | 856.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 907.40 | 853.21 | 858.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 910.10 | 853.78 | 858.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 913.70 | 862.72 | 862.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 913.70 | 862.72 | 862.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-05 12:15:00)

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
| BUY | retest1 | 2025-07-21 14:15:00 | 859.65 | 2025-07-25 09:15:00 | 840.86 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest1 | 2025-07-24 09:15:00 | 855.55 | 2025-07-25 09:15:00 | 840.86 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-01-27 09:15:00 | 862.27 | STOP_HIT | 1.00 | 2.65% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-01-27 09:15:00 | 862.27 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-01-27 09:15:00 | 862.27 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-01-27 09:15:00 | 862.27 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2026-01-27 15:15:00 | 894.75 | 2026-03-12 09:15:00 | 851.75 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2026-01-28 12:15:00 | 895.15 | 2026-03-12 09:15:00 | 851.75 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2026-02-02 14:15:00 | 910.95 | 2026-03-12 09:15:00 | 851.75 | STOP_HIT | 1.00 | -6.50% |
| BUY | retest2 | 2026-03-10 10:15:00 | 898.55 | 2026-03-12 09:15:00 | 851.75 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest2 | 2026-04-29 10:15:00 | 916.10 | 2026-05-05 12:15:00 | 913.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-30 11:15:00 | 910.10 | 2026-05-05 12:15:00 | 913.70 | STOP_HIT | 1.00 | 0.40% |

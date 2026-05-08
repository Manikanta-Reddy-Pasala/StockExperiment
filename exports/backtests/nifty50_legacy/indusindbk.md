# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 950.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 10 |
| PENDING | 40 |
| PENDING_CANCEL | 14 |
| ENTRY1 | 14 |
| ENTRY2 | 12 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 20
- **Target hits / Stop hits / Partials:** 1 / 25 / 5
- **Avg / median % per leg:** 2.08% / -2.31%
- **Sum % (uncompounded):** 64.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 8 | 57.1% | 0 | 10 | 4 | 4.71% | 65.9% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 0 | 6 | 4 | 9.37% | 93.7% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.93% | -27.7% |
| SELL (all) | 17 | 3 | 17.6% | 1 | 15 | 1 | -0.08% | -1.4% |
| SELL @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 8 | 0 | -2.75% | -22.0% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 1 | 7 | 1 | 2.29% | 20.6% |
| retest1 (combined) | 18 | 9 | 50.0% | 0 | 14 | 4 | 3.98% | 71.7% |
| retest2 (combined) | 13 | 2 | 15.4% | 1 | 11 | 1 | -0.55% | -7.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 14:15:00 | 1475.15 | 1532.12 | 1532.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 09:15:00 | 1469.70 | 1531.00 | 1531.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 14:15:00 | 1521.95 | 1520.52 | 1525.89 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-02-22 10:15:00 | 1486.15 | 1520.01 | 1525.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 11:15:00 | 1474.50 | 1519.56 | 1525.30 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 1516.20 | 1504.32 | 1515.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-01 13:15:00 | 1516.20 | 1504.32 | 1515.70 | SL hit (close>ema400) qty=1.00 sl=1515.70 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-13 12:15:00 | 1508.00 | 1518.58 | 1521.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 13:15:00 | 1501.35 | 1518.41 | 1521.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-22 15:15:00 | 1518.00 | 1501.17 | 1510.81 | SL hit (close>static) qty=1.00 sl=1516.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-26 10:15:00 | 1505.30 | 1501.32 | 1510.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-26 11:15:00 | 1508.85 | 1501.39 | 1510.78 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1553.75 | 1518.07 | 1518.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1558.50 | 1518.81 | 1518.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.27 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 13:15:00 | 1471.05 | 1519.50 | 1519.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 1469.00 | 1515.48 | 1517.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 12:15:00 | 1507.90 | 1506.41 | 1512.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 13:15:00 | 1531.20 | 1506.66 | 1512.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 1531.20 | 1506.66 | 1512.48 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-02 11:15:00 | 1502.50 | 1506.83 | 1512.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 12:15:00 | 1503.30 | 1506.80 | 1512.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-03 11:15:00 | 1487.10 | 1506.51 | 1512.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:15:00 | 1479.05 | 1506.24 | 1511.91 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-04 09:15:00 | 1471.00 | 1460.01 | 1476.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 1390.20 | 1459.32 | 1475.79 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-14 13:15:00 | 1499.15 | 1469.42 | 1476.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-14 14:15:00 | 1504.90 | 1469.77 | 1477.12 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-14 15:15:00 | 1502.35 | 1470.09 | 1477.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-18 09:15:00 | 1504.10 | 1470.43 | 1477.38 | ENTRY2 sustain failed after 5400m |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1542.90 | 1473.39 | 1478.64 | SL hit (close>static) qty=1.00 sl=1531.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1542.90 | 1473.39 | 1478.64 | SL hit (close>static) qty=1.00 sl=1531.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1542.90 | 1473.39 | 1478.64 | SL hit (close>static) qty=1.00 sl=1531.20 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 1527.15 | 1483.64 | 1483.53 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 1456.00 | 1483.60 | 1483.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 1440.55 | 1483.17 | 1483.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 1458.80 | 1457.60 | 1467.89 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-07-18 15:15:00 | 1452.00 | 1457.55 | 1467.81 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1441.40 | 1457.39 | 1467.68 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 1419.00 | 1391.63 | 1415.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 1419.00 | 1391.63 | 1415.39 | SL hit (close>ema400) qty=1.00 sl=1415.39 alert=retest1 |
| Cross detected — sustain check pending | 2024-09-06 09:15:00 | 1407.65 | 1406.06 | 1418.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-06 10:15:00 | 1413.40 | 1406.13 | 1418.39 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-09 09:15:00 | 1409.00 | 1406.40 | 1418.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-09 10:15:00 | 1418.40 | 1406.52 | 1418.17 | ENTRY2 sustain failed after 60m |

### Cycle 6 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1491.00 | 1426.86 | 1426.54 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1349.20 | 1428.84 | 1429.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 1342.05 | 1419.65 | 1424.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 996.00 | 992.78 | 1069.27 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-06 10:15:00 | 978.50 | 993.26 | 1066.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 974.90 | 993.08 | 1065.70 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 12:15:00 | 989.95 | 991.91 | 1062.25 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-07 13:15:00 | 992.00 | 991.91 | 1061.90 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-07 14:15:00 | 982.55 | 991.82 | 1061.50 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 15:15:00 | 975.05 | 991.65 | 1061.07 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-31 13:15:00 | 985.20 | 969.12 | 1014.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 14:15:00 | 989.75 | 969.33 | 1014.58 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1019.55 | 970.46 | 1014.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1019.55 | 970.46 | 1014.48 | SL hit (close>ema400) qty=1.00 sl=1014.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1019.55 | 970.46 | 1014.48 | SL hit (close>ema400) qty=1.00 sl=1014.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1019.55 | 970.46 | 1014.48 | SL hit (close>ema400) qty=1.00 sl=1014.48 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-28 09:15:00 | 997.65 | 1021.89 | 1028.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 10:15:00 | 991.00 | 1021.58 | 1027.83 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 842.35 | 995.14 | 1012.17 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-11 11:15:00 | 693.70 | 989.13 | 1008.98 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.18 | 825.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.14 | 839.77 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-21 13:15:00 | 852.95 | 850.07 | 839.89 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 14:15:00 | 859.65 | 850.16 | 839.99 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 13:15:00 | 851.70 | 849.80 | 840.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-23 14:15:00 | 849.95 | 849.81 | 840.49 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-23 15:15:00 | 851.15 | 849.82 | 840.54 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 855.55 | 849.88 | 840.62 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.73 | 840.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 835.75 | 849.73 | 840.86 | SL hit (close<ema400) qty=1.00 sl=840.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 835.75 | 849.73 | 840.86 | SL hit (close<ema400) qty=1.00 sl=840.86 alert=retest1 |

### Cycle 9 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 796.10 | 833.76 | 833.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.54 | 828.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.31 | 770.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 765.60 | 750.16 | 767.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 765.60 | 750.16 | 767.01 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-13 15:15:00 | 758.20 | 750.78 | 766.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 753.45 | 750.81 | 766.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 768.20 | 749.67 | 763.98 | SL hit (close>static) qty=1.00 sl=767.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-20 15:15:00 | 758.90 | 750.29 | 763.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 13:15:00 | 758.75 | 750.37 | 763.92 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1320m) |
| Cross detected — sustain check pending | 2025-10-23 15:15:00 | 758.45 | 751.17 | 763.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 754.80 | 751.21 | 763.75 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 769.65 | 752.07 | 763.46 | SL hit (close>static) qty=1.00 sl=767.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 769.65 | 752.07 | 763.46 | SL hit (close>static) qty=1.00 sl=767.95 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.99 | 771.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 819.85 | 773.01 | 772.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 830.35 | 833.02 | 813.71 | EMA200 retest candle locked (from upside) |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.23 | 862.27 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-27 14:15:00 | 894.60 | 888.21 | 862.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 894.75 | 888.28 | 863.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-28 11:15:00 | 894.00 | 888.30 | 863.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 895.15 | 888.37 | 863.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 11:15:00 | 891.85 | 889.93 | 866.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-02 12:15:00 | 888.45 | 889.92 | 866.89 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 893.25 | 889.95 | 867.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:15:00 | 910.95 | 890.16 | 867.24 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 14:15:00 | 964.22 | 918.19 | 895.65 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 14:15:00 | 964.74 | 918.19 | 895.65 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 966.00 | 918.66 | 896.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 967.84 | 918.66 | 896.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 921.10 | 922.84 | 899.82 | SL hit (close<ema200) qty=0.50 sl=922.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 921.10 | 922.84 | 899.82 | SL hit (close<ema200) qty=0.50 sl=922.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 921.10 | 922.84 | 899.82 | SL hit (close<ema200) qty=0.50 sl=922.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 921.10 | 922.84 | 899.82 | SL hit (close<ema200) qty=0.50 sl=922.84 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-10 09:15:00 | 890.05 | 920.96 | 901.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:15:00 | 898.55 | 920.74 | 901.90 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 897.60 | 920.51 | 901.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.49 | 900.91 | SL hit (close<static) qty=1.00 sl=851.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.49 | 900.91 | SL hit (close<static) qty=1.00 sl=851.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.49 | 900.91 | SL hit (close<static) qty=1.00 sl=851.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.49 | 900.91 | SL hit (close<static) qty=1.00 sl=851.75 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 823.90 | 887.83 | 887.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 817.35 | 886.51 | 887.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.75 | 835.30 | 856.98 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 12:15:00 | 831.40 | 835.28 | 856.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 13:15:00 | 830.40 | 835.23 | 856.62 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 826.10 | 835.14 | 856.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 828.00 | 835.07 | 856.12 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 14:15:00 | 830.85 | 833.95 | 854.42 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 15:15:00 | 830.00 | 833.92 | 854.29 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.28 | 852.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.28 | 852.47 | SL hit (close>ema400) qty=1.00 sl=852.47 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.28 | 852.47 | SL hit (close>ema400) qty=1.00 sl=852.47 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.28 | 852.47 | SL hit (close>ema400) qty=1.00 sl=852.47 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-17 09:15:00 | 841.80 | 834.22 | 852.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-17 10:15:00 | 843.80 | 834.31 | 852.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 840.50 | 841.12 | 853.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-24 14:15:00 | 848.85 | 841.20 | 853.11 | ENTRY2 sustain failed after 60m |

### Cycle 12 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 913.70 | 862.72 | 862.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 925.35 | 864.77 | 863.69 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-02-22 11:15:00 | 1474.50 | 2024-03-01 13:15:00 | 1516.20 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-03-13 13:15:00 | 1501.35 | 2024-03-22 15:15:00 | 1518.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-02 12:15:00 | 1503.30 | 2024-06-19 09:15:00 | 1542.90 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-05-03 12:15:00 | 1479.05 | 2024-06-19 09:15:00 | 1542.90 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2024-06-04 10:15:00 | 1390.20 | 2024-06-19 09:15:00 | 1542.90 | STOP_HIT | 1.00 | -10.98% |
| SELL | retest1 | 2024-07-19 09:15:00 | 1441.40 | 2024-08-28 11:15:00 | 1419.00 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest1 | 2025-01-06 11:15:00 | 974.90 | 2025-02-03 10:15:00 | 1019.55 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest1 | 2025-01-07 15:15:00 | 975.05 | 2025-02-03 10:15:00 | 1019.55 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest1 | 2025-01-31 14:15:00 | 989.75 | 2025-02-03 10:15:00 | 1019.55 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-02-28 10:15:00 | 991.00 | 2025-03-11 09:15:00 | 842.35 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-28 10:15:00 | 991.00 | 2025-03-11 11:15:00 | 693.70 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest1 | 2025-07-21 14:15:00 | 859.65 | 2025-07-25 09:15:00 | 835.75 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-07-24 09:15:00 | 855.55 | 2025-07-25 09:15:00 | 835.75 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-10-14 09:15:00 | 753.45 | 2025-10-20 10:15:00 | 768.20 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-10-21 13:15:00 | 758.75 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-10-24 09:15:00 | 754.80 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-02-26 14:15:00 | 964.22 | PARTIAL | 0.50 | 14.79% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-02-26 14:15:00 | 964.74 | PARTIAL | 0.50 | 15.06% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-02-26 15:15:00 | 966.00 | PARTIAL | 0.50 | 15.15% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-02-26 15:15:00 | 967.84 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-03-04 09:15:00 | 921.10 | STOP_HIT | 0.50 | 9.65% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-03-04 09:15:00 | 921.10 | STOP_HIT | 0.50 | 9.86% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-03-04 09:15:00 | 921.10 | STOP_HIT | 0.50 | 9.80% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-03-04 09:15:00 | 921.10 | STOP_HIT | 0.50 | 9.45% |
| BUY | retest2 | 2026-01-27 15:15:00 | 894.75 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.40% |
| BUY | retest2 | 2026-01-28 12:15:00 | 895.15 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest2 | 2026-02-02 14:15:00 | 910.95 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -8.07% |
| BUY | retest2 | 2026-03-10 10:15:00 | 898.55 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.80% |
| SELL | retest1 | 2026-04-08 13:15:00 | 830.40 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest1 | 2026-04-09 10:15:00 | 828.00 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest1 | 2026-04-10 15:15:00 | 830.00 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -2.79% |

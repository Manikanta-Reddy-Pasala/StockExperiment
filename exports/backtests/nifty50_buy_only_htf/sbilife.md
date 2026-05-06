# SBILIFE (SBILIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1859.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 9 |
| PENDING | 32 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 21
- **Target hits / Stop hits / Partials:** 0 / 24 / 0
- **Avg / median % per leg:** -1.36% / -1.50%
- **Sum % (uncompounded):** -32.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 7 | 0 | -0.12% | -0.8% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.70% | 5.1% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.48% | -5.9% |
| SELL (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -1.87% | -31.7% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.49% | -5.0% |
| SELL @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.78% | -26.7% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.02% | 0.1% |
| retest2 (combined) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.72% | -32.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 1460.90 | 1468.30 | 1468.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1453.75 | 1468.08 | 1468.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 1453.15 | 1451.71 | 1458.76 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-05-21 09:15:00 | 1428.05 | 1450.59 | 1457.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 10:15:00 | 1427.50 | 1450.36 | 1457.73 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-27 13:15:00 | 1431.90 | 1446.61 | 1454.62 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 14:15:00 | 1409.20 | 1446.23 | 1454.40 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1452.00 | 1445.30 | 1453.59 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 1453.59 | 1445.30 | 1453.59 | SL hit qty=1.00 sl=1453.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 1453.59 | 1445.30 | 1453.59 | SL hit qty=1.00 sl=1453.59 alert=retest1 |
| Cross detected — sustain check pending | 2024-05-29 09:15:00 | 1420.90 | 1445.05 | 1453.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 1417.05 | 1444.77 | 1453.25 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-12 12:15:00 | 1454.95 | 1426.48 | 1439.33 | SL hit qty=1.00 sl=1454.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-13 10:15:00 | 1443.50 | 1427.70 | 1439.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-13 11:15:00 | 1449.95 | 1427.92 | 1439.68 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-13 12:15:00 | 1444.95 | 1428.09 | 1439.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 13:15:00 | 1438.35 | 1428.19 | 1439.70 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-14 10:15:00 | 1454.95 | 1429.14 | 1439.95 | SL hit qty=1.00 sl=1454.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-19 14:15:00 | 1448.35 | 1434.76 | 1441.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-19 15:15:00 | 1450.00 | 1434.91 | 1442.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-25 09:15:00 | 1446.80 | 1439.33 | 1443.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 1433.50 | 1439.28 | 1443.55 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1454.95 | 1439.38 | 1443.56 | SL hit qty=1.00 sl=1454.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1504.35 | 1447.54 | 1447.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 1511.45 | 1455.71 | 1451.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 1836.05 | 1838.72 | 1768.58 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 1760.40 | 1830.77 | 1772.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1760.40 | 1830.77 | 1772.98 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 13:15:00 | 1604.45 | 1743.43 | 1743.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1593.40 | 1717.71 | 1730.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 1448.85 | 1446.49 | 1514.77 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 14:15:00 | 1429.95 | 1445.99 | 1511.18 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-06 15:15:00 | 1434.65 | 1445.87 | 1510.80 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 09:15:00 | 1472.40 | 1455.85 | 1503.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1472.40 | 1455.85 | 1503.67 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-21 14:15:00 | 1466.75 | 1468.69 | 1503.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:15:00 | 1465.00 | 1468.66 | 1503.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-30 11:15:00 | 1459.15 | 1459.93 | 1491.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:15:00 | 1465.15 | 1459.98 | 1491.28 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 1436.55 | 1461.44 | 1490.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1461.50 | 1461.44 | 1490.21 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-05 15:15:00 | 1469.15 | 1462.46 | 1488.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 1465.15 | 1462.49 | 1488.01 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1473.10 | 1459.76 | 1480.82 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-18 09:15:00 | 1463.70 | 1460.59 | 1480.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1464.80 | 1460.63 | 1480.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 1481.50 | 1461.46 | 1480.37 | SL hit qty=1.00 sl=1481.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 12:15:00 | 1467.65 | 1462.75 | 1480.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 13:15:00 | 1469.00 | 1462.81 | 1480.06 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-21 12:15:00 | 1481.50 | 1463.36 | 1479.82 | SL hit qty=1.00 sl=1481.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-25 10:15:00 | 1469.15 | 1465.84 | 1480.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 1467.00 | 1465.85 | 1480.08 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-27 09:15:00 | 1469.45 | 1466.04 | 1479.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 10:15:00 | 1466.90 | 1466.05 | 1479.76 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1471.00 | 1440.19 | 1458.93 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 1481.50 | 1440.88 | 1459.09 | SL hit qty=1.00 sl=1481.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 1481.50 | 1440.88 | 1459.09 | SL hit qty=1.00 sl=1481.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 1504.25 | 1443.65 | 1459.96 | SL hit qty=1.00 sl=1504.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 1504.25 | 1443.65 | 1459.96 | SL hit qty=1.00 sl=1504.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 1504.25 | 1443.65 | 1459.96 | SL hit qty=1.00 sl=1504.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 1504.25 | 1443.65 | 1459.96 | SL hit qty=1.00 sl=1504.25 alert=retest2 |
| CROSSOVER_SKIP | 2025-03-27 10:15:00 | 1543.25 | 1473.88 | 1473.68 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-04-07 12:15:00 | 1449.10 | 1495.10 | 1485.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 13:15:00 | 1449.75 | 1494.65 | 1485.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 1472.00 | 1493.66 | 1485.19 | SL hit qty=1.00 sl=1472.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1784.10 | 1809.06 | 1809.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1777.90 | 1808.75 | 1809.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.40 | 1805.63 | 1807.38 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 1808.40 | 1805.63 | 1807.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1808.40 | 1805.63 | 1807.38 | EMA400 retest candle locked |

### Cycle 5 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.14 | 1815.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.26 | 1915.85 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-01 15:15:00 | 1974.00 | 1964.25 | 1917.74 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 09:15:00 | 1960.80 | 1964.22 | 1917.95 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-12-02 13:15:00 | 1975.40 | 1964.32 | 1918.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1980.30 | 1964.48 | 1919.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 1976.80 | 1965.01 | 1920.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 12:15:00 | 1974.70 | 1965.10 | 1920.66 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 14:15:00 | 1972.90 | 1965.20 | 1921.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 15:15:00 | 1974.00 | 1965.29 | 1921.41 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.38 | 2009.85 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 2009.85 | 2046.38 | 2009.85 | SL hit qty=1.00 sl=2009.85 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 2009.85 | 2046.38 | 2009.85 | SL hit qty=1.00 sl=2009.85 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 2009.85 | 2046.38 | 2009.85 | SL hit qty=1.00 sl=2009.85 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 2030.30 | 2043.03 | 2009.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-27 10:15:00 | 2025.20 | 2042.85 | 2009.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 2028.90 | 2042.71 | 2009.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 2034.00 | 2042.63 | 2010.04 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 2009.70 | 2043.00 | 2011.97 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 2051.70 | 2033.71 | 2010.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2050.30 | 2033.87 | 2010.35 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2009.70 | 2033.71 | 2010.62 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-04 12:15:00 | 2041.40 | 2032.85 | 2010.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 2042.50 | 2032.95 | 2011.02 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 2009.70 | 2032.55 | 2011.47 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-09 10:15:00 | 2033.40 | 2029.25 | 2011.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 11:15:00 | 2025.40 | 2029.21 | 2011.07 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-11 13:15:00 | 2028.60 | 2027.62 | 2011.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-11 14:15:00 | 2026.40 | 2027.61 | 2011.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2031.30 | 2026.79 | 2011.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-13 10:15:00 | 2020.70 | 2026.73 | 2012.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 13:15:00 | 2031.90 | 2026.65 | 2012.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 2032.60 | 2026.71 | 2012.31 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2011.90 | 2047.42 | 2028.63 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2009.70 | 2047.42 | 2028.63 | SL hit qty=1.00 sl=2009.70 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.70 | 2013.66 | 2013.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.93 | 2013.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1913.90 | 1898.43 | 1942.72 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1942.30 | 1900.06 | 1940.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1942.30 | 1900.06 | 1940.77 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1909.70 | 1902.06 | 1940.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1919.00 | 1902.22 | 1940.29 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 13:15:00 | 1918.90 | 1902.88 | 1940.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:15:00 | 1914.90 | 1903.00 | 1939.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1946.30 | 1903.68 | 1939.90 | SL hit qty=1.00 sl=1946.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1946.30 | 1903.68 | 1939.90 | SL hit qty=1.00 sl=1946.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-21 09:15:00 | 1900.00 | 1917.62 | 1943.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 1910.00 | 1917.54 | 1943.03 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 1946.30 | 1917.36 | 1942.82 | SL hit qty=1.00 sl=1946.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-21 10:15:00 | 1427.50 | 2024-05-28 15:15:00 | 1453.59 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest1 | 2024-05-27 14:15:00 | 1409.20 | 2024-05-28 15:15:00 | 1453.59 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-05-29 10:15:00 | 1417.05 | 2024-06-12 12:15:00 | 1454.95 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-06-13 13:15:00 | 1438.35 | 2024-06-14 10:15:00 | 1454.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-06-25 10:15:00 | 1433.50 | 2024-06-25 12:15:00 | 1454.95 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-01-21 15:15:00 | 1465.00 | 2025-02-19 09:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-01-30 12:15:00 | 1465.15 | 2025-02-21 12:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1461.50 | 2025-03-19 11:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1465.15 | 2025-03-19 11:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1464.80 | 2025-03-20 10:15:00 | 1504.25 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-02-20 13:15:00 | 1469.00 | 2025-03-20 10:15:00 | 1504.25 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-02-25 11:15:00 | 1467.00 | 2025-03-20 10:15:00 | 1504.25 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-02-27 10:15:00 | 1466.90 | 2025-03-20 10:15:00 | 1504.25 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-04-07 13:15:00 | 1449.75 | 2025-04-08 09:15:00 | 1472.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2025-12-02 14:15:00 | 1980.30 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest1 | 2025-12-03 12:15:00 | 1974.70 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest1 | 2025-12-03 15:15:00 | 1974.00 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-27 12:15:00 | 2034.00 | 2026-01-29 09:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2050.30 | 2026-02-03 13:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2042.50 | 2026-02-05 12:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2032.60 | 2026-03-02 09:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1919.00 | 2026-04-15 09:15:00 | 1946.30 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-04-13 14:15:00 | 1914.90 | 2026-04-15 09:15:00 | 1946.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1910.00 | 2026-04-21 11:15:00 | 1946.30 | STOP_HIT | 1.00 | -1.90% |

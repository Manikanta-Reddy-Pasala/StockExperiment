# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-11-06 15:25:00 (9183 bars)
- **Last close:** 1511.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 4
- **Avg / median % per leg:** 0.34% / 0.00%
- **Sum % (uncompounded):** 4.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.37% | 3.7% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.37% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.20% | 0.6% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.20% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.34% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:15:00 | 1714.03 | 1706.31 | 0.00 | ORB-long ORB[1690.03,1714.00] vol=4.9x ATR=7.45 |
| Stop hit — per-position SL triggered | 2024-06-12 10:25:00 | 1706.58 | 1706.50 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 1724.10 | 1716.68 | 0.00 | ORB-long ORB[1703.28,1718.58] vol=2.4x ATR=7.59 |
| Stop hit — per-position SL triggered | 2024-06-14 09:40:00 | 1716.51 | 1716.13 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 1811.90 | 1809.02 | 0.00 | ORB-long ORB[1783.65,1807.05] vol=13.9x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:45:00 | 1821.75 | 1809.38 | 0.00 | T1 1.5R @ 1821.75 |
| Stop hit — per-position SL triggered | 2024-07-03 12:30:00 | 1811.90 | 1813.54 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:35:00 | 1893.48 | 1879.24 | 0.00 | ORB-long ORB[1862.55,1884.48] vol=5.3x ATR=9.42 |
| Stop hit — per-position SL triggered | 2024-07-08 10:40:00 | 1884.06 | 1879.83 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-08-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:50:00 | 1690.65 | 1675.89 | 0.00 | ORB-long ORB[1656.83,1675.00] vol=4.5x ATR=9.06 |
| Stop hit — per-position SL triggered | 2024-08-09 10:55:00 | 1681.59 | 1676.48 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 1760.45 | 1771.27 | 0.00 | ORB-short ORB[1769.03,1792.45] vol=1.5x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:10:00 | 1748.11 | 1766.39 | 0.00 | T1 1.5R @ 1748.11 |
| Target hit | 2024-08-21 12:35:00 | 1754.95 | 1754.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2024-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:00:00 | 1773.95 | 1759.82 | 0.00 | ORB-long ORB[1751.03,1769.73] vol=2.7x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 11:20:00 | 1783.05 | 1765.27 | 0.00 | T1 1.5R @ 1783.05 |
| Target hit | 2024-08-22 15:20:00 | 1846.00 | 1817.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-09-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:10:00 | 1822.08 | 1844.71 | 0.00 | ORB-short ORB[1842.53,1868.13] vol=1.9x ATR=7.29 |
| Stop hit — per-position SL triggered | 2024-09-03 10:45:00 | 1829.37 | 1835.37 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:55:00 | 1812.35 | 1795.89 | 0.00 | ORB-long ORB[1780.00,1806.95] vol=3.1x ATR=6.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:35:00 | 1822.11 | 1802.84 | 0.00 | T1 1.5R @ 1822.11 |
| Stop hit — per-position SL triggered | 2024-09-26 12:05:00 | 1812.35 | 1805.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-12 10:15:00 | 1714.03 | 2024-06-12 10:25:00 | 1706.58 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-06-14 09:30:00 | 1724.10 | 2024-06-14 09:40:00 | 1716.51 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1811.90 | 2024-07-03 09:45:00 | 1821.75 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1811.90 | 2024-07-03 12:30:00 | 1811.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 10:35:00 | 1893.48 | 2024-07-08 10:40:00 | 1884.06 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-08-09 10:50:00 | 1690.65 | 2024-08-09 10:55:00 | 1681.59 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-08-21 09:45:00 | 1760.45 | 2024-08-21 10:10:00 | 1748.11 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-08-21 09:45:00 | 1760.45 | 2024-08-21 12:35:00 | 1754.95 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-22 11:00:00 | 1773.95 | 2024-08-22 11:20:00 | 1783.05 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-22 11:00:00 | 1773.95 | 2024-08-22 15:20:00 | 1846.00 | TARGET_HIT | 0.50 | 4.06% |
| SELL | retest1 | 2024-09-03 10:10:00 | 1822.08 | 2024-09-03 10:45:00 | 1829.37 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-26 10:55:00 | 1812.35 | 2024-09-26 11:35:00 | 1822.11 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-26 10:55:00 | 1812.35 | 2024-09-26 12:05:00 | 1812.35 | STOP_HIT | 0.50 | 0.00% |

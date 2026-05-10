# Sun Pharmaceutical Industries Ltd. (SUNPHARMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1845.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 3
- **Avg / median % per leg:** -0.06% / -0.17%
- **Sum % (uncompounded):** -0.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.13% | -0.9% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.13% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 4 | 26.7% | 1 | 11 | 3 | -0.06% | -0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 1712.00 | 1705.21 | 0.00 | ORB-long ORB[1696.00,1709.70] vol=1.7x ATR=6.50 |
| Stop hit — per-position SL triggered | 2026-02-09 11:45:00 | 1705.50 | 1708.10 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:10:00 | 1726.30 | 1727.22 | 0.00 | ORB-short ORB[1726.50,1733.00] vol=1.8x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-02-24 10:30:00 | 1728.63 | 1727.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 1816.60 | 1821.63 | 0.00 | ORB-short ORB[1820.90,1830.90] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 1820.82 | 1821.34 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 1782.60 | 1794.21 | 0.00 | ORB-short ORB[1798.30,1818.00] vol=2.7x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 11:35:00 | 1775.67 | 1792.64 | 0.00 | T1 1.5R @ 1775.67 |
| Stop hit — per-position SL triggered | 2026-03-16 14:25:00 | 1782.60 | 1783.99 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 1758.70 | 1774.63 | 0.00 | ORB-short ORB[1762.00,1787.20] vol=2.9x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-04-01 11:05:00 | 1763.69 | 1774.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:45:00 | 1675.80 | 1669.18 | 0.00 | ORB-long ORB[1657.70,1675.00] vol=1.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-04-15 11:00:00 | 1672.87 | 1669.58 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 1679.30 | 1688.13 | 0.00 | ORB-short ORB[1682.10,1696.90] vol=1.6x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:30:00 | 1674.24 | 1684.33 | 0.00 | T1 1.5R @ 1674.24 |
| Target hit | 2026-04-17 15:20:00 | 1676.90 | 1678.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 1654.60 | 1660.02 | 0.00 | ORB-short ORB[1657.20,1672.00] vol=2.0x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 1657.44 | 1659.94 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 1693.80 | 1680.44 | 0.00 | ORB-long ORB[1661.20,1676.10] vol=1.7x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-04-23 09:55:00 | 1689.24 | 1684.91 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 1736.00 | 1729.68 | 0.00 | ORB-long ORB[1714.30,1735.50] vol=3.4x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:45:00 | 1742.72 | 1731.34 | 0.00 | T1 1.5R @ 1742.72 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 1736.00 | 1731.75 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 1844.30 | 1830.19 | 0.00 | ORB-long ORB[1818.40,1837.00] vol=3.3x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 1839.25 | 1833.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 1849.70 | 1841.48 | 0.00 | ORB-long ORB[1819.20,1844.10] vol=2.6x ATR=3.91 |
| Stop hit — per-position SL triggered | 2026-05-08 12:00:00 | 1845.79 | 1843.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 1712.00 | 2026-02-09 11:45:00 | 1705.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-24 10:10:00 | 1726.30 | 2026-02-24 10:30:00 | 1728.63 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2026-03-13 10:25:00 | 1816.60 | 2026-03-13 10:50:00 | 1820.82 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-16 11:15:00 | 1782.60 | 2026-03-16 11:35:00 | 1775.67 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-16 11:15:00 | 1782.60 | 2026-03-16 14:25:00 | 1782.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-01 11:00:00 | 1758.70 | 2026-04-01 11:05:00 | 1763.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 10:45:00 | 1675.80 | 2026-04-15 11:00:00 | 1672.87 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-04-17 11:05:00 | 1679.30 | 2026-04-17 12:30:00 | 1674.24 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-17 11:05:00 | 1679.30 | 2026-04-17 15:20:00 | 1676.90 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-04-22 10:55:00 | 1654.60 | 2026-04-22 11:00:00 | 1657.44 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-04-23 09:40:00 | 1693.80 | 2026-04-23 09:55:00 | 1689.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-28 09:40:00 | 1736.00 | 2026-04-28 09:45:00 | 1742.72 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-28 09:40:00 | 1736.00 | 2026-04-28 09:55:00 | 1736.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:55:00 | 1844.30 | 2026-05-06 11:05:00 | 1839.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-08 11:05:00 | 1849.70 | 2026-05-08 12:00:00 | 1845.79 | STOP_HIT | 1.00 | -0.21% |

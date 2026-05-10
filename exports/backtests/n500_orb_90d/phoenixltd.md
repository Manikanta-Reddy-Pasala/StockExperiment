# Phoenix Mills Ltd. (PHOENIXLTD)

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
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 3
- **Avg / median % per leg:** -0.00% / -0.26%
- **Sum % (uncompounded):** -0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 3 | 4 | 2 | 0.08% | 0.8% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 3 | 4 | 2 | 0.08% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 6 | 40.0% | 3 | 9 | 3 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 1748.90 | 1745.77 | 0.00 | ORB-long ORB[1725.10,1747.90] vol=5.4x ATR=10.04 |
| Target hit | 2026-02-09 15:20:00 | 1750.00 | 1749.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:50:00 | 1800.90 | 1770.53 | 0.00 | ORB-long ORB[1743.80,1768.20] vol=2.9x ATR=6.00 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 1794.90 | 1777.99 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:50:00 | 1747.70 | 1756.68 | 0.00 | ORB-short ORB[1755.00,1772.50] vol=2.5x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:00:00 | 1739.88 | 1754.22 | 0.00 | T1 1.5R @ 1739.88 |
| Stop hit — per-position SL triggered | 2026-02-13 11:40:00 | 1747.70 | 1745.48 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:10:00 | 1746.00 | 1740.36 | 0.00 | ORB-long ORB[1720.00,1744.80] vol=2.6x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 1741.44 | 1740.56 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 1741.30 | 1753.51 | 0.00 | ORB-short ORB[1755.00,1775.00] vol=2.0x ATR=5.36 |
| Stop hit — per-position SL triggered | 2026-02-18 10:05:00 | 1746.66 | 1751.26 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1737.20 | 1750.61 | 0.00 | ORB-short ORB[1753.90,1768.20] vol=1.9x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-02-19 12:00:00 | 1742.00 | 1743.98 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1694.20 | 1694.44 | 0.00 | ORB-short ORB[1697.50,1710.70] vol=3.3x ATR=6.23 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 1700.43 | 1694.51 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 1603.80 | 1615.13 | 0.00 | ORB-short ORB[1604.10,1626.00] vol=1.9x ATR=4.12 |
| Stop hit — per-position SL triggered | 2026-03-06 11:30:00 | 1607.92 | 1614.11 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 1564.20 | 1557.71 | 0.00 | ORB-long ORB[1533.90,1552.30] vol=1.6x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 1557.68 | 1558.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 1744.10 | 1729.42 | 0.00 | ORB-long ORB[1712.60,1735.30] vol=1.6x ATR=9.43 |
| Stop hit — per-position SL triggered | 2026-04-10 09:45:00 | 1734.67 | 1732.89 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 1787.00 | 1779.75 | 0.00 | ORB-long ORB[1768.90,1784.20] vol=1.8x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 1796.02 | 1784.06 | 0.00 | T1 1.5R @ 1796.02 |
| Target hit | 2026-04-29 14:10:00 | 1807.00 | 1812.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 1802.20 | 1791.47 | 0.00 | ORB-long ORB[1782.70,1798.00] vol=2.1x ATR=6.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:45:00 | 1811.27 | 1799.23 | 0.00 | T1 1.5R @ 1811.27 |
| Target hit | 2026-05-06 13:00:00 | 1804.40 | 1804.98 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 1748.90 | 2026-02-09 15:20:00 | 1750.00 | TARGET_HIT | 1.00 | 0.06% |
| BUY | retest1 | 2026-02-11 10:50:00 | 1800.90 | 2026-02-11 11:30:00 | 1794.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-13 09:50:00 | 1747.70 | 2026-02-13 10:00:00 | 1739.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-13 09:50:00 | 1747.70 | 2026-02-13 11:40:00 | 1747.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 11:10:00 | 1746.00 | 2026-02-16 11:25:00 | 1741.44 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1741.30 | 2026-02-18 10:05:00 | 1746.66 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1737.20 | 2026-02-19 12:00:00 | 1742.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1694.20 | 2026-02-27 10:35:00 | 1700.43 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1603.80 | 2026-03-06 11:30:00 | 1607.92 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-17 10:35:00 | 1564.20 | 2026-03-17 11:25:00 | 1557.68 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-10 09:35:00 | 1744.10 | 2026-04-10 09:45:00 | 1734.67 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-04-29 11:05:00 | 1787.00 | 2026-04-29 11:15:00 | 1796.02 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-29 11:05:00 | 1787.00 | 2026-04-29 14:10:00 | 1807.00 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1802.20 | 2026-05-06 10:45:00 | 1811.27 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1802.20 | 2026-05-06 13:00:00 | 1804.40 | TARGET_HIT | 0.50 | 0.12% |

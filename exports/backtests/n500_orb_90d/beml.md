# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1952.00
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** -0.31% / -0.40%
- **Sum % (uncompounded):** -4.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.43% | -3.4% |
| BUY @ 2nd Alert (retest1) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.43% | -3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.12% | -0.6% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.12% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 1 | 7.7% | 0 | 12 | 1 | -0.31% | -4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 1771.40 | 1759.76 | 0.00 | ORB-long ORB[1746.00,1770.00] vol=5.1x ATR=7.21 |
| Stop hit — per-position SL triggered | 2026-02-11 11:40:00 | 1764.19 | 1763.18 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 1744.90 | 1748.76 | 0.00 | ORB-short ORB[1746.00,1767.40] vol=1.5x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 1751.04 | 1748.75 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1748.50 | 1743.39 | 0.00 | ORB-long ORB[1733.00,1748.10] vol=2.4x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 1743.52 | 1743.88 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1742.10 | 1756.75 | 0.00 | ORB-short ORB[1752.00,1774.40] vol=1.6x ATR=5.38 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 1747.48 | 1755.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:55:00 | 1744.90 | 1720.25 | 0.00 | ORB-long ORB[1701.00,1725.00] vol=2.2x ATR=8.66 |
| Stop hit — per-position SL triggered | 2026-02-20 10:00:00 | 1736.24 | 1721.43 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:10:00 | 1696.00 | 1709.73 | 0.00 | ORB-short ORB[1700.60,1720.80] vol=2.0x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:35:00 | 1688.20 | 1705.85 | 0.00 | T1 1.5R @ 1688.20 |
| Stop hit — per-position SL triggered | 2026-02-23 11:35:00 | 1696.00 | 1702.88 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 1729.80 | 1715.30 | 0.00 | ORB-long ORB[1703.10,1724.30] vol=2.1x ATR=6.93 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 1722.87 | 1716.50 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 1683.00 | 1664.85 | 0.00 | ORB-long ORB[1647.50,1669.90] vol=1.9x ATR=8.58 |
| Stop hit — per-position SL triggered | 2026-04-15 10:20:00 | 1674.42 | 1667.04 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 1772.00 | 1758.99 | 0.00 | ORB-long ORB[1738.70,1764.90] vol=2.0x ATR=8.36 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 1763.64 | 1765.05 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:20:00 | 1773.80 | 1797.63 | 0.00 | ORB-short ORB[1796.80,1823.10] vol=2.1x ATR=7.46 |
| Stop hit — per-position SL triggered | 2026-04-30 10:30:00 | 1781.26 | 1797.28 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 1859.00 | 1844.72 | 0.00 | ORB-long ORB[1826.00,1847.00] vol=4.0x ATR=8.31 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 1850.69 | 1846.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 1890.80 | 1879.93 | 0.00 | ORB-long ORB[1866.50,1882.00] vol=2.5x ATR=7.37 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 1883.43 | 1880.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 11:05:00 | 1771.40 | 2026-02-11 11:40:00 | 1764.19 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-12 10:45:00 | 1744.90 | 2026-02-12 11:00:00 | 1751.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-18 09:45:00 | 1748.50 | 2026-02-18 09:50:00 | 1743.52 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 09:30:00 | 1742.10 | 2026-02-19 09:35:00 | 1747.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-20 09:55:00 | 1744.90 | 2026-02-20 10:00:00 | 1736.24 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-02-23 10:10:00 | 1696.00 | 2026-02-23 10:35:00 | 1688.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-23 10:10:00 | 1696.00 | 2026-02-23 11:35:00 | 1696.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:55:00 | 1729.80 | 2026-02-25 10:00:00 | 1722.87 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-15 10:05:00 | 1683.00 | 2026-04-15 10:20:00 | 1674.42 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1772.00 | 2026-04-21 10:25:00 | 1763.64 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-30 10:20:00 | 1773.80 | 2026-04-30 10:30:00 | 1781.26 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-05 09:35:00 | 1859.00 | 2026-05-05 09:40:00 | 1850.69 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1890.80 | 2026-05-06 09:35:00 | 1883.43 | STOP_HIT | 1.00 | -0.39% |

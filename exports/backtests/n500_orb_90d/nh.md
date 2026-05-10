# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1820.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 3
- **Avg / median % per leg:** 0.05% / -0.25%
- **Sum % (uncompounded):** 0.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.39% | -1.9% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.39% | -1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.29% | 2.6% |
| SELL @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.29% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.05% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1853.20 | 1843.72 | 0.00 | ORB-long ORB[1831.40,1850.00] vol=4.1x ATR=6.32 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 1846.88 | 1845.91 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 1686.60 | 1691.19 | 0.00 | ORB-short ORB[1687.50,1710.00] vol=1.5x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 1678.11 | 1688.45 | 0.00 | T1 1.5R @ 1678.11 |
| Target hit | 2026-03-13 12:45:00 | 1672.10 | 1669.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2026-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:30:00 | 1656.30 | 1643.31 | 0.00 | ORB-long ORB[1634.00,1654.00] vol=1.6x ATR=7.71 |
| Stop hit — per-position SL triggered | 2026-03-24 09:40:00 | 1648.59 | 1644.37 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 1663.90 | 1680.37 | 0.00 | ORB-short ORB[1674.00,1698.60] vol=2.1x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 1654.49 | 1674.95 | 0.00 | T1 1.5R @ 1654.49 |
| Target hit | 2026-03-27 14:10:00 | 1641.80 | 1638.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 1611.50 | 1629.81 | 0.00 | ORB-short ORB[1629.00,1652.00] vol=3.6x ATR=6.51 |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 1618.01 | 1628.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1812.40 | 1794.07 | 0.00 | ORB-long ORB[1773.90,1800.40] vol=2.8x ATR=8.05 |
| Stop hit — per-position SL triggered | 2026-04-15 09:55:00 | 1804.35 | 1797.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:50:00 | 1819.70 | 1809.58 | 0.00 | ORB-long ORB[1798.50,1817.20] vol=1.5x ATR=4.63 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 1815.07 | 1809.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 1793.10 | 1801.62 | 0.00 | ORB-short ORB[1798.20,1815.90] vol=2.2x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:25:00 | 1786.33 | 1798.38 | 0.00 | T1 1.5R @ 1786.33 |
| Stop hit — per-position SL triggered | 2026-04-23 12:35:00 | 1793.10 | 1797.39 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:40:00 | 1780.70 | 1790.22 | 0.00 | ORB-short ORB[1792.10,1810.50] vol=2.0x ATR=5.02 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 1785.72 | 1788.94 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:00:00 | 1749.90 | 1761.49 | 0.00 | ORB-short ORB[1750.30,1767.00] vol=3.4x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 1755.55 | 1759.69 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:50:00 | 1806.50 | 1786.66 | 0.00 | ORB-long ORB[1768.00,1793.00] vol=1.7x ATR=8.01 |
| Stop hit — per-position SL triggered | 2026-05-04 10:30:00 | 1798.49 | 1792.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-19 09:30:00 | 1853.20 | 2026-02-19 09:35:00 | 1846.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1686.60 | 2026-03-13 10:00:00 | 1678.11 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1686.60 | 2026-03-13 12:45:00 | 1672.10 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2026-03-24 09:30:00 | 1656.30 | 2026-03-24 09:40:00 | 1648.59 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-27 11:05:00 | 1663.90 | 2026-03-27 11:15:00 | 1654.49 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-27 11:05:00 | 1663.90 | 2026-03-27 14:10:00 | 1641.80 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-04-01 10:55:00 | 1611.50 | 2026-04-01 11:15:00 | 1618.01 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1812.40 | 2026-04-15 09:55:00 | 1804.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-21 10:50:00 | 1819.70 | 2026-04-21 11:00:00 | 1815.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-23 11:15:00 | 1793.10 | 2026-04-23 12:25:00 | 1786.33 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-23 11:15:00 | 1793.10 | 2026-04-23 12:35:00 | 1793.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 10:40:00 | 1780.70 | 2026-04-28 11:20:00 | 1785.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-30 11:00:00 | 1749.90 | 2026-04-30 11:25:00 | 1755.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-04 09:50:00 | 1806.50 | 2026-05-04 10:30:00 | 1798.49 | STOP_HIT | 1.00 | -0.44% |

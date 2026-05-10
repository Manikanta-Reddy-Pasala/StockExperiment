# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1671.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 8
- **Target hits / Stop hits / Partials:** 5 / 8 / 8
- **Avg / median % per leg:** 0.38% / 0.40%
- **Sum % (uncompounded):** 7.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.20% | 1.4% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.20% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 9 | 64.3% | 3 | 5 | 6 | 0.47% | 6.6% |
| SELL @ 2nd Alert (retest1) | 14 | 9 | 64.3% | 3 | 5 | 6 | 0.47% | 6.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 13 | 61.9% | 5 | 8 | 8 | 0.38% | 8.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 1770.10 | 1760.02 | 0.00 | ORB-long ORB[1751.00,1764.90] vol=2.0x ATR=7.46 |
| Stop hit — per-position SL triggered | 2026-02-09 15:20:00 | 1770.10 | 1768.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1732.90 | 1725.51 | 0.00 | ORB-long ORB[1718.80,1732.00] vol=2.3x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-02-11 12:05:00 | 1729.42 | 1727.39 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 1720.70 | 1708.36 | 0.00 | ORB-long ORB[1701.00,1720.00] vol=1.8x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:10:00 | 1727.64 | 1709.61 | 0.00 | T1 1.5R @ 1727.64 |
| Target hit | 2026-02-12 15:20:00 | 1734.90 | 1726.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:35:00 | 1714.70 | 1714.78 | 0.00 | ORB-short ORB[1718.20,1731.00] vol=1.7x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:00:00 | 1707.81 | 1714.02 | 0.00 | T1 1.5R @ 1707.81 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 1714.70 | 1712.88 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:25:00 | 1701.20 | 1696.18 | 0.00 | ORB-long ORB[1672.70,1690.60] vol=1.9x ATR=4.45 |
| Stop hit — per-position SL triggered | 2026-02-23 10:35:00 | 1696.75 | 1696.52 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:05:00 | 1674.50 | 1680.84 | 0.00 | ORB-short ORB[1675.30,1698.00] vol=2.0x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 1666.33 | 1674.41 | 0.00 | T1 1.5R @ 1666.33 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 1674.50 | 1674.58 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:00:00 | 1748.40 | 1764.13 | 0.00 | ORB-short ORB[1751.00,1770.00] vol=2.3x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:05:00 | 1737.96 | 1751.81 | 0.00 | T1 1.5R @ 1737.96 |
| Target hit | 2026-02-26 10:35:00 | 1702.40 | 1684.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1643.90 | 1653.68 | 0.00 | ORB-short ORB[1650.60,1673.00] vol=1.6x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 14:45:00 | 1634.95 | 1645.10 | 0.00 | T1 1.5R @ 1634.95 |
| Target hit | 2026-03-06 15:20:00 | 1626.10 | 1640.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:10:00 | 1516.30 | 1529.07 | 0.00 | ORB-short ORB[1530.50,1547.30] vol=2.9x ATR=5.58 |
| Stop hit — per-position SL triggered | 2026-03-18 12:35:00 | 1521.88 | 1522.58 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:55:00 | 1399.30 | 1412.97 | 0.00 | ORB-short ORB[1411.60,1429.70] vol=2.1x ATR=6.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:10:00 | 1389.59 | 1410.84 | 0.00 | T1 1.5R @ 1389.59 |
| Target hit | 2026-03-23 14:00:00 | 1395.60 | 1394.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2026-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:45:00 | 1381.70 | 1392.13 | 0.00 | ORB-short ORB[1400.00,1417.80] vol=2.9x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-03-24 10:55:00 | 1388.05 | 1390.97 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:30:00 | 1384.10 | 1396.32 | 0.00 | ORB-short ORB[1395.80,1409.60] vol=1.8x ATR=6.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:00:00 | 1374.37 | 1394.60 | 0.00 | T1 1.5R @ 1374.37 |
| Stop hit — per-position SL triggered | 2026-04-01 11:45:00 | 1384.10 | 1392.33 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1549.10 | 1544.83 | 0.00 | ORB-long ORB[1530.80,1548.80] vol=2.0x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 12:10:00 | 1555.83 | 1546.93 | 0.00 | T1 1.5R @ 1555.83 |
| Target hit | 2026-04-29 14:25:00 | 1551.90 | 1553.92 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 1770.10 | 2026-02-09 15:20:00 | 1770.10 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1732.90 | 2026-02-11 12:05:00 | 1729.42 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-12 10:50:00 | 1720.70 | 2026-02-12 11:10:00 | 1727.64 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-12 10:50:00 | 1720.70 | 2026-02-12 15:20:00 | 1734.90 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2026-02-13 10:35:00 | 1714.70 | 2026-02-13 11:00:00 | 1707.81 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-13 10:35:00 | 1714.70 | 2026-02-13 11:50:00 | 1714.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 10:25:00 | 1701.20 | 2026-02-23 10:35:00 | 1696.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-24 10:05:00 | 1674.50 | 2026-02-24 10:15:00 | 1666.33 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-24 10:05:00 | 1674.50 | 2026-02-24 10:20:00 | 1674.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:00:00 | 1748.40 | 2026-02-26 10:05:00 | 1737.96 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-26 10:00:00 | 1748.40 | 2026-02-26 10:35:00 | 1702.40 | TARGET_HIT | 0.50 | 2.63% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1643.90 | 2026-03-06 14:45:00 | 1634.95 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1643.90 | 2026-03-06 15:20:00 | 1626.10 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2026-03-18 10:10:00 | 1516.30 | 2026-03-18 12:35:00 | 1521.88 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-23 09:55:00 | 1399.30 | 2026-03-23 10:10:00 | 1389.59 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-23 09:55:00 | 1399.30 | 2026-03-23 14:00:00 | 1395.60 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2026-03-24 10:45:00 | 1381.70 | 2026-03-24 10:55:00 | 1388.05 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-01 10:30:00 | 1384.10 | 2026-04-01 11:00:00 | 1374.37 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-01 10:30:00 | 1384.10 | 2026-04-01 11:45:00 | 1384.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1549.10 | 2026-04-29 12:10:00 | 1555.83 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1549.10 | 2026-04-29 14:25:00 | 1551.90 | TARGET_HIT | 0.50 | 0.18% |

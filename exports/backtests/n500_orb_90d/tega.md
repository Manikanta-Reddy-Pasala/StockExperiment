# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1659.00
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
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 10
- **Target hits / Stop hits / Partials:** 3 / 10 / 6
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 5.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.24% | 2.7% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.24% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.31% | 2.5% |
| SELL @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.31% | 2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.27% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 1655.90 | 1634.24 | 0.00 | ORB-long ORB[1600.00,1613.80] vol=3.1x ATR=8.52 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 1647.38 | 1641.53 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1648.00 | 1634.41 | 0.00 | ORB-long ORB[1619.60,1643.90] vol=2.7x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:45:00 | 1657.27 | 1636.97 | 0.00 | T1 1.5R @ 1657.27 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 1648.00 | 1640.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:00:00 | 1729.90 | 1715.37 | 0.00 | ORB-long ORB[1698.80,1723.40] vol=1.6x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:45:00 | 1742.10 | 1723.67 | 0.00 | T1 1.5R @ 1742.10 |
| Target hit | 2026-02-24 15:20:00 | 1783.50 | 1747.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 1771.00 | 1756.56 | 0.00 | ORB-long ORB[1735.60,1759.90] vol=1.6x ATR=7.62 |
| Stop hit — per-position SL triggered | 2026-03-06 10:00:00 | 1763.38 | 1766.98 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:10:00 | 1783.60 | 1748.73 | 0.00 | ORB-long ORB[1707.50,1728.20] vol=2.1x ATR=8.11 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 1775.49 | 1754.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1713.20 | 1720.72 | 0.00 | ORB-short ORB[1713.40,1727.60] vol=2.4x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:50:00 | 1704.61 | 1715.66 | 0.00 | T1 1.5R @ 1704.61 |
| Target hit | 2026-04-09 11:25:00 | 1710.00 | 1709.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 1703.00 | 1711.12 | 0.00 | ORB-short ORB[1705.00,1727.50] vol=1.7x ATR=5.51 |
| Stop hit — per-position SL triggered | 2026-04-15 10:10:00 | 1708.51 | 1708.95 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:50:00 | 1716.50 | 1723.04 | 0.00 | ORB-short ORB[1717.40,1738.00] vol=1.8x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-04-20 10:00:00 | 1721.55 | 1721.17 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:25:00 | 1734.20 | 1728.93 | 0.00 | ORB-long ORB[1710.80,1725.00] vol=1.6x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:50:00 | 1739.36 | 1730.19 | 0.00 | T1 1.5R @ 1739.36 |
| Stop hit — per-position SL triggered | 2026-04-21 12:40:00 | 1734.20 | 1732.28 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 1683.00 | 1688.50 | 0.00 | ORB-short ORB[1686.10,1703.70] vol=4.0x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 1677.80 | 1686.11 | 0.00 | T1 1.5R @ 1677.80 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 1683.00 | 1685.42 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 1680.80 | 1676.52 | 0.00 | ORB-long ORB[1672.10,1680.00] vol=2.2x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-05-04 09:45:00 | 1675.15 | 1676.80 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 1646.50 | 1651.30 | 0.00 | ORB-short ORB[1646.80,1665.70] vol=2.5x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:40:00 | 1639.23 | 1647.53 | 0.00 | T1 1.5R @ 1639.23 |
| Target hit | 2026-05-05 15:20:00 | 1618.90 | 1634.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 1651.60 | 1646.67 | 0.00 | ORB-long ORB[1641.10,1650.00] vol=1.9x ATR=3.91 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 1647.69 | 1647.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:55:00 | 1655.90 | 2026-02-17 10:45:00 | 1647.38 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1648.00 | 2026-02-20 10:45:00 | 1657.27 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1648.00 | 2026-02-20 11:05:00 | 1648.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:00:00 | 1729.90 | 2026-02-24 10:45:00 | 1742.10 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-24 10:00:00 | 1729.90 | 2026-02-24 15:20:00 | 1783.50 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2026-03-06 09:30:00 | 1771.00 | 2026-03-06 10:00:00 | 1763.38 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-10 10:10:00 | 1783.60 | 2026-03-10 10:15:00 | 1775.49 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-09 09:30:00 | 1713.20 | 2026-04-09 09:50:00 | 1704.61 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-09 09:30:00 | 1713.20 | 2026-04-09 11:25:00 | 1710.00 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-04-15 09:50:00 | 1703.00 | 2026-04-15 10:10:00 | 1708.51 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-20 09:50:00 | 1716.50 | 2026-04-20 10:00:00 | 1721.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-21 10:25:00 | 1734.20 | 2026-04-21 10:50:00 | 1739.36 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-21 10:25:00 | 1734.20 | 2026-04-21 12:40:00 | 1734.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 11:15:00 | 1683.00 | 2026-04-28 11:25:00 | 1677.80 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-28 11:15:00 | 1683.00 | 2026-04-28 11:45:00 | 1683.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:30:00 | 1680.80 | 2026-05-04 09:45:00 | 1675.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-05 10:00:00 | 1646.50 | 2026-05-05 10:40:00 | 1639.23 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-05 10:00:00 | 1646.50 | 2026-05-05 15:20:00 | 1618.90 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2026-05-08 10:45:00 | 1651.60 | 2026-05-08 11:00:00 | 1647.69 | STOP_HIT | 1.00 | -0.24% |

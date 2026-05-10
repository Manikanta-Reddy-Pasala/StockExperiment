# Piramal Finance Ltd. (PIRAMALFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2015.00
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 5
- **Avg / median % per leg:** 0.26% / 0.13%
- **Sum % (uncompounded):** 4.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.24% | 1.9% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.24% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.28% | 2.3% |
| SELL @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.28% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 8 | 50.0% | 3 | 8 | 5 | 0.26% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:50:00 | 1750.90 | 1737.51 | 0.00 | ORB-long ORB[1718.60,1741.90] vol=3.6x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 1761.40 | 1741.36 | 0.00 | T1 1.5R @ 1761.40 |
| Stop hit — per-position SL triggered | 2026-02-10 13:35:00 | 1750.90 | 1757.66 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 1751.90 | 1762.85 | 0.00 | ORB-short ORB[1762.30,1784.90] vol=2.9x ATR=6.20 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 1758.10 | 1762.44 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 1790.40 | 1772.29 | 0.00 | ORB-long ORB[1747.30,1770.70] vol=2.8x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 1782.96 | 1776.34 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 1756.00 | 1769.01 | 0.00 | ORB-short ORB[1772.80,1792.20] vol=5.5x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:40:00 | 1747.01 | 1766.35 | 0.00 | T1 1.5R @ 1747.01 |
| Stop hit — per-position SL triggered | 2026-03-06 13:05:00 | 1756.00 | 1754.77 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 1802.00 | 1777.02 | 0.00 | ORB-long ORB[1770.10,1797.00] vol=2.2x ATR=8.22 |
| Stop hit — per-position SL triggered | 2026-03-17 11:10:00 | 1793.78 | 1778.30 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 11:00:00 | 1719.30 | 1731.75 | 0.00 | ORB-short ORB[1722.00,1744.90] vol=8.2x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 14:50:00 | 1709.80 | 1723.49 | 0.00 | T1 1.5R @ 1709.80 |
| Target hit | 2026-04-07 15:20:00 | 1696.80 | 1719.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 1746.80 | 1753.11 | 0.00 | ORB-short ORB[1750.00,1767.80] vol=2.0x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:20:00 | 1736.76 | 1743.91 | 0.00 | T1 1.5R @ 1736.76 |
| Target hit | 2026-04-15 10:30:00 | 1744.50 | 1743.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 1654.50 | 1647.45 | 0.00 | ORB-long ORB[1632.20,1650.00] vol=1.9x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:30:00 | 1661.85 | 1650.12 | 0.00 | T1 1.5R @ 1661.85 |
| Target hit | 2026-04-21 15:20:00 | 1696.00 | 1685.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 1945.00 | 1927.12 | 0.00 | ORB-long ORB[1906.70,1934.90] vol=3.0x ATR=9.27 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 1935.73 | 1931.59 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 1915.70 | 1924.98 | 0.00 | ORB-short ORB[1925.00,1950.00] vol=1.8x ATR=8.63 |
| Stop hit — per-position SL triggered | 2026-05-07 11:40:00 | 1924.33 | 1921.86 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 1950.00 | 1937.21 | 0.00 | ORB-long ORB[1918.00,1936.50] vol=3.4x ATR=5.49 |
| Stop hit — per-position SL triggered | 2026-05-08 09:45:00 | 1944.51 | 1944.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:50:00 | 1750.90 | 2026-02-10 10:10:00 | 1761.40 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-10 09:50:00 | 1750.90 | 2026-02-10 13:35:00 | 1750.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:15:00 | 1751.90 | 2026-02-13 10:30:00 | 1758.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-25 10:25:00 | 1790.40 | 2026-02-25 10:55:00 | 1782.96 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1756.00 | 2026-03-06 11:40:00 | 1747.01 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1756.00 | 2026-03-06 13:05:00 | 1756.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 11:05:00 | 1802.00 | 2026-03-17 11:10:00 | 1793.78 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-07 11:00:00 | 1719.30 | 2026-04-07 14:50:00 | 1709.80 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-07 11:00:00 | 1719.30 | 2026-04-07 15:20:00 | 1696.80 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2026-04-15 09:35:00 | 1746.80 | 2026-04-15 10:20:00 | 1736.76 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-15 09:35:00 | 1746.80 | 2026-04-15 10:30:00 | 1744.50 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2026-04-21 10:45:00 | 1654.50 | 2026-04-21 11:30:00 | 1661.85 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-21 10:45:00 | 1654.50 | 2026-04-21 15:20:00 | 1696.00 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1945.00 | 2026-05-06 09:55:00 | 1935.73 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-07 10:45:00 | 1915.70 | 2026-05-07 11:40:00 | 1924.33 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-08 09:30:00 | 1950.00 | 2026-05-08 09:45:00 | 1944.51 | STOP_HIT | 1.00 | -0.28% |

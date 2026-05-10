# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1691.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.19% | 1.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.19% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.08% | 0.6% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.08% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.12% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 1917.00 | 1914.07 | 0.00 | ORB-long ORB[1889.30,1904.00] vol=4.0x ATR=10.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:40:00 | 1932.86 | 1917.58 | 0.00 | T1 1.5R @ 1932.86 |
| Target hit | 2026-02-09 15:20:00 | 1938.60 | 1924.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:45:00 | 1984.00 | 1971.50 | 0.00 | ORB-long ORB[1960.00,1974.00] vol=1.8x ATR=6.66 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 1977.34 | 1973.58 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 1998.80 | 1989.42 | 0.00 | ORB-long ORB[1976.40,1989.20] vol=2.9x ATR=6.05 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 1992.75 | 1995.37 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 1988.60 | 2002.87 | 0.00 | ORB-short ORB[1992.80,2019.40] vol=2.2x ATR=7.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:55:00 | 1977.94 | 1999.46 | 0.00 | T1 1.5R @ 1977.94 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 1988.60 | 1988.47 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1993.80 | 2000.15 | 0.00 | ORB-short ORB[2005.00,2017.70] vol=1.5x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 1985.91 | 1998.08 | 0.00 | T1 1.5R @ 1985.91 |
| Stop hit — per-position SL triggered | 2026-02-19 12:20:00 | 1993.80 | 1994.38 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 1976.80 | 1962.81 | 0.00 | ORB-long ORB[1949.10,1974.90] vol=1.8x ATR=6.97 |
| Stop hit — per-position SL triggered | 2026-02-25 10:45:00 | 1969.83 | 1965.05 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 1935.60 | 1951.61 | 0.00 | ORB-short ORB[1947.10,1968.00] vol=1.5x ATR=5.98 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 1941.58 | 1949.88 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:50:00 | 1615.20 | 1636.39 | 0.00 | ORB-short ORB[1623.30,1646.60] vol=2.3x ATR=7.61 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 1622.81 | 1627.96 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 1902.00 | 1907.64 | 0.00 | ORB-short ORB[1910.70,1928.90] vol=1.7x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:05:00 | 1893.41 | 1907.25 | 0.00 | T1 1.5R @ 1893.41 |
| Stop hit — per-position SL triggered | 2026-04-28 14:40:00 | 1902.00 | 1901.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 1917.00 | 2026-02-09 10:40:00 | 1932.86 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-09 10:30:00 | 1917.00 | 2026-02-09 15:20:00 | 1938.60 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2026-02-13 10:45:00 | 1984.00 | 2026-02-13 11:00:00 | 1977.34 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1998.80 | 2026-02-17 10:40:00 | 1992.75 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1988.60 | 2026-02-18 09:55:00 | 1977.94 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1988.60 | 2026-02-18 10:50:00 | 1988.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1993.80 | 2026-02-19 11:15:00 | 1985.91 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1993.80 | 2026-02-19 12:20:00 | 1993.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:05:00 | 1976.80 | 2026-02-25 10:45:00 | 1969.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 10:10:00 | 1935.60 | 2026-02-27 10:20:00 | 1941.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-30 10:50:00 | 1615.20 | 2026-03-30 12:00:00 | 1622.81 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-28 11:00:00 | 1902.00 | 2026-04-28 11:05:00 | 1893.41 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-28 11:00:00 | 1902.00 | 2026-04-28 14:40:00 | 1902.00 | STOP_HIT | 0.50 | 0.00% |

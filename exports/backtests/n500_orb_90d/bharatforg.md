# Bharat Forge Ltd. (BHARATFORG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1984.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 1
- **Avg / median % per leg:** -0.18% / -0.32%
- **Sum % (uncompounded):** -1.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.35% | -1.8% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.35% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.11% | 0.3% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.11% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.18% | -1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 1612.00 | 1600.05 | 0.00 | ORB-long ORB[1582.20,1600.00] vol=2.3x ATR=5.12 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 1606.88 | 1601.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 1745.00 | 1740.32 | 0.00 | ORB-long ORB[1726.10,1743.00] vol=2.4x ATR=5.72 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 1739.28 | 1740.53 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 1766.60 | 1761.73 | 0.00 | ORB-long ORB[1747.30,1765.70] vol=3.0x ATR=4.90 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 1761.70 | 1761.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 1728.10 | 1711.32 | 0.00 | ORB-long ORB[1691.20,1710.20] vol=2.0x ATR=8.37 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 1719.73 | 1715.06 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:35:00 | 1843.00 | 1831.82 | 0.00 | ORB-long ORB[1825.30,1841.90] vol=2.1x ATR=6.47 |
| Stop hit — per-position SL triggered | 2026-04-15 11:05:00 | 1836.53 | 1836.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 1863.10 | 1873.00 | 0.00 | ORB-short ORB[1874.50,1892.10] vol=1.9x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:40:00 | 1850.76 | 1868.84 | 0.00 | T1 1.5R @ 1850.76 |
| Target hit | 2026-04-24 14:35:00 | 1862.00 | 1857.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 1824.10 | 1844.54 | 0.00 | ORB-short ORB[1829.30,1853.90] vol=3.8x ATR=7.08 |
| Stop hit — per-position SL triggered | 2026-05-05 11:20:00 | 1831.18 | 1841.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:35:00 | 1612.00 | 2026-02-10 09:45:00 | 1606.88 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-17 10:25:00 | 1745.00 | 2026-02-17 10:30:00 | 1739.28 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-18 09:40:00 | 1766.60 | 2026-02-18 09:50:00 | 1761.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-17 10:00:00 | 1728.10 | 2026-03-17 10:30:00 | 1719.73 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-15 10:35:00 | 1843.00 | 2026-04-15 11:05:00 | 1836.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 09:40:00 | 1863.10 | 2026-04-24 10:40:00 | 1850.76 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-04-24 09:40:00 | 1863.10 | 2026-04-24 14:35:00 | 1862.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-05-05 10:55:00 | 1824.10 | 2026-05-05 11:20:00 | 1831.18 | STOP_HIT | 1.00 | -0.39% |

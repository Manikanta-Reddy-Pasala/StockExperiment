# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1669.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** 0.22% / -0.38%
- **Sum % (uncompounded):** 1.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.44% | -0.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.44% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.33% | 2.0% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.33% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.22% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1781.25 | 1799.75 | 0.00 | ORB-short ORB[1802.40,1823.35] vol=2.0x ATR=8.14 |
| Stop hit — per-position SL triggered | 2026-02-18 11:30:00 | 1789.39 | 1796.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 1798.40 | 1787.98 | 0.00 | ORB-long ORB[1777.60,1792.50] vol=1.9x ATR=7.89 |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 1790.51 | 1790.41 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 1763.95 | 1776.93 | 0.00 | ORB-short ORB[1775.25,1797.45] vol=1.6x ATR=5.44 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 1769.39 | 1776.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:30:00 | 1458.10 | 1468.19 | 0.00 | ORB-short ORB[1464.00,1478.30] vol=2.3x ATR=6.85 |
| Stop hit — per-position SL triggered | 2026-03-19 09:45:00 | 1464.95 | 1465.57 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 1606.90 | 1619.17 | 0.00 | ORB-short ORB[1616.10,1636.80] vol=3.2x ATR=8.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:40:00 | 1593.60 | 1609.86 | 0.00 | T1 1.5R @ 1593.60 |
| Target hit | 2026-04-21 15:20:00 | 1562.70 | 1590.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-04-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:40:00 | 1455.00 | 1466.97 | 0.00 | ORB-short ORB[1466.00,1487.90] vol=2.6x ATR=5.53 |
| Stop hit — per-position SL triggered | 2026-04-28 10:55:00 | 1460.53 | 1466.31 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:55:00 | 1781.25 | 2026-02-18 11:30:00 | 1789.39 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-20 09:45:00 | 1798.40 | 2026-02-20 10:15:00 | 1790.51 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-23 10:55:00 | 1763.95 | 2026-02-23 11:10:00 | 1769.39 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-19 09:30:00 | 1458.10 | 2026-03-19 09:45:00 | 1464.95 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-21 09:50:00 | 1606.90 | 2026-04-21 10:40:00 | 1593.60 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2026-04-21 09:50:00 | 1606.90 | 2026-04-21 15:20:00 | 1562.70 | TARGET_HIT | 0.50 | 2.75% |
| SELL | retest1 | 2026-04-28 10:40:00 | 1455.00 | 2026-04-28 10:55:00 | 1460.53 | STOP_HIT | 1.00 | -0.38% |

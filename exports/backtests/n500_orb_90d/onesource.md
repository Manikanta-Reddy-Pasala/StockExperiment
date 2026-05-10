# Onesource Specialty Pharma Ltd. (ONESOURCE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1836.10
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.82% | -0.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.82% | -0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.19% | 0.9% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.19% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.02% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:15:00 | 1430.00 | 1440.85 | 0.00 | ORB-short ORB[1442.00,1463.00] vol=1.8x ATR=5.24 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 1435.24 | 1440.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:40:00 | 1482.20 | 1468.32 | 0.00 | ORB-long ORB[1453.80,1473.60] vol=1.7x ATR=12.22 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 1469.98 | 1468.66 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 1789.40 | 1816.20 | 0.00 | ORB-short ORB[1811.50,1835.90] vol=5.3x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:30:00 | 1778.83 | 1801.71 | 0.00 | T1 1.5R @ 1778.83 |
| Stop hit — per-position SL triggered | 2026-04-28 13:10:00 | 1789.40 | 1797.29 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:55:00 | 1728.80 | 1756.58 | 0.00 | ORB-short ORB[1744.00,1769.90] vol=1.8x ATR=8.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:20:00 | 1716.34 | 1747.87 | 0.00 | T1 1.5R @ 1716.34 |
| Stop hit — per-position SL triggered | 2026-04-30 11:45:00 | 1728.80 | 1740.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-13 11:15:00 | 1430.00 | 2026-03-13 11:25:00 | 1435.24 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-20 09:40:00 | 1482.20 | 2026-03-20 09:50:00 | 1469.98 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest1 | 2026-04-28 11:15:00 | 1789.40 | 2026-04-28 12:30:00 | 1778.83 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-28 11:15:00 | 1789.40 | 2026-04-28 13:10:00 | 1789.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:55:00 | 1728.80 | 2026-04-30 11:20:00 | 1716.34 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-04-30 10:55:00 | 1728.80 | 2026-04-30 11:45:00 | 1728.80 | STOP_HIT | 0.50 | 0.00% |

# Onesource Specialty Pharma Ltd. (ONESOURCE)

## Backtest Summary

- **Window:** 2025-01-24 09:35:00 → 2026-05-08 15:25:00 (23709 bars)
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 0.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.50% | -1.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.50% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.39% | 1.5% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.39% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.01% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 11:05:00 | 1406.20 | 1426.14 | 0.00 | ORB-short ORB[1416.00,1435.05] vol=2.9x ATR=5.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 11:15:00 | 1397.52 | 1422.60 | 0.00 | T1 1.5R @ 1397.52 |
| Stop hit — per-position SL triggered | 2025-03-06 12:35:00 | 1406.20 | 1412.79 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-03-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:00:00 | 1439.25 | 1436.77 | 0.00 | ORB-long ORB[1423.00,1439.15] vol=2.4x ATR=5.53 |
| Stop hit — per-position SL triggered | 2025-03-07 11:05:00 | 1433.72 | 1436.75 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:45:00 | 1497.75 | 1483.61 | 0.00 | ORB-long ORB[1473.00,1487.00] vol=1.6x ATR=9.51 |
| Stop hit — per-position SL triggered | 2025-03-18 12:05:00 | 1488.24 | 1498.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-03-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:35:00 | 1676.65 | 1683.78 | 0.00 | ORB-short ORB[1683.20,1707.75] vol=1.5x ATR=10.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:50:00 | 1661.02 | 1681.61 | 0.00 | T1 1.5R @ 1661.02 |
| Stop hit — per-position SL triggered | 2025-03-27 14:35:00 | 1676.65 | 1665.62 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-04-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:25:00 | 1613.70 | 1597.96 | 0.00 | ORB-long ORB[1575.90,1598.70] vol=6.8x ATR=7.84 |
| Stop hit — per-position SL triggered | 2025-04-24 10:30:00 | 1605.86 | 1603.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-03-06 11:05:00 | 1406.20 | 2025-03-06 11:15:00 | 1397.52 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-03-06 11:05:00 | 1406.20 | 2025-03-06 12:35:00 | 1406.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 11:00:00 | 1439.25 | 2025-03-07 11:05:00 | 1433.72 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-18 09:45:00 | 1497.75 | 2025-03-18 12:05:00 | 1488.24 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2025-03-27 10:35:00 | 1676.65 | 2025-03-27 10:50:00 | 1661.02 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2025-03-27 10:35:00 | 1676.65 | 2025-03-27 14:35:00 | 1676.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 10:25:00 | 1613.70 | 2025-04-24 10:30:00 | 1605.86 | STOP_HIT | 1.00 | -0.49% |

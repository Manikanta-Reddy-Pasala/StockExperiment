# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1589.10
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 0.46% / 0.43%
- **Sum % (uncompounded):** 2.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.87% | 2.6% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.87% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.05% | 0.2% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.05% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.46% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1307.00 | 1311.19 | 0.00 | ORB-short ORB[1308.30,1323.40] vol=3.1x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:40:00 | 1301.35 | 1308.05 | 0.00 | T1 1.5R @ 1301.35 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 1307.00 | 1304.47 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:45:00 | 1309.90 | 1303.76 | 0.00 | ORB-long ORB[1291.10,1309.20] vol=1.9x ATR=5.66 |
| Stop hit — per-position SL triggered | 2026-02-19 09:55:00 | 1304.24 | 1304.33 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:55:00 | 1296.10 | 1302.44 | 0.00 | ORB-short ORB[1298.90,1315.50] vol=2.5x ATR=3.52 |
| Stop hit — per-position SL triggered | 2026-02-24 10:00:00 | 1299.62 | 1302.28 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 1558.00 | 1553.55 | 0.00 | ORB-long ORB[1538.00,1557.40] vol=3.3x ATR=5.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:40:00 | 1566.96 | 1555.85 | 0.00 | T1 1.5R @ 1566.96 |
| Target hit | 2026-05-08 11:55:00 | 1596.30 | 1598.08 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:30:00 | 1307.00 | 2026-02-18 09:40:00 | 1301.35 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-18 09:30:00 | 1307.00 | 2026-02-18 10:15:00 | 1307.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 09:45:00 | 1309.90 | 2026-02-19 09:55:00 | 1304.24 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-24 09:55:00 | 1296.10 | 2026-02-24 10:00:00 | 1299.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-08 09:35:00 | 1558.00 | 2026-05-08 09:40:00 | 1566.96 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-05-08 09:35:00 | 1558.00 | 2026-05-08 11:55:00 | 1596.30 | TARGET_HIT | 0.50 | 2.46% |

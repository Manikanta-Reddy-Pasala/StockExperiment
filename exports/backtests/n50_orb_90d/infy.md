# INFY (INFY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1179.50
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 1.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.38% | 1.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.38% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.24% | -0.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.24% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.20% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 1487.50 | 1493.82 | 0.00 | ORB-short ORB[1497.00,1505.90] vol=1.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 1490.28 | 1493.46 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 1401.40 | 1392.64 | 0.00 | ORB-long ORB[1382.20,1398.00] vol=2.3x ATR=4.42 |
| Stop hit — per-position SL triggered | 2026-02-19 11:10:00 | 1396.98 | 1392.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 1227.20 | 1238.25 | 0.00 | ORB-short ORB[1236.00,1254.30] vol=2.0x ATR=3.71 |
| Stop hit — per-position SL triggered | 2026-03-16 11:10:00 | 1230.91 | 1237.60 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:00:00 | 1281.10 | 1272.20 | 0.00 | ORB-long ORB[1257.10,1272.60] vol=2.0x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:05:00 | 1287.79 | 1276.09 | 0.00 | T1 1.5R @ 1287.79 |
| Stop hit — per-position SL triggered | 2026-03-25 14:20:00 | 1281.10 | 1280.95 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:05:00 | 1322.00 | 1314.73 | 0.00 | ORB-long ORB[1293.50,1310.90] vol=1.9x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:15:00 | 1327.88 | 1316.64 | 0.00 | T1 1.5R @ 1327.88 |
| Target hit | 2026-04-07 15:20:00 | 1338.70 | 1329.53 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:10:00 | 1487.50 | 2026-02-11 11:30:00 | 1490.28 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-19 11:00:00 | 1401.40 | 2026-02-19 11:10:00 | 1396.98 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-16 11:00:00 | 1227.20 | 2026-03-16 11:10:00 | 1230.91 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-25 10:00:00 | 1281.10 | 2026-03-25 11:05:00 | 1287.79 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-25 10:00:00 | 1281.10 | 2026-03-25 14:20:00 | 1281.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 11:05:00 | 1322.00 | 2026-04-07 11:15:00 | 1327.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-07 11:05:00 | 1322.00 | 2026-04-07 15:20:00 | 1338.70 | TARGET_HIT | 0.50 | 1.26% |

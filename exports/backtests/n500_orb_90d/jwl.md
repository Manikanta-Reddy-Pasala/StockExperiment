# Jupiter Wagons Ltd. (JWL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 298.90
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.02% | -0.1% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.02% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.32% | 0.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.32% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.06% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 285.25 | 286.02 | 0.00 | ORB-short ORB[285.50,289.45] vol=2.3x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:00:00 | 283.62 | 285.50 | 0.00 | T1 1.5R @ 283.62 |
| Target hit | 2026-02-24 15:05:00 | 285.05 | 284.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 274.08 | 272.59 | 0.00 | ORB-long ORB[270.10,273.36] vol=2.8x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 272.96 | 272.65 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 284.03 | 282.53 | 0.00 | ORB-long ORB[281.11,283.68] vol=2.7x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 283.05 | 282.79 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 286.20 | 283.61 | 0.00 | ORB-long ORB[280.88,284.07] vol=2.0x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:45:00 | 287.81 | 284.61 | 0.00 | T1 1.5R @ 287.81 |
| Stop hit — per-position SL triggered | 2026-04-27 11:20:00 | 286.20 | 285.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 288.55 | 286.41 | 0.00 | ORB-long ORB[285.05,288.15] vol=3.4x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:40:00 | 290.36 | 286.87 | 0.00 | T1 1.5R @ 290.36 |
| Stop hit — per-position SL triggered | 2026-05-04 12:05:00 | 288.55 | 287.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 291.60 | 289.05 | 0.00 | ORB-long ORB[284.40,288.50] vol=6.3x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 289.98 | 289.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-24 09:35:00 | 285.25 | 2026-02-24 12:00:00 | 283.62 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-24 09:35:00 | 285.25 | 2026-02-24 15:05:00 | 285.05 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2026-04-10 09:30:00 | 274.08 | 2026-04-10 09:35:00 | 272.96 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-22 09:35:00 | 284.03 | 2026-04-22 09:40:00 | 283.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-27 10:40:00 | 286.20 | 2026-04-27 10:45:00 | 287.81 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-27 10:40:00 | 286.20 | 2026-04-27 11:20:00 | 286.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:00:00 | 288.55 | 2026-05-04 11:40:00 | 290.36 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-05-04 11:00:00 | 288.55 | 2026-05-04 12:05:00 | 288.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:35:00 | 291.60 | 2026-05-05 09:45:00 | 289.98 | STOP_HIT | 1.00 | -0.56% |

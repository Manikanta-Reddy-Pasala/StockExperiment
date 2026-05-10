# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4630.00
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 2 / 0 / 2
- **Avg / median % per leg:** 0.47% / 0.58%
- **Sum % (uncompounded):** 1.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.46% | 0.9% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.46% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.48% | 1.0% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.48% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 0.47% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:10:00 | 3780.10 | 3744.58 | 0.00 | ORB-long ORB[3711.50,3753.50] vol=2.3x ATR=15.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 3802.75 | 3753.87 | 0.00 | T1 1.5R @ 3802.75 |
| Target hit | 2026-02-25 13:05:00 | 3792.50 | 3793.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 4150.00 | 4176.17 | 0.00 | ORB-short ORB[4166.70,4204.60] vol=2.6x ATR=10.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:45:00 | 4134.30 | 4172.65 | 0.00 | T1 1.5R @ 4134.30 |
| Target hit | 2026-04-15 15:20:00 | 4126.00 | 4157.29 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-25 11:10:00 | 3780.10 | 2026-02-25 11:25:00 | 3802.75 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-25 11:10:00 | 3780.10 | 2026-02-25 13:05:00 | 3792.50 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-15 11:15:00 | 4150.00 | 2026-04-15 11:45:00 | 4134.30 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-15 11:15:00 | 4150.00 | 2026-04-15 15:20:00 | 4126.00 | TARGET_HIT | 0.50 | 0.58% |

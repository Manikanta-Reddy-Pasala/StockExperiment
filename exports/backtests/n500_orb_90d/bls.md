# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 290.00
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 4
- **Avg / median % per leg:** 0.23% / 0.47%
- **Sum % (uncompounded):** 2.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.32% | 1.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.32% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.13% | 0.6% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.13% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.23% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 256.70 | 258.64 | 0.00 | ORB-short ORB[257.45,260.70] vol=3.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 255.48 | 258.39 | 0.00 | T1 1.5R @ 255.48 |
| Stop hit — per-position SL triggered | 2026-03-05 12:40:00 | 256.70 | 257.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 249.50 | 247.19 | 0.00 | ORB-long ORB[244.80,247.70] vol=3.0x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:35:00 | 250.99 | 247.53 | 0.00 | T1 1.5R @ 250.99 |
| Target hit | 2026-03-12 15:20:00 | 251.40 | 249.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 246.95 | 247.87 | 0.00 | ORB-short ORB[247.15,250.50] vol=1.8x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-03-13 11:05:00 | 247.87 | 247.36 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:40:00 | 298.00 | 299.57 | 0.00 | ORB-short ORB[299.01,302.76] vol=3.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:00:00 | 296.38 | 299.11 | 0.00 | T1 1.5R @ 296.38 |
| Stop hit — per-position SL triggered | 2026-04-17 14:05:00 | 298.00 | 298.77 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:05:00 | 297.08 | 295.09 | 0.00 | ORB-long ORB[293.05,296.99] vol=2.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 296.20 | 295.14 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 283.00 | 280.98 | 0.00 | ORB-long ORB[277.95,281.80] vol=2.0x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 284.59 | 281.89 | 0.00 | T1 1.5R @ 284.59 |
| Stop hit — per-position SL triggered | 2026-05-05 09:55:00 | 283.00 | 283.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-05 11:15:00 | 256.70 | 2026-03-05 11:25:00 | 255.48 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-05 11:15:00 | 256.70 | 2026-03-05 12:40:00 | 256.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 11:15:00 | 249.50 | 2026-03-12 11:35:00 | 250.99 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-12 11:15:00 | 249.50 | 2026-03-12 15:20:00 | 251.40 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2026-03-13 10:00:00 | 246.95 | 2026-03-13 11:05:00 | 247.87 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-17 10:40:00 | 298.00 | 2026-04-17 12:00:00 | 296.38 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-04-17 10:40:00 | 298.00 | 2026-04-17 14:05:00 | 298.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 11:05:00 | 297.08 | 2026-04-21 11:15:00 | 296.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-05 09:30:00 | 283.00 | 2026-05-05 09:35:00 | 284.59 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-05-05 09:30:00 | 283.00 | 2026-05-05 09:55:00 | 283.00 | STOP_HIT | 0.50 | 0.00% |

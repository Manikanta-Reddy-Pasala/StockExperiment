# Power Grid Corporation of India Ltd. (POWERGRID)

## Backtest Summary

- **Window:** 2025-02-05 09:15:00 → 2026-05-08 15:25:00 (23038 bars)
- **Last close:** 313.90
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
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 5
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.04% | 0.3% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.04% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.12% | 0.6% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.12% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 0 | 9 | 5 | 0.06% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:20:00 | 270.20 | 271.98 | 0.00 | ORB-short ORB[272.05,276.00] vol=1.5x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:55:00 | 268.89 | 271.55 | 0.00 | T1 1.5R @ 268.89 |
| Stop hit — per-position SL triggered | 2025-02-10 12:30:00 | 270.20 | 270.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:55:00 | 264.50 | 263.68 | 0.00 | ORB-long ORB[261.30,264.00] vol=2.3x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 12:20:00 | 265.35 | 264.23 | 0.00 | T1 1.5R @ 265.35 |
| Stop hit — per-position SL triggered | 2025-02-20 12:25:00 | 264.50 | 264.33 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 10:40:00 | 266.45 | 267.38 | 0.00 | ORB-short ORB[267.10,270.65] vol=1.9x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 10:50:00 | 265.45 | 267.35 | 0.00 | T1 1.5R @ 265.45 |
| Stop hit — per-position SL triggered | 2025-03-17 11:10:00 | 266.45 | 267.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:10:00 | 271.45 | 269.81 | 0.00 | ORB-long ORB[267.30,269.40] vol=1.7x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:35:00 | 272.19 | 270.03 | 0.00 | T1 1.5R @ 272.19 |
| Stop hit — per-position SL triggered | 2025-03-18 11:50:00 | 271.45 | 270.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-03-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:50:00 | 274.00 | 272.36 | 0.00 | ORB-long ORB[269.80,272.80] vol=2.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-03-19 11:30:00 | 273.39 | 272.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-03-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:10:00 | 284.45 | 282.36 | 0.00 | ORB-long ORB[280.05,282.85] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 283.67 | 282.40 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:55:00 | 310.05 | 312.96 | 0.00 | ORB-short ORB[313.35,316.10] vol=2.1x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-04-23 12:15:00 | 310.92 | 312.04 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:15:00 | 309.60 | 307.94 | 0.00 | ORB-long ORB[303.55,307.80] vol=3.2x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 11:35:00 | 310.92 | 308.04 | 0.00 | T1 1.5R @ 310.92 |
| Stop hit — per-position SL triggered | 2025-04-28 11:55:00 | 309.60 | 308.17 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 11:15:00 | 309.35 | 308.09 | 0.00 | ORB-long ORB[303.60,308.05] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-04-30 11:20:00 | 308.73 | 308.10 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-02-10 10:20:00 | 270.20 | 2025-02-10 10:55:00 | 268.89 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-02-10 10:20:00 | 270.20 | 2025-02-10 12:30:00 | 270.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 10:55:00 | 264.50 | 2025-02-20 12:20:00 | 265.35 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-02-20 10:55:00 | 264.50 | 2025-02-20 12:25:00 | 264.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-17 10:40:00 | 266.45 | 2025-03-17 10:50:00 | 265.45 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-03-17 10:40:00 | 266.45 | 2025-03-17 11:10:00 | 266.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 11:10:00 | 271.45 | 2025-03-18 11:35:00 | 272.19 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-03-18 11:10:00 | 271.45 | 2025-03-18 11:50:00 | 271.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 10:50:00 | 274.00 | 2025-03-19 11:30:00 | 273.39 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-21 10:10:00 | 284.45 | 2025-03-21 10:15:00 | 283.67 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-23 10:55:00 | 310.05 | 2025-04-23 12:15:00 | 310.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-28 11:15:00 | 309.60 | 2025-04-28 11:35:00 | 310.92 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-28 11:15:00 | 309.60 | 2025-04-28 11:55:00 | 309.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 11:15:00 | 309.35 | 2025-04-30 11:20:00 | 308.73 | STOP_HIT | 1.00 | -0.20% |

# City Union Bank Ltd. (CUB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 258.95
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 7
- **Avg / median % per leg:** 0.41% / 0.41%
- **Sum % (uncompounded):** 6.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.33% | 1.3% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.33% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.43% | 5.2% |
| SELL @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.43% | 5.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 10 | 62.5% | 3 | 6 | 7 | 0.41% | 6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 277.90 | 279.78 | 0.00 | ORB-short ORB[279.05,282.50] vol=3.8x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:25:00 | 276.26 | 278.81 | 0.00 | T1 1.5R @ 276.26 |
| Stop hit — per-position SL triggered | 2026-02-16 11:00:00 | 277.90 | 277.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 278.10 | 280.52 | 0.00 | ORB-short ORB[281.65,284.55] vol=1.7x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 279.08 | 279.92 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 287.60 | 285.72 | 0.00 | ORB-long ORB[281.65,285.50] vol=2.0x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:15:00 | 289.42 | 287.63 | 0.00 | T1 1.5R @ 289.42 |
| Target hit | 2026-02-23 11:25:00 | 288.30 | 288.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 291.40 | 290.75 | 0.00 | ORB-long ORB[287.70,291.00] vol=3.7x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:15:00 | 292.73 | 291.44 | 0.00 | T1 1.5R @ 292.73 |
| Stop hit — per-position SL triggered | 2026-02-25 10:45:00 | 291.40 | 292.27 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:05:00 | 243.60 | 244.32 | 0.00 | ORB-short ORB[243.80,247.40] vol=2.3x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:30:00 | 241.91 | 243.62 | 0.00 | T1 1.5R @ 241.91 |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 243.60 | 243.32 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 251.03 | 252.57 | 0.00 | ORB-short ORB[251.63,254.54] vol=3.2x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:20:00 | 250.01 | 252.20 | 0.00 | T1 1.5R @ 250.01 |
| Target hit | 2026-04-16 15:20:00 | 250.27 | 249.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 267.30 | 269.55 | 0.00 | ORB-short ORB[269.55,271.95] vol=1.6x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:00:00 | 265.32 | 268.59 | 0.00 | T1 1.5R @ 265.32 |
| Stop hit — per-position SL triggered | 2026-05-05 10:55:00 | 267.30 | 267.38 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:15:00 | 269.50 | 270.99 | 0.00 | ORB-short ORB[271.70,273.90] vol=1.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-05-07 13:45:00 | 270.21 | 270.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 266.70 | 267.93 | 0.00 | ORB-short ORB[268.60,271.55] vol=2.3x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:20:00 | 265.23 | 266.88 | 0.00 | T1 1.5R @ 265.23 |
| Target hit | 2026-05-08 15:20:00 | 259.90 | 262.63 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 10:15:00 | 277.90 | 2026-02-16 10:25:00 | 276.26 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-16 10:15:00 | 277.90 | 2026-02-16 11:00:00 | 277.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-20 11:00:00 | 278.10 | 2026-02-20 11:30:00 | 279.08 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-23 09:45:00 | 287.60 | 2026-02-23 10:15:00 | 289.42 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-23 09:45:00 | 287.60 | 2026-02-23 11:25:00 | 288.30 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-02-25 10:05:00 | 291.40 | 2026-02-25 10:15:00 | 292.73 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-25 10:05:00 | 291.40 | 2026-02-25 10:45:00 | 291.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 10:05:00 | 243.60 | 2026-03-19 13:30:00 | 241.91 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-19 10:05:00 | 243.60 | 2026-03-19 14:15:00 | 243.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 11:05:00 | 251.03 | 2026-04-16 11:20:00 | 250.01 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-16 11:05:00 | 251.03 | 2026-04-16 15:20:00 | 250.27 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-05-05 09:45:00 | 267.30 | 2026-05-05 10:00:00 | 265.32 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-05-05 09:45:00 | 267.30 | 2026-05-05 10:55:00 | 267.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 11:15:00 | 269.50 | 2026-05-07 13:45:00 | 270.21 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-08 10:15:00 | 266.70 | 2026-05-08 10:20:00 | 265.23 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-08 10:15:00 | 266.70 | 2026-05-08 15:20:00 | 259.90 | TARGET_HIT | 0.50 | 2.55% |

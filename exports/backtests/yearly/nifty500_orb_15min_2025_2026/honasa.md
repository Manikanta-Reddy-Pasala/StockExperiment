# Honasa Consumer Ltd. (HONASA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-07-09 15:25:00 (3225 bars)
- **Last close:** 302.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 3
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 1.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 1 | 1 | 0 | -0.03% | -0.1% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 1 | 1 | 0 | -0.03% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.14% | 1.6% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.14% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 6 | 46.2% | 3 | 7 | 3 | 0.12% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-12 10:30:00 | 249.50 | 250.23 | 0.00 | ORB-short ORB[250.03,253.00] vol=2.0x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:45:00 | 247.55 | 249.87 | 0.00 | T1 1.5R @ 247.55 |
| Target hit | 2025-05-12 15:05:00 | 248.66 | 248.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2025-05-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 11:00:00 | 266.76 | 267.91 | 0.00 | ORB-short ORB[267.10,271.00] vol=3.4x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-05-20 11:25:00 | 267.64 | 267.82 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 310.60 | 311.79 | 0.00 | ORB-short ORB[311.07,313.98] vol=2.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-05-29 11:30:00 | 311.50 | 311.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:50:00 | 314.35 | 316.39 | 0.00 | ORB-short ORB[317.36,320.87] vol=1.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 315.58 | 315.66 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:45:00 | 312.85 | 314.77 | 0.00 | ORB-short ORB[313.25,316.90] vol=1.8x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:30:00 | 311.11 | 313.47 | 0.00 | T1 1.5R @ 311.11 |
| Stop hit — per-position SL triggered | 2025-06-03 11:45:00 | 312.85 | 313.30 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:05:00 | 314.15 | 316.76 | 0.00 | ORB-short ORB[315.10,318.00] vol=3.6x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-06-06 11:30:00 | 315.35 | 316.36 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 11:00:00 | 307.30 | 306.06 | 0.00 | ORB-long ORB[302.55,305.55] vol=2.3x ATR=1.23 |
| Target hit | 2025-06-18 15:20:00 | 307.90 | 307.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 11:15:00 | 303.20 | 304.48 | 0.00 | ORB-short ORB[303.25,307.00] vol=7.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-06-24 11:35:00 | 304.17 | 304.43 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:40:00 | 305.50 | 306.94 | 0.00 | ORB-short ORB[306.50,311.00] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:55:00 | 304.27 | 306.62 | 0.00 | T1 1.5R @ 304.27 |
| Target hit | 2025-07-01 15:20:00 | 301.80 | 304.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-07-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:55:00 | 300.25 | 299.01 | 0.00 | ORB-long ORB[297.10,299.85] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-07-09 11:45:00 | 299.47 | 299.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-12 10:30:00 | 249.50 | 2025-05-12 10:45:00 | 247.55 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-05-12 10:30:00 | 249.50 | 2025-05-12 15:05:00 | 248.66 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-05-20 11:00:00 | 266.76 | 2025-05-20 11:25:00 | 267.64 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-29 11:10:00 | 310.60 | 2025-05-29 11:30:00 | 311.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-30 10:50:00 | 314.35 | 2025-05-30 12:15:00 | 315.58 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-06-03 09:45:00 | 312.85 | 2025-06-03 11:30:00 | 311.11 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-06-03 09:45:00 | 312.85 | 2025-06-03 11:45:00 | 312.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 11:05:00 | 314.15 | 2025-06-06 11:30:00 | 315.35 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-18 11:00:00 | 307.30 | 2025-06-18 15:20:00 | 307.90 | TARGET_HIT | 1.00 | 0.20% |
| SELL | retest1 | 2025-06-24 11:15:00 | 303.20 | 2025-06-24 11:35:00 | 304.17 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-01 10:40:00 | 305.50 | 2025-07-01 10:55:00 | 304.27 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-01 10:40:00 | 305.50 | 2025-07-01 15:20:00 | 301.80 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2025-07-09 10:55:00 | 300.25 | 2025-07-09 11:45:00 | 299.47 | STOP_HIT | 1.00 | -0.26% |

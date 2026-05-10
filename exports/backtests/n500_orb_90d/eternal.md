# Eternal Ltd. (ETERNAL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 256.15
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
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 6
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 2.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.34% | 2.4% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.34% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.05% | 0.5% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.05% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 7 | 43.8% | 1 | 9 | 6 | 0.18% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 275.55 | 279.46 | 0.00 | ORB-short ORB[281.20,284.65] vol=2.3x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 276.43 | 279.32 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 275.70 | 277.52 | 0.00 | ORB-short ORB[276.10,279.80] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:55:00 | 274.29 | 277.23 | 0.00 | T1 1.5R @ 274.29 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 275.70 | 277.21 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 253.05 | 254.19 | 0.00 | ORB-short ORB[253.85,256.60] vol=2.3x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-02-25 10:30:00 | 254.10 | 253.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 238.94 | 240.08 | 0.00 | ORB-short ORB[239.80,241.48] vol=5.1x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:05:00 | 237.55 | 239.63 | 0.00 | T1 1.5R @ 237.55 |
| Stop hit — per-position SL triggered | 2026-04-10 10:30:00 | 238.94 | 239.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:25:00 | 239.32 | 236.94 | 0.00 | ORB-long ORB[234.53,237.36] vol=2.8x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:45:00 | 240.78 | 237.96 | 0.00 | T1 1.5R @ 240.78 |
| Stop hit — per-position SL triggered | 2026-04-13 11:30:00 | 239.32 | 238.62 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:00:00 | 243.79 | 242.32 | 0.00 | ORB-long ORB[240.05,243.66] vol=2.9x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:10:00 | 245.25 | 242.90 | 0.00 | T1 1.5R @ 245.25 |
| Target hit | 2026-04-15 15:20:00 | 246.15 | 245.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 258.13 | 256.47 | 0.00 | ORB-long ORB[254.13,257.21] vol=1.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-04-21 10:50:00 | 257.23 | 257.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 256.50 | 258.43 | 0.00 | ORB-short ORB[257.50,261.27] vol=1.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-04-27 09:35:00 | 257.51 | 258.29 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:05:00 | 248.39 | 246.62 | 0.00 | ORB-long ORB[245.00,247.40] vol=1.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:45:00 | 249.79 | 247.18 | 0.00 | T1 1.5R @ 249.79 |
| Stop hit — per-position SL triggered | 2026-05-04 13:25:00 | 248.39 | 248.82 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 247.19 | 250.13 | 0.00 | ORB-short ORB[249.25,252.49] vol=3.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:05:00 | 245.94 | 249.41 | 0.00 | T1 1.5R @ 245.94 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 247.19 | 249.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:50:00 | 275.55 | 2026-02-18 10:55:00 | 276.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-19 10:50:00 | 275.70 | 2026-02-19 10:55:00 | 274.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-19 10:50:00 | 275.70 | 2026-02-19 11:00:00 | 275.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:45:00 | 253.05 | 2026-02-25 10:30:00 | 254.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-10 09:40:00 | 238.94 | 2026-04-10 10:05:00 | 237.55 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-10 09:40:00 | 238.94 | 2026-04-10 10:30:00 | 238.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:25:00 | 239.32 | 2026-04-13 10:45:00 | 240.78 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-13 10:25:00 | 239.32 | 2026-04-13 11:30:00 | 239.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 10:00:00 | 243.79 | 2026-04-15 10:10:00 | 245.25 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-15 10:00:00 | 243.79 | 2026-04-15 15:20:00 | 246.15 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2026-04-21 09:40:00 | 258.13 | 2026-04-21 10:50:00 | 257.23 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-27 09:30:00 | 256.50 | 2026-04-27 09:35:00 | 257.51 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-04 10:05:00 | 248.39 | 2026-05-04 10:45:00 | 249.79 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-05-04 10:05:00 | 248.39 | 2026-05-04 13:25:00 | 248.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 11:00:00 | 247.19 | 2026-05-05 11:05:00 | 245.94 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-05 11:00:00 | 247.19 | 2026-05-05 11:10:00 | 247.19 | STOP_HIT | 0.50 | 0.00% |

# IDFC First Bank Ltd. (IDFCFIRSTB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 71.19
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 5
- **Target hits / Stop hits / Partials:** 6 / 5 / 7
- **Avg / median % per leg:** 0.47% / 0.42%
- **Sum % (uncompounded):** 8.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.54% | 4.3% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.54% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 0.42% | 4.2% |
| SELL @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 4 | 2 | 4 | 0.42% | 4.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 13 | 72.2% | 6 | 5 | 7 | 0.47% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:15:00 | 83.00 | 83.21 | 0.00 | ORB-short ORB[83.55,84.09] vol=1.8x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:15:00 | 82.72 | 83.11 | 0.00 | T1 1.5R @ 82.72 |
| Target hit | 2026-02-11 15:20:00 | 82.44 | 82.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 80.79 | 81.09 | 0.00 | ORB-short ORB[80.90,81.85] vol=2.0x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:00:00 | 80.45 | 80.91 | 0.00 | T1 1.5R @ 80.45 |
| Target hit | 2026-02-13 10:30:00 | 80.76 | 80.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 70.71 | 70.24 | 0.00 | ORB-long ORB[69.85,70.65] vol=1.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:05:00 | 71.07 | 70.47 | 0.00 | T1 1.5R @ 71.07 |
| Target hit | 2026-02-26 15:20:00 | 72.83 | 72.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 73.04 | 72.46 | 0.00 | ORB-long ORB[72.25,72.91] vol=2.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:15:00 | 73.41 | 72.55 | 0.00 | T1 1.5R @ 73.41 |
| Stop hit — per-position SL triggered | 2026-02-27 12:00:00 | 73.04 | 72.76 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 63.49 | 64.05 | 0.00 | ORB-short ORB[64.00,64.60] vol=1.7x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 63.14 | 63.79 | 0.00 | T1 1.5R @ 63.14 |
| Target hit | 2026-03-13 15:20:00 | 62.51 | 62.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:05:00 | 61.70 | 62.28 | 0.00 | ORB-short ORB[62.30,62.95] vol=1.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-03-16 11:20:00 | 61.95 | 62.21 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:10:00 | 68.20 | 67.99 | 0.00 | ORB-long ORB[67.11,68.09] vol=7.8x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 68.01 | 68.05 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:20:00 | 68.20 | 68.00 | 0.00 | ORB-long ORB[67.50,68.14] vol=2.6x ATR=0.18 |
| Stop hit — per-position SL triggered | 2026-04-23 10:40:00 | 68.02 | 68.07 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 67.73 | 67.96 | 0.00 | ORB-short ORB[67.83,68.64] vol=2.2x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:10:00 | 67.46 | 67.93 | 0.00 | T1 1.5R @ 67.46 |
| Target hit | 2026-04-24 15:20:00 | 67.18 | 67.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 70.00 | 69.23 | 0.00 | ORB-long ORB[68.61,69.38] vol=2.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:10:00 | 70.37 | 69.65 | 0.00 | T1 1.5R @ 70.37 |
| Target hit | 2026-04-29 14:15:00 | 70.21 | 70.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 68.92 | 69.22 | 0.00 | ORB-short ORB[69.14,69.59] vol=2.1x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 69.07 | 69.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:15:00 | 83.00 | 2026-02-11 12:15:00 | 82.72 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-11 11:15:00 | 83.00 | 2026-02-11 15:20:00 | 82.44 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-13 09:30:00 | 80.79 | 2026-02-13 10:00:00 | 80.45 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-13 09:30:00 | 80.79 | 2026-02-13 10:30:00 | 80.76 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-02-26 09:45:00 | 70.71 | 2026-02-26 10:05:00 | 71.07 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-26 09:45:00 | 70.71 | 2026-02-26 15:20:00 | 72.83 | TARGET_HIT | 0.50 | 3.00% |
| BUY | retest1 | 2026-02-27 11:10:00 | 73.04 | 2026-02-27 11:15:00 | 73.41 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-27 11:10:00 | 73.04 | 2026-02-27 12:00:00 | 73.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:55:00 | 63.49 | 2026-03-13 10:20:00 | 63.14 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 09:55:00 | 63.49 | 2026-03-13 15:20:00 | 62.51 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2026-03-16 11:05:00 | 61.70 | 2026-03-16 11:20:00 | 61.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-16 10:10:00 | 68.20 | 2026-04-16 10:30:00 | 68.01 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-23 10:20:00 | 68.20 | 2026-04-23 10:40:00 | 68.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 10:50:00 | 67.73 | 2026-04-24 11:10:00 | 67.46 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-24 10:50:00 | 67.73 | 2026-04-24 15:20:00 | 67.18 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2026-04-29 10:00:00 | 70.00 | 2026-04-29 11:10:00 | 70.37 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-29 10:00:00 | 70.00 | 2026-04-29 14:15:00 | 70.21 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-05-05 10:50:00 | 68.92 | 2026-05-05 11:05:00 | 69.07 | STOP_HIT | 1.00 | -0.21% |

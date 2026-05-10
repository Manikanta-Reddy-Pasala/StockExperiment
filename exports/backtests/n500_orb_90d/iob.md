# Indian Overseas Bank (IOB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 34.75
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 4
- **Avg / median % per leg:** 0.07% / -0.24%
- **Sum % (uncompounded):** 1.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.27% | 2.7% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.27% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.15% | -1.4% |
| SELL @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.15% | -1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 6 | 31.6% | 2 | 13 | 4 | 0.07% | 1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 34.73 | 34.89 | 0.00 | ORB-short ORB[34.78,35.20] vol=1.8x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 34.83 | 34.85 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 35.53 | 35.35 | 0.00 | ORB-long ORB[35.07,35.33] vol=2.1x ATR=0.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:25:00 | 35.68 | 35.38 | 0.00 | T1 1.5R @ 35.68 |
| Target hit | 2026-02-17 14:55:00 | 35.93 | 35.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 36.55 | 36.39 | 0.00 | ORB-long ORB[36.19,36.49] vol=2.3x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:35:00 | 36.75 | 36.54 | 0.00 | T1 1.5R @ 36.75 |
| Stop hit — per-position SL triggered | 2026-02-18 15:20:00 | 36.55 | 36.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 36.19 | 36.36 | 0.00 | ORB-short ORB[36.27,36.74] vol=1.7x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 36.30 | 36.35 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 37.25 | 36.81 | 0.00 | ORB-long ORB[36.50,36.81] vol=5.2x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-02-25 09:55:00 | 37.10 | 36.94 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 33.68 | 33.86 | 0.00 | ORB-short ORB[33.74,34.11] vol=1.6x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-03-11 12:05:00 | 33.79 | 33.78 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 33.39 | 33.20 | 0.00 | ORB-long ORB[32.91,33.29] vol=1.7x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:55:00 | 33.56 | 33.24 | 0.00 | T1 1.5R @ 33.56 |
| Target hit | 2026-03-12 15:20:00 | 33.86 | 33.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 32.03 | 32.16 | 0.00 | ORB-short ORB[32.08,32.54] vol=1.7x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-03-17 14:10:00 | 32.14 | 32.12 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 35.07 | 35.25 | 0.00 | ORB-short ORB[35.11,35.48] vol=1.6x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 35.15 | 35.23 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 35.22 | 35.09 | 0.00 | ORB-long ORB[34.91,35.15] vol=3.4x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-04-21 10:05:00 | 35.10 | 35.11 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 35.26 | 35.08 | 0.00 | ORB-long ORB[34.71,35.19] vol=1.7x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 35.15 | 35.12 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 35.15 | 35.30 | 0.00 | ORB-short ORB[35.16,35.48] vol=1.8x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:10:00 | 34.92 | 35.25 | 0.00 | T1 1.5R @ 34.92 |
| Stop hit — per-position SL triggered | 2026-04-30 13:40:00 | 35.15 | 35.18 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 34.94 | 35.12 | 0.00 | ORB-short ORB[35.08,35.38] vol=2.0x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-05-04 09:45:00 | 35.04 | 35.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 34.70 | 34.83 | 0.00 | ORB-short ORB[34.71,35.00] vol=2.2x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-05-05 13:20:00 | 34.78 | 34.75 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 35.06 | 34.95 | 0.00 | ORB-long ORB[34.75,35.04] vol=2.1x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-05-06 10:00:00 | 34.96 | 34.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 34.73 | 2026-02-13 09:40:00 | 34.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:20:00 | 35.53 | 2026-02-17 10:25:00 | 35.68 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:20:00 | 35.53 | 2026-02-17 14:55:00 | 35.93 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2026-02-18 09:30:00 | 36.55 | 2026-02-18 12:35:00 | 36.75 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-18 09:30:00 | 36.55 | 2026-02-18 15:20:00 | 36.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 36.19 | 2026-02-24 09:35:00 | 36.30 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-25 09:50:00 | 37.25 | 2026-02-25 09:55:00 | 37.10 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-11 09:55:00 | 33.68 | 2026-03-11 12:05:00 | 33.79 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-12 11:15:00 | 33.39 | 2026-03-12 11:55:00 | 33.56 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-12 11:15:00 | 33.39 | 2026-03-12 15:20:00 | 33.86 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2026-03-17 11:15:00 | 32.03 | 2026-03-17 14:10:00 | 32.14 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-16 09:55:00 | 35.07 | 2026-04-16 10:15:00 | 35.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-21 09:45:00 | 35.22 | 2026-04-21 10:05:00 | 35.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-22 09:40:00 | 35.26 | 2026-04-22 10:00:00 | 35.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-30 09:50:00 | 35.15 | 2026-04-30 10:10:00 | 34.92 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-04-30 09:50:00 | 35.15 | 2026-04-30 13:40:00 | 35.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 09:40:00 | 34.94 | 2026-05-04 09:45:00 | 35.04 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-05 11:05:00 | 34.70 | 2026-05-05 13:20:00 | 34.78 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-06 09:50:00 | 35.06 | 2026-05-06 10:00:00 | 34.96 | STOP_HIT | 1.00 | -0.28% |

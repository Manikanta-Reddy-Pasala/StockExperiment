# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 134.13
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
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 6
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 0 | 9 | 3 | -0.04% | -0.5% |
| BUY @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 0 | 9 | 3 | -0.04% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.07% | 0.6% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.07% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 7 | 33.3% | 1 | 14 | 6 | 0.00% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 144.79 | 145.81 | 0.00 | ORB-short ORB[145.21,147.34] vol=1.9x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 144.27 | 145.24 | 0.00 | T1 1.5R @ 144.27 |
| Stop hit — per-position SL triggered | 2026-02-11 10:25:00 | 144.79 | 144.88 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:05:00 | 144.49 | 144.96 | 0.00 | ORB-short ORB[144.50,145.80] vol=2.1x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-02-12 10:10:00 | 144.87 | 144.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 141.98 | 142.79 | 0.00 | ORB-short ORB[142.61,143.80] vol=1.5x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 142.38 | 142.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 146.70 | 145.97 | 0.00 | ORB-long ORB[144.80,146.17] vol=2.5x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 147.30 | 146.21 | 0.00 | T1 1.5R @ 147.30 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 146.70 | 146.31 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 151.92 | 150.92 | 0.00 | ORB-long ORB[149.38,150.49] vol=2.4x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 151.46 | 150.95 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 156.89 | 156.11 | 0.00 | ORB-long ORB[154.74,156.30] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 156.46 | 156.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 157.24 | 156.08 | 0.00 | ORB-long ORB[155.00,155.95] vol=2.1x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:20:00 | 157.80 | 156.18 | 0.00 | T1 1.5R @ 157.80 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 157.24 | 156.48 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 158.63 | 158.34 | 0.00 | ORB-long ORB[156.90,158.57] vol=5.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-02-25 09:35:00 | 158.24 | 158.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 135.60 | 137.50 | 0.00 | ORB-short ORB[137.80,139.67] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 136.11 | 137.39 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 144.60 | 143.76 | 0.00 | ORB-long ORB[142.84,144.00] vol=2.2x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 144.22 | 143.85 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 145.37 | 144.65 | 0.00 | ORB-long ORB[143.13,144.90] vol=2.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 144.95 | 144.70 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 139.68 | 140.56 | 0.00 | ORB-short ORB[140.05,141.86] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:45:00 | 138.91 | 140.26 | 0.00 | T1 1.5R @ 138.91 |
| Stop hit — per-position SL triggered | 2026-04-24 12:10:00 | 139.68 | 139.44 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 141.10 | 141.61 | 0.00 | ORB-short ORB[141.18,142.42] vol=1.8x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:30:00 | 140.40 | 141.22 | 0.00 | T1 1.5R @ 140.40 |
| Target hit | 2026-04-27 13:30:00 | 140.90 | 140.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 138.03 | 137.45 | 0.00 | ORB-long ORB[136.60,137.86] vol=4.6x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:55:00 | 138.71 | 137.61 | 0.00 | T1 1.5R @ 138.71 |
| Stop hit — per-position SL triggered | 2026-04-29 13:35:00 | 138.03 | 137.94 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 137.66 | 136.92 | 0.00 | ORB-long ORB[136.26,137.49] vol=3.1x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 137.09 | 137.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 144.79 | 2026-02-11 09:45:00 | 144.27 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-11 09:30:00 | 144.79 | 2026-02-11 10:25:00 | 144.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:05:00 | 144.49 | 2026-02-12 10:10:00 | 144.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-13 09:35:00 | 141.98 | 2026-02-13 09:40:00 | 142.38 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-17 10:20:00 | 146.70 | 2026-02-17 10:30:00 | 147.30 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-17 10:20:00 | 146.70 | 2026-02-17 10:40:00 | 146.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:10:00 | 151.92 | 2026-02-18 10:15:00 | 151.46 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-23 10:55:00 | 156.89 | 2026-02-23 11:05:00 | 156.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 11:10:00 | 157.24 | 2026-02-24 11:20:00 | 157.80 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-24 11:10:00 | 157.24 | 2026-02-24 11:45:00 | 157.24 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:30:00 | 158.63 | 2026-02-25 09:35:00 | 158.24 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-13 10:40:00 | 135.60 | 2026-03-13 10:50:00 | 136.11 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-21 09:30:00 | 144.60 | 2026-04-21 09:40:00 | 144.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-22 09:45:00 | 145.37 | 2026-04-22 09:55:00 | 144.95 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-24 09:30:00 | 139.68 | 2026-04-24 09:45:00 | 138.91 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-24 09:30:00 | 139.68 | 2026-04-24 12:10:00 | 139.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 09:30:00 | 141.10 | 2026-04-27 10:30:00 | 140.40 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-27 09:30:00 | 141.10 | 2026-04-27 13:30:00 | 140.90 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2026-04-29 10:20:00 | 138.03 | 2026-04-29 10:55:00 | 138.71 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-29 10:20:00 | 138.03 | 2026-04-29 13:35:00 | 138.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:45:00 | 137.66 | 2026-05-06 09:55:00 | 137.09 | STOP_HIT | 1.00 | -0.41% |

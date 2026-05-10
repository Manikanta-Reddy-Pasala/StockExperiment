# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 83.90
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
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 7
- **Target hits / Stop hits / Partials:** 4 / 7 / 5
- **Avg / median % per leg:** 0.42% / 0.43%
- **Sum % (uncompounded):** 6.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.51% | 5.6% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.51% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.24% | 1.2% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.24% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 9 | 56.2% | 4 | 7 | 5 | 0.42% | 6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 66.37 | 66.13 | 0.00 | ORB-long ORB[65.73,66.29] vol=2.2x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:55:00 | 66.65 | 66.27 | 0.00 | T1 1.5R @ 66.65 |
| Target hit | 2026-02-12 12:00:00 | 66.55 | 66.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 66.23 | 65.91 | 0.00 | ORB-long ORB[65.44,65.98] vol=2.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:25:00 | 66.52 | 66.02 | 0.00 | T1 1.5R @ 66.52 |
| Target hit | 2026-02-17 15:20:00 | 67.35 | 67.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 68.93 | 69.06 | 0.00 | ORB-short ORB[68.97,69.85] vol=1.5x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:40:00 | 68.63 | 69.05 | 0.00 | T1 1.5R @ 68.63 |
| Target hit | 2026-02-19 15:20:00 | 67.70 | 68.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 73.95 | 72.83 | 0.00 | ORB-long ORB[71.68,72.76] vol=4.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:00:00 | 74.53 | 73.26 | 0.00 | T1 1.5R @ 74.53 |
| Stop hit — per-position SL triggered | 2026-02-25 10:20:00 | 73.95 | 73.58 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 66.69 | 66.02 | 0.00 | ORB-long ORB[65.45,66.24] vol=1.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-03-12 09:50:00 | 66.37 | 66.06 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 70.44 | 70.69 | 0.00 | ORB-short ORB[70.45,71.45] vol=1.9x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 70.68 | 70.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 70.96 | 71.53 | 0.00 | ORB-short ORB[71.25,72.24] vol=1.6x ATR=0.21 |
| Stop hit — per-position SL triggered | 2026-04-15 11:50:00 | 71.17 | 71.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 72.09 | 71.64 | 0.00 | ORB-long ORB[71.28,71.99] vol=3.0x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 71.85 | 71.67 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 79.91 | 79.48 | 0.00 | ORB-long ORB[78.93,79.80] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 79.60 | 79.54 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 79.22 | 78.65 | 0.00 | ORB-long ORB[78.13,78.90] vol=1.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:00:00 | 79.70 | 78.88 | 0.00 | T1 1.5R @ 79.70 |
| Target hit | 2026-05-05 15:20:00 | 81.25 | 80.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 83.85 | 84.65 | 0.00 | ORB-short ORB[84.40,85.31] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 84.16 | 84.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:35:00 | 66.37 | 2026-02-12 09:55:00 | 66.65 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-12 09:35:00 | 66.37 | 2026-02-12 12:00:00 | 66.55 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-17 10:20:00 | 66.23 | 2026-02-17 10:25:00 | 66.52 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:20:00 | 66.23 | 2026-02-17 15:20:00 | 67.35 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2026-02-19 11:05:00 | 68.93 | 2026-02-19 11:40:00 | 68.63 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-19 11:05:00 | 68.93 | 2026-02-19 15:20:00 | 67.70 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2026-02-25 09:50:00 | 73.95 | 2026-02-25 10:00:00 | 74.53 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-25 09:50:00 | 73.95 | 2026-02-25 10:20:00 | 73.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 09:45:00 | 66.69 | 2026-03-12 09:50:00 | 66.37 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-04-10 10:05:00 | 70.44 | 2026-04-10 10:15:00 | 70.68 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-15 11:05:00 | 70.96 | 2026-04-15 11:50:00 | 71.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-16 10:15:00 | 72.09 | 2026-04-16 10:20:00 | 71.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-04 09:35:00 | 79.91 | 2026-05-04 09:50:00 | 79.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-05 09:45:00 | 79.22 | 2026-05-05 10:00:00 | 79.70 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-05 09:45:00 | 79.22 | 2026-05-05 15:20:00 | 81.25 | TARGET_HIT | 0.50 | 2.56% |
| SELL | retest1 | 2026-05-08 09:40:00 | 83.85 | 2026-05-08 10:10:00 | 84.16 | STOP_HIT | 1.00 | -0.37% |

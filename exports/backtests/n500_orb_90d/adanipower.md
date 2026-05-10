# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 225.02
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 5
- **Avg / median % per leg:** 0.59% / 0.49%
- **Sum % (uncompounded):** 7.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.05% | -0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.05% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 7 | 77.8% | 3 | 2 | 4 | 0.87% | 7.8% |
| SELL @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 3 | 2 | 4 | 0.87% | 7.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.59% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 149.09 | 150.43 | 0.00 | ORB-short ORB[150.32,152.00] vol=1.9x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 148.32 | 150.09 | 0.00 | T1 1.5R @ 148.32 |
| Stop hit — per-position SL triggered | 2026-02-10 10:45:00 | 149.09 | 150.02 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 149.02 | 149.61 | 0.00 | ORB-short ORB[149.25,150.75] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-02-12 11:45:00 | 149.33 | 149.58 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 143.70 | 145.08 | 0.00 | ORB-short ORB[144.76,146.43] vol=1.9x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:30:00 | 142.97 | 144.47 | 0.00 | T1 1.5R @ 142.97 |
| Target hit | 2026-02-19 15:20:00 | 140.25 | 142.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 144.04 | 144.26 | 0.00 | ORB-short ORB[144.30,145.60] vol=4.2x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:55:00 | 143.41 | 144.22 | 0.00 | T1 1.5R @ 143.41 |
| Target hit | 2026-02-25 15:20:00 | 140.49 | 142.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:30:00 | 170.23 | 167.82 | 0.00 | ORB-long ORB[166.21,168.38] vol=5.4x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:35:00 | 171.45 | 169.25 | 0.00 | T1 1.5R @ 171.45 |
| Stop hit — per-position SL triggered | 2026-04-08 10:45:00 | 170.23 | 169.87 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 187.34 | 185.15 | 0.00 | ORB-long ORB[183.27,185.60] vol=3.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 186.40 | 185.46 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:25:00 | 232.84 | 231.46 | 0.00 | ORB-long ORB[229.88,232.00] vol=3.3x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-05-07 10:50:00 | 231.87 | 231.72 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 227.62 | 229.58 | 0.00 | ORB-short ORB[228.71,231.43] vol=1.7x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:40:00 | 226.51 | 229.12 | 0.00 | T1 1.5R @ 226.51 |
| Target hit | 2026-05-08 15:20:00 | 224.85 | 227.85 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 149.09 | 2026-02-10 10:40:00 | 148.32 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-10 10:35:00 | 149.09 | 2026-02-10 10:45:00 | 149.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 11:15:00 | 149.02 | 2026-02-12 11:45:00 | 149.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-19 09:40:00 | 143.70 | 2026-02-19 10:30:00 | 142.97 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-19 09:40:00 | 143.70 | 2026-02-19 15:20:00 | 140.25 | TARGET_HIT | 0.50 | 2.40% |
| SELL | retest1 | 2026-02-25 10:50:00 | 144.04 | 2026-02-25 10:55:00 | 143.41 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-25 10:50:00 | 144.04 | 2026-02-25 15:20:00 | 140.49 | TARGET_HIT | 0.50 | 2.46% |
| BUY | retest1 | 2026-04-08 10:30:00 | 170.23 | 2026-04-08 10:35:00 | 171.45 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-08 10:30:00 | 170.23 | 2026-04-08 10:45:00 | 170.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 09:45:00 | 187.34 | 2026-04-16 09:55:00 | 186.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-07 10:25:00 | 232.84 | 2026-05-07 10:50:00 | 231.87 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-08 10:50:00 | 227.62 | 2026-05-08 11:40:00 | 226.51 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-05-08 10:50:00 | 227.62 | 2026-05-08 15:20:00 | 224.85 | TARGET_HIT | 0.50 | 1.22% |

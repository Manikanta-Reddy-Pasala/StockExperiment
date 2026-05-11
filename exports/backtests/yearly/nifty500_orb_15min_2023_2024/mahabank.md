# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53685 bars)
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 7
- **Avg / median % per leg:** 0.45% / 0.56%
- **Sum % (uncompounded):** 8.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.50% | 6.9% |
| BUY @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.50% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.32% | 1.6% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.32% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.45% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 09:55:00 | 54.20 | 53.71 | 0.00 | ORB-long ORB[53.30,53.95] vol=2.1x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-01-25 10:10:00 | 53.88 | 53.76 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-01-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:45:00 | 55.00 | 54.22 | 0.00 | ORB-long ORB[53.80,54.55] vol=2.7x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 10:50:00 | 55.48 | 54.41 | 0.00 | T1 1.5R @ 55.48 |
| Stop hit — per-position SL triggered | 2024-01-29 10:55:00 | 55.00 | 54.44 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-01-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 11:10:00 | 54.75 | 54.00 | 0.00 | ORB-long ORB[53.65,54.40] vol=3.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 11:15:00 | 55.14 | 54.19 | 0.00 | T1 1.5R @ 55.14 |
| Stop hit — per-position SL triggered | 2024-01-31 11:20:00 | 54.75 | 54.24 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:15:00 | 54.60 | 55.31 | 0.00 | ORB-short ORB[55.05,55.80] vol=1.5x ATR=0.24 |
| Stop hit — per-position SL triggered | 2024-02-01 11:20:00 | 54.84 | 55.29 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:30:00 | 62.60 | 62.23 | 0.00 | ORB-long ORB[61.65,62.55] vol=1.9x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:35:00 | 63.02 | 62.49 | 0.00 | T1 1.5R @ 63.02 |
| Stop hit — per-position SL triggered | 2024-03-11 09:55:00 | 62.60 | 62.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-03-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 09:40:00 | 58.40 | 58.94 | 0.00 | ORB-short ORB[58.85,59.40] vol=1.7x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:10:00 | 57.97 | 58.64 | 0.00 | T1 1.5R @ 57.97 |
| Stop hit — per-position SL triggered | 2024-03-20 11:05:00 | 58.40 | 58.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-03-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:40:00 | 60.20 | 59.70 | 0.00 | ORB-long ORB[59.30,59.75] vol=2.0x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 09:45:00 | 60.60 | 60.22 | 0.00 | T1 1.5R @ 60.60 |
| Target hit | 2024-03-28 15:20:00 | 62.35 | 61.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:00:00 | 65.15 | 64.54 | 0.00 | ORB-long ORB[64.05,64.90] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-04-09 10:15:00 | 64.86 | 64.77 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-04-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:35:00 | 64.75 | 64.26 | 0.00 | ORB-long ORB[63.80,64.65] vol=1.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2024-04-10 10:40:00 | 64.49 | 64.27 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-04-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:25:00 | 65.00 | 64.49 | 0.00 | ORB-long ORB[64.10,64.90] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-04-12 10:35:00 | 64.72 | 64.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 11:15:00 | 63.30 | 62.73 | 0.00 | ORB-long ORB[62.00,62.95] vol=3.4x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 13:25:00 | 63.65 | 63.00 | 0.00 | T1 1.5R @ 63.65 |
| Target hit | 2024-04-22 15:20:00 | 64.40 | 63.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-05-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:20:00 | 68.85 | 69.59 | 0.00 | ORB-short ORB[69.85,70.35] vol=2.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 10:40:00 | 68.45 | 69.36 | 0.00 | T1 1.5R @ 68.45 |
| Target hit | 2024-05-03 15:20:00 | 68.35 | 68.72 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-25 09:55:00 | 54.20 | 2024-01-25 10:10:00 | 53.88 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-01-29 10:45:00 | 55.00 | 2024-01-29 10:50:00 | 55.48 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-01-29 10:45:00 | 55.00 | 2024-01-29 10:55:00 | 55.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-31 11:10:00 | 54.75 | 2024-01-31 11:15:00 | 55.14 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-01-31 11:10:00 | 54.75 | 2024-01-31 11:20:00 | 54.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-01 11:15:00 | 54.60 | 2024-02-01 11:20:00 | 54.84 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-03-11 09:30:00 | 62.60 | 2024-03-11 09:35:00 | 63.02 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-03-11 09:30:00 | 62.60 | 2024-03-11 09:55:00 | 62.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 09:40:00 | 58.40 | 2024-03-20 10:10:00 | 57.97 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-03-20 09:40:00 | 58.40 | 2024-03-20 11:05:00 | 58.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-28 09:40:00 | 60.20 | 2024-03-28 09:45:00 | 60.60 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-03-28 09:40:00 | 60.20 | 2024-03-28 15:20:00 | 62.35 | TARGET_HIT | 0.50 | 3.57% |
| BUY | retest1 | 2024-04-09 10:00:00 | 65.15 | 2024-04-09 10:15:00 | 64.86 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-04-10 10:35:00 | 64.75 | 2024-04-10 10:40:00 | 64.49 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-04-12 10:25:00 | 65.00 | 2024-04-12 10:35:00 | 64.72 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-04-22 11:15:00 | 63.30 | 2024-04-22 13:25:00 | 63.65 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-04-22 11:15:00 | 63.30 | 2024-04-22 15:20:00 | 64.40 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2024-05-03 10:20:00 | 68.85 | 2024-05-03 10:40:00 | 68.45 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-05-03 10:20:00 | 68.85 | 2024-05-03 15:20:00 | 68.35 | TARGET_HIT | 0.50 | 0.73% |

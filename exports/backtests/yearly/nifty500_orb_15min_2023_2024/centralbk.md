# Central Bank of India (CENTRALBK)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-07-29 15:25:00 (41167 bars)
- **Last close:** 36.97
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 13
- **Target hits / Stop hits / Partials:** 4 / 13 / 7
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 5.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 8 | 53.3% | 3 | 7 | 5 | 0.34% | 5.1% |
| BUY @ 2nd Alert (retest1) | 15 | 8 | 53.3% | 3 | 7 | 5 | 0.34% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.05% | 0.4% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.05% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 11 | 45.8% | 4 | 13 | 7 | 0.23% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 09:50:00 | 51.40 | 50.75 | 0.00 | ORB-long ORB[50.20,50.80] vol=2.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-12-29 09:55:00 | 51.07 | 50.82 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 51.60 | 50.63 | 0.00 | ORB-long ORB[49.90,50.40] vol=4.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-01-02 09:40:00 | 51.31 | 51.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-01-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:45:00 | 52.70 | 52.47 | 0.00 | ORB-long ORB[52.00,52.50] vol=4.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-01-04 09:55:00 | 52.42 | 52.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-01-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:45:00 | 51.75 | 52.67 | 0.00 | ORB-short ORB[52.70,53.25] vol=2.2x ATR=0.20 |
| Stop hit — per-position SL triggered | 2024-01-05 10:50:00 | 51.95 | 52.53 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-01-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 11:00:00 | 50.60 | 50.94 | 0.00 | ORB-short ORB[50.75,51.45] vol=2.1x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-01-09 12:35:00 | 50.77 | 50.90 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-01-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:25:00 | 51.00 | 50.55 | 0.00 | ORB-long ORB[50.25,50.80] vol=3.2x ATR=0.21 |
| Stop hit — per-position SL triggered | 2024-01-11 10:35:00 | 50.79 | 50.61 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:40:00 | 51.15 | 50.87 | 0.00 | ORB-long ORB[50.70,51.05] vol=3.9x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 09:50:00 | 51.47 | 51.10 | 0.00 | T1 1.5R @ 51.47 |
| Target hit | 2024-01-12 11:55:00 | 51.75 | 51.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:20:00 | 55.30 | 54.80 | 0.00 | ORB-long ORB[54.50,55.15] vol=2.1x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 10:25:00 | 55.70 | 55.08 | 0.00 | T1 1.5R @ 55.70 |
| Target hit | 2024-01-29 13:05:00 | 56.20 | 56.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-01-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 11:05:00 | 56.70 | 56.26 | 0.00 | ORB-long ORB[55.70,56.45] vol=3.8x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 11:10:00 | 57.14 | 56.33 | 0.00 | T1 1.5R @ 57.14 |
| Target hit | 2024-01-31 13:10:00 | 57.30 | 57.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:15:00 | 56.20 | 57.34 | 0.00 | ORB-short ORB[57.40,57.95] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-02-01 11:20:00 | 56.51 | 57.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-02-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:45:00 | 66.75 | 66.34 | 0.00 | ORB-long ORB[65.90,66.65] vol=2.8x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 09:50:00 | 67.24 | 66.44 | 0.00 | T1 1.5R @ 67.24 |
| Stop hit — per-position SL triggered | 2024-02-21 10:25:00 | 66.75 | 67.01 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 61.50 | 62.69 | 0.00 | ORB-short ORB[63.15,63.85] vol=1.7x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:55:00 | 61.02 | 62.60 | 0.00 | T1 1.5R @ 61.02 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 61.50 | 62.48 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 11:10:00 | 58.25 | 58.75 | 0.00 | ORB-short ORB[58.55,59.40] vol=1.7x ATR=0.21 |
| Stop hit — per-position SL triggered | 2024-03-27 11:55:00 | 58.46 | 58.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-03-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:45:00 | 59.00 | 58.35 | 0.00 | ORB-long ORB[57.75,58.35] vol=2.0x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 09:55:00 | 59.53 | 58.58 | 0.00 | T1 1.5R @ 59.53 |
| Stop hit — per-position SL triggered | 2024-03-28 10:00:00 | 59.00 | 58.61 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 11:15:00 | 64.30 | 65.93 | 0.00 | ORB-short ORB[66.20,67.00] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-04-04 11:20:00 | 64.67 | 65.84 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 11:10:00 | 64.10 | 63.30 | 0.00 | ORB-long ORB[62.75,63.70] vol=3.7x ATR=0.24 |
| Stop hit — per-position SL triggered | 2024-04-22 11:15:00 | 63.86 | 63.33 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-05-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 09:35:00 | 66.00 | 66.35 | 0.00 | ORB-short ORB[66.10,66.70] vol=2.1x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 09:55:00 | 65.67 | 66.18 | 0.00 | T1 1.5R @ 65.67 |
| Target hit | 2024-05-03 15:20:00 | 65.10 | 65.40 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-29 09:50:00 | 51.40 | 2023-12-29 09:55:00 | 51.07 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-01-02 09:35:00 | 51.60 | 2024-01-02 09:40:00 | 51.31 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-01-04 09:45:00 | 52.70 | 2024-01-04 09:55:00 | 52.42 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-01-05 10:45:00 | 51.75 | 2024-01-05 10:50:00 | 51.95 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-01-09 11:00:00 | 50.60 | 2024-01-09 12:35:00 | 50.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-01-11 10:25:00 | 51.00 | 2024-01-11 10:35:00 | 50.79 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-01-12 09:40:00 | 51.15 | 2024-01-12 09:50:00 | 51.47 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-01-12 09:40:00 | 51.15 | 2024-01-12 11:55:00 | 51.75 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2024-01-29 10:20:00 | 55.30 | 2024-01-29 10:25:00 | 55.70 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-01-29 10:20:00 | 55.30 | 2024-01-29 13:05:00 | 56.20 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2024-01-31 11:05:00 | 56.70 | 2024-01-31 11:10:00 | 57.14 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-01-31 11:05:00 | 56.70 | 2024-01-31 13:10:00 | 57.30 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2024-02-01 11:15:00 | 56.20 | 2024-02-01 11:20:00 | 56.51 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-02-21 09:45:00 | 66.75 | 2024-02-21 09:50:00 | 67.24 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-02-21 09:45:00 | 66.75 | 2024-02-21 10:25:00 | 66.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:50:00 | 61.50 | 2024-02-28 10:55:00 | 61.02 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-02-28 10:50:00 | 61.50 | 2024-02-28 11:00:00 | 61.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-27 11:10:00 | 58.25 | 2024-03-27 11:55:00 | 58.46 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-03-28 09:45:00 | 59.00 | 2024-03-28 09:55:00 | 59.53 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-03-28 09:45:00 | 59.00 | 2024-03-28 10:00:00 | 59.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 11:15:00 | 64.30 | 2024-04-04 11:20:00 | 64.67 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-04-22 11:10:00 | 64.10 | 2024-04-22 11:15:00 | 63.86 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-05-03 09:35:00 | 66.00 | 2024-05-03 09:55:00 | 65.67 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-03 09:35:00 | 66.00 | 2024-05-03 15:20:00 | 65.10 | TARGET_HIT | 0.50 | 1.36% |

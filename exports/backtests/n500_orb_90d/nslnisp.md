# NMDC Steel Ltd. (NSLNISP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 43.60
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 7
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 1.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.08% | 1.2% |
| BUY @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.08% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | -0.01% | -0.1% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 2 | 6 | 2 | -0.01% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 10 | 40.0% | 3 | 15 | 7 | 0.04% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 41.90 | 41.45 | 0.00 | ORB-long ORB[41.14,41.75] vol=5.9x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:10:00 | 42.24 | 41.62 | 0.00 | T1 1.5R @ 42.24 |
| Stop hit — per-position SL triggered | 2026-02-09 12:15:00 | 41.90 | 41.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:00:00 | 42.66 | 42.35 | 0.00 | ORB-long ORB[41.91,42.47] vol=3.0x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 42.92 | 42.44 | 0.00 | T1 1.5R @ 42.92 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 42.66 | 42.62 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:50:00 | 40.00 | 40.16 | 0.00 | ORB-short ORB[40.03,40.41] vol=2.4x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-02-17 11:40:00 | 40.10 | 40.13 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 40.80 | 40.57 | 0.00 | ORB-long ORB[40.18,40.77] vol=2.7x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-02-18 09:35:00 | 40.68 | 40.61 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 39.78 | 39.89 | 0.00 | ORB-short ORB[39.90,40.23] vol=1.8x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 39.90 | 39.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 38.94 | 39.13 | 0.00 | ORB-short ORB[39.00,39.33] vol=1.7x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 38.76 | 39.02 | 0.00 | T1 1.5R @ 38.76 |
| Target hit | 2026-02-23 15:05:00 | 38.93 | 38.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 38.47 | 38.67 | 0.00 | ORB-short ORB[38.61,38.82] vol=1.6x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 38.58 | 38.64 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 39.49 | 39.25 | 0.00 | ORB-long ORB[38.88,39.35] vol=1.8x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:05:00 | 39.72 | 39.32 | 0.00 | T1 1.5R @ 39.72 |
| Target hit | 2026-02-25 11:30:00 | 39.55 | 39.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 39.92 | 39.60 | 0.00 | ORB-long ORB[39.32,39.69] vol=3.2x ATR=0.16 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 39.76 | 39.76 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 41.68 | 42.11 | 0.00 | ORB-short ORB[42.01,42.63] vol=2.9x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:20:00 | 41.42 | 42.00 | 0.00 | T1 1.5R @ 41.42 |
| Target hit | 2026-04-15 15:20:00 | 41.41 | 41.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:05:00 | 41.48 | 41.89 | 0.00 | ORB-short ORB[41.73,42.25] vol=1.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 41.66 | 41.73 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 42.34 | 42.15 | 0.00 | ORB-long ORB[41.90,42.30] vol=2.0x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:45:00 | 42.56 | 42.34 | 0.00 | T1 1.5R @ 42.56 |
| Stop hit — per-position SL triggered | 2026-04-17 12:25:00 | 42.34 | 42.46 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 42.49 | 42.24 | 0.00 | ORB-long ORB[41.78,42.28] vol=3.6x ATR=0.16 |
| Stop hit — per-position SL triggered | 2026-04-22 10:20:00 | 42.33 | 42.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 42.29 | 41.96 | 0.00 | ORB-long ORB[41.58,42.20] vol=2.4x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:10:00 | 42.54 | 42.11 | 0.00 | T1 1.5R @ 42.54 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 42.29 | 42.14 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 42.57 | 41.83 | 0.00 | ORB-long ORB[41.23,41.80] vol=4.5x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 42.34 | 41.95 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 42.72 | 43.00 | 0.00 | ORB-short ORB[42.94,43.35] vol=2.8x ATR=0.14 |
| Stop hit — per-position SL triggered | 2026-05-04 11:05:00 | 42.86 | 42.99 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 42.67 | 43.05 | 0.00 | ORB-short ORB[42.71,43.34] vol=1.8x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-05-06 11:45:00 | 42.78 | 43.01 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 43.73 | 43.42 | 0.00 | ORB-long ORB[43.10,43.50] vol=2.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-05-07 09:45:00 | 43.54 | 43.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 41.90 | 2026-02-09 11:10:00 | 42.24 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-02-09 11:05:00 | 41.90 | 2026-02-09 12:15:00 | 41.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 10:00:00 | 42.66 | 2026-02-10 10:15:00 | 42.92 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-10 10:00:00 | 42.66 | 2026-02-10 10:40:00 | 42.66 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 10:50:00 | 40.00 | 2026-02-17 11:40:00 | 40.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-18 09:30:00 | 40.80 | 2026-02-18 09:35:00 | 40.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:05:00 | 39.78 | 2026-02-19 10:15:00 | 39.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-23 09:45:00 | 38.94 | 2026-02-23 10:40:00 | 38.76 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-23 09:45:00 | 38.94 | 2026-02-23 15:05:00 | 38.93 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2026-02-24 09:30:00 | 38.47 | 2026-02-24 09:35:00 | 38.58 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-25 09:55:00 | 39.49 | 2026-02-25 10:05:00 | 39.72 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-25 09:55:00 | 39.49 | 2026-02-25 11:30:00 | 39.55 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-02-26 09:30:00 | 39.92 | 2026-02-26 09:55:00 | 39.76 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-15 10:50:00 | 41.68 | 2026-04-15 11:20:00 | 41.42 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-15 10:50:00 | 41.68 | 2026-04-15 15:20:00 | 41.41 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2026-04-16 10:05:00 | 41.48 | 2026-04-16 10:30:00 | 41.66 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-17 09:45:00 | 42.34 | 2026-04-17 10:45:00 | 42.56 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-17 09:45:00 | 42.34 | 2026-04-17 12:25:00 | 42.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:30:00 | 42.49 | 2026-04-22 10:20:00 | 42.33 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-28 09:55:00 | 42.29 | 2026-04-28 10:10:00 | 42.54 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-28 09:55:00 | 42.29 | 2026-04-28 10:25:00 | 42.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:15:00 | 42.57 | 2026-04-29 10:20:00 | 42.34 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-05-04 11:00:00 | 42.72 | 2026-05-04 11:05:00 | 42.86 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-06 11:10:00 | 42.67 | 2026-05-06 11:45:00 | 42.78 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-07 09:40:00 | 43.73 | 2026-05-07 09:45:00 | 43.54 | STOP_HIT | 1.00 | -0.43% |

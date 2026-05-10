# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 74.79
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
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 9
- **Avg / median % per leg:** 0.47% / 0.43%
- **Sum % (uncompounded):** 9.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.55% | 4.4% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.55% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 7 | 53.8% | 1 | 6 | 6 | 0.43% | 5.6% |
| SELL @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 1 | 6 | 6 | 0.43% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 12 | 57.1% | 3 | 9 | 9 | 0.47% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 105.55 | 104.87 | 0.00 | ORB-long ORB[104.06,105.13] vol=2.9x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 106.16 | 105.10 | 0.00 | T1 1.5R @ 106.16 |
| Target hit | 2026-02-10 10:10:00 | 108.20 | 108.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 113.30 | 112.60 | 0.00 | ORB-long ORB[111.40,112.65] vol=5.0x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 114.03 | 113.29 | 0.00 | T1 1.5R @ 114.03 |
| Target hit | 2026-02-17 11:00:00 | 114.07 | 114.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 111.98 | 112.82 | 0.00 | ORB-short ORB[112.56,113.75] vol=2.3x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:35:00 | 111.36 | 112.49 | 0.00 | T1 1.5R @ 111.36 |
| Stop hit — per-position SL triggered | 2026-02-19 10:55:00 | 111.98 | 112.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:05:00 | 114.33 | 113.73 | 0.00 | ORB-long ORB[112.70,114.00] vol=3.1x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 115.11 | 114.11 | 0.00 | T1 1.5R @ 115.11 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 114.33 | 114.34 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 114.94 | 114.21 | 0.00 | ORB-long ORB[113.40,114.68] vol=3.1x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 114.45 | 114.29 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 102.00 | 103.04 | 0.00 | ORB-short ORB[103.02,104.05] vol=1.8x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:05:00 | 101.29 | 102.65 | 0.00 | T1 1.5R @ 101.29 |
| Target hit | 2026-03-11 15:20:00 | 99.12 | 100.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 74.02 | 74.71 | 0.00 | ORB-short ORB[74.47,75.47] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-04-16 10:35:00 | 74.24 | 74.52 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 75.10 | 74.80 | 0.00 | ORB-long ORB[74.30,74.97] vol=2.0x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-04-21 10:00:00 | 74.87 | 74.83 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 74.05 | 74.59 | 0.00 | ORB-short ORB[74.62,75.39] vol=5.7x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:00:00 | 73.70 | 74.33 | 0.00 | T1 1.5R @ 73.70 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 74.05 | 74.24 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 72.96 | 73.48 | 0.00 | ORB-short ORB[73.10,74.19] vol=1.9x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:40:00 | 72.65 | 73.35 | 0.00 | T1 1.5R @ 72.65 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 72.96 | 73.08 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 74.97 | 75.39 | 0.00 | ORB-short ORB[75.18,76.10] vol=1.8x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 74.62 | 75.26 | 0.00 | T1 1.5R @ 74.62 |
| Stop hit — per-position SL triggered | 2026-05-06 13:25:00 | 74.97 | 75.12 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:50:00 | 75.97 | 76.41 | 0.00 | ORB-short ORB[76.15,77.25] vol=3.6x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:05:00 | 75.65 | 76.34 | 0.00 | T1 1.5R @ 75.65 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 75.97 | 76.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:35:00 | 105.55 | 2026-02-10 09:45:00 | 106.16 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-10 09:35:00 | 105.55 | 2026-02-10 10:10:00 | 108.20 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2026-02-17 10:20:00 | 113.30 | 2026-02-17 10:30:00 | 114.03 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-17 10:20:00 | 113.30 | 2026-02-17 11:00:00 | 114.07 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2026-02-19 10:15:00 | 111.98 | 2026-02-19 10:35:00 | 111.36 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-19 10:15:00 | 111.98 | 2026-02-19 10:55:00 | 111.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:05:00 | 114.33 | 2026-02-24 10:15:00 | 115.11 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-02-24 10:05:00 | 114.33 | 2026-02-24 11:45:00 | 114.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:45:00 | 114.94 | 2026-02-26 09:55:00 | 114.45 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-11 09:55:00 | 102.00 | 2026-03-11 10:05:00 | 101.29 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-11 09:55:00 | 102.00 | 2026-03-11 15:20:00 | 99.12 | TARGET_HIT | 0.50 | 2.82% |
| SELL | retest1 | 2026-04-16 09:55:00 | 74.02 | 2026-04-16 10:35:00 | 74.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-21 09:45:00 | 75.10 | 2026-04-21 10:00:00 | 74.87 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-22 09:30:00 | 74.05 | 2026-04-22 10:00:00 | 73.70 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-22 09:30:00 | 74.05 | 2026-04-22 10:45:00 | 74.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:30:00 | 72.96 | 2026-04-24 09:40:00 | 72.65 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-24 09:30:00 | 72.96 | 2026-04-24 10:00:00 | 72.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 10:05:00 | 74.97 | 2026-05-06 11:00:00 | 74.62 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-05-06 10:05:00 | 74.97 | 2026-05-06 13:25:00 | 74.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 10:50:00 | 75.97 | 2026-05-07 11:05:00 | 75.65 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-07 10:50:00 | 75.97 | 2026-05-07 11:30:00 | 75.97 | STOP_HIT | 0.50 | 0.00% |

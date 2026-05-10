# NBCC (India) Ltd. (NBCC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 101.10
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
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 15
- **Target hits / Stop hits / Partials:** 0 / 15 / 4
- **Avg / median % per leg:** -0.09% / -0.29%
- **Sum % (uncompounded):** -1.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.11% | -1.1% |
| BUY @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.11% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.08% | -0.7% |
| SELL @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.08% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 4 | 21.1% | 0 | 15 | 4 | -0.09% | -1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 100.20 | 100.91 | 0.00 | ORB-short ORB[100.61,101.75] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-02-11 09:55:00 | 100.49 | 100.64 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 98.48 | 98.95 | 0.00 | ORB-short ORB[98.76,100.12] vol=2.4x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 98.78 | 98.87 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 97.09 | 97.80 | 0.00 | ORB-short ORB[97.70,98.67] vol=1.7x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 97.39 | 97.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 94.65 | 95.39 | 0.00 | ORB-short ORB[95.10,96.29] vol=1.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:45:00 | 94.26 | 95.28 | 0.00 | T1 1.5R @ 94.26 |
| Stop hit — per-position SL triggered | 2026-02-23 12:10:00 | 94.65 | 95.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:35:00 | 85.95 | 85.00 | 0.00 | ORB-long ORB[84.50,85.50] vol=2.0x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-03-12 10:50:00 | 85.57 | 85.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 79.91 | 79.25 | 0.00 | ORB-long ORB[78.64,79.70] vol=1.7x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:45:00 | 80.47 | 79.50 | 0.00 | T1 1.5R @ 80.47 |
| Stop hit — per-position SL triggered | 2026-03-30 10:05:00 | 79.91 | 79.63 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 91.56 | 92.35 | 0.00 | ORB-short ORB[91.91,93.20] vol=1.5x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:00:00 | 91.11 | 92.26 | 0.00 | T1 1.5R @ 91.11 |
| Stop hit — per-position SL triggered | 2026-04-16 10:05:00 | 91.56 | 92.23 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 94.29 | 93.98 | 0.00 | ORB-long ORB[93.32,94.25] vol=2.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 94.00 | 94.03 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 94.12 | 93.36 | 0.00 | ORB-long ORB[92.94,93.80] vol=3.1x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-04-22 10:50:00 | 93.85 | 93.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 92.65 | 93.39 | 0.00 | ORB-short ORB[93.20,94.18] vol=3.3x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-04-24 10:05:00 | 93.00 | 93.00 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:55:00 | 90.90 | 92.02 | 0.00 | ORB-short ORB[91.87,93.00] vol=2.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 91.19 | 91.89 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 93.13 | 92.75 | 0.00 | ORB-long ORB[92.29,93.12] vol=1.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:25:00 | 93.50 | 92.85 | 0.00 | T1 1.5R @ 93.50 |
| Stop hit — per-position SL triggered | 2026-05-04 12:05:00 | 93.13 | 92.96 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 94.53 | 94.12 | 0.00 | ORB-long ORB[93.70,94.49] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 94.25 | 94.13 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 95.63 | 94.94 | 0.00 | ORB-long ORB[94.37,95.54] vol=2.9x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 95.34 | 94.96 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 99.28 | 98.01 | 0.00 | ORB-long ORB[96.70,97.89] vol=4.5x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 98.75 | 98.25 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 100.20 | 2026-02-11 09:55:00 | 100.49 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-13 11:15:00 | 98.48 | 2026-02-13 11:50:00 | 98.78 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 09:40:00 | 97.09 | 2026-02-19 09:45:00 | 97.39 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-23 11:15:00 | 94.65 | 2026-02-23 11:45:00 | 94.26 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-23 11:15:00 | 94.65 | 2026-02-23 12:10:00 | 94.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 10:35:00 | 85.95 | 2026-03-12 10:50:00 | 85.57 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-30 09:30:00 | 79.91 | 2026-03-30 09:45:00 | 80.47 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-30 09:30:00 | 79.91 | 2026-03-30 10:05:00 | 79.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:50:00 | 91.56 | 2026-04-16 10:00:00 | 91.11 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-16 09:50:00 | 91.56 | 2026-04-16 10:05:00 | 91.56 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 94.29 | 2026-04-21 09:40:00 | 94.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-22 10:45:00 | 94.12 | 2026-04-22 10:50:00 | 93.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-24 09:30:00 | 92.65 | 2026-04-24 10:05:00 | 93.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-30 10:55:00 | 90.90 | 2026-04-30 11:05:00 | 91.19 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-04 11:15:00 | 93.13 | 2026-05-04 11:25:00 | 93.50 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-05-04 11:15:00 | 93.13 | 2026-05-04 12:05:00 | 93.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:00:00 | 94.53 | 2026-05-06 10:05:00 | 94.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-07 11:10:00 | 95.63 | 2026-05-07 11:15:00 | 95.34 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-08 09:40:00 | 99.28 | 2026-05-08 09:55:00 | 98.75 | STOP_HIT | 1.00 | -0.54% |

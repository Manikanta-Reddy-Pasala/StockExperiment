# SJVN Ltd. (SJVN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 78.69
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
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 8
- **Avg / median % per leg:** 0.31% / 0.29%
- **Sum % (uncompounded):** 5.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.23% | 1.9% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.23% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 1 | 5 | 5 | 0.37% | 4.0% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 1 | 5 | 5 | 0.37% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 2 | 9 | 8 | 0.31% | 5.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 72.27 | 72.56 | 0.00 | ORB-short ORB[72.40,73.09] vol=2.0x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 72.02 | 72.42 | 0.00 | T1 1.5R @ 72.02 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 72.27 | 72.41 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 76.85 | 76.15 | 0.00 | ORB-long ORB[75.61,76.65] vol=1.8x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:15:00 | 77.27 | 76.32 | 0.00 | T1 1.5R @ 77.27 |
| Target hit | 2026-02-16 15:20:00 | 77.68 | 77.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:00:00 | 78.33 | 77.89 | 0.00 | ORB-long ORB[77.44,78.16] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 78.05 | 77.91 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 76.55 | 76.82 | 0.00 | ORB-short ORB[76.57,77.28] vol=1.7x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:45:00 | 76.33 | 76.77 | 0.00 | T1 1.5R @ 76.33 |
| Target hit | 2026-02-19 15:20:00 | 74.75 | 75.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 74.30 | 74.75 | 0.00 | ORB-short ORB[74.60,75.49] vol=1.8x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 13:20:00 | 73.97 | 74.59 | 0.00 | T1 1.5R @ 73.97 |
| Stop hit — per-position SL triggered | 2026-02-23 15:05:00 | 74.30 | 74.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 74.12 | 73.66 | 0.00 | ORB-long ORB[73.15,73.99] vol=1.9x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:30:00 | 74.46 | 73.76 | 0.00 | T1 1.5R @ 74.46 |
| Stop hit — per-position SL triggered | 2026-02-24 12:20:00 | 74.12 | 73.86 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:30:00 | 70.15 | 69.67 | 0.00 | ORB-long ORB[69.11,70.10] vol=1.8x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:45:00 | 70.54 | 69.76 | 0.00 | T1 1.5R @ 70.54 |
| Stop hit — per-position SL triggered | 2026-04-08 10:55:00 | 70.15 | 69.78 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 75.30 | 75.73 | 0.00 | ORB-short ORB[75.50,76.28] vol=1.5x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:55:00 | 74.92 | 75.52 | 0.00 | T1 1.5R @ 74.92 |
| Stop hit — per-position SL triggered | 2026-04-16 10:05:00 | 75.30 | 75.49 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 81.75 | 81.34 | 0.00 | ORB-long ORB[80.18,81.35] vol=5.3x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 81.41 | 81.40 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 78.91 | 79.39 | 0.00 | ORB-short ORB[79.30,80.10] vol=2.2x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:35:00 | 78.61 | 79.21 | 0.00 | T1 1.5R @ 78.61 |
| Stop hit — per-position SL triggered | 2026-05-06 12:00:00 | 78.91 | 79.17 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 79.15 | 79.52 | 0.00 | ORB-short ORB[79.26,80.09] vol=1.8x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 79.38 | 79.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 72.27 | 2026-02-11 09:40:00 | 72.02 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-11 09:30:00 | 72.27 | 2026-02-11 09:45:00 | 72.27 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:45:00 | 76.85 | 2026-02-16 11:15:00 | 77.27 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-16 10:45:00 | 76.85 | 2026-02-16 15:20:00 | 77.68 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2026-02-17 10:00:00 | 78.33 | 2026-02-17 10:05:00 | 78.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-19 11:15:00 | 76.55 | 2026-02-19 11:45:00 | 76.33 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-19 11:15:00 | 76.55 | 2026-02-19 15:20:00 | 74.75 | TARGET_HIT | 0.50 | 2.35% |
| SELL | retest1 | 2026-02-23 11:05:00 | 74.30 | 2026-02-23 13:20:00 | 73.97 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-23 11:05:00 | 74.30 | 2026-02-23 15:05:00 | 74.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 11:00:00 | 74.12 | 2026-02-24 11:30:00 | 74.46 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-24 11:00:00 | 74.12 | 2026-02-24 12:20:00 | 74.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 10:30:00 | 70.15 | 2026-04-08 10:45:00 | 70.54 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-08 10:30:00 | 70.15 | 2026-04-08 10:55:00 | 70.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:35:00 | 75.30 | 2026-04-16 09:55:00 | 74.92 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-16 09:35:00 | 75.30 | 2026-04-16 10:05:00 | 75.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:45:00 | 81.75 | 2026-04-28 09:55:00 | 81.41 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-06 10:45:00 | 78.91 | 2026-05-06 11:35:00 | 78.61 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 10:45:00 | 78.91 | 2026-05-06 12:00:00 | 78.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 09:40:00 | 79.15 | 2026-05-08 09:50:00 | 79.38 | STOP_HIT | 1.00 | -0.29% |

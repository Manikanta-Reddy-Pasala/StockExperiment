# Zee Entertainment Enterprises Ltd. (ZEEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 95.22
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** -0.26% / -0.36%
- **Sum % (uncompounded):** -3.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.22% | -1.8% |
| BUY @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.22% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.33% | -1.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.33% | -1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 1 | 8.3% | 0 | 11 | 1 | -0.26% | -3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 92.39 | 92.95 | 0.00 | ORB-short ORB[92.67,94.00] vol=2.1x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 92.70 | 92.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 93.44 | 92.52 | 0.00 | ORB-long ORB[91.89,93.25] vol=2.1x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 93.09 | 92.62 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 95.55 | 94.01 | 0.00 | ORB-long ORB[92.71,94.01] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:25:00 | 96.43 | 94.34 | 0.00 | T1 1.5R @ 96.43 |
| Stop hit — per-position SL triggered | 2026-02-13 10:55:00 | 95.55 | 95.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 96.06 | 95.00 | 0.00 | ORB-long ORB[93.64,94.52] vol=6.2x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-02-17 11:25:00 | 95.66 | 95.18 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 94.32 | 94.86 | 0.00 | ORB-short ORB[94.35,95.54] vol=1.5x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 94.62 | 94.78 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 92.09 | 91.48 | 0.00 | ORB-long ORB[90.71,91.80] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-02-20 10:10:00 | 91.70 | 91.65 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 89.47 | 90.09 | 0.00 | ORB-short ORB[89.76,91.05] vol=2.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 89.73 | 90.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 83.36 | 82.21 | 0.00 | ORB-long ORB[81.30,82.35] vol=4.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 82.92 | 82.38 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 89.62 | 88.81 | 0.00 | ORB-long ORB[88.10,89.20] vol=2.3x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-04-27 09:35:00 | 89.22 | 88.86 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 92.01 | 91.65 | 0.00 | ORB-long ORB[90.64,92.00] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 91.56 | 91.67 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 90.87 | 91.26 | 0.00 | ORB-short ORB[91.03,92.23] vol=4.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-05-06 09:50:00 | 91.20 | 91.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:35:00 | 92.39 | 2026-02-11 09:45:00 | 92.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-12 10:00:00 | 93.44 | 2026-02-12 10:15:00 | 93.09 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-13 10:20:00 | 95.55 | 2026-02-13 10:25:00 | 96.43 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2026-02-13 10:20:00 | 95.55 | 2026-02-13 10:55:00 | 95.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:30:00 | 96.06 | 2026-02-17 11:25:00 | 95.66 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-18 09:50:00 | 94.32 | 2026-02-18 10:20:00 | 94.62 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-20 09:35:00 | 92.09 | 2026-02-20 10:10:00 | 91.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-24 09:30:00 | 89.47 | 2026-02-24 09:35:00 | 89.73 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-15 10:05:00 | 83.36 | 2026-04-15 10:25:00 | 82.92 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-27 09:30:00 | 89.62 | 2026-04-27 09:35:00 | 89.22 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-28 09:40:00 | 92.01 | 2026-04-28 09:50:00 | 91.56 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-05-06 09:40:00 | 90.87 | 2026-05-06 09:50:00 | 91.20 | STOP_HIT | 1.00 | -0.36% |

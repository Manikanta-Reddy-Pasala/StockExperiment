# MMTC Ltd. (MMTC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 68.15
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
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 8
- **Avg / median % per leg:** 0.35% / 0.33%
- **Sum % (uncompounded):** 6.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.20% | 1.8% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.20% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 1 | 5 | 5 | 0.47% | 5.1% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 1 | 5 | 5 | 0.47% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 10 | 50.0% | 2 | 10 | 8 | 0.35% | 7.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:15:00 | 65.34 | 64.53 | 0.00 | ORB-long ORB[64.07,64.86] vol=2.1x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:20:00 | 65.87 | 64.71 | 0.00 | T1 1.5R @ 65.87 |
| Stop hit — per-position SL triggered | 2026-02-09 11:25:00 | 65.34 | 64.77 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:05:00 | 66.84 | 66.06 | 0.00 | ORB-long ORB[65.64,66.44] vol=3.2x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 66.54 | 66.13 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 63.70 | 63.94 | 0.00 | ORB-short ORB[63.71,64.26] vol=1.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:00:00 | 63.41 | 63.84 | 0.00 | T1 1.5R @ 63.41 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 63.70 | 63.78 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 63.05 | 63.35 | 0.00 | ORB-short ORB[63.30,63.81] vol=2.1x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:45:00 | 62.80 | 63.16 | 0.00 | T1 1.5R @ 62.80 |
| Stop hit — per-position SL triggered | 2026-02-19 15:05:00 | 63.05 | 63.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 61.45 | 61.74 | 0.00 | ORB-short ORB[61.62,62.30] vol=1.7x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 61.65 | 61.69 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:15:00 | 61.48 | 61.89 | 0.00 | ORB-short ORB[61.86,62.64] vol=2.4x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:40:00 | 61.28 | 61.81 | 0.00 | T1 1.5R @ 61.28 |
| Stop hit — per-position SL triggered | 2026-02-26 15:05:00 | 61.48 | 61.59 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 61.65 | 61.19 | 0.00 | ORB-long ORB[60.80,61.35] vol=2.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:55:00 | 62.01 | 62.55 | 0.00 | T1 1.5R @ 62.01 |
| Target hit | 2026-02-27 10:00:00 | 62.55 | 62.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 54.88 | 55.13 | 0.00 | ORB-short ORB[54.95,55.62] vol=2.2x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:50:00 | 54.51 | 55.01 | 0.00 | T1 1.5R @ 54.51 |
| Target hit | 2026-03-13 15:20:00 | 53.20 | 53.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:30:00 | 57.17 | 56.60 | 0.00 | ORB-long ORB[55.93,56.73] vol=3.3x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-03-24 09:35:00 | 56.76 | 56.53 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 60.15 | 59.88 | 0.00 | ORB-long ORB[59.07,59.80] vol=8.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 59.86 | 59.89 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 67.35 | 67.09 | 0.00 | ORB-long ORB[66.67,67.21] vol=3.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:50:00 | 67.78 | 67.25 | 0.00 | T1 1.5R @ 67.78 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 67.35 | 67.24 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 65.94 | 66.27 | 0.00 | ORB-short ORB[66.10,66.60] vol=1.8x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:05:00 | 65.59 | 66.18 | 0.00 | T1 1.5R @ 65.59 |
| Stop hit — per-position SL triggered | 2026-05-06 14:25:00 | 65.94 | 65.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:15:00 | 65.34 | 2026-02-09 11:20:00 | 65.87 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2026-02-09 11:15:00 | 65.34 | 2026-02-09 11:25:00 | 65.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:05:00 | 66.84 | 2026-02-11 10:10:00 | 66.54 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-18 09:50:00 | 63.70 | 2026-02-18 10:00:00 | 63.41 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-18 09:50:00 | 63.70 | 2026-02-18 10:50:00 | 63.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 09:40:00 | 63.05 | 2026-02-19 11:45:00 | 62.80 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 09:40:00 | 63.05 | 2026-02-19 15:05:00 | 63.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:30:00 | 61.45 | 2026-02-25 09:45:00 | 61.65 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-26 11:15:00 | 61.48 | 2026-02-26 11:40:00 | 61.28 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-26 11:15:00 | 61.48 | 2026-02-26 15:05:00 | 61.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 09:50:00 | 61.65 | 2026-02-27 09:55:00 | 62.01 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-27 09:50:00 | 61.65 | 2026-02-27 10:00:00 | 62.55 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2026-03-13 09:40:00 | 54.88 | 2026-03-13 09:50:00 | 54.51 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-03-13 09:40:00 | 54.88 | 2026-03-13 15:20:00 | 53.20 | TARGET_HIT | 0.50 | 3.06% |
| BUY | retest1 | 2026-03-24 09:30:00 | 57.17 | 2026-03-24 09:35:00 | 56.76 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest1 | 2026-04-10 09:30:00 | 60.15 | 2026-04-10 09:35:00 | 59.86 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-28 09:45:00 | 67.35 | 2026-04-28 09:50:00 | 67.78 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-28 09:45:00 | 67.35 | 2026-04-28 09:55:00 | 67.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 09:40:00 | 65.94 | 2026-05-06 10:05:00 | 65.59 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-06 09:40:00 | 65.94 | 2026-05-06 14:25:00 | 65.94 | STOP_HIT | 0.50 | 0.00% |

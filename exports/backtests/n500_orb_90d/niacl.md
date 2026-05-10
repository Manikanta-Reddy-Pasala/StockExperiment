# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 163.20
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 7
- **Avg / median % per leg:** 0.33% / 0.46%
- **Sum % (uncompounded):** 6.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.39% | 6.2% |
| BUY @ 2nd Alert (retest1) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.39% | 6.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.13% | 0.7% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.13% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.33% | 6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 151.94 | 150.50 | 0.00 | ORB-long ORB[149.51,151.26] vol=3.5x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 13:35:00 | 152.93 | 151.38 | 0.00 | T1 1.5R @ 152.93 |
| Target hit | 2026-02-09 15:20:00 | 152.82 | 152.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:25:00 | 154.96 | 153.70 | 0.00 | ORB-long ORB[152.47,154.70] vol=2.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:30:00 | 155.88 | 154.04 | 0.00 | T1 1.5R @ 155.88 |
| Target hit | 2026-02-10 15:20:00 | 159.16 | 158.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 153.88 | 154.43 | 0.00 | ORB-short ORB[154.01,155.96] vol=1.9x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-12 09:50:00 | 154.32 | 154.30 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 152.00 | 151.69 | 0.00 | ORB-long ORB[150.88,151.85] vol=4.9x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:45:00 | 152.70 | 152.18 | 0.00 | T1 1.5R @ 152.70 |
| Stop hit — per-position SL triggered | 2026-02-17 09:55:00 | 152.00 | 152.30 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:50:00 | 151.12 | 150.49 | 0.00 | ORB-long ORB[149.00,151.04] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-02-20 10:20:00 | 150.56 | 150.67 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 151.97 | 150.62 | 0.00 | ORB-long ORB[149.38,150.27] vol=3.4x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-02-26 09:45:00 | 151.41 | 150.98 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 147.30 | 147.87 | 0.00 | ORB-short ORB[147.77,149.04] vol=1.7x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-02-27 10:00:00 | 147.77 | 147.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:05:00 | 138.89 | 139.40 | 0.00 | ORB-short ORB[139.00,140.50] vol=2.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:30:00 | 138.05 | 139.29 | 0.00 | T1 1.5R @ 138.05 |
| Target hit | 2026-03-04 15:20:00 | 137.45 | 138.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:40:00 | 139.45 | 138.94 | 0.00 | ORB-long ORB[138.00,139.26] vol=1.5x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:50:00 | 140.19 | 139.14 | 0.00 | T1 1.5R @ 140.19 |
| Target hit | 2026-03-06 10:30:00 | 140.44 | 140.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:55:00 | 137.69 | 136.19 | 0.00 | ORB-long ORB[135.44,136.79] vol=1.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-03-10 11:05:00 | 137.19 | 136.22 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 166.43 | 164.35 | 0.00 | ORB-long ORB[162.42,164.19] vol=7.1x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:05:00 | 167.72 | 165.19 | 0.00 | T1 1.5R @ 167.72 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 166.43 | 165.33 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 164.13 | 165.14 | 0.00 | ORB-short ORB[164.51,166.80] vol=1.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-04-28 09:35:00 | 164.75 | 165.07 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 162.79 | 161.77 | 0.00 | ORB-long ORB[160.80,162.40] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 163.86 | 162.21 | 0.00 | T1 1.5R @ 163.86 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 162.79 | 162.93 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 163.67 | 163.15 | 0.00 | ORB-long ORB[161.96,163.56] vol=3.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 163.03 | 163.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 151.94 | 2026-02-09 13:35:00 | 152.93 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-02-09 11:05:00 | 151.94 | 2026-02-09 15:20:00 | 152.82 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-10 10:25:00 | 154.96 | 2026-02-10 10:30:00 | 155.88 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-10 10:25:00 | 154.96 | 2026-02-10 15:20:00 | 159.16 | TARGET_HIT | 0.50 | 2.71% |
| SELL | retest1 | 2026-02-12 09:35:00 | 153.88 | 2026-02-12 09:50:00 | 154.32 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 09:30:00 | 152.00 | 2026-02-17 09:45:00 | 152.70 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 09:30:00 | 152.00 | 2026-02-17 09:55:00 | 152.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:50:00 | 151.12 | 2026-02-20 10:20:00 | 150.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-26 09:35:00 | 151.97 | 2026-02-26 09:45:00 | 151.41 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 09:55:00 | 147.30 | 2026-02-27 10:00:00 | 147.77 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-04 11:05:00 | 138.89 | 2026-03-04 11:30:00 | 138.05 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-04 11:05:00 | 138.89 | 2026-03-04 15:20:00 | 137.45 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2026-03-06 09:40:00 | 139.45 | 2026-03-06 09:50:00 | 140.19 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-06 09:40:00 | 139.45 | 2026-03-06 10:30:00 | 140.44 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-10 10:55:00 | 137.69 | 2026-03-10 11:05:00 | 137.19 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-23 10:00:00 | 166.43 | 2026-04-23 10:05:00 | 167.72 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-04-23 10:00:00 | 166.43 | 2026-04-23 10:10:00 | 166.43 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:30:00 | 164.13 | 2026-04-28 09:35:00 | 164.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-05-05 09:30:00 | 162.79 | 2026-05-05 09:35:00 | 163.86 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-05-05 09:30:00 | 162.79 | 2026-05-05 10:10:00 | 162.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:30:00 | 163.67 | 2026-05-06 09:35:00 | 163.03 | STOP_HIT | 1.00 | -0.39% |

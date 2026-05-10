# Canara HSBC Life Insurance Company Ltd. (CANHLIFE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 142.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 7
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 2.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.11% | 1.6% |
| BUY @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.11% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.32% | 1.3% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.32% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 8 | 42.1% | 1 | 11 | 7 | 0.15% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 147.87 | 146.12 | 0.00 | ORB-long ORB[144.55,146.05] vol=3.2x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 147.02 | 146.96 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 150.00 | 148.22 | 0.00 | ORB-long ORB[146.19,147.25] vol=6.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 149.46 | 148.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:40:00 | 144.96 | 144.60 | 0.00 | ORB-long ORB[143.50,144.90] vol=1.6x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:15:00 | 146.03 | 144.96 | 0.00 | T1 1.5R @ 146.03 |
| Stop hit — per-position SL triggered | 2026-03-06 10:30:00 | 144.96 | 145.12 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 147.88 | 148.38 | 0.00 | ORB-short ORB[147.96,149.98] vol=2.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:20:00 | 147.22 | 148.30 | 0.00 | T1 1.5R @ 147.22 |
| Stop hit — per-position SL triggered | 2026-03-18 12:30:00 | 147.88 | 148.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:35:00 | 142.00 | 142.93 | 0.00 | ORB-short ORB[142.20,144.00] vol=3.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 09:45:00 | 140.82 | 142.05 | 0.00 | T1 1.5R @ 140.82 |
| Stop hit — per-position SL triggered | 2026-03-24 09:55:00 | 142.00 | 141.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 10:45:00 | 144.18 | 143.55 | 0.00 | ORB-long ORB[142.06,144.00] vol=1.5x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:40:00 | 145.02 | 143.72 | 0.00 | T1 1.5R @ 145.02 |
| Stop hit — per-position SL triggered | 2026-04-01 11:45:00 | 144.18 | 143.73 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:40:00 | 145.68 | 144.18 | 0.00 | ORB-long ORB[142.45,144.33] vol=2.1x ATR=0.66 |
| Stop hit — per-position SL triggered | 2026-04-06 09:50:00 | 145.02 | 144.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:50:00 | 146.80 | 145.78 | 0.00 | ORB-long ORB[144.40,146.20] vol=1.7x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-04-09 12:40:00 | 146.34 | 146.34 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 145.56 | 145.19 | 0.00 | ORB-long ORB[143.50,145.38] vol=2.8x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 146.66 | 145.34 | 0.00 | T1 1.5R @ 146.66 |
| Target hit | 2026-04-22 10:45:00 | 145.80 | 145.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 146.25 | 145.03 | 0.00 | ORB-long ORB[143.46,145.45] vol=3.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:40:00 | 147.20 | 145.68 | 0.00 | T1 1.5R @ 147.20 |
| Stop hit — per-position SL triggered | 2026-04-23 09:50:00 | 146.25 | 145.85 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 144.71 | 143.52 | 0.00 | ORB-long ORB[141.93,143.77] vol=2.6x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:55:00 | 145.87 | 145.02 | 0.00 | T1 1.5R @ 145.87 |
| Stop hit — per-position SL triggered | 2026-05-05 13:50:00 | 144.71 | 145.11 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:35:00 | 143.75 | 142.15 | 0.00 | ORB-long ORB[140.88,141.98] vol=2.3x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-05-07 10:45:00 | 143.25 | 142.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 147.87 | 2026-02-09 10:45:00 | 147.02 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-02-13 10:30:00 | 150.00 | 2026-02-13 10:35:00 | 149.46 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-06 09:40:00 | 144.96 | 2026-03-06 10:15:00 | 146.03 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-03-06 09:40:00 | 144.96 | 2026-03-06 10:30:00 | 144.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-18 11:00:00 | 147.88 | 2026-03-18 11:20:00 | 147.22 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-18 11:00:00 | 147.88 | 2026-03-18 12:30:00 | 147.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 09:35:00 | 142.00 | 2026-03-24 09:45:00 | 140.82 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2026-03-24 09:35:00 | 142.00 | 2026-03-24 09:55:00 | 142.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-01 10:45:00 | 144.18 | 2026-04-01 11:40:00 | 145.02 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-01 10:45:00 | 144.18 | 2026-04-01 11:45:00 | 144.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-06 09:40:00 | 145.68 | 2026-04-06 09:50:00 | 145.02 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-09 10:50:00 | 146.80 | 2026-04-09 12:40:00 | 146.34 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-22 10:05:00 | 145.56 | 2026-04-22 10:15:00 | 146.66 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-04-22 10:05:00 | 145.56 | 2026-04-22 10:45:00 | 145.80 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-04-23 09:30:00 | 146.25 | 2026-04-23 09:40:00 | 147.20 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-23 09:30:00 | 146.25 | 2026-04-23 09:50:00 | 146.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 10:00:00 | 144.71 | 2026-05-05 11:55:00 | 145.87 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-05-05 10:00:00 | 144.71 | 2026-05-05 13:50:00 | 144.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:35:00 | 143.75 | 2026-05-07 10:45:00 | 143.25 | STOP_HIT | 1.00 | -0.35% |

# IRB Infrastructure Developers Ltd. (IRB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 21.46
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
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 7
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 2.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.07% | 0.6% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.07% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 0 | 5 | 4 | 0.17% | 1.5% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 5 | 4 | 0.17% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 8 | 44.4% | 1 | 10 | 7 | 0.12% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:25:00 | 20.11 | 20.27 | 0.00 | ORB-short ORB[20.26,20.53] vol=2.9x ATR=0.07 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 20.18 | 20.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 20.29 | 20.15 | 0.00 | ORB-long ORB[20.06,20.27] vol=1.5x ATR=0.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:20:00 | 20.38 | 20.22 | 0.00 | T1 1.5R @ 20.38 |
| Stop hit — per-position SL triggered | 2026-02-25 12:45:00 | 20.29 | 20.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 20.79 | 20.64 | 0.00 | ORB-long ORB[20.50,20.62] vol=2.4x ATR=0.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:40:00 | 20.90 | 20.69 | 0.00 | T1 1.5R @ 20.90 |
| Target hit | 2026-02-26 10:40:00 | 20.81 | 20.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-03-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:35:00 | 20.87 | 20.96 | 0.00 | ORB-short ORB[20.88,21.04] vol=5.0x ATR=0.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:00:00 | 20.75 | 20.94 | 0.00 | T1 1.5R @ 20.75 |
| Stop hit — per-position SL triggered | 2026-03-20 11:35:00 | 20.87 | 20.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 22.24 | 22.13 | 0.00 | ORB-long ORB[21.93,22.23] vol=3.5x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:35:00 | 22.37 | 22.17 | 0.00 | T1 1.5R @ 22.37 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 22.24 | 22.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 22.04 | 21.86 | 0.00 | ORB-long ORB[21.72,21.92] vol=4.6x ATR=0.06 |
| Stop hit — per-position SL triggered | 2026-04-22 11:45:00 | 21.98 | 21.88 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 22.02 | 22.09 | 0.00 | ORB-short ORB[22.03,22.25] vol=2.0x ATR=0.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:25:00 | 21.93 | 22.08 | 0.00 | T1 1.5R @ 21.93 |
| Stop hit — per-position SL triggered | 2026-04-23 13:05:00 | 22.02 | 22.04 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 22.09 | 21.84 | 0.00 | ORB-long ORB[21.61,21.92] vol=2.9x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-04-27 10:00:00 | 21.99 | 21.90 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 21.90 | 21.75 | 0.00 | ORB-long ORB[21.60,21.83] vol=2.7x ATR=0.07 |
| Stop hit — per-position SL triggered | 2026-05-04 11:55:00 | 21.83 | 21.76 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 21.58 | 21.64 | 0.00 | ORB-short ORB[21.61,21.84] vol=2.3x ATR=0.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:00:00 | 21.51 | 21.62 | 0.00 | T1 1.5R @ 21.51 |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 21.58 | 21.62 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 21.47 | 21.59 | 0.00 | ORB-short ORB[21.57,21.77] vol=1.8x ATR=0.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:45:00 | 21.36 | 21.53 | 0.00 | T1 1.5R @ 21.36 |
| Stop hit — per-position SL triggered | 2026-05-08 10:25:00 | 21.47 | 21.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-23 10:25:00 | 20.11 | 2026-02-23 10:55:00 | 20.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-25 10:15:00 | 20.29 | 2026-02-25 10:20:00 | 20.38 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-25 10:15:00 | 20.29 | 2026-02-25 12:45:00 | 20.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:35:00 | 20.79 | 2026-02-26 09:40:00 | 20.90 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-26 09:35:00 | 20.79 | 2026-02-26 10:40:00 | 20.81 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2026-03-20 10:35:00 | 20.87 | 2026-03-20 11:00:00 | 20.75 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-20 10:35:00 | 20.87 | 2026-03-20 11:35:00 | 20.87 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 09:30:00 | 22.24 | 2026-04-16 09:35:00 | 22.37 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-16 09:30:00 | 22.24 | 2026-04-16 09:40:00 | 22.24 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 11:15:00 | 22.04 | 2026-04-22 11:45:00 | 21.98 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-23 11:10:00 | 22.02 | 2026-04-23 11:25:00 | 21.93 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-23 11:10:00 | 22.02 | 2026-04-23 13:05:00 | 22.02 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:45:00 | 22.09 | 2026-04-27 10:00:00 | 21.99 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-05-04 11:15:00 | 21.90 | 2026-05-04 11:55:00 | 21.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-06 11:00:00 | 21.58 | 2026-05-06 13:00:00 | 21.51 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-06 11:00:00 | 21.58 | 2026-05-06 13:15:00 | 21.58 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 09:35:00 | 21.47 | 2026-05-08 09:45:00 | 21.36 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-08 09:35:00 | 21.47 | 2026-05-08 10:25:00 | 21.47 | STOP_HIT | 0.50 | 0.00% |

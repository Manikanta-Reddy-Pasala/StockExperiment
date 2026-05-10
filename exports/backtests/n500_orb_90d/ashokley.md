# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 168.77
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 3.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.34% | -1.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.34% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.58% | 4.1% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.58% | 4.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.30% | 3.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 207.10 | 205.95 | 0.00 | ORB-long ORB[204.92,206.79] vol=2.1x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 206.48 | 206.33 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 200.73 | 201.52 | 0.00 | ORB-short ORB[201.04,203.60] vol=3.4x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:30:00 | 199.76 | 201.24 | 0.00 | T1 1.5R @ 199.76 |
| Stop hit — per-position SL triggered | 2026-03-05 12:05:00 | 200.73 | 200.67 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 198.89 | 200.60 | 0.00 | ORB-short ORB[200.31,202.49] vol=1.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 199.54 | 200.50 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 190.20 | 193.01 | 0.00 | ORB-short ORB[192.37,194.58] vol=1.6x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 189.28 | 192.54 | 0.00 | T1 1.5R @ 189.28 |
| Target hit | 2026-03-11 15:20:00 | 184.49 | 188.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 178.56 | 177.68 | 0.00 | ORB-long ORB[175.98,178.30] vol=1.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 177.99 | 177.82 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 161.32 | 162.78 | 0.00 | ORB-short ORB[163.09,164.72] vol=1.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:40:00 | 160.66 | 162.47 | 0.00 | T1 1.5R @ 160.66 |
| Stop hit — per-position SL triggered | 2026-04-30 12:30:00 | 161.32 | 162.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 160.50 | 159.63 | 0.00 | ORB-long ORB[158.59,160.27] vol=1.5x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-05-05 10:45:00 | 159.86 | 159.97 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 207.10 | 2026-02-10 09:50:00 | 206.48 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-05 11:00:00 | 200.73 | 2026-03-05 11:30:00 | 199.76 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-05 11:00:00 | 200.73 | 2026-03-05 12:05:00 | 200.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:55:00 | 198.89 | 2026-03-06 11:00:00 | 199.54 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-11 11:05:00 | 190.20 | 2026-03-11 11:20:00 | 189.28 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-11 11:05:00 | 190.20 | 2026-03-11 15:20:00 | 184.49 | TARGET_HIT | 0.50 | 3.00% |
| BUY | retest1 | 2026-04-21 09:55:00 | 178.56 | 2026-04-21 10:25:00 | 177.99 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-30 11:15:00 | 161.32 | 2026-04-30 11:40:00 | 160.66 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-30 11:15:00 | 161.32 | 2026-04-30 12:30:00 | 161.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 160.50 | 2026-05-05 10:45:00 | 159.86 | STOP_HIT | 1.00 | -0.40% |

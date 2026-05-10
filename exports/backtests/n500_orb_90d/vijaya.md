# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1275.00
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 4
- **Avg / median % per leg:** 0.24% / 0.16%
- **Sum % (uncompounded):** 2.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.07% | 0.5% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.07% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 0.52% | 2.1% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 0.52% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.24% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:50:00 | 1025.55 | 1022.99 | 0.00 | ORB-long ORB[1015.55,1025.30] vol=2.4x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-02-11 11:05:00 | 1022.42 | 1022.97 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1006.35 | 1010.01 | 0.00 | ORB-short ORB[1013.20,1022.20] vol=4.0x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:00:00 | 1002.62 | 1008.96 | 0.00 | T1 1.5R @ 1002.62 |
| Target hit | 2026-02-18 14:20:00 | 1004.70 | 1002.85 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 1002.05 | 1004.30 | 0.00 | ORB-short ORB[1002.45,1012.10] vol=1.5x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:20:00 | 996.96 | 1001.87 | 0.00 | T1 1.5R @ 996.96 |
| Target hit | 2026-02-24 14:50:00 | 991.55 | 990.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 955.90 | 951.05 | 0.00 | ORB-long ORB[943.10,954.80] vol=3.3x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-03-11 10:20:00 | 953.20 | 951.69 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:10:00 | 1010.00 | 993.83 | 0.00 | ORB-long ORB[982.20,994.25] vol=2.3x ATR=3.74 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 1006.26 | 998.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:20:00 | 1036.70 | 1025.91 | 0.00 | ORB-long ORB[1014.30,1029.00] vol=3.2x ATR=4.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:30:00 | 1043.88 | 1028.83 | 0.00 | T1 1.5R @ 1043.88 |
| Stop hit — per-position SL triggered | 2026-04-17 11:40:00 | 1036.70 | 1036.57 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 1151.90 | 1144.64 | 0.00 | ORB-long ORB[1136.20,1150.80] vol=1.7x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:00:00 | 1160.86 | 1146.50 | 0.00 | T1 1.5R @ 1160.86 |
| Stop hit — per-position SL triggered | 2026-05-04 12:30:00 | 1151.90 | 1153.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:50:00 | 1025.55 | 2026-02-11 11:05:00 | 1022.42 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-18 10:55:00 | 1006.35 | 2026-02-18 11:00:00 | 1002.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-18 10:55:00 | 1006.35 | 2026-02-18 14:20:00 | 1004.70 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1002.05 | 2026-02-24 10:20:00 | 996.96 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1002.05 | 2026-02-24 14:50:00 | 991.55 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2026-03-11 09:45:00 | 955.90 | 2026-03-11 10:20:00 | 953.20 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 10:10:00 | 1010.00 | 2026-04-15 10:25:00 | 1006.26 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-17 10:20:00 | 1036.70 | 2026-04-17 10:30:00 | 1043.88 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-17 10:20:00 | 1036.70 | 2026-04-17 11:40:00 | 1036.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:55:00 | 1151.90 | 2026-05-04 11:00:00 | 1160.86 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-05-04 10:55:00 | 1151.90 | 2026-05-04 12:30:00 | 1151.90 | STOP_HIT | 0.50 | 0.00% |

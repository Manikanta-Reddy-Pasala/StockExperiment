# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1350.00
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 4
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 1.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.27% | 2.4% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.27% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.15% | -0.9% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.15% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 6 | 40.0% | 2 | 9 | 4 | 0.10% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 969.25 | 976.47 | 0.00 | ORB-short ORB[976.00,985.80] vol=2.5x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 964.22 | 972.87 | 0.00 | T1 1.5R @ 964.22 |
| Stop hit — per-position SL triggered | 2026-02-10 10:45:00 | 969.25 | 972.49 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 983.65 | 982.12 | 0.00 | ORB-long ORB[975.05,983.45] vol=2.4x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:35:00 | 987.48 | 982.82 | 0.00 | T1 1.5R @ 987.48 |
| Target hit | 2026-02-11 10:05:00 | 984.60 | 984.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 981.25 | 983.33 | 0.00 | ORB-short ORB[983.00,993.00] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-02-12 11:35:00 | 983.20 | 983.29 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 981.50 | 988.40 | 0.00 | ORB-short ORB[982.65,991.80] vol=3.3x ATR=4.65 |
| Stop hit — per-position SL triggered | 2026-02-17 11:05:00 | 986.15 | 987.86 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 989.75 | 1002.03 | 0.00 | ORB-short ORB[1006.00,1019.95] vol=1.7x ATR=3.77 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 993.52 | 1000.06 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 965.20 | 973.79 | 0.00 | ORB-short ORB[971.00,980.10] vol=2.1x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-02-20 11:25:00 | 968.68 | 973.32 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 980.50 | 975.36 | 0.00 | ORB-long ORB[966.65,979.00] vol=1.6x ATR=3.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:40:00 | 985.86 | 978.52 | 0.00 | T1 1.5R @ 985.86 |
| Stop hit — per-position SL triggered | 2026-02-23 10:25:00 | 980.50 | 980.39 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:05:00 | 972.15 | 968.03 | 0.00 | ORB-long ORB[964.80,972.00] vol=1.9x ATR=2.98 |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 969.17 | 968.14 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:00:00 | 850.05 | 846.50 | 0.00 | ORB-long ORB[837.40,849.60] vol=3.8x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:35:00 | 856.57 | 850.50 | 0.00 | T1 1.5R @ 856.57 |
| Target hit | 2026-03-12 15:20:00 | 866.20 | 863.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1312.80 | 1295.95 | 0.00 | ORB-long ORB[1284.20,1299.00] vol=2.0x ATR=6.56 |
| Stop hit — per-position SL triggered | 2026-05-05 10:30:00 | 1306.24 | 1305.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:35:00 | 1376.70 | 1363.09 | 0.00 | ORB-long ORB[1353.10,1368.20] vol=1.9x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-05-07 10:40:00 | 1370.42 | 1363.90 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 969.25 | 2026-02-10 10:40:00 | 964.22 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-10 10:35:00 | 969.25 | 2026-02-10 10:45:00 | 969.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 09:30:00 | 983.65 | 2026-02-11 09:35:00 | 987.48 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-11 09:30:00 | 983.65 | 2026-02-11 10:05:00 | 984.60 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2026-02-12 11:15:00 | 981.25 | 2026-02-12 11:35:00 | 983.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-17 10:45:00 | 981.50 | 2026-02-17 11:05:00 | 986.15 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-02-18 10:40:00 | 989.75 | 2026-02-18 11:00:00 | 993.52 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-20 11:10:00 | 965.20 | 2026-02-20 11:25:00 | 968.68 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-23 09:30:00 | 980.50 | 2026-02-23 09:40:00 | 985.86 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-23 09:30:00 | 980.50 | 2026-02-23 10:25:00 | 980.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:05:00 | 972.15 | 2026-02-24 10:15:00 | 969.17 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-12 10:00:00 | 850.05 | 2026-03-12 10:35:00 | 856.57 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-12 10:00:00 | 850.05 | 2026-03-12 15:20:00 | 866.20 | TARGET_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2026-05-05 09:30:00 | 1312.80 | 2026-05-05 10:30:00 | 1306.24 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-07 10:35:00 | 1376.70 | 2026-05-07 10:40:00 | 1370.42 | STOP_HIT | 1.00 | -0.46% |

# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1198.00
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 3
- **Avg / median % per leg:** 0.04% / -0.16%
- **Sum % (uncompounded):** 0.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | 0.05% | 0.5% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 2 | 6 | 2 | 0.05% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.04% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1598.00 | 1594.31 | 0.00 | ORB-long ORB[1582.70,1595.00] vol=2.1x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 1594.22 | 1594.48 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1563.40 | 1568.56 | 0.00 | ORB-short ORB[1564.40,1578.10] vol=1.5x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 1558.18 | 1564.77 | 0.00 | T1 1.5R @ 1558.18 |
| Target hit | 2026-02-11 14:10:00 | 1558.30 | 1558.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:05:00 | 1449.30 | 1465.17 | 0.00 | ORB-short ORB[1461.90,1483.20] vol=2.2x ATR=6.18 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 1455.48 | 1464.11 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 1417.50 | 1430.35 | 0.00 | ORB-short ORB[1422.00,1441.60] vol=1.6x ATR=4.21 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 1421.71 | 1429.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 10:55:00 | 1361.90 | 1346.39 | 0.00 | ORB-long ORB[1328.00,1342.00] vol=2.0x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 11:15:00 | 1368.70 | 1351.70 | 0.00 | T1 1.5R @ 1368.70 |
| Stop hit — per-position SL triggered | 2026-03-09 12:05:00 | 1361.90 | 1355.40 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1348.10 | 1352.35 | 0.00 | ORB-short ORB[1348.50,1364.20] vol=1.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-03-10 11:30:00 | 1351.21 | 1352.23 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 1354.50 | 1357.92 | 0.00 | ORB-short ORB[1361.20,1376.00] vol=5.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 1358.10 | 1357.78 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 11:15:00 | 1433.30 | 1434.40 | 0.00 | ORB-short ORB[1436.80,1451.90] vol=2.9x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-04-20 11:20:00 | 1436.30 | 1434.48 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1209.40 | 1219.50 | 0.00 | ORB-short ORB[1220.40,1232.70] vol=1.6x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:25:00 | 1205.33 | 1214.07 | 0.00 | T1 1.5R @ 1205.33 |
| Target hit | 2026-04-28 15:20:00 | 1195.90 | 1202.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 1206.20 | 1201.92 | 0.00 | ORB-long ORB[1193.00,1203.40] vol=2.1x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-04-29 11:40:00 | 1203.20 | 1203.09 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 1197.40 | 1201.72 | 0.00 | ORB-short ORB[1203.60,1212.00] vol=3.3x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1199.36 | 1201.42 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:45:00 | 1598.00 | 2026-02-10 09:55:00 | 1594.22 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-11 09:30:00 | 1563.40 | 2026-02-11 09:45:00 | 1558.18 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-11 09:30:00 | 1563.40 | 2026-02-11 14:10:00 | 1558.30 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-18 10:05:00 | 1449.30 | 2026-02-18 10:15:00 | 1455.48 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-23 10:40:00 | 1417.50 | 2026-02-23 10:55:00 | 1421.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-09 10:55:00 | 1361.90 | 2026-03-09 11:15:00 | 1368.70 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-09 10:55:00 | 1361.90 | 2026-03-09 12:05:00 | 1361.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 11:15:00 | 1348.10 | 2026-03-10 11:30:00 | 1351.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-11 10:40:00 | 1354.50 | 2026-03-11 11:00:00 | 1358.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-20 11:15:00 | 1433.30 | 2026-04-20 11:20:00 | 1436.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1209.40 | 2026-04-28 12:25:00 | 1205.33 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1209.40 | 2026-04-28 15:20:00 | 1195.90 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-04-29 10:55:00 | 1206.20 | 2026-04-29 11:40:00 | 1203.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-06 11:10:00 | 1197.40 | 2026-05-06 11:15:00 | 1199.36 | STOP_HIT | 1.00 | -0.16% |

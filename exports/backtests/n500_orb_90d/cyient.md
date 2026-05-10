# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 902.50
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 4
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 3.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.05% | 0.3% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.05% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 4 | 66.7% | 1 | 2 | 3 | 0.46% | 2.8% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 2 | 3 | 0.46% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.24% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 1091.00 | 1086.94 | 0.00 | ORB-long ORB[1080.30,1088.30] vol=1.5x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 1088.01 | 1087.90 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 1010.90 | 1000.08 | 0.00 | ORB-long ORB[987.70,1002.10] vol=1.6x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 1006.76 | 1004.02 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:50:00 | 983.00 | 989.06 | 0.00 | ORB-short ORB[986.40,998.80] vol=1.7x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:20:00 | 977.89 | 984.61 | 0.00 | T1 1.5R @ 977.89 |
| Stop hit — per-position SL triggered | 2026-02-23 13:20:00 | 983.00 | 980.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:35:00 | 836.25 | 838.60 | 0.00 | ORB-short ORB[836.40,848.75] vol=1.5x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:25:00 | 831.52 | 837.69 | 0.00 | T1 1.5R @ 831.52 |
| Target hit | 2026-03-20 15:20:00 | 825.80 | 832.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:55:00 | 911.15 | 897.54 | 0.00 | ORB-long ORB[884.25,897.85] vol=3.5x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-04-10 11:45:00 | 906.92 | 901.46 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:25:00 | 961.55 | 954.94 | 0.00 | ORB-long ORB[949.25,957.75] vol=1.7x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 958.47 | 956.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 857.90 | 849.40 | 0.00 | ORB-long ORB[841.00,853.40] vol=6.0x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:45:00 | 862.23 | 851.71 | 0.00 | T1 1.5R @ 862.23 |
| Target hit | 2026-04-30 15:20:00 | 871.95 | 864.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:15:00 | 867.00 | 871.41 | 0.00 | ORB-short ORB[869.75,878.00] vol=1.7x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:40:00 | 863.20 | 869.51 | 0.00 | T1 1.5R @ 863.20 |
| Stop hit — per-position SL triggered | 2026-05-05 12:35:00 | 867.00 | 868.89 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 904.15 | 899.89 | 0.00 | ORB-long ORB[894.00,903.25] vol=1.5x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-05-08 10:00:00 | 901.16 | 900.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 1091.00 | 2026-02-10 10:00:00 | 1088.01 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 09:55:00 | 1010.90 | 2026-02-17 10:45:00 | 1006.76 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-23 09:50:00 | 983.00 | 2026-02-23 10:20:00 | 977.89 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-23 09:50:00 | 983.00 | 2026-02-23 13:20:00 | 983.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 10:35:00 | 836.25 | 2026-03-20 12:25:00 | 831.52 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-20 10:35:00 | 836.25 | 2026-03-20 15:20:00 | 825.80 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2026-04-10 10:55:00 | 911.15 | 2026-04-10 11:45:00 | 906.92 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-04-21 10:25:00 | 961.55 | 2026-04-21 11:00:00 | 958.47 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-30 11:15:00 | 857.90 | 2026-04-30 11:45:00 | 862.23 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-30 11:15:00 | 857.90 | 2026-04-30 15:20:00 | 871.95 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2026-05-05 10:15:00 | 867.00 | 2026-05-05 11:40:00 | 863.20 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-05 10:15:00 | 867.00 | 2026-05-05 12:35:00 | 867.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:50:00 | 904.15 | 2026-05-08 10:00:00 | 901.16 | STOP_HIT | 1.00 | -0.33% |

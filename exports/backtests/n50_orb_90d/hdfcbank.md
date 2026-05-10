# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 781.25
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
- **Avg / median % per leg:** 0.27% / 0.17%
- **Sum % (uncompounded):** 5.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.22% | 2.5% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.22% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.33% | 3.3% |
| SELL @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.33% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.27% | 5.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 925.70 | 923.59 | 0.00 | ORB-long ORB[919.55,925.30] vol=3.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-17 12:30:00 | 924.12 | 924.74 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:05:00 | 919.65 | 922.44 | 0.00 | ORB-short ORB[921.50,927.95] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-24 11:20:00 | 921.10 | 922.28 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 905.55 | 907.58 | 0.00 | ORB-short ORB[905.95,912.00] vol=1.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 906.88 | 907.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 890.50 | 892.88 | 0.00 | ORB-short ORB[892.50,898.95] vol=4.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:40:00 | 888.51 | 892.36 | 0.00 | T1 1.5R @ 888.51 |
| Stop hit — per-position SL triggered | 2026-02-27 12:00:00 | 890.50 | 892.14 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 874.50 | 869.31 | 0.00 | ORB-long ORB[863.50,873.85] vol=2.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-03-05 11:30:00 | 872.43 | 870.06 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:50:00 | 836.20 | 841.39 | 0.00 | ORB-short ORB[840.60,848.85] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 833.67 | 840.22 | 0.00 | T1 1.5R @ 833.67 |
| Target hit | 2026-03-11 15:05:00 | 834.80 | 834.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 831.90 | 830.30 | 0.00 | ORB-long ORB[820.10,830.90] vol=3.4x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 12:05:00 | 835.53 | 831.33 | 0.00 | T1 1.5R @ 835.53 |
| Stop hit — per-position SL triggered | 2026-03-12 14:40:00 | 831.90 | 832.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:10:00 | 731.65 | 730.53 | 0.00 | ORB-long ORB[726.65,731.00] vol=1.7x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:55:00 | 734.41 | 730.99 | 0.00 | T1 1.5R @ 734.41 |
| Target hit | 2026-04-02 15:20:00 | 750.00 | 740.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 810.80 | 806.67 | 0.00 | ORB-long ORB[801.00,807.60] vol=1.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 808.07 | 807.21 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 810.95 | 813.13 | 0.00 | ORB-short ORB[814.85,820.05] vol=2.7x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:10:00 | 808.81 | 812.80 | 0.00 | T1 1.5R @ 808.81 |
| Target hit | 2026-04-16 15:20:00 | 793.65 | 802.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 785.95 | 781.04 | 0.00 | ORB-long ORB[777.30,784.05] vol=2.4x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-04-24 11:10:00 | 784.25 | 781.59 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 791.50 | 789.03 | 0.00 | ORB-long ORB[783.30,790.00] vol=1.6x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:05:00 | 793.86 | 790.13 | 0.00 | T1 1.5R @ 793.86 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 791.50 | 790.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 787.80 | 785.06 | 0.00 | ORB-long ORB[780.90,786.00] vol=2.4x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-04-29 11:35:00 | 786.19 | 785.28 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 782.95 | 784.83 | 0.00 | ORB-short ORB[784.00,788.75] vol=2.7x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:30:00 | 780.81 | 784.55 | 0.00 | T1 1.5R @ 780.81 |
| Target hit | 2026-05-08 15:10:00 | 781.45 | 781.44 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 11:00:00 | 925.70 | 2026-02-17 12:30:00 | 924.12 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-24 11:05:00 | 919.65 | 2026-02-24 11:20:00 | 921.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-02-26 11:00:00 | 905.55 | 2026-02-26 11:25:00 | 906.88 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-02-27 11:10:00 | 890.50 | 2026-02-27 11:40:00 | 888.51 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2026-02-27 11:10:00 | 890.50 | 2026-02-27 12:00:00 | 890.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:55:00 | 874.50 | 2026-03-05 11:30:00 | 872.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-11 10:50:00 | 836.20 | 2026-03-11 11:15:00 | 833.67 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-03-11 10:50:00 | 836.20 | 2026-03-11 15:05:00 | 834.80 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-03-12 10:20:00 | 831.90 | 2026-03-12 12:05:00 | 835.53 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-12 10:20:00 | 831.90 | 2026-03-12 14:40:00 | 831.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-02 11:10:00 | 731.65 | 2026-04-02 11:55:00 | 734.41 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-02 11:10:00 | 731.65 | 2026-04-02 15:20:00 | 750.00 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2026-04-10 09:45:00 | 810.80 | 2026-04-10 10:05:00 | 808.07 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 11:00:00 | 810.95 | 2026-04-16 11:10:00 | 808.81 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-16 11:00:00 | 810.95 | 2026-04-16 15:20:00 | 793.65 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2026-04-24 10:50:00 | 785.95 | 2026-04-24 11:10:00 | 784.25 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-28 09:45:00 | 791.50 | 2026-04-28 10:05:00 | 793.86 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-28 09:45:00 | 791.50 | 2026-04-28 10:35:00 | 791.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 11:15:00 | 787.80 | 2026-04-29 11:35:00 | 786.19 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-05-08 11:05:00 | 782.95 | 2026-05-08 11:30:00 | 780.81 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-05-08 11:05:00 | 782.95 | 2026-05-08 15:10:00 | 781.45 | TARGET_HIT | 0.50 | 0.19% |

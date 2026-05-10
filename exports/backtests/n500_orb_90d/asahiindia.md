# Asahi India Glass Ltd. (ASAHIINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 836.30
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 13
- **Target hits / Stop hits / Partials:** 4 / 13 / 10
- **Avg / median % per leg:** 0.28% / 0.32%
- **Sum % (uncompounded):** 7.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.40% | 6.8% |
| BUY @ 2nd Alert (retest1) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.40% | 6.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.08% | 0.8% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.08% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 14 | 51.9% | 4 | 13 | 10 | 0.28% | 7.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 938.50 | 933.44 | 0.00 | ORB-long ORB[930.20,937.40] vol=1.6x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 942.10 | 935.25 | 0.00 | T1 1.5R @ 942.10 |
| Stop hit — per-position SL triggered | 2026-02-17 10:35:00 | 938.50 | 935.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 940.00 | 934.99 | 0.00 | ORB-long ORB[928.50,936.80] vol=6.9x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 14:05:00 | 944.81 | 936.64 | 0.00 | T1 1.5R @ 944.81 |
| Target hit | 2026-02-20 15:20:00 | 953.90 | 939.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 924.00 | 930.62 | 0.00 | ORB-short ORB[931.00,941.00] vol=1.9x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-02-24 10:00:00 | 929.01 | 929.55 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 941.60 | 937.35 | 0.00 | ORB-long ORB[933.00,939.00] vol=2.0x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 938.36 | 937.51 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 932.30 | 933.16 | 0.00 | ORB-short ORB[932.80,939.00] vol=2.9x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:55:00 | 928.41 | 931.92 | 0.00 | T1 1.5R @ 928.41 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 932.30 | 931.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:40:00 | 849.90 | 844.13 | 0.00 | ORB-long ORB[832.35,844.30] vol=1.5x ATR=3.49 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 846.41 | 845.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:30:00 | 840.25 | 834.11 | 0.00 | ORB-long ORB[827.75,840.00] vol=2.2x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:40:00 | 848.37 | 836.91 | 0.00 | T1 1.5R @ 848.37 |
| Target hit | 2026-04-08 15:20:00 | 855.00 | 849.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 877.10 | 871.00 | 0.00 | ORB-long ORB[864.05,869.00] vol=3.1x ATR=3.96 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 873.14 | 872.61 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 843.00 | 845.62 | 0.00 | ORB-short ORB[844.05,855.20] vol=1.7x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:55:00 | 839.35 | 842.82 | 0.00 | T1 1.5R @ 839.35 |
| Stop hit — per-position SL triggered | 2026-04-17 12:30:00 | 843.00 | 841.87 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 850.65 | 847.93 | 0.00 | ORB-long ORB[842.80,850.20] vol=2.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 854.51 | 850.61 | 0.00 | T1 1.5R @ 854.51 |
| Target hit | 2026-04-21 14:40:00 | 856.00 | 856.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 864.80 | 859.70 | 0.00 | ORB-long ORB[853.90,858.90] vol=1.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 862.07 | 859.91 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 868.10 | 865.61 | 0.00 | ORB-long ORB[858.25,867.45] vol=2.7x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 865.32 | 865.54 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 843.25 | 840.66 | 0.00 | ORB-long ORB[835.95,842.70] vol=1.9x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 847.57 | 841.94 | 0.00 | T1 1.5R @ 847.57 |
| Target hit | 2026-04-27 15:20:00 | 856.10 | 851.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 862.70 | 857.75 | 0.00 | ORB-long ORB[850.00,861.50] vol=1.8x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:30:00 | 866.63 | 859.68 | 0.00 | T1 1.5R @ 866.63 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 862.70 | 860.08 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:25:00 | 832.45 | 840.63 | 0.00 | ORB-short ORB[842.40,852.25] vol=4.0x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-05-04 10:30:00 | 835.69 | 840.22 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 838.60 | 842.91 | 0.00 | ORB-short ORB[844.05,849.25] vol=1.5x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 833.68 | 840.99 | 0.00 | T1 1.5R @ 833.68 |
| Stop hit — per-position SL triggered | 2026-05-06 13:30:00 | 838.60 | 838.52 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 840.75 | 845.21 | 0.00 | ORB-short ORB[842.20,852.20] vol=1.6x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:35:00 | 838.05 | 844.60 | 0.00 | T1 1.5R @ 838.05 |
| Stop hit — per-position SL triggered | 2026-05-08 11:45:00 | 840.75 | 844.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:20:00 | 938.50 | 2026-02-17 10:30:00 | 942.10 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-17 10:20:00 | 938.50 | 2026-02-17 10:35:00 | 938.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 11:10:00 | 940.00 | 2026-02-20 14:05:00 | 944.81 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-20 11:10:00 | 940.00 | 2026-02-20 15:20:00 | 953.90 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2026-02-24 09:30:00 | 924.00 | 2026-02-24 10:00:00 | 929.01 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-02-26 09:40:00 | 941.60 | 2026-02-26 09:50:00 | 938.36 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-27 10:10:00 | 932.30 | 2026-02-27 10:55:00 | 928.41 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 10:10:00 | 932.30 | 2026-02-27 11:00:00 | 932.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:40:00 | 849.90 | 2026-03-17 11:05:00 | 846.41 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-08 09:30:00 | 840.25 | 2026-04-08 09:40:00 | 848.37 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2026-04-08 09:30:00 | 840.25 | 2026-04-08 15:20:00 | 855.00 | TARGET_HIT | 0.50 | 1.76% |
| BUY | retest1 | 2026-04-10 09:40:00 | 877.10 | 2026-04-10 09:55:00 | 873.14 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-17 09:40:00 | 843.00 | 2026-04-17 11:55:00 | 839.35 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-17 09:40:00 | 843.00 | 2026-04-17 12:30:00 | 843.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:55:00 | 850.65 | 2026-04-21 10:05:00 | 854.51 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-21 09:55:00 | 850.65 | 2026-04-21 14:40:00 | 856.00 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-22 09:35:00 | 864.80 | 2026-04-22 09:40:00 | 862.07 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-23 09:50:00 | 868.10 | 2026-04-23 10:10:00 | 865.32 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-27 09:30:00 | 843.25 | 2026-04-27 09:50:00 | 847.57 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-27 09:30:00 | 843.25 | 2026-04-27 15:20:00 | 856.10 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2026-04-28 09:55:00 | 862.70 | 2026-04-28 10:30:00 | 866.63 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-28 09:55:00 | 862.70 | 2026-04-28 10:35:00 | 862.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:25:00 | 832.45 | 2026-05-04 10:30:00 | 835.69 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-06 10:05:00 | 838.60 | 2026-05-06 11:00:00 | 833.68 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-05-06 10:05:00 | 838.60 | 2026-05-06 13:30:00 | 838.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:15:00 | 840.75 | 2026-05-08 11:35:00 | 838.05 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-05-08 11:15:00 | 840.75 | 2026-05-08 11:45:00 | 840.75 | STOP_HIT | 0.50 | 0.00% |

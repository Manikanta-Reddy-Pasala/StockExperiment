# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 903.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 4
- **Avg / median % per leg:** 0.09% / -0.26%
- **Sum % (uncompounded):** 1.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.39% | -1.6% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.39% | -1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.22% | 3.4% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.22% | 3.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 6 | 30.0% | 2 | 14 | 4 | 0.09% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 963.15 | 953.32 | 0.00 | ORB-long ORB[945.90,956.20] vol=3.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 959.09 | 956.12 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 1068.00 | 1072.30 | 0.00 | ORB-short ORB[1069.65,1081.15] vol=2.0x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 1071.08 | 1071.52 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1059.40 | 1061.80 | 0.00 | ORB-short ORB[1060.00,1070.00] vol=2.3x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:55:00 | 1054.59 | 1058.61 | 0.00 | T1 1.5R @ 1054.59 |
| Target hit | 2026-02-19 15:20:00 | 1035.50 | 1053.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 947.95 | 944.41 | 0.00 | ORB-long ORB[934.50,945.00] vol=7.0x ATR=4.59 |
| Stop hit — per-position SL triggered | 2026-02-26 10:45:00 | 943.36 | 945.62 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:00:00 | 915.15 | 918.58 | 0.00 | ORB-short ORB[920.55,926.00] vol=9.8x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:35:00 | 908.63 | 914.67 | 0.00 | T1 1.5R @ 908.63 |
| Target hit | 2026-03-06 15:20:00 | 900.30 | 903.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:15:00 | 779.00 | 784.28 | 0.00 | ORB-short ORB[783.00,793.05] vol=2.9x ATR=3.19 |
| Stop hit — per-position SL triggered | 2026-03-19 10:30:00 | 782.19 | 784.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:45:00 | 728.20 | 730.44 | 0.00 | ORB-short ORB[732.25,741.40] vol=9.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-03-24 09:55:00 | 732.14 | 730.34 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 802.15 | 804.83 | 0.00 | ORB-short ORB[809.10,817.75] vol=5.2x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 805.47 | 804.67 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:40:00 | 798.00 | 801.04 | 0.00 | ORB-short ORB[799.80,808.60] vol=4.1x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:45:00 | 794.01 | 800.66 | 0.00 | T1 1.5R @ 794.01 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 798.00 | 800.56 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:25:00 | 861.00 | 859.90 | 0.00 | ORB-long ORB[852.05,860.45] vol=2.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-04-23 10:35:00 | 858.78 | 859.84 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:20:00 | 842.00 | 850.13 | 0.00 | ORB-short ORB[850.80,862.75] vol=4.0x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:40:00 | 836.97 | 848.49 | 0.00 | T1 1.5R @ 836.97 |
| Stop hit — per-position SL triggered | 2026-04-24 14:30:00 | 842.00 | 839.91 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 854.00 | 854.97 | 0.00 | ORB-short ORB[854.50,859.25] vol=4.1x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-04-29 10:00:00 | 856.79 | 855.01 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:35:00 | 851.05 | 856.93 | 0.00 | ORB-short ORB[857.20,866.25] vol=4.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 853.68 | 856.75 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 885.00 | 875.21 | 0.00 | ORB-long ORB[869.65,880.30] vol=2.6x ATR=3.47 |
| Stop hit — per-position SL triggered | 2026-05-04 10:45:00 | 881.53 | 879.20 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:05:00 | 868.55 | 877.92 | 0.00 | ORB-short ORB[876.10,885.20] vol=3.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-05-05 15:20:00 | 869.00 | 870.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 899.90 | 900.37 | 0.00 | ORB-short ORB[900.00,904.90] vol=6.1x ATR=3.27 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 903.17 | 900.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:30:00 | 963.15 | 2026-02-11 10:40:00 | 959.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-18 11:15:00 | 1068.00 | 2026-02-18 12:15:00 | 1071.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 09:30:00 | 1059.40 | 2026-02-19 11:55:00 | 1054.59 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-19 09:30:00 | 1059.40 | 2026-02-19 15:20:00 | 1035.50 | TARGET_HIT | 0.50 | 2.26% |
| BUY | retest1 | 2026-02-26 09:40:00 | 947.95 | 2026-02-26 10:45:00 | 943.36 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-06 10:00:00 | 915.15 | 2026-03-06 10:35:00 | 908.63 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-06 10:00:00 | 915.15 | 2026-03-06 15:20:00 | 900.30 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2026-03-19 10:15:00 | 779.00 | 2026-03-19 10:30:00 | 782.19 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-24 09:45:00 | 728.20 | 2026-03-24 09:55:00 | 732.14 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-04-16 09:50:00 | 802.15 | 2026-04-16 10:40:00 | 805.47 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-20 09:40:00 | 798.00 | 2026-04-20 09:45:00 | 794.01 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-20 09:40:00 | 798.00 | 2026-04-20 09:50:00 | 798.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:25:00 | 861.00 | 2026-04-23 10:35:00 | 858.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-24 10:20:00 | 842.00 | 2026-04-24 10:40:00 | 836.97 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-24 10:20:00 | 842.00 | 2026-04-24 14:30:00 | 842.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:45:00 | 854.00 | 2026-04-29 10:00:00 | 856.79 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-30 10:35:00 | 851.05 | 2026-04-30 11:05:00 | 853.68 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-04 10:00:00 | 885.00 | 2026-05-04 10:45:00 | 881.53 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-05 10:05:00 | 868.55 | 2026-05-05 15:20:00 | 869.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest1 | 2026-05-08 09:45:00 | 899.90 | 2026-05-08 09:50:00 | 903.17 | STOP_HIT | 1.00 | -0.36% |

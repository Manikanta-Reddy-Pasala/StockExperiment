# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 955.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 4
- **Avg / median % per leg:** -0.01% / 0.00%
- **Sum % (uncompounded):** -0.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.18% | -1.1% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.18% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.11% | 1.0% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.11% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 4 | 26.7% | 0 | 11 | 4 | -0.01% | -0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 911.10 | 914.60 | 0.00 | ORB-short ORB[912.45,920.00] vol=1.6x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:10:00 | 907.02 | 912.67 | 0.00 | T1 1.5R @ 907.02 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 911.10 | 911.86 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 914.00 | 906.21 | 0.00 | ORB-long ORB[900.85,912.95] vol=4.1x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-02-19 11:05:00 | 911.29 | 906.66 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 918.50 | 910.74 | 0.00 | ORB-long ORB[907.95,917.60] vol=1.9x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 913.44 | 912.29 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 903.00 | 907.02 | 0.00 | ORB-short ORB[907.55,920.80] vol=2.5x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:40:00 | 895.41 | 903.08 | 0.00 | T1 1.5R @ 895.41 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 903.00 | 901.99 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 899.65 | 905.84 | 0.00 | ORB-short ORB[905.10,914.35] vol=3.9x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:35:00 | 894.81 | 902.58 | 0.00 | T1 1.5R @ 894.81 |
| Stop hit — per-position SL triggered | 2026-02-27 14:10:00 | 899.65 | 901.29 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 829.15 | 831.70 | 0.00 | ORB-short ORB[831.25,838.90] vol=2.4x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-03-11 09:45:00 | 831.54 | 831.60 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 850.30 | 856.04 | 0.00 | ORB-short ORB[853.20,864.40] vol=3.0x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-04-09 10:30:00 | 853.09 | 853.41 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 897.35 | 893.66 | 0.00 | ORB-long ORB[887.15,896.75] vol=3.1x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:00:00 | 900.86 | 894.32 | 0.00 | T1 1.5R @ 900.86 |
| Stop hit — per-position SL triggered | 2026-04-17 11:05:00 | 897.35 | 894.67 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 930.60 | 923.27 | 0.00 | ORB-long ORB[917.05,922.35] vol=1.8x ATR=3.39 |
| Stop hit — per-position SL triggered | 2026-04-21 10:20:00 | 927.21 | 924.82 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 929.40 | 928.38 | 0.00 | ORB-long ORB[921.65,928.00] vol=2.3x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-04-22 15:00:00 | 926.92 | 928.74 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 922.70 | 926.97 | 0.00 | ORB-short ORB[922.75,931.95] vol=2.3x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 924.91 | 926.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:35:00 | 911.10 | 2026-02-18 10:10:00 | 907.02 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-18 09:35:00 | 911.10 | 2026-02-18 10:50:00 | 911.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 11:00:00 | 914.00 | 2026-02-19 11:05:00 | 911.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-24 09:45:00 | 918.50 | 2026-02-24 11:15:00 | 913.44 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-02-25 09:30:00 | 903.00 | 2026-02-25 09:40:00 | 895.41 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2026-02-25 09:30:00 | 903.00 | 2026-02-25 10:00:00 | 903.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:15:00 | 899.65 | 2026-02-27 11:35:00 | 894.81 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-27 10:15:00 | 899.65 | 2026-02-27 14:10:00 | 899.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 09:40:00 | 829.15 | 2026-03-11 09:45:00 | 831.54 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-09 09:30:00 | 850.30 | 2026-04-09 10:30:00 | 853.09 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-17 10:55:00 | 897.35 | 2026-04-17 11:00:00 | 900.86 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-17 10:55:00 | 897.35 | 2026-04-17 11:05:00 | 897.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 930.60 | 2026-04-21 10:20:00 | 927.21 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-22 10:30:00 | 929.40 | 2026-04-22 15:00:00 | 926.92 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-23 10:55:00 | 922.70 | 2026-04-23 11:05:00 | 924.91 | STOP_HIT | 1.00 | -0.24% |

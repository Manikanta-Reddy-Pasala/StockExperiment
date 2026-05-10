# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1088.90
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
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 6
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 0.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.01% | -0.1% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.01% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 7 | 31.8% | 1 | 15 | 6 | 0.01% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 870.30 | 862.41 | 0.00 | ORB-long ORB[852.00,857.55] vol=1.9x ATR=3.84 |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 866.46 | 863.82 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 853.20 | 860.55 | 0.00 | ORB-short ORB[861.65,869.85] vol=2.1x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-10 10:10:00 | 855.98 | 859.88 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 831.75 | 837.43 | 0.00 | ORB-short ORB[838.50,848.00] vol=4.4x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 834.59 | 837.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 846.35 | 841.20 | 0.00 | ORB-long ORB[830.15,840.40] vol=3.1x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-16 11:30:00 | 844.01 | 841.87 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 853.95 | 848.77 | 0.00 | ORB-long ORB[841.00,853.55] vol=3.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:10:00 | 856.66 | 851.95 | 0.00 | T1 1.5R @ 856.66 |
| Target hit | 2026-02-17 15:20:00 | 863.60 | 860.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 854.15 | 858.95 | 0.00 | ORB-short ORB[859.00,868.55] vol=2.5x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 851.24 | 856.93 | 0.00 | T1 1.5R @ 851.24 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 854.15 | 856.56 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 850.15 | 847.65 | 0.00 | ORB-long ORB[838.40,848.80] vol=4.4x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 855.09 | 849.51 | 0.00 | T1 1.5R @ 855.09 |
| Stop hit — per-position SL triggered | 2026-02-20 13:25:00 | 850.15 | 850.77 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 846.75 | 850.20 | 0.00 | ORB-short ORB[848.50,854.25] vol=3.5x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:05:00 | 843.69 | 849.93 | 0.00 | T1 1.5R @ 843.69 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 846.75 | 846.43 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 824.95 | 827.52 | 0.00 | ORB-short ORB[827.05,838.00] vol=2.5x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-02-24 10:40:00 | 827.38 | 827.30 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 759.00 | 763.82 | 0.00 | ORB-short ORB[762.65,774.00] vol=3.6x ATR=3.39 |
| Stop hit — per-position SL triggered | 2026-03-12 09:45:00 | 762.39 | 762.77 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 770.50 | 774.47 | 0.00 | ORB-short ORB[772.60,778.50] vol=1.9x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:55:00 | 766.16 | 772.55 | 0.00 | T1 1.5R @ 766.16 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 770.50 | 771.13 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 772.15 | 764.54 | 0.00 | ORB-long ORB[758.40,765.50] vol=2.3x ATR=3.67 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 768.48 | 769.58 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:10:00 | 782.20 | 772.73 | 0.00 | ORB-long ORB[767.15,773.15] vol=2.0x ATR=2.94 |
| Stop hit — per-position SL triggered | 2026-03-17 10:40:00 | 779.26 | 774.95 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1055.05 | 1043.82 | 0.00 | ORB-long ORB[1036.45,1050.10] vol=2.2x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:25:00 | 1060.63 | 1048.07 | 0.00 | T1 1.5R @ 1060.63 |
| Stop hit — per-position SL triggered | 2026-04-29 14:45:00 | 1055.05 | 1056.56 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1049.50 | 1044.65 | 0.00 | ORB-long ORB[1036.60,1045.00] vol=1.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2026-05-05 10:30:00 | 1046.78 | 1048.58 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 1060.90 | 1054.76 | 0.00 | ORB-long ORB[1048.30,1058.80] vol=1.8x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 1056.85 | 1056.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 870.30 | 2026-02-09 11:15:00 | 866.46 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-10 10:05:00 | 853.20 | 2026-02-10 10:10:00 | 855.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-13 10:20:00 | 831.75 | 2026-02-13 10:30:00 | 834.59 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-16 10:50:00 | 846.35 | 2026-02-16 11:30:00 | 844.01 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-17 11:05:00 | 853.95 | 2026-02-17 11:10:00 | 856.66 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 11:05:00 | 853.95 | 2026-02-17 15:20:00 | 863.60 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-02-19 10:50:00 | 854.15 | 2026-02-19 11:15:00 | 851.24 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 10:50:00 | 854.15 | 2026-02-19 11:25:00 | 854.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:35:00 | 850.15 | 2026-02-20 12:15:00 | 855.09 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-20 09:35:00 | 850.15 | 2026-02-20 13:25:00 | 850.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 846.75 | 2026-02-23 11:05:00 | 843.69 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-23 11:00:00 | 846.75 | 2026-02-23 11:15:00 | 846.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 10:25:00 | 824.95 | 2026-02-24 10:40:00 | 827.38 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-12 09:30:00 | 759.00 | 2026-03-12 09:45:00 | 762.39 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-13 09:45:00 | 770.50 | 2026-03-13 09:55:00 | 766.16 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-13 09:45:00 | 770.50 | 2026-03-13 10:30:00 | 770.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-16 09:30:00 | 772.15 | 2026-03-16 10:10:00 | 768.48 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-17 10:10:00 | 782.20 | 2026-03-17 10:40:00 | 779.26 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1055.05 | 2026-04-29 11:25:00 | 1060.63 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1055.05 | 2026-04-29 14:45:00 | 1055.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 10:15:00 | 1049.50 | 2026-05-05 10:30:00 | 1046.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-06 09:35:00 | 1060.90 | 2026-05-06 09:55:00 | 1056.85 | STOP_HIT | 1.00 | -0.38% |

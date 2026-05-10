# Premier Energies Ltd. (PREMIERENE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1014.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 12
- **Target hits / Stop hits / Partials:** 1 / 12 / 6
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 0.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 4 | 30.8% | 0 | 9 | 4 | -0.07% | -0.9% |
| BUY @ 2nd Alert (retest1) | 13 | 4 | 30.8% | 0 | 9 | 4 | -0.07% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.19% | 1.1% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.19% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 7 | 36.8% | 1 | 12 | 6 | 0.01% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 764.05 | 758.57 | 0.00 | ORB-long ORB[751.40,761.95] vol=1.9x ATR=3.34 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 760.71 | 760.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 761.25 | 757.89 | 0.00 | ORB-long ORB[754.20,759.95] vol=2.6x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 758.43 | 758.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 750.30 | 756.50 | 0.00 | ORB-short ORB[758.40,767.80] vol=2.1x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:35:00 | 746.09 | 754.27 | 0.00 | T1 1.5R @ 746.09 |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 750.30 | 751.68 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 761.60 | 766.74 | 0.00 | ORB-short ORB[764.30,772.60] vol=2.7x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 764.39 | 766.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 721.30 | 699.36 | 0.00 | ORB-long ORB[699.35,706.20] vol=5.2x ATR=10.88 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 710.42 | 702.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:30:00 | 740.80 | 731.28 | 0.00 | ORB-long ORB[721.05,728.90] vol=3.3x ATR=3.53 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 737.27 | 732.10 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:25:00 | 904.50 | 901.73 | 0.00 | ORB-long ORB[891.45,900.50] vol=2.5x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:00:00 | 911.37 | 903.92 | 0.00 | T1 1.5R @ 911.37 |
| Stop hit — per-position SL triggered | 2026-03-25 11:20:00 | 904.50 | 905.82 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:10:00 | 970.70 | 963.30 | 0.00 | ORB-long ORB[956.60,968.35] vol=2.5x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:35:00 | 975.53 | 965.25 | 0.00 | T1 1.5R @ 975.53 |
| Stop hit — per-position SL triggered | 2026-04-10 11:45:00 | 970.70 | 965.68 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 1019.10 | 1013.14 | 0.00 | ORB-long ORB[1004.00,1016.95] vol=2.0x ATR=4.45 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 1014.65 | 1014.64 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 1002.50 | 1009.02 | 0.00 | ORB-short ORB[1004.30,1017.10] vol=1.8x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 13:15:00 | 996.30 | 1006.74 | 0.00 | T1 1.5R @ 996.30 |
| Target hit | 2026-04-21 15:20:00 | 995.90 | 1003.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 1040.70 | 1032.86 | 0.00 | ORB-long ORB[1025.00,1037.35] vol=3.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:10:00 | 1045.71 | 1036.33 | 0.00 | T1 1.5R @ 1045.71 |
| Stop hit — per-position SL triggered | 2026-04-28 12:00:00 | 1040.70 | 1038.91 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 1018.60 | 1025.46 | 0.00 | ORB-short ORB[1023.50,1033.40] vol=1.9x ATR=3.40 |
| Stop hit — per-position SL triggered | 2026-05-07 09:40:00 | 1022.00 | 1024.76 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:25:00 | 1031.40 | 1025.04 | 0.00 | ORB-long ORB[1020.00,1029.60] vol=4.0x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:00:00 | 1037.42 | 1027.09 | 0.00 | T1 1.5R @ 1037.42 |
| Stop hit — per-position SL triggered | 2026-05-08 11:05:00 | 1031.40 | 1027.18 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:45:00 | 764.05 | 2026-02-17 10:40:00 | 760.71 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-18 09:35:00 | 761.25 | 2026-02-18 09:40:00 | 758.43 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-19 10:15:00 | 750.30 | 2026-02-19 10:35:00 | 746.09 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-19 10:15:00 | 750.30 | 2026-02-19 11:15:00 | 750.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:05:00 | 761.60 | 2026-02-23 11:15:00 | 764.39 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-25 09:35:00 | 721.30 | 2026-02-25 09:40:00 | 710.42 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2026-02-27 10:30:00 | 740.80 | 2026-02-27 10:35:00 | 737.27 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-25 10:25:00 | 904.50 | 2026-03-25 11:00:00 | 911.37 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-03-25 10:25:00 | 904.50 | 2026-03-25 11:20:00 | 904.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 11:10:00 | 970.70 | 2026-04-10 11:35:00 | 975.53 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-10 11:10:00 | 970.70 | 2026-04-10 11:45:00 | 970.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 09:35:00 | 1019.10 | 2026-04-17 09:55:00 | 1014.65 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-21 10:40:00 | 1002.50 | 2026-04-21 13:15:00 | 996.30 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-21 10:40:00 | 1002.50 | 2026-04-21 15:20:00 | 995.90 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-28 10:55:00 | 1040.70 | 2026-04-28 11:10:00 | 1045.71 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-28 10:55:00 | 1040.70 | 2026-04-28 12:00:00 | 1040.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 09:35:00 | 1018.60 | 2026-05-07 09:40:00 | 1022.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-08 10:25:00 | 1031.40 | 2026-05-08 11:00:00 | 1037.42 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-05-08 10:25:00 | 1031.40 | 2026-05-08 11:05:00 | 1031.40 | STOP_HIT | 0.50 | 0.00% |

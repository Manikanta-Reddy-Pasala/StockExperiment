# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1144.95
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 6
- **Avg / median % per leg:** 0.36% / 0.00%
- **Sum % (uncompounded):** 6.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.65% | 5.9% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.65% | 5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.07% | 0.6% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.07% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 1 | 11 | 6 | 0.36% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 806.40 | 802.15 | 0.00 | ORB-long ORB[794.10,803.95] vol=3.5x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 810.93 | 803.00 | 0.00 | T1 1.5R @ 810.93 |
| Stop hit — per-position SL triggered | 2026-02-10 09:40:00 | 806.40 | 803.05 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 804.10 | 809.10 | 0.00 | ORB-short ORB[807.60,819.00] vol=3.2x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 806.89 | 808.54 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 814.35 | 808.29 | 0.00 | ORB-long ORB[798.15,808.95] vol=2.2x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 811.53 | 808.46 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 819.90 | 823.81 | 0.00 | ORB-short ORB[820.00,828.95] vol=1.7x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-02-18 11:05:00 | 822.40 | 822.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 809.35 | 815.33 | 0.00 | ORB-short ORB[816.00,825.20] vol=2.6x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:00:00 | 805.54 | 814.19 | 0.00 | T1 1.5R @ 805.54 |
| Stop hit — per-position SL triggered | 2026-02-24 13:20:00 | 809.35 | 811.87 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 825.70 | 819.68 | 0.00 | ORB-long ORB[812.35,821.90] vol=2.5x ATR=2.60 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 823.10 | 821.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 835.00 | 831.94 | 0.00 | ORB-long ORB[826.20,832.45] vol=4.3x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:35:00 | 838.80 | 833.67 | 0.00 | T1 1.5R @ 838.80 |
| Target hit | 2026-02-26 15:20:00 | 880.00 | 855.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 987.90 | 977.12 | 0.00 | ORB-long ORB[967.50,981.40] vol=2.3x ATR=5.56 |
| Stop hit — per-position SL triggered | 2026-03-10 09:55:00 | 982.34 | 981.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 947.70 | 930.74 | 0.00 | ORB-long ORB[914.00,928.00] vol=2.3x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:25:00 | 954.39 | 935.72 | 0.00 | T1 1.5R @ 954.39 |
| Stop hit — per-position SL triggered | 2026-04-17 12:50:00 | 947.70 | 937.55 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:55:00 | 965.80 | 968.03 | 0.00 | ORB-short ORB[967.80,982.10] vol=1.9x ATR=4.82 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 970.62 | 968.50 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:05:00 | 1006.55 | 1010.31 | 0.00 | ORB-short ORB[1007.55,1019.00] vol=2.2x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:40:00 | 1000.32 | 1009.50 | 0.00 | T1 1.5R @ 1000.32 |
| Stop hit — per-position SL triggered | 2026-04-28 10:50:00 | 1006.55 | 1009.33 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 996.00 | 1006.22 | 0.00 | ORB-short ORB[1002.05,1015.25] vol=1.6x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:55:00 | 989.49 | 1003.72 | 0.00 | T1 1.5R @ 989.49 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 996.00 | 1002.84 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 806.40 | 2026-02-10 09:35:00 | 810.93 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-10 09:30:00 | 806.40 | 2026-02-10 09:40:00 | 806.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 09:30:00 | 804.10 | 2026-02-13 09:40:00 | 806.89 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-17 09:55:00 | 814.35 | 2026-02-17 10:05:00 | 811.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-18 09:55:00 | 819.90 | 2026-02-18 11:05:00 | 822.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-24 11:10:00 | 809.35 | 2026-02-24 12:00:00 | 805.54 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-24 11:10:00 | 809.35 | 2026-02-24 13:20:00 | 809.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:40:00 | 825.70 | 2026-02-25 11:20:00 | 823.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 09:30:00 | 835.00 | 2026-02-26 09:35:00 | 838.80 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-26 09:30:00 | 835.00 | 2026-02-26 15:20:00 | 880.00 | TARGET_HIT | 0.50 | 5.39% |
| BUY | retest1 | 2026-03-10 09:35:00 | 987.90 | 2026-03-10 09:55:00 | 982.34 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-04-17 11:00:00 | 947.70 | 2026-04-17 12:25:00 | 954.39 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-04-17 11:00:00 | 947.70 | 2026-04-17 12:50:00 | 947.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:55:00 | 965.80 | 2026-04-24 10:00:00 | 970.62 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-28 10:05:00 | 1006.55 | 2026-04-28 10:40:00 | 1000.32 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-28 10:05:00 | 1006.55 | 2026-04-28 10:50:00 | 1006.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:10:00 | 996.00 | 2026-04-30 10:55:00 | 989.49 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-04-30 10:10:00 | 996.00 | 2026-04-30 11:10:00 | 996.00 | STOP_HIT | 0.50 | 0.00% |

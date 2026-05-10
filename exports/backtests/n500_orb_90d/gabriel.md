# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1136.50
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
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 4
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.10% | 0.9% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.10% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.3% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.08% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1000.80 | 989.14 | 0.00 | ORB-long ORB[970.00,982.15] vol=4.3x ATR=5.59 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 995.21 | 989.67 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 967.50 | 972.52 | 0.00 | ORB-short ORB[969.30,978.20] vol=2.0x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 970.37 | 972.23 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 978.25 | 972.08 | 0.00 | ORB-long ORB[965.80,973.70] vol=4.2x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:45:00 | 983.54 | 974.62 | 0.00 | T1 1.5R @ 983.54 |
| Target hit | 2026-02-25 12:35:00 | 998.35 | 1000.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 947.90 | 938.95 | 0.00 | ORB-long ORB[933.25,941.70] vol=1.7x ATR=5.22 |
| Stop hit — per-position SL triggered | 2026-03-06 10:20:00 | 942.68 | 941.23 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 916.00 | 905.71 | 0.00 | ORB-long ORB[898.95,911.30] vol=1.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-03-11 10:10:00 | 912.06 | 909.36 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 881.35 | 874.32 | 0.00 | ORB-long ORB[868.85,880.70] vol=2.6x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:40:00 | 886.33 | 876.67 | 0.00 | T1 1.5R @ 886.33 |
| Stop hit — per-position SL triggered | 2026-03-12 11:45:00 | 881.35 | 877.02 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 861.60 | 867.91 | 0.00 | ORB-short ORB[869.00,878.40] vol=1.6x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 856.44 | 865.94 | 0.00 | T1 1.5R @ 856.44 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 861.60 | 864.16 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 1010.75 | 1019.58 | 0.00 | ORB-short ORB[1015.75,1026.60] vol=3.4x ATR=4.47 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 1015.22 | 1014.74 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 989.90 | 987.43 | 0.00 | ORB-long ORB[976.45,988.90] vol=3.1x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-04-27 11:00:00 | 985.96 | 987.93 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:25:00 | 1018.55 | 1012.29 | 0.00 | ORB-long ORB[1011.55,1017.00] vol=2.2x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-04-28 10:30:00 | 1015.24 | 1013.45 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 1022.15 | 1028.32 | 0.00 | ORB-short ORB[1025.60,1037.95] vol=1.8x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:35:00 | 1017.23 | 1027.15 | 0.00 | T1 1.5R @ 1017.23 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 1022.15 | 1026.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:45:00 | 1000.80 | 2026-02-10 09:50:00 | 995.21 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2026-02-19 09:30:00 | 967.50 | 2026-02-19 09:40:00 | 970.37 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-25 09:40:00 | 978.25 | 2026-02-25 09:45:00 | 983.54 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-25 09:40:00 | 978.25 | 2026-02-25 12:35:00 | 998.35 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2026-03-06 09:45:00 | 947.90 | 2026-03-06 10:20:00 | 942.68 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-03-11 09:55:00 | 916.00 | 2026-03-11 10:10:00 | 912.06 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-12 11:15:00 | 881.35 | 2026-03-12 11:40:00 | 886.33 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-12 11:15:00 | 881.35 | 2026-03-12 11:45:00 | 881.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:55:00 | 861.60 | 2026-03-13 10:20:00 | 856.44 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-13 09:55:00 | 861.60 | 2026-03-13 10:50:00 | 861.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 09:45:00 | 1010.75 | 2026-04-22 11:05:00 | 1015.22 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-27 10:35:00 | 989.90 | 2026-04-27 11:00:00 | 985.96 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-28 10:25:00 | 1018.55 | 2026-04-28 10:30:00 | 1015.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-29 10:15:00 | 1022.15 | 2026-04-29 10:35:00 | 1017.23 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-29 10:15:00 | 1022.15 | 2026-04-29 10:50:00 | 1022.15 | STOP_HIT | 0.50 | 0.00% |

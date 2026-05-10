# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1122.00
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 4
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 2.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.31% | 1.9% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.31% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.03% | 0.2% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.03% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.17% | 2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 990.90 | 996.03 | 0.00 | ORB-short ORB[994.85,1002.75] vol=2.5x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:45:00 | 986.35 | 992.03 | 0.00 | T1 1.5R @ 986.35 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 990.90 | 992.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 1041.30 | 1037.89 | 0.00 | ORB-long ORB[1027.20,1041.25] vol=1.7x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 1047.34 | 1040.00 | 0.00 | T1 1.5R @ 1047.34 |
| Stop hit — per-position SL triggered | 2026-02-26 11:40:00 | 1041.30 | 1040.39 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1018.00 | 1026.90 | 0.00 | ORB-short ORB[1027.10,1040.00] vol=1.8x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:45:00 | 1012.95 | 1024.87 | 0.00 | T1 1.5R @ 1012.95 |
| Stop hit — per-position SL triggered | 2026-03-06 11:25:00 | 1018.00 | 1022.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 1052.90 | 1050.33 | 0.00 | ORB-long ORB[1040.00,1049.40] vol=6.0x ATR=4.07 |
| Stop hit — per-position SL triggered | 2026-03-11 09:35:00 | 1048.83 | 1050.44 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:40:00 | 1045.40 | 1038.78 | 0.00 | ORB-long ORB[1027.20,1038.40] vol=2.0x ATR=4.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 12:15:00 | 1052.06 | 1043.93 | 0.00 | T1 1.5R @ 1052.06 |
| Target hit | 2026-03-12 15:20:00 | 1060.00 | 1052.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 1038.80 | 1044.00 | 0.00 | ORB-short ORB[1041.30,1050.90] vol=1.7x ATR=5.14 |
| Stop hit — per-position SL triggered | 2026-03-30 11:15:00 | 1043.94 | 1043.86 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1106.40 | 1114.71 | 0.00 | ORB-short ORB[1114.40,1129.20] vol=2.1x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 1109.57 | 1114.16 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 1121.80 | 1112.52 | 0.00 | ORB-long ORB[1104.90,1112.00] vol=2.3x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 1117.66 | 1118.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 09:30:00 | 990.90 | 2026-02-19 09:45:00 | 986.35 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-19 09:30:00 | 990.90 | 2026-02-19 09:50:00 | 990.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:50:00 | 1041.30 | 2026-02-26 11:05:00 | 1047.34 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-26 09:50:00 | 1041.30 | 2026-02-26 11:40:00 | 1041.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1018.00 | 2026-03-06 10:45:00 | 1012.95 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1018.00 | 2026-03-06 11:25:00 | 1018.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:30:00 | 1052.90 | 2026-03-11 09:35:00 | 1048.83 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-12 09:40:00 | 1045.40 | 2026-03-12 12:15:00 | 1052.06 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-03-12 09:40:00 | 1045.40 | 2026-03-12 15:20:00 | 1060.00 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2026-03-30 10:55:00 | 1038.80 | 2026-03-30 11:15:00 | 1043.94 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1106.40 | 2026-04-28 11:20:00 | 1109.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-29 09:40:00 | 1121.80 | 2026-04-29 10:35:00 | 1117.66 | STOP_HIT | 1.00 | -0.37% |

# Bajaj Auto Ltd. (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 10696.50
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 5 / 10 / 7
- **Avg / median % per leg:** 0.41% / 0.30%
- **Sum % (uncompounded):** 9.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 10 | 71.4% | 4 | 4 | 6 | 0.59% | 8.3% |
| BUY @ 2nd Alert (retest1) | 14 | 10 | 71.4% | 4 | 4 | 6 | 0.59% | 8.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.10% | 0.8% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.10% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 12 | 54.5% | 5 | 10 | 7 | 0.41% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 9680.50 | 9657.57 | 0.00 | ORB-long ORB[9582.50,9672.50] vol=1.9x ATR=19.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 9710.06 | 9670.79 | 0.00 | T1 1.5R @ 9710.06 |
| Target hit | 2026-02-10 14:40:00 | 9769.00 | 9773.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 9873.50 | 9834.66 | 0.00 | ORB-long ORB[9804.50,9842.00] vol=1.5x ATR=19.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:35:00 | 9903.16 | 9850.93 | 0.00 | T1 1.5R @ 9903.16 |
| Target hit | 2026-02-18 10:10:00 | 9889.50 | 9891.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 9972.00 | 9917.84 | 0.00 | ORB-long ORB[9839.50,9928.00] vol=2.2x ATR=24.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:35:00 | 10008.80 | 9936.61 | 0.00 | T1 1.5R @ 10008.80 |
| Target hit | 2026-02-25 13:10:00 | 10039.00 | 10044.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 10044.50 | 10111.76 | 0.00 | ORB-short ORB[10103.00,10187.00] vol=2.0x ATR=20.96 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 10065.46 | 10110.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 10010.00 | 10043.74 | 0.00 | ORB-short ORB[10037.00,10091.50] vol=2.1x ATR=19.19 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 10029.19 | 10038.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:05:00 | 9507.00 | 9595.08 | 0.00 | ORB-short ORB[9540.50,9650.00] vol=1.9x ATR=39.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:40:00 | 9448.42 | 9569.20 | 0.00 | T1 1.5R @ 9448.42 |
| Target hit | 2026-03-11 15:20:00 | 9327.50 | 9414.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 8887.00 | 8993.50 | 0.00 | ORB-short ORB[9053.00,9113.50] vol=1.6x ATR=31.66 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 8918.66 | 8983.26 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-25 09:40:00 | 8995.00 | 9014.22 | 0.00 | ORB-short ORB[9002.00,9062.00] vol=1.7x ATR=33.32 |
| Stop hit — per-position SL triggered | 2026-03-25 09:45:00 | 9028.32 | 9014.66 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:00:00 | 8876.50 | 8844.64 | 0.00 | ORB-long ORB[8755.00,8851.00] vol=2.2x ATR=33.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:05:00 | 8926.60 | 8854.10 | 0.00 | T1 1.5R @ 8926.60 |
| Stop hit — per-position SL triggered | 2026-04-06 10:25:00 | 8876.50 | 8873.28 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:15:00 | 8955.00 | 8892.69 | 0.00 | ORB-long ORB[8800.50,8915.50] vol=2.1x ATR=29.28 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 8925.72 | 8922.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 9850.00 | 9900.62 | 0.00 | ORB-short ORB[9880.00,9976.00] vol=3.7x ATR=35.55 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 9885.55 | 9893.85 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 9796.50 | 9833.26 | 0.00 | ORB-short ORB[9837.00,9918.50] vol=3.0x ATR=23.74 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 9820.24 | 9813.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 9645.00 | 9632.12 | 0.00 | ORB-long ORB[9496.00,9640.00] vol=1.9x ATR=26.47 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 9618.53 | 9633.01 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:35:00 | 9573.00 | 9438.77 | 0.00 | ORB-long ORB[9411.00,9530.00] vol=4.7x ATR=35.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:40:00 | 9626.56 | 9569.12 | 0.00 | T1 1.5R @ 9626.56 |
| Target hit | 2026-04-30 15:20:00 | 10024.50 | 9808.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 10721.50 | 10645.93 | 0.00 | ORB-long ORB[10540.00,10647.50] vol=1.7x ATR=24.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:45:00 | 10758.93 | 10673.38 | 0.00 | T1 1.5R @ 10758.93 |
| Stop hit — per-position SL triggered | 2026-05-08 12:45:00 | 10721.50 | 10692.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:35:00 | 9680.50 | 2026-02-10 09:45:00 | 9710.06 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-10 09:35:00 | 9680.50 | 2026-02-10 14:40:00 | 9769.00 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2026-02-18 09:30:00 | 9873.50 | 2026-02-18 09:35:00 | 9903.16 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-18 09:30:00 | 9873.50 | 2026-02-18 10:10:00 | 9889.50 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-02-25 10:25:00 | 9972.00 | 2026-02-25 10:35:00 | 10008.80 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-25 10:25:00 | 9972.00 | 2026-02-25 13:10:00 | 10039.00 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-26 11:00:00 | 10044.50 | 2026-02-26 11:05:00 | 10065.46 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 10:20:00 | 10010.00 | 2026-02-27 10:35:00 | 10029.19 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-03-11 10:05:00 | 9507.00 | 2026-03-11 10:40:00 | 9448.42 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-03-11 10:05:00 | 9507.00 | 2026-03-11 15:20:00 | 9327.50 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2026-03-13 10:35:00 | 8887.00 | 2026-03-13 10:50:00 | 8918.66 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-25 09:40:00 | 8995.00 | 2026-03-25 09:45:00 | 9028.32 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-06 10:00:00 | 8876.50 | 2026-04-06 10:05:00 | 8926.60 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-06 10:00:00 | 8876.50 | 2026-04-06 10:25:00 | 8876.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 10:15:00 | 8955.00 | 2026-04-07 10:50:00 | 8925.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-15 09:30:00 | 9850.00 | 2026-04-15 09:40:00 | 9885.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-16 09:50:00 | 9796.50 | 2026-04-16 11:40:00 | 9820.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-29 10:20:00 | 9645.00 | 2026-04-29 10:35:00 | 9618.53 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-30 10:35:00 | 9573.00 | 2026-04-30 10:40:00 | 9626.56 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-30 10:35:00 | 9573.00 | 2026-04-30 15:20:00 | 10024.50 | TARGET_HIT | 0.50 | 4.72% |
| BUY | retest1 | 2026-05-08 10:45:00 | 10721.50 | 2026-05-08 11:45:00 | 10758.93 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-05-08 10:45:00 | 10721.50 | 2026-05-08 12:45:00 | 10721.50 | STOP_HIT | 0.50 | 0.00% |

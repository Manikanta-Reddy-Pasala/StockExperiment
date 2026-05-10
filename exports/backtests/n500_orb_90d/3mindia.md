# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 32070.00
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
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 5
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 0.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.13% | -1.2% |
| BUY @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.13% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.17% | 1.9% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.17% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 7 | 35.0% | 2 | 13 | 5 | 0.04% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 37620.00 | 37035.21 | 0.00 | ORB-long ORB[36375.00,36815.00] vol=1.8x ATR=251.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:25:00 | 37997.56 | 37292.75 | 0.00 | T1 1.5R @ 37997.56 |
| Stop hit — per-position SL triggered | 2026-02-10 13:35:00 | 37620.00 | 37603.55 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 35660.00 | 35992.51 | 0.00 | ORB-short ORB[35800.00,36335.00] vol=12.2x ATR=101.12 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 35761.12 | 35970.54 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 36320.00 | 36250.86 | 0.00 | ORB-long ORB[36000.00,36300.00] vol=3.6x ATR=97.35 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 36222.65 | 36251.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 36460.00 | 36338.91 | 0.00 | ORB-long ORB[36175.00,36345.00] vol=2.1x ATR=77.25 |
| Stop hit — per-position SL triggered | 2026-02-27 10:40:00 | 36382.75 | 36375.98 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 33550.00 | 33621.27 | 0.00 | ORB-short ORB[33555.00,33795.00] vol=2.4x ATR=89.63 |
| Stop hit — per-position SL triggered | 2026-03-10 11:00:00 | 33639.63 | 33619.26 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 34500.00 | 34327.19 | 0.00 | ORB-long ORB[34045.00,34350.00] vol=5.8x ATR=109.56 |
| Stop hit — per-position SL triggered | 2026-03-11 13:05:00 | 34390.44 | 34396.52 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:30:00 | 32250.00 | 32639.58 | 0.00 | ORB-short ORB[32595.00,32985.00] vol=1.9x ATR=142.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:50:00 | 32036.35 | 32574.21 | 0.00 | T1 1.5R @ 32036.35 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 32250.00 | 32506.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 33595.00 | 33334.52 | 0.00 | ORB-long ORB[32820.00,33200.00] vol=1.7x ATR=137.42 |
| Stop hit — per-position SL triggered | 2026-03-17 13:00:00 | 33457.58 | 33460.53 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 33435.00 | 33142.09 | 0.00 | ORB-long ORB[32730.00,33105.00] vol=2.5x ATR=132.81 |
| Stop hit — per-position SL triggered | 2026-03-20 09:40:00 | 33302.19 | 33175.83 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 31260.00 | 31422.34 | 0.00 | ORB-short ORB[31330.00,31785.00] vol=3.8x ATR=143.66 |
| Stop hit — per-position SL triggered | 2026-03-27 09:45:00 | 31403.66 | 31390.40 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 33630.00 | 33524.26 | 0.00 | ORB-long ORB[33280.00,33550.00] vol=2.1x ATR=104.93 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 33525.07 | 33537.09 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 32755.00 | 32945.47 | 0.00 | ORB-short ORB[32890.00,33115.00] vol=3.5x ATR=79.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:45:00 | 32635.89 | 32866.81 | 0.00 | T1 1.5R @ 32635.89 |
| Target hit | 2026-05-05 15:20:00 | 32440.00 | 32580.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 32295.00 | 32465.26 | 0.00 | ORB-short ORB[32410.00,32795.00] vol=2.6x ATR=83.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:10:00 | 32169.85 | 32341.71 | 0.00 | T1 1.5R @ 32169.85 |
| Stop hit — per-position SL triggered | 2026-05-06 12:25:00 | 32295.00 | 32216.90 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:15:00 | 32805.00 | 32710.41 | 0.00 | ORB-long ORB[32445.00,32800.00] vol=10.1x ATR=79.59 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 32725.41 | 32712.11 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:55:00 | 32270.00 | 32373.46 | 0.00 | ORB-short ORB[32425.00,32700.00] vol=3.9x ATR=93.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:30:00 | 32129.45 | 32267.41 | 0.00 | T1 1.5R @ 32129.45 |
| Target hit | 2026-05-08 13:30:00 | 32250.00 | 32248.87 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:10:00 | 37620.00 | 2026-02-10 10:25:00 | 37997.56 | PARTIAL | 0.50 | 1.00% |
| BUY | retest1 | 2026-02-10 10:10:00 | 37620.00 | 2026-02-10 13:35:00 | 37620.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:15:00 | 35660.00 | 2026-02-17 11:35:00 | 35761.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-25 10:30:00 | 36320.00 | 2026-02-25 10:40:00 | 36222.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-27 09:55:00 | 36460.00 | 2026-02-27 10:40:00 | 36382.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-10 10:50:00 | 33550.00 | 2026-03-10 11:00:00 | 33639.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-11 10:55:00 | 34500.00 | 2026-03-11 13:05:00 | 34390.44 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-16 10:30:00 | 32250.00 | 2026-03-16 10:50:00 | 32036.35 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-16 10:30:00 | 32250.00 | 2026-03-16 11:15:00 | 32250.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:35:00 | 33595.00 | 2026-03-17 13:00:00 | 33457.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-20 09:35:00 | 33435.00 | 2026-03-20 09:40:00 | 33302.19 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-27 09:30:00 | 31260.00 | 2026-03-27 09:45:00 | 31403.66 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-05-04 10:00:00 | 33630.00 | 2026-05-04 10:20:00 | 33525.07 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-05 10:10:00 | 32755.00 | 2026-05-05 10:45:00 | 32635.89 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-05-05 10:10:00 | 32755.00 | 2026-05-05 15:20:00 | 32440.00 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-05-06 09:40:00 | 32295.00 | 2026-05-06 10:10:00 | 32169.85 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 09:40:00 | 32295.00 | 2026-05-06 12:25:00 | 32295.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 11:15:00 | 32805.00 | 2026-05-07 11:25:00 | 32725.41 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-08 09:55:00 | 32270.00 | 2026-05-08 12:30:00 | 32129.45 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-08 09:55:00 | 32270.00 | 2026-05-08 13:30:00 | 32250.00 | TARGET_HIT | 0.50 | 0.06% |

# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 8188.00
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 17
- **Target hits / Stop hits / Partials:** 3 / 17 / 7
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 1.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 1 | 12 | 5 | 0.03% | 0.5% |
| BUY @ 2nd Alert (retest1) | 18 | 6 | 33.3% | 1 | 12 | 5 | 0.03% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.09% | 0.8% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.09% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 10 | 37.0% | 3 | 17 | 7 | 0.05% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 8711.00 | 8742.08 | 0.00 | ORB-short ORB[8731.00,8797.00] vol=3.5x ATR=30.83 |
| Stop hit — per-position SL triggered | 2026-02-10 12:55:00 | 8741.83 | 8717.84 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 8530.00 | 8501.98 | 0.00 | ORB-long ORB[8440.00,8527.50] vol=1.8x ATR=21.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 8562.42 | 8521.08 | 0.00 | T1 1.5R @ 8562.42 |
| Target hit | 2026-02-17 15:20:00 | 8650.00 | 8621.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 8562.00 | 8505.37 | 0.00 | ORB-long ORB[8469.50,8530.00] vol=2.1x ATR=25.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:25:00 | 8600.44 | 8535.22 | 0.00 | T1 1.5R @ 8600.44 |
| Stop hit — per-position SL triggered | 2026-02-23 11:45:00 | 8562.00 | 8549.08 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 8402.00 | 8431.50 | 0.00 | ORB-short ORB[8411.00,8474.50] vol=1.7x ATR=12.77 |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 8414.77 | 8430.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 7990.00 | 8041.36 | 0.00 | ORB-short ORB[8030.00,8098.00] vol=5.2x ATR=30.82 |
| Stop hit — per-position SL triggered | 2026-03-05 09:35:00 | 8020.82 | 8039.47 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 8099.50 | 8164.80 | 0.00 | ORB-short ORB[8143.00,8200.00] vol=2.1x ATR=21.35 |
| Stop hit — per-position SL triggered | 2026-03-06 11:45:00 | 8120.85 | 8149.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 8165.00 | 8109.27 | 0.00 | ORB-long ORB[8054.50,8144.00] vol=3.3x ATR=22.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:55:00 | 8198.59 | 8128.33 | 0.00 | T1 1.5R @ 8198.59 |
| Stop hit — per-position SL triggered | 2026-03-10 12:05:00 | 8165.00 | 8133.73 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 8077.50 | 8127.61 | 0.00 | ORB-short ORB[8130.00,8194.50] vol=2.2x ATR=19.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 12:25:00 | 8048.13 | 8107.72 | 0.00 | T1 1.5R @ 8048.13 |
| Target hit | 2026-03-11 15:20:00 | 7973.50 | 8050.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 8076.50 | 7995.94 | 0.00 | ORB-long ORB[7946.00,8000.00] vol=3.1x ATR=28.13 |
| Stop hit — per-position SL triggered | 2026-03-17 10:55:00 | 8048.37 | 8020.94 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:50:00 | 8101.50 | 8022.10 | 0.00 | ORB-long ORB[7936.50,7999.50] vol=7.4x ATR=25.17 |
| Stop hit — per-position SL triggered | 2026-03-18 11:20:00 | 8076.33 | 8048.74 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 7907.00 | 7892.74 | 0.00 | ORB-long ORB[7851.00,7900.50] vol=1.7x ATR=14.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:20:00 | 7928.78 | 7893.93 | 0.00 | T1 1.5R @ 7928.78 |
| Stop hit — per-position SL triggered | 2026-03-19 12:05:00 | 7907.00 | 7896.34 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 8127.00 | 8062.53 | 0.00 | ORB-long ORB[7995.50,8088.50] vol=2.0x ATR=48.81 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 8078.19 | 8099.42 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 09:50:00 | 7489.50 | 7452.04 | 0.00 | ORB-long ORB[7395.00,7481.50] vol=2.1x ATR=34.48 |
| Stop hit — per-position SL triggered | 2026-04-01 10:20:00 | 7455.02 | 7460.48 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 7933.50 | 7898.02 | 0.00 | ORB-long ORB[7855.50,7905.00] vol=4.0x ATR=19.40 |
| Stop hit — per-position SL triggered | 2026-04-21 09:55:00 | 7914.10 | 7902.31 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 7990.00 | 7951.27 | 0.00 | ORB-long ORB[7910.00,7984.50] vol=4.2x ATR=17.66 |
| Stop hit — per-position SL triggered | 2026-04-23 11:35:00 | 7972.34 | 7961.74 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 8244.00 | 8186.95 | 0.00 | ORB-long ORB[8118.00,8207.00] vol=3.8x ATR=25.44 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 8218.56 | 8212.31 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:00:00 | 8213.00 | 8192.29 | 0.00 | ORB-long ORB[8170.50,8211.50] vol=2.0x ATR=22.69 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 8190.31 | 8203.45 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 8169.50 | 8190.47 | 0.00 | ORB-short ORB[8215.50,8272.00] vol=1.6x ATR=17.87 |
| Stop hit — per-position SL triggered | 2026-04-29 11:05:00 | 8187.37 | 8175.74 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 7949.00 | 7974.66 | 0.00 | ORB-short ORB[7965.00,8031.50] vol=1.6x ATR=20.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:35:00 | 7917.53 | 7956.02 | 0.00 | T1 1.5R @ 7917.53 |
| Target hit | 2026-05-06 14:35:00 | 7936.50 | 7923.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 8048.00 | 7986.64 | 0.00 | ORB-long ORB[7942.50,8017.50] vol=3.9x ATR=19.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:30:00 | 8077.68 | 8028.23 | 0.00 | T1 1.5R @ 8077.68 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 8048.00 | 8035.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:40:00 | 8711.00 | 2026-02-10 12:55:00 | 8741.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-17 09:35:00 | 8530.00 | 2026-02-17 10:10:00 | 8562.42 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-17 09:35:00 | 8530.00 | 2026-02-17 15:20:00 | 8650.00 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2026-02-23 10:50:00 | 8562.00 | 2026-02-23 11:25:00 | 8600.44 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-23 10:50:00 | 8562.00 | 2026-02-23 11:45:00 | 8562.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:55:00 | 8402.00 | 2026-02-26 11:15:00 | 8414.77 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-03-05 09:30:00 | 7990.00 | 2026-03-05 09:35:00 | 8020.82 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-06 11:00:00 | 8099.50 | 2026-03-06 11:45:00 | 8120.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-10 11:10:00 | 8165.00 | 2026-03-10 11:55:00 | 8198.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-10 11:10:00 | 8165.00 | 2026-03-10 12:05:00 | 8165.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:40:00 | 8077.50 | 2026-03-11 12:25:00 | 8048.13 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-11 10:40:00 | 8077.50 | 2026-03-11 15:20:00 | 7973.50 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2026-03-17 10:35:00 | 8076.50 | 2026-03-17 10:55:00 | 8048.37 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-18 10:50:00 | 8101.50 | 2026-03-18 11:20:00 | 8076.33 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-19 11:10:00 | 7907.00 | 2026-03-19 11:20:00 | 7928.78 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-03-19 11:10:00 | 7907.00 | 2026-03-19 12:05:00 | 7907.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 09:30:00 | 8127.00 | 2026-03-20 09:50:00 | 8078.19 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-01 09:50:00 | 7489.50 | 2026-04-01 10:20:00 | 7455.02 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-04-21 09:50:00 | 7933.50 | 2026-04-21 09:55:00 | 7914.10 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-23 11:00:00 | 7990.00 | 2026-04-23 11:35:00 | 7972.34 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-27 09:45:00 | 8244.00 | 2026-04-27 10:05:00 | 8218.56 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-28 10:00:00 | 8213.00 | 2026-04-28 11:15:00 | 8190.31 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-29 10:35:00 | 8169.50 | 2026-04-29 11:05:00 | 8187.37 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-06 09:55:00 | 7949.00 | 2026-05-06 10:35:00 | 7917.53 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-05-06 09:55:00 | 7949.00 | 2026-05-06 14:35:00 | 7936.50 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-05-08 11:15:00 | 8048.00 | 2026-05-08 11:30:00 | 8077.68 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-05-08 11:15:00 | 8048.00 | 2026-05-08 11:40:00 | 8048.00 | STOP_HIT | 0.50 | 0.00% |

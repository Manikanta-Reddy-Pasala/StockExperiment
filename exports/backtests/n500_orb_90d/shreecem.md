# Shree Cement Ltd. (SHREECEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 25445.00
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
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 17
- **Target hits / Stop hits / Partials:** 3 / 17 / 10
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 2.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.02% | 0.3% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.02% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.11% | 1.8% |
| SELL @ 2nd Alert (retest1) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.11% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 13 | 43.3% | 3 | 17 | 10 | 0.07% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 26655.00 | 26784.62 | 0.00 | ORB-short ORB[26725.00,27090.00] vol=2.7x ATR=63.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:40:00 | 26559.27 | 26724.67 | 0.00 | T1 1.5R @ 26559.27 |
| Stop hit — per-position SL triggered | 2026-02-10 12:15:00 | 26655.00 | 26695.12 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:40:00 | 26440.00 | 26557.62 | 0.00 | ORB-short ORB[26530.00,26800.00] vol=1.7x ATR=46.63 |
| Stop hit — per-position SL triggered | 2026-02-12 10:50:00 | 26486.63 | 26545.26 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 25775.00 | 25929.52 | 0.00 | ORB-short ORB[25935.00,26190.00] vol=2.9x ATR=61.87 |
| Stop hit — per-position SL triggered | 2026-02-13 11:35:00 | 25836.87 | 25917.98 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 26145.00 | 26019.37 | 0.00 | ORB-long ORB[25915.00,26115.00] vol=4.5x ATR=46.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 26214.18 | 26109.10 | 0.00 | T1 1.5R @ 26214.18 |
| Target hit | 2026-02-16 15:20:00 | 26265.00 | 26215.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 26360.00 | 26295.52 | 0.00 | ORB-long ORB[26125.00,26350.00] vol=2.0x ATR=54.06 |
| Stop hit — per-position SL triggered | 2026-02-17 10:50:00 | 26305.94 | 26297.36 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:50:00 | 26790.00 | 26744.49 | 0.00 | ORB-long ORB[26510.00,26780.00] vol=1.6x ATR=68.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:55:00 | 26892.16 | 26798.48 | 0.00 | T1 1.5R @ 26892.16 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 26790.00 | 26832.42 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 26530.00 | 26409.23 | 0.00 | ORB-long ORB[26270.00,26525.00] vol=3.8x ATR=53.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:10:00 | 26610.41 | 26437.78 | 0.00 | T1 1.5R @ 26610.41 |
| Stop hit — per-position SL triggered | 2026-02-24 11:55:00 | 26530.00 | 26541.25 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 26780.00 | 26596.13 | 0.00 | ORB-long ORB[26400.00,26740.00] vol=1.9x ATR=71.43 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 26708.57 | 26653.91 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 24965.00 | 25094.81 | 0.00 | ORB-short ORB[25235.00,25520.00] vol=1.7x ATR=61.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:30:00 | 24872.51 | 25038.37 | 0.00 | T1 1.5R @ 24872.51 |
| Target hit | 2026-03-05 13:10:00 | 24950.00 | 24948.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 23850.00 | 23892.20 | 0.00 | ORB-short ORB[23940.00,24130.00] vol=3.9x ATR=52.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:10:00 | 23771.59 | 23855.87 | 0.00 | T1 1.5R @ 23771.59 |
| Stop hit — per-position SL triggered | 2026-03-11 13:25:00 | 23850.00 | 23853.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 23200.00 | 23248.25 | 0.00 | ORB-short ORB[23210.00,23505.00] vol=2.1x ATR=67.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:35:00 | 23098.21 | 23184.93 | 0.00 | T1 1.5R @ 23098.21 |
| Stop hit — per-position SL triggered | 2026-03-13 11:55:00 | 23200.00 | 23161.68 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:50:00 | 23960.00 | 23881.21 | 0.00 | ORB-long ORB[23705.00,23930.00] vol=1.5x ATR=62.93 |
| Stop hit — per-position SL triggered | 2026-03-18 11:50:00 | 23897.07 | 23891.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:25:00 | 23570.00 | 23536.98 | 0.00 | ORB-long ORB[23360.00,23555.00] vol=2.4x ATR=60.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:35:00 | 23661.30 | 23541.15 | 0.00 | T1 1.5R @ 23661.30 |
| Stop hit — per-position SL triggered | 2026-03-20 11:55:00 | 23570.00 | 23559.16 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:50:00 | 24580.00 | 24328.31 | 0.00 | ORB-long ORB[23895.00,24220.00] vol=2.3x ATR=101.73 |
| Stop hit — per-position SL triggered | 2026-04-08 11:50:00 | 24478.27 | 24381.85 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 23920.00 | 24118.03 | 0.00 | ORB-short ORB[23975.00,24320.00] vol=1.6x ATR=102.40 |
| Stop hit — per-position SL triggered | 2026-04-09 12:25:00 | 24022.40 | 23955.30 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 25170.00 | 25327.48 | 0.00 | ORB-short ORB[25335.00,25550.00] vol=3.9x ATR=54.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 12:30:00 | 25087.52 | 25283.95 | 0.00 | T1 1.5R @ 25087.52 |
| Target hit | 2026-04-24 15:20:00 | 24940.00 | 25096.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 25350.00 | 25256.90 | 0.00 | ORB-long ORB[25070.00,25340.00] vol=1.7x ATR=89.43 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 25260.57 | 25320.73 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 24840.00 | 24954.94 | 0.00 | ORB-short ORB[24870.00,25230.00] vol=2.1x ATR=57.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 24753.63 | 24940.62 | 0.00 | T1 1.5R @ 24753.63 |
| Stop hit — per-position SL triggered | 2026-04-28 11:40:00 | 24840.00 | 24908.41 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 24880.00 | 24973.66 | 0.00 | ORB-short ORB[24935.00,25200.00] vol=1.5x ATR=64.55 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 24944.55 | 24970.26 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 25250.00 | 25383.01 | 0.00 | ORB-short ORB[25350.00,25590.00] vol=2.1x ATR=60.22 |
| Stop hit — per-position SL triggered | 2026-05-08 12:00:00 | 25310.22 | 25287.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 26655.00 | 2026-02-10 11:40:00 | 26559.27 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-10 10:35:00 | 26655.00 | 2026-02-10 12:15:00 | 26655.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:40:00 | 26440.00 | 2026-02-12 10:50:00 | 26486.63 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-13 11:15:00 | 25775.00 | 2026-02-13 11:35:00 | 25836.87 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 11:15:00 | 26145.00 | 2026-02-16 11:20:00 | 26214.18 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-16 11:15:00 | 26145.00 | 2026-02-16 15:20:00 | 26265.00 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 10:30:00 | 26360.00 | 2026-02-17 10:50:00 | 26305.94 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-23 09:50:00 | 26790.00 | 2026-02-23 09:55:00 | 26892.16 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-23 09:50:00 | 26790.00 | 2026-02-23 10:10:00 | 26790.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 11:00:00 | 26530.00 | 2026-02-24 11:10:00 | 26610.41 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-24 11:00:00 | 26530.00 | 2026-02-24 11:55:00 | 26530.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:40:00 | 26780.00 | 2026-02-25 11:15:00 | 26708.57 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-05 11:00:00 | 24965.00 | 2026-03-05 11:30:00 | 24872.51 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-05 11:00:00 | 24965.00 | 2026-03-05 13:10:00 | 24950.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-03-11 10:55:00 | 23850.00 | 2026-03-11 13:10:00 | 23771.59 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-03-11 10:55:00 | 23850.00 | 2026-03-11 13:25:00 | 23850.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:50:00 | 23200.00 | 2026-03-13 10:35:00 | 23098.21 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-13 09:50:00 | 23200.00 | 2026-03-13 11:55:00 | 23200.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:50:00 | 23960.00 | 2026-03-18 11:50:00 | 23897.07 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-20 10:25:00 | 23570.00 | 2026-03-20 10:35:00 | 23661.30 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-20 10:25:00 | 23570.00 | 2026-03-20 11:55:00 | 23570.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 10:50:00 | 24580.00 | 2026-04-08 11:50:00 | 24478.27 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-09 09:45:00 | 23920.00 | 2026-04-09 12:25:00 | 24022.40 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-24 10:50:00 | 25170.00 | 2026-04-24 12:30:00 | 25087.52 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-24 10:50:00 | 25170.00 | 2026-04-24 15:20:00 | 24940.00 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2026-04-27 09:30:00 | 25350.00 | 2026-04-27 10:05:00 | 25260.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 11:00:00 | 24840.00 | 2026-04-28 11:15:00 | 24753.63 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-28 11:00:00 | 24840.00 | 2026-04-28 11:40:00 | 24840.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:10:00 | 24880.00 | 2026-05-06 11:15:00 | 24944.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-08 11:05:00 | 25250.00 | 2026-05-08 12:00:00 | 25310.22 | STOP_HIT | 1.00 | -0.24% |

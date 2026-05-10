# MARUTI (MARUTI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 13733.00
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
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 6
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 0.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 0.18% | 0.9% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 0.18% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | -0.01% | -0.1% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | -0.01% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 10 | 47.6% | 4 | 11 | 6 | 0.04% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 15136.00 | 15059.38 | 0.00 | ORB-long ORB[14983.00,15065.00] vol=1.9x ATR=25.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 15174.39 | 15098.97 | 0.00 | T1 1.5R @ 15174.39 |
| Target hit | 2026-02-10 13:00:00 | 15212.00 | 15214.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:25:00 | 15100.00 | 15190.61 | 0.00 | ORB-short ORB[15158.00,15299.00] vol=1.8x ATR=37.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:45:00 | 15043.60 | 15163.00 | 0.00 | T1 1.5R @ 15043.60 |
| Target hit | 2026-02-16 15:10:00 | 15064.00 | 15061.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 14922.00 | 14975.43 | 0.00 | ORB-short ORB[14933.00,15062.00] vol=2.1x ATR=26.09 |
| Stop hit — per-position SL triggered | 2026-02-24 10:55:00 | 14948.09 | 14973.37 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 14999.00 | 14965.72 | 0.00 | ORB-long ORB[14837.00,14948.00] vol=1.6x ATR=25.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:05:00 | 15037.03 | 14994.64 | 0.00 | T1 1.5R @ 15037.03 |
| Target hit | 2026-02-25 12:55:00 | 15025.00 | 15030.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 13767.00 | 13807.78 | 0.00 | ORB-short ORB[13770.00,13948.00] vol=2.4x ATR=36.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:40:00 | 13711.62 | 13800.16 | 0.00 | T1 1.5R @ 13711.62 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 13767.00 | 13793.06 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:15:00 | 12701.00 | 12797.67 | 0.00 | ORB-short ORB[12847.00,12962.00] vol=1.9x ATR=31.38 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 12732.38 | 12794.39 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:55:00 | 12441.00 | 12575.92 | 0.00 | ORB-short ORB[12469.00,12654.00] vol=2.2x ATR=58.32 |
| Stop hit — per-position SL triggered | 2026-03-16 10:00:00 | 12499.32 | 12567.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 12643.00 | 12722.68 | 0.00 | ORB-short ORB[12700.00,12838.00] vol=1.7x ATR=28.32 |
| Stop hit — per-position SL triggered | 2026-03-19 11:20:00 | 12671.32 | 12716.16 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:40:00 | 12338.00 | 12466.76 | 0.00 | ORB-short ORB[12486.00,12650.00] vol=2.1x ATR=42.34 |
| Stop hit — per-position SL triggered | 2026-03-24 11:00:00 | 12380.34 | 12430.26 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:20:00 | 12468.00 | 12502.43 | 0.00 | ORB-short ORB[12502.00,12646.00] vol=1.7x ATR=41.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:25:00 | 12405.61 | 12481.78 | 0.00 | T1 1.5R @ 12405.61 |
| Target hit | 2026-03-27 15:20:00 | 12420.00 | 12416.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:05:00 | 13176.00 | 13261.84 | 0.00 | ORB-short ORB[13251.00,13420.00] vol=2.4x ATR=52.86 |
| Stop hit — per-position SL triggered | 2026-04-13 10:45:00 | 13228.86 | 13236.67 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 13277.00 | 13331.20 | 0.00 | ORB-short ORB[13311.00,13435.00] vol=1.5x ATR=31.59 |
| Stop hit — per-position SL triggered | 2026-04-16 10:10:00 | 13308.59 | 13317.54 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:45:00 | 13070.00 | 13110.78 | 0.00 | ORB-short ORB[13093.00,13189.00] vol=1.5x ATR=31.94 |
| Stop hit — per-position SL triggered | 2026-04-27 11:00:00 | 13101.94 | 13107.32 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 13456.00 | 13549.01 | 0.00 | ORB-short ORB[13522.00,13619.00] vol=1.7x ATR=31.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 13408.90 | 13534.30 | 0.00 | T1 1.5R @ 13408.90 |
| Stop hit — per-position SL triggered | 2026-05-06 11:20:00 | 13456.00 | 13517.87 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 13848.00 | 13788.81 | 0.00 | ORB-long ORB[13698.00,13805.00] vol=1.6x ATR=42.42 |
| Stop hit — per-position SL triggered | 2026-05-07 09:45:00 | 13805.58 | 13791.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 15136.00 | 2026-02-10 09:45:00 | 15174.39 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2026-02-10 09:40:00 | 15136.00 | 2026-02-10 13:00:00 | 15212.00 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-16 10:25:00 | 15100.00 | 2026-02-16 10:45:00 | 15043.60 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-16 10:25:00 | 15100.00 | 2026-02-16 15:10:00 | 15064.00 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-24 10:45:00 | 14922.00 | 2026-02-24 10:55:00 | 14948.09 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-02-25 10:30:00 | 14999.00 | 2026-02-25 11:05:00 | 15037.03 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2026-02-25 10:30:00 | 14999.00 | 2026-02-25 12:55:00 | 15025.00 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-03-11 10:35:00 | 13767.00 | 2026-03-11 10:40:00 | 13711.62 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-03-11 10:35:00 | 13767.00 | 2026-03-11 11:00:00 | 13767.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 11:15:00 | 12701.00 | 2026-03-13 11:20:00 | 12732.38 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-16 09:55:00 | 12441.00 | 2026-03-16 10:00:00 | 12499.32 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-19 11:10:00 | 12643.00 | 2026-03-19 11:20:00 | 12671.32 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-24 10:40:00 | 12338.00 | 2026-03-24 11:00:00 | 12380.34 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-27 10:20:00 | 12468.00 | 2026-03-27 11:25:00 | 12405.61 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-27 10:20:00 | 12468.00 | 2026-03-27 15:20:00 | 12420.00 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-13 10:05:00 | 13176.00 | 2026-04-13 10:45:00 | 13228.86 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-16 09:50:00 | 13277.00 | 2026-04-16 10:10:00 | 13308.59 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-27 10:45:00 | 13070.00 | 2026-04-27 11:00:00 | 13101.94 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-06 10:50:00 | 13456.00 | 2026-05-06 11:00:00 | 13408.90 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-06 10:50:00 | 13456.00 | 2026-05-06 11:20:00 | 13456.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 09:40:00 | 13848.00 | 2026-05-07 09:45:00 | 13805.58 | STOP_HIT | 1.00 | -0.31% |

# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 8990.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 19
- **Target hits / Stop hits / Partials:** 2 / 19 / 5
- **Avg / median % per leg:** -0.03% / -0.27%
- **Sum % (uncompounded):** -0.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.09% | -1.2% |
| BUY @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.09% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 4 | 30.8% | 1 | 9 | 3 | 0.03% | 0.4% |
| SELL @ 2nd Alert (retest1) | 13 | 4 | 30.8% | 1 | 9 | 3 | 0.03% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 7 | 26.9% | 2 | 19 | 5 | -0.03% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 8000.00 | 8047.14 | 0.00 | ORB-short ORB[8040.50,8150.00] vol=8.6x ATR=23.31 |
| Stop hit — per-position SL triggered | 2026-02-12 12:10:00 | 8023.31 | 8032.26 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 7847.00 | 7780.63 | 0.00 | ORB-long ORB[7653.00,7763.00] vol=3.2x ATR=28.85 |
| Stop hit — per-position SL triggered | 2026-02-16 11:10:00 | 7818.15 | 7786.28 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:25:00 | 7822.00 | 7789.69 | 0.00 | ORB-long ORB[7730.50,7799.00] vol=2.2x ATR=19.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:40:00 | 7850.87 | 7817.63 | 0.00 | T1 1.5R @ 7850.87 |
| Target hit | 2026-02-18 12:00:00 | 7918.50 | 7933.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:00:00 | 7849.50 | 7806.83 | 0.00 | ORB-long ORB[7767.00,7824.00] vol=1.6x ATR=24.56 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 7824.94 | 7825.22 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 7584.00 | 7639.23 | 0.00 | ORB-short ORB[7640.00,7750.00] vol=2.9x ATR=21.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:00:00 | 7552.04 | 7628.38 | 0.00 | T1 1.5R @ 7552.04 |
| Target hit | 2026-02-27 15:20:00 | 7511.00 | 7546.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 7468.00 | 7399.17 | 0.00 | ORB-long ORB[7360.50,7442.50] vol=2.4x ATR=27.56 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 7440.44 | 7404.56 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 7295.00 | 7338.93 | 0.00 | ORB-short ORB[7297.00,7395.00] vol=2.8x ATR=23.43 |
| Stop hit — per-position SL triggered | 2026-03-10 10:55:00 | 7318.43 | 7336.97 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:15:00 | 7252.50 | 7332.79 | 0.00 | ORB-short ORB[7345.50,7425.00] vol=3.1x ATR=23.39 |
| Stop hit — per-position SL triggered | 2026-03-11 10:25:00 | 7275.89 | 7319.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:10:00 | 6869.00 | 6824.30 | 0.00 | ORB-long ORB[6794.50,6840.50] vol=2.5x ATR=26.88 |
| Stop hit — per-position SL triggered | 2026-03-19 11:05:00 | 6842.12 | 6829.48 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:00:00 | 7080.50 | 7000.40 | 0.00 | ORB-long ORB[6888.00,6968.00] vol=5.6x ATR=24.61 |
| Stop hit — per-position SL triggered | 2026-03-25 11:40:00 | 7055.89 | 7012.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:10:00 | 6999.00 | 7090.11 | 0.00 | ORB-short ORB[7075.50,7156.00] vol=1.5x ATR=31.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:30:00 | 6951.61 | 7078.56 | 0.00 | T1 1.5R @ 6951.61 |
| Stop hit — per-position SL triggered | 2026-04-01 13:05:00 | 6999.00 | 7016.51 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 7505.00 | 7491.67 | 0.00 | ORB-long ORB[7439.50,7500.00] vol=2.6x ATR=26.35 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 7478.65 | 7492.60 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 7600.50 | 7557.77 | 0.00 | ORB-long ORB[7488.50,7534.00] vol=1.6x ATR=21.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:15:00 | 7632.71 | 7583.04 | 0.00 | T1 1.5R @ 7632.71 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 7600.50 | 7592.71 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 11:15:00 | 7670.00 | 7741.47 | 0.00 | ORB-short ORB[7721.50,7782.00] vol=7.7x ATR=18.88 |
| Stop hit — per-position SL triggered | 2026-04-21 14:15:00 | 7688.88 | 7705.83 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 7750.00 | 7719.72 | 0.00 | ORB-long ORB[7640.00,7724.50] vol=2.9x ATR=21.32 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 7728.68 | 7720.41 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 7611.50 | 7671.16 | 0.00 | ORB-short ORB[7653.50,7766.00] vol=1.7x ATR=25.35 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 7636.85 | 7664.69 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 7766.50 | 7676.73 | 0.00 | ORB-long ORB[7531.00,7644.00] vol=1.7x ATR=33.52 |
| Stop hit — per-position SL triggered | 2026-04-27 10:10:00 | 7732.98 | 7722.12 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:05:00 | 7560.00 | 7609.72 | 0.00 | ORB-short ORB[7613.00,7649.00] vol=1.6x ATR=21.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 7527.93 | 7579.57 | 0.00 | T1 1.5R @ 7527.93 |
| Stop hit — per-position SL triggered | 2026-04-28 13:00:00 | 7560.00 | 7565.20 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 7549.00 | 7562.38 | 0.00 | ORB-short ORB[7561.50,7636.00] vol=12.2x ATR=20.27 |
| Stop hit — per-position SL triggered | 2026-04-29 10:55:00 | 7569.27 | 7562.40 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 7834.50 | 7808.38 | 0.00 | ORB-long ORB[7729.00,7795.00] vol=1.6x ATR=31.81 |
| Stop hit — per-position SL triggered | 2026-05-04 10:35:00 | 7802.69 | 7833.80 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:35:00 | 7714.00 | 7766.84 | 0.00 | ORB-short ORB[7750.00,7843.50] vol=3.6x ATR=20.35 |
| Stop hit — per-position SL triggered | 2026-05-06 10:45:00 | 7734.35 | 7765.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:10:00 | 8000.00 | 2026-02-12 12:10:00 | 8023.31 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-16 10:45:00 | 7847.00 | 2026-02-16 11:10:00 | 7818.15 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-18 10:25:00 | 7822.00 | 2026-02-18 10:40:00 | 7850.87 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-18 10:25:00 | 7822.00 | 2026-02-18 12:00:00 | 7918.50 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2026-02-26 10:00:00 | 7849.50 | 2026-02-26 10:55:00 | 7824.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 10:50:00 | 7584.00 | 2026-02-27 11:00:00 | 7552.04 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 10:50:00 | 7584.00 | 2026-02-27 15:20:00 | 7511.00 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2026-03-05 10:50:00 | 7468.00 | 2026-03-05 11:20:00 | 7440.44 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-10 10:40:00 | 7295.00 | 2026-03-10 10:55:00 | 7318.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-11 10:15:00 | 7252.50 | 2026-03-11 10:25:00 | 7275.89 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-19 10:10:00 | 6869.00 | 2026-03-19 11:05:00 | 6842.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-25 11:00:00 | 7080.50 | 2026-03-25 11:40:00 | 7055.89 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-01 11:10:00 | 6999.00 | 2026-04-01 11:30:00 | 6951.61 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-01 11:10:00 | 6999.00 | 2026-04-01 13:05:00 | 6999.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 7505.00 | 2026-04-10 10:05:00 | 7478.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-17 10:00:00 | 7600.50 | 2026-04-17 10:15:00 | 7632.71 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-17 10:00:00 | 7600.50 | 2026-04-17 11:00:00 | 7600.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 11:15:00 | 7670.00 | 2026-04-21 14:15:00 | 7688.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-22 09:35:00 | 7750.00 | 2026-04-22 09:40:00 | 7728.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-24 11:10:00 | 7611.50 | 2026-04-24 11:20:00 | 7636.85 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-27 09:55:00 | 7766.50 | 2026-04-27 10:10:00 | 7732.98 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-28 10:05:00 | 7560.00 | 2026-04-28 11:25:00 | 7527.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-28 10:05:00 | 7560.00 | 2026-04-28 13:00:00 | 7560.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:35:00 | 7549.00 | 2026-04-29 10:55:00 | 7569.27 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-04 09:35:00 | 7834.50 | 2026-05-04 10:35:00 | 7802.69 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-06 10:35:00 | 7714.00 | 2026-05-06 10:45:00 | 7734.35 | STOP_HIT | 1.00 | -0.26% |

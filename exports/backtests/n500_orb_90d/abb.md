# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 7010.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 7
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 3.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.15% | 2.0% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.15% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.3% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 10 | 40.0% | 3 | 15 | 7 | 0.13% | 3.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 5810.00 | 5831.56 | 0.00 | ORB-short ORB[5830.50,5861.00] vol=2.2x ATR=11.22 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 5821.22 | 5831.02 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 5791.00 | 5816.57 | 0.00 | ORB-short ORB[5803.50,5864.50] vol=2.4x ATR=11.51 |
| Stop hit — per-position SL triggered | 2026-02-12 11:10:00 | 5802.51 | 5815.16 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 5804.00 | 5771.23 | 0.00 | ORB-long ORB[5750.00,5798.00] vol=3.4x ATR=15.75 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 5788.25 | 5771.73 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 5815.00 | 5794.93 | 0.00 | ORB-long ORB[5742.50,5811.50] vol=1.6x ATR=12.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 5834.45 | 5797.93 | 0.00 | T1 1.5R @ 5834.45 |
| Target hit | 2026-02-16 15:20:00 | 5894.00 | 5860.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 5789.50 | 5816.19 | 0.00 | ORB-short ORB[5801.00,5858.50] vol=1.7x ATR=11.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 5771.69 | 5809.62 | 0.00 | T1 1.5R @ 5771.69 |
| Stop hit — per-position SL triggered | 2026-02-18 11:35:00 | 5789.50 | 5808.63 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 5922.00 | 5868.92 | 0.00 | ORB-long ORB[5833.50,5896.00] vol=2.8x ATR=18.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:00:00 | 5950.10 | 5887.32 | 0.00 | T1 1.5R @ 5950.10 |
| Stop hit — per-position SL triggered | 2026-03-05 11:10:00 | 5922.00 | 5889.58 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:10:00 | 6016.50 | 6006.64 | 0.00 | ORB-long ORB[5869.00,5959.00] vol=1.9x ATR=19.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:35:00 | 6046.27 | 6011.06 | 0.00 | T1 1.5R @ 6046.27 |
| Target hit | 2026-03-06 15:20:00 | 6069.00 | 6047.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:40:00 | 6392.00 | 6354.01 | 0.00 | ORB-long ORB[6287.50,6343.00] vol=2.1x ATR=21.36 |
| Stop hit — per-position SL triggered | 2026-03-18 11:35:00 | 6370.64 | 6370.16 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:25:00 | 6506.00 | 6430.44 | 0.00 | ORB-long ORB[6370.00,6465.00] vol=1.8x ATR=26.58 |
| Stop hit — per-position SL triggered | 2026-04-08 10:35:00 | 6479.42 | 6435.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:30:00 | 6510.00 | 6526.50 | 0.00 | ORB-short ORB[6511.50,6592.00] vol=1.5x ATR=17.20 |
| Stop hit — per-position SL triggered | 2026-04-09 10:35:00 | 6527.20 | 6526.45 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 6975.50 | 6925.86 | 0.00 | ORB-long ORB[6875.00,6949.50] vol=4.6x ATR=29.07 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 6946.43 | 6947.76 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 6806.50 | 6854.11 | 0.00 | ORB-short ORB[6814.00,6907.00] vol=2.1x ATR=22.25 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 6828.75 | 6843.99 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 7293.50 | 7265.43 | 0.00 | ORB-long ORB[7184.00,7287.50] vol=1.6x ATR=16.63 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 7276.87 | 7269.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 7417.00 | 7356.94 | 0.00 | ORB-long ORB[7325.00,7398.00] vol=2.7x ATR=26.27 |
| Stop hit — per-position SL triggered | 2026-04-27 11:00:00 | 7390.73 | 7358.26 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:15:00 | 7371.00 | 7416.77 | 0.00 | ORB-short ORB[7402.50,7490.00] vol=2.9x ATR=21.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:35:00 | 7338.59 | 7406.46 | 0.00 | T1 1.5R @ 7338.59 |
| Target hit | 2026-04-28 15:20:00 | 7285.00 | 7321.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 7236.50 | 7274.58 | 0.00 | ORB-short ORB[7260.50,7340.00] vol=2.1x ATR=22.00 |
| Stop hit — per-position SL triggered | 2026-04-29 11:30:00 | 7258.50 | 7258.11 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 7289.50 | 7275.51 | 0.00 | ORB-long ORB[7215.00,7278.50] vol=2.3x ATR=25.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:45:00 | 7327.91 | 7288.38 | 0.00 | T1 1.5R @ 7327.91 |
| Stop hit — per-position SL triggered | 2026-05-05 10:20:00 | 7289.50 | 7304.94 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 7086.00 | 7203.49 | 0.00 | ORB-short ORB[7183.00,7243.50] vol=5.8x ATR=32.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:15:00 | 7038.01 | 7173.41 | 0.00 | T1 1.5R @ 7038.01 |
| Stop hit — per-position SL triggered | 2026-05-07 11:20:00 | 7086.00 | 7166.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:05:00 | 5810.00 | 2026-02-11 11:15:00 | 5821.22 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-12 10:55:00 | 5791.00 | 2026-02-12 11:10:00 | 5802.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-13 10:55:00 | 5804.00 | 2026-02-13 11:00:00 | 5788.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-16 11:05:00 | 5815.00 | 2026-02-16 11:20:00 | 5834.45 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-16 11:05:00 | 5815.00 | 2026-02-16 15:20:00 | 5894.00 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2026-02-18 10:55:00 | 5789.50 | 2026-02-18 11:25:00 | 5771.69 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 10:55:00 | 5789.50 | 2026-02-18 11:35:00 | 5789.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:45:00 | 5922.00 | 2026-03-05 11:00:00 | 5950.10 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-05 10:45:00 | 5922.00 | 2026-03-05 11:10:00 | 5922.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 11:10:00 | 6016.50 | 2026-03-06 11:35:00 | 6046.27 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-06 11:10:00 | 6016.50 | 2026-03-06 15:20:00 | 6069.00 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-03-18 10:40:00 | 6392.00 | 2026-03-18 11:35:00 | 6370.64 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-08 10:25:00 | 6506.00 | 2026-04-08 10:35:00 | 6479.42 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-09 10:30:00 | 6510.00 | 2026-04-09 10:35:00 | 6527.20 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-15 09:30:00 | 6975.50 | 2026-04-15 10:50:00 | 6946.43 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-16 09:40:00 | 6806.50 | 2026-04-16 09:50:00 | 6828.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 11:10:00 | 7293.50 | 2026-04-21 11:35:00 | 7276.87 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-27 10:55:00 | 7417.00 | 2026-04-27 11:00:00 | 7390.73 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 10:15:00 | 7371.00 | 2026-04-28 10:35:00 | 7338.59 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-28 10:15:00 | 7371.00 | 2026-04-28 15:20:00 | 7285.00 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2026-04-29 10:00:00 | 7236.50 | 2026-04-29 11:30:00 | 7258.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:35:00 | 7289.50 | 2026-05-05 09:45:00 | 7327.91 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-05-05 09:35:00 | 7289.50 | 2026-05-05 10:20:00 | 7289.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 11:10:00 | 7086.00 | 2026-05-07 11:15:00 | 7038.01 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-05-07 11:10:00 | 7086.00 | 2026-05-07 11:20:00 | 7086.00 | STOP_HIT | 0.50 | 0.00% |

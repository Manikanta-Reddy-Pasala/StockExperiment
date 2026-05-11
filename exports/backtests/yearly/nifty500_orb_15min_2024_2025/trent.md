# Trent Ltd. (TRENT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (33646 bars)
- **Last close:** 4249.10
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 13 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 102 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 54
- **Target hits / Stop hits / Partials:** 13 / 54 / 35
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 23.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 20 | 39.2% | 6 | 31 | 14 | 0.16% | 7.9% |
| BUY @ 2nd Alert (retest1) | 51 | 20 | 39.2% | 6 | 31 | 14 | 0.16% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 28 | 54.9% | 7 | 23 | 21 | 0.30% | 15.1% |
| SELL @ 2nd Alert (retest1) | 51 | 28 | 54.9% | 7 | 23 | 21 | 0.30% | 15.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 102 | 48 | 47.1% | 13 | 54 | 35 | 0.23% | 23.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 4570.00 | 4558.34 | 0.00 | ORB-long ORB[4532.20,4565.40] vol=1.9x ATR=12.23 |
| Stop hit — per-position SL triggered | 2024-05-15 09:50:00 | 4557.77 | 4560.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:55:00 | 4705.00 | 4666.02 | 0.00 | ORB-long ORB[4623.65,4690.00] vol=2.9x ATR=20.32 |
| Stop hit — per-position SL triggered | 2024-05-21 11:20:00 | 4684.68 | 4685.34 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 4612.95 | 4636.24 | 0.00 | ORB-short ORB[4620.50,4685.40] vol=1.5x ATR=19.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:00:00 | 4583.34 | 4625.34 | 0.00 | T1 1.5R @ 4583.34 |
| Target hit | 2024-05-31 13:20:00 | 4603.90 | 4582.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-07 11:15:00 | 4854.45 | 4879.11 | 0.00 | ORB-short ORB[4857.85,4916.50] vol=1.9x ATR=13.43 |
| Stop hit — per-position SL triggered | 2024-06-07 11:55:00 | 4867.88 | 4875.93 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:40:00 | 4974.30 | 4959.23 | 0.00 | ORB-long ORB[4909.05,4945.00] vol=1.7x ATR=13.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:40:00 | 4994.36 | 4967.23 | 0.00 | T1 1.5R @ 4994.36 |
| Target hit | 2024-06-12 13:55:00 | 5020.80 | 5031.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-06-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:55:00 | 5366.25 | 5408.59 | 0.00 | ORB-short ORB[5418.30,5459.00] vol=1.6x ATR=16.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:15:00 | 5341.51 | 5402.25 | 0.00 | T1 1.5R @ 5341.51 |
| Stop hit — per-position SL triggered | 2024-06-25 11:50:00 | 5366.25 | 5396.15 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:45:00 | 5482.45 | 5423.04 | 0.00 | ORB-long ORB[5358.55,5394.35] vol=4.8x ATR=20.51 |
| Stop hit — per-position SL triggered | 2024-06-28 11:20:00 | 5461.94 | 5444.52 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:25:00 | 5553.65 | 5499.56 | 0.00 | ORB-long ORB[5476.35,5533.00] vol=2.0x ATR=17.44 |
| Stop hit — per-position SL triggered | 2024-07-02 10:40:00 | 5536.21 | 5510.47 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:10:00 | 5518.45 | 5538.85 | 0.00 | ORB-short ORB[5528.85,5577.75] vol=2.6x ATR=10.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:25:00 | 5502.99 | 5535.56 | 0.00 | T1 1.5R @ 5502.99 |
| Stop hit — per-position SL triggered | 2024-07-03 11:35:00 | 5518.45 | 5533.95 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 5578.15 | 5553.15 | 0.00 | ORB-long ORB[5493.75,5562.80] vol=3.3x ATR=17.98 |
| Stop hit — per-position SL triggered | 2024-07-04 09:35:00 | 5560.17 | 5557.01 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:10:00 | 5559.35 | 5586.73 | 0.00 | ORB-short ORB[5584.25,5634.60] vol=1.7x ATR=14.11 |
| Stop hit — per-position SL triggered | 2024-07-09 11:35:00 | 5573.46 | 5584.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:15:00 | 5531.75 | 5570.78 | 0.00 | ORB-short ORB[5571.00,5626.45] vol=1.6x ATR=18.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 5504.27 | 5567.06 | 0.00 | T1 1.5R @ 5504.27 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 5531.75 | 5545.96 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:30:00 | 5392.15 | 5355.27 | 0.00 | ORB-long ORB[5328.10,5387.85] vol=2.7x ATR=15.53 |
| Stop hit — per-position SL triggered | 2024-07-26 12:00:00 | 5376.62 | 5371.52 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-08-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:35:00 | 5349.55 | 5318.39 | 0.00 | ORB-long ORB[5274.60,5340.40] vol=1.7x ATR=24.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:00:00 | 5386.34 | 5334.45 | 0.00 | T1 1.5R @ 5386.34 |
| Stop hit — per-position SL triggered | 2024-08-07 12:00:00 | 5349.55 | 5341.52 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 6311.50 | 6352.33 | 0.00 | ORB-short ORB[6353.45,6401.70] vol=3.3x ATR=23.45 |
| Stop hit — per-position SL triggered | 2024-08-14 09:40:00 | 6334.95 | 6343.59 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:30:00 | 6607.05 | 6576.04 | 0.00 | ORB-long ORB[6522.30,6594.95] vol=4.6x ATR=23.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 09:40:00 | 6642.45 | 6592.72 | 0.00 | T1 1.5R @ 6642.45 |
| Target hit | 2024-08-19 12:20:00 | 6659.00 | 6674.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-08-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:25:00 | 6881.60 | 6837.27 | 0.00 | ORB-long ORB[6790.00,6834.90] vol=1.5x ATR=16.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:30:00 | 6905.64 | 6845.88 | 0.00 | T1 1.5R @ 6905.64 |
| Stop hit — per-position SL triggered | 2024-08-22 10:35:00 | 6881.60 | 6847.33 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 7050.00 | 7019.02 | 0.00 | ORB-long ORB[6985.10,7031.90] vol=3.4x ATR=21.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:45:00 | 7082.54 | 7031.41 | 0.00 | T1 1.5R @ 7082.54 |
| Stop hit — per-position SL triggered | 2024-08-26 10:00:00 | 7050.00 | 7034.12 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:55:00 | 6905.35 | 6929.50 | 0.00 | ORB-short ORB[6910.00,6973.00] vol=1.5x ATR=20.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:00:00 | 6875.00 | 6925.89 | 0.00 | T1 1.5R @ 6875.00 |
| Stop hit — per-position SL triggered | 2024-08-27 10:45:00 | 6905.35 | 6918.73 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:00:00 | 7150.00 | 7211.63 | 0.00 | ORB-short ORB[7210.30,7284.95] vol=1.9x ATR=25.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:30:00 | 7111.38 | 7188.02 | 0.00 | T1 1.5R @ 7111.38 |
| Stop hit — per-position SL triggered | 2024-08-29 12:50:00 | 7150.00 | 7180.96 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:40:00 | 7223.55 | 7198.74 | 0.00 | ORB-long ORB[7159.50,7223.50] vol=1.6x ATR=26.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:45:00 | 7262.99 | 7203.73 | 0.00 | T1 1.5R @ 7262.99 |
| Stop hit — per-position SL triggered | 2024-08-30 09:50:00 | 7223.55 | 7205.13 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:10:00 | 7200.00 | 7143.33 | 0.00 | ORB-long ORB[7092.20,7169.90] vol=1.6x ATR=25.65 |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 7174.35 | 7146.54 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:40:00 | 7313.75 | 7259.97 | 0.00 | ORB-long ORB[7228.65,7282.15] vol=1.5x ATR=17.99 |
| Stop hit — per-position SL triggered | 2024-09-16 10:50:00 | 7295.76 | 7262.88 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:10:00 | 7375.00 | 7343.31 | 0.00 | ORB-long ORB[7282.10,7350.00] vol=1.6x ATR=19.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:35:00 | 7404.64 | 7356.78 | 0.00 | T1 1.5R @ 7404.64 |
| Stop hit — per-position SL triggered | 2024-09-20 14:55:00 | 7375.00 | 7393.10 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:15:00 | 7590.75 | 7538.55 | 0.00 | ORB-long ORB[7481.10,7539.00] vol=2.4x ATR=19.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:30:00 | 7620.71 | 7550.41 | 0.00 | T1 1.5R @ 7620.71 |
| Stop hit — per-position SL triggered | 2024-09-23 11:45:00 | 7590.75 | 7554.86 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 7721.55 | 7677.97 | 0.00 | ORB-long ORB[7590.75,7700.00] vol=1.8x ATR=24.61 |
| Stop hit — per-position SL triggered | 2024-09-25 10:00:00 | 7696.94 | 7696.10 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:35:00 | 8181.10 | 8116.01 | 0.00 | ORB-long ORB[8035.00,8150.00] vol=2.2x ATR=42.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 09:45:00 | 8245.38 | 8138.66 | 0.00 | T1 1.5R @ 8245.38 |
| Target hit | 2024-10-09 15:10:00 | 8211.75 | 8214.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:15:00 | 8335.00 | 8267.67 | 0.00 | ORB-long ORB[8196.05,8295.70] vol=1.6x ATR=33.86 |
| Stop hit — per-position SL triggered | 2024-10-14 10:20:00 | 8301.14 | 8271.37 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:50:00 | 8132.20 | 8201.66 | 0.00 | ORB-short ORB[8180.00,8260.00] vol=1.8x ATR=24.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 10:25:00 | 8095.78 | 8180.08 | 0.00 | T1 1.5R @ 8095.78 |
| Stop hit — per-position SL triggered | 2024-10-15 14:55:00 | 8132.20 | 8145.23 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 7522.00 | 7589.80 | 0.00 | ORB-short ORB[7589.30,7666.50] vol=2.2x ATR=31.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:45:00 | 7475.15 | 7566.31 | 0.00 | T1 1.5R @ 7475.15 |
| Stop hit — per-position SL triggered | 2024-10-22 11:10:00 | 7522.00 | 7557.19 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:45:00 | 7208.80 | 7361.23 | 0.00 | ORB-short ORB[7410.00,7511.20] vol=1.9x ATR=38.58 |
| Stop hit — per-position SL triggered | 2024-10-25 10:50:00 | 7247.38 | 7341.58 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 11:15:00 | 6321.35 | 6368.87 | 0.00 | ORB-short ORB[6380.10,6453.00] vol=1.6x ATR=22.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 11:35:00 | 6288.24 | 6357.31 | 0.00 | T1 1.5R @ 6288.24 |
| Stop hit — per-position SL triggered | 2024-11-18 12:25:00 | 6321.35 | 6340.62 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:10:00 | 6510.05 | 6484.05 | 0.00 | ORB-long ORB[6430.00,6490.00] vol=2.6x ATR=25.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 11:00:00 | 6547.68 | 6496.83 | 0.00 | T1 1.5R @ 6547.68 |
| Target hit | 2024-11-22 15:20:00 | 6659.00 | 6593.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-11-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:45:00 | 6751.55 | 6693.31 | 0.00 | ORB-long ORB[6631.35,6695.00] vol=2.2x ATR=19.49 |
| Stop hit — per-position SL triggered | 2024-11-27 11:00:00 | 6732.06 | 6701.57 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 6744.75 | 6791.96 | 0.00 | ORB-short ORB[6763.65,6859.15] vol=1.9x ATR=19.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:40:00 | 6715.96 | 6766.10 | 0.00 | T1 1.5R @ 6715.96 |
| Stop hit — per-position SL triggered | 2024-12-03 13:55:00 | 6744.75 | 6727.90 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 6750.85 | 6807.77 | 0.00 | ORB-short ORB[6813.05,6869.95] vol=1.7x ATR=19.81 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 6770.66 | 6785.79 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 7010.00 | 7083.69 | 0.00 | ORB-short ORB[7050.00,7140.00] vol=1.5x ATR=45.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 12:50:00 | 6941.67 | 7024.20 | 0.00 | T1 1.5R @ 6941.67 |
| Stop hit — per-position SL triggered | 2024-12-06 15:00:00 | 7010.00 | 7014.26 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:00:00 | 6884.95 | 6921.97 | 0.00 | ORB-short ORB[6958.55,7025.00] vol=2.6x ATR=22.22 |
| Stop hit — per-position SL triggered | 2024-12-09 11:05:00 | 6907.17 | 6921.11 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 7067.10 | 7046.57 | 0.00 | ORB-long ORB[6980.00,7059.95] vol=2.5x ATR=30.25 |
| Stop hit — per-position SL triggered | 2024-12-12 11:05:00 | 7036.85 | 7054.03 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:00:00 | 6914.65 | 6926.06 | 0.00 | ORB-short ORB[6950.00,7000.00] vol=1.5x ATR=21.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 11:25:00 | 6882.88 | 6921.06 | 0.00 | T1 1.5R @ 6882.88 |
| Stop hit — per-position SL triggered | 2024-12-13 11:40:00 | 6914.65 | 6918.88 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:50:00 | 6964.00 | 6973.90 | 0.00 | ORB-short ORB[6967.40,7030.00] vol=1.5x ATR=18.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:40:00 | 6936.70 | 6969.49 | 0.00 | T1 1.5R @ 6936.70 |
| Target hit | 2024-12-17 15:00:00 | 6954.90 | 6943.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2024-12-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:00:00 | 6996.00 | 6929.02 | 0.00 | ORB-long ORB[6848.80,6947.95] vol=1.6x ATR=26.33 |
| Stop hit — per-position SL triggered | 2024-12-18 10:45:00 | 6969.67 | 6958.06 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:15:00 | 7140.00 | 7063.73 | 0.00 | ORB-long ORB[6991.60,7080.00] vol=1.9x ATR=31.72 |
| Stop hit — per-position SL triggered | 2024-12-19 11:30:00 | 7108.28 | 7098.01 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 7154.00 | 7107.23 | 0.00 | ORB-long ORB[7055.95,7119.00] vol=2.2x ATR=17.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:15:00 | 7180.30 | 7117.28 | 0.00 | T1 1.5R @ 7180.30 |
| Stop hit — per-position SL triggered | 2025-01-02 11:50:00 | 7154.00 | 7131.23 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 7144.90 | 7238.42 | 0.00 | ORB-short ORB[7272.00,7338.60] vol=2.9x ATR=24.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:40:00 | 7107.71 | 7221.10 | 0.00 | T1 1.5R @ 7107.71 |
| Target hit | 2025-01-06 15:20:00 | 6974.35 | 7062.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 6135.75 | 6169.45 | 0.00 | ORB-short ORB[6153.25,6225.00] vol=2.2x ATR=19.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:50:00 | 6107.17 | 6145.79 | 0.00 | T1 1.5R @ 6107.17 |
| Stop hit — per-position SL triggered | 2025-01-20 10:30:00 | 6135.75 | 6135.19 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 5672.05 | 5699.13 | 0.00 | ORB-short ORB[5692.85,5755.05] vol=1.7x ATR=23.08 |
| Stop hit — per-position SL triggered | 2025-01-24 09:40:00 | 5695.13 | 5699.11 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 11:10:00 | 5345.90 | 5382.14 | 0.00 | ORB-short ORB[5433.05,5490.00] vol=2.8x ATR=23.80 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 5369.70 | 5381.59 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:55:00 | 5558.65 | 5517.94 | 0.00 | ORB-long ORB[5420.50,5498.25] vol=1.8x ATR=31.82 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 5526.83 | 5544.69 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-02-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-11 10:45:00 | 5228.00 | 5222.03 | 0.00 | ORB-long ORB[5171.10,5221.50] vol=2.1x ATR=17.37 |
| Stop hit — per-position SL triggered | 2025-02-11 11:00:00 | 5210.63 | 5221.95 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-02-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 10:40:00 | 5348.05 | 5289.10 | 0.00 | ORB-long ORB[5204.30,5278.00] vol=1.7x ATR=21.78 |
| Stop hit — per-position SL triggered | 2025-02-13 11:30:00 | 5326.27 | 5312.08 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 11:10:00 | 4990.20 | 5038.31 | 0.00 | ORB-short ORB[5051.65,5119.45] vol=1.7x ATR=17.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:30:00 | 4964.51 | 5022.74 | 0.00 | T1 1.5R @ 4964.51 |
| Stop hit — per-position SL triggered | 2025-02-18 15:00:00 | 4990.20 | 4987.75 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:00:00 | 5011.60 | 4987.71 | 0.00 | ORB-long ORB[4951.45,5002.05] vol=1.9x ATR=14.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:10:00 | 5033.62 | 4990.21 | 0.00 | T1 1.5R @ 5033.62 |
| Target hit | 2025-02-20 15:20:00 | 5099.90 | 5050.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 09:40:00 | 5065.75 | 5087.30 | 0.00 | ORB-short ORB[5067.80,5129.25] vol=1.8x ATR=21.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:05:00 | 5033.52 | 5079.83 | 0.00 | T1 1.5R @ 5033.52 |
| Target hit | 2025-02-25 15:20:00 | 4957.85 | 5014.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-03-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:50:00 | 5038.60 | 4972.50 | 0.00 | ORB-long ORB[4865.70,4939.10] vol=2.0x ATR=23.33 |
| Stop hit — per-position SL triggered | 2025-03-04 12:40:00 | 5015.27 | 5003.70 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-10 10:50:00 | 4888.00 | 4914.46 | 0.00 | ORB-short ORB[4924.50,4997.85] vol=5.3x ATR=13.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 12:05:00 | 4867.72 | 4903.43 | 0.00 | T1 1.5R @ 4867.72 |
| Target hit | 2025-03-10 15:20:00 | 4797.90 | 4867.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-03-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:40:00 | 5050.65 | 5010.86 | 0.00 | ORB-long ORB[4960.00,5021.10] vol=1.5x ATR=22.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 09:50:00 | 5084.95 | 5031.96 | 0.00 | T1 1.5R @ 5084.95 |
| Stop hit — per-position SL triggered | 2025-03-12 10:10:00 | 5050.65 | 5047.33 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:30:00 | 5096.20 | 5051.03 | 0.00 | ORB-long ORB[4970.10,5044.95] vol=1.5x ATR=19.00 |
| Stop hit — per-position SL triggered | 2025-03-13 10:35:00 | 5077.20 | 5053.08 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:55:00 | 5130.00 | 5093.62 | 0.00 | ORB-long ORB[5014.00,5088.00] vol=1.8x ATR=15.14 |
| Stop hit — per-position SL triggered | 2025-03-17 11:05:00 | 5114.86 | 5096.99 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:50:00 | 5230.00 | 5206.59 | 0.00 | ORB-long ORB[5171.45,5228.00] vol=1.6x ATR=17.44 |
| Stop hit — per-position SL triggered | 2025-03-18 10:20:00 | 5212.56 | 5214.35 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 11:10:00 | 5099.80 | 5103.65 | 0.00 | ORB-short ORB[5113.95,5168.15] vol=1.6x ATR=11.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 12:05:00 | 5081.86 | 5101.33 | 0.00 | T1 1.5R @ 5081.86 |
| Target hit | 2025-03-24 15:20:00 | 5059.50 | 5086.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:35:00 | 5137.00 | 5107.08 | 0.00 | ORB-long ORB[5071.90,5121.50] vol=1.6x ATR=16.22 |
| Stop hit — per-position SL triggered | 2025-03-25 09:40:00 | 5120.78 | 5110.43 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:25:00 | 5241.60 | 5180.73 | 0.00 | ORB-long ORB[5158.75,5218.95] vol=2.7x ATR=23.93 |
| Stop hit — per-position SL triggered | 2025-03-26 10:30:00 | 5217.67 | 5185.46 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 5201.50 | 5162.54 | 0.00 | ORB-long ORB[5126.00,5180.00] vol=1.8x ATR=20.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:45:00 | 5232.24 | 5179.02 | 0.00 | T1 1.5R @ 5232.24 |
| Target hit | 2025-04-21 15:20:00 | 5355.00 | 5287.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 5302.50 | 5338.54 | 0.00 | ORB-short ORB[5320.00,5382.00] vol=1.9x ATR=18.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:55:00 | 5274.60 | 5320.06 | 0.00 | T1 1.5R @ 5274.60 |
| Target hit | 2025-04-23 11:15:00 | 5294.00 | 5280.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — SELL (started 2025-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 10:35:00 | 5270.50 | 5306.92 | 0.00 | ORB-short ORB[5280.00,5348.50] vol=1.6x ATR=16.92 |
| Stop hit — per-position SL triggered | 2025-04-24 10:50:00 | 5287.42 | 5302.08 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:50:00 | 5190.50 | 5261.56 | 0.00 | ORB-short ORB[5300.50,5356.00] vol=2.9x ATR=20.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:00:00 | 5159.78 | 5239.83 | 0.00 | T1 1.5R @ 5159.78 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 5190.50 | 5228.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:35:00 | 4570.00 | 2024-05-15 09:50:00 | 4557.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-21 09:55:00 | 4705.00 | 2024-05-21 11:20:00 | 4684.68 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-05-31 09:45:00 | 4612.95 | 2024-05-31 10:00:00 | 4583.34 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-05-31 09:45:00 | 4612.95 | 2024-05-31 13:20:00 | 4603.90 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-06-07 11:15:00 | 4854.45 | 2024-06-07 11:55:00 | 4867.88 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-12 10:40:00 | 4974.30 | 2024-06-12 11:40:00 | 4994.36 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-12 10:40:00 | 4974.30 | 2024-06-12 13:55:00 | 5020.80 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2024-06-25 10:55:00 | 5366.25 | 2024-06-25 11:15:00 | 5341.51 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-06-25 10:55:00 | 5366.25 | 2024-06-25 11:50:00 | 5366.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 10:45:00 | 5482.45 | 2024-06-28 11:20:00 | 5461.94 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-02 10:25:00 | 5553.65 | 2024-07-02 10:40:00 | 5536.21 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-03 11:10:00 | 5518.45 | 2024-07-03 11:25:00 | 5502.99 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-07-03 11:10:00 | 5518.45 | 2024-07-03 11:35:00 | 5518.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 09:30:00 | 5578.15 | 2024-07-04 09:35:00 | 5560.17 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-09 11:10:00 | 5559.35 | 2024-07-09 11:35:00 | 5573.46 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-10 10:15:00 | 5531.75 | 2024-07-10 10:20:00 | 5504.27 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-07-10 10:15:00 | 5531.75 | 2024-07-10 10:55:00 | 5531.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:30:00 | 5392.15 | 2024-07-26 12:00:00 | 5376.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-07 10:35:00 | 5349.55 | 2024-08-07 11:00:00 | 5386.34 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-08-07 10:35:00 | 5349.55 | 2024-08-07 12:00:00 | 5349.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 09:30:00 | 6311.50 | 2024-08-14 09:40:00 | 6334.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-19 09:30:00 | 6607.05 | 2024-08-19 09:40:00 | 6642.45 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-19 09:30:00 | 6607.05 | 2024-08-19 12:20:00 | 6659.00 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2024-08-22 10:25:00 | 6881.60 | 2024-08-22 10:30:00 | 6905.64 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-22 10:25:00 | 6881.60 | 2024-08-22 10:35:00 | 6881.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:40:00 | 7050.00 | 2024-08-26 09:45:00 | 7082.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-26 09:40:00 | 7050.00 | 2024-08-26 10:00:00 | 7050.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 09:55:00 | 6905.35 | 2024-08-27 10:00:00 | 6875.00 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-27 09:55:00 | 6905.35 | 2024-08-27 10:45:00 | 6905.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-29 11:00:00 | 7150.00 | 2024-08-29 11:30:00 | 7111.38 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-08-29 11:00:00 | 7150.00 | 2024-08-29 12:50:00 | 7150.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 09:40:00 | 7223.55 | 2024-08-30 09:45:00 | 7262.99 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-30 09:40:00 | 7223.55 | 2024-08-30 09:50:00 | 7223.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-05 10:10:00 | 7200.00 | 2024-09-05 10:15:00 | 7174.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-16 10:40:00 | 7313.75 | 2024-09-16 10:50:00 | 7295.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-20 11:10:00 | 7375.00 | 2024-09-20 11:35:00 | 7404.64 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-20 11:10:00 | 7375.00 | 2024-09-20 14:55:00 | 7375.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-23 11:15:00 | 7590.75 | 2024-09-23 11:30:00 | 7620.71 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-23 11:15:00 | 7590.75 | 2024-09-23 11:45:00 | 7590.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 09:30:00 | 7721.55 | 2024-09-25 10:00:00 | 7696.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-09 09:35:00 | 8181.10 | 2024-10-09 09:45:00 | 8245.38 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-10-09 09:35:00 | 8181.10 | 2024-10-09 15:10:00 | 8211.75 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-10-14 10:15:00 | 8335.00 | 2024-10-14 10:20:00 | 8301.14 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-15 09:50:00 | 8132.20 | 2024-10-15 10:25:00 | 8095.78 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-15 09:50:00 | 8132.20 | 2024-10-15 14:55:00 | 8132.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:30:00 | 7522.00 | 2024-10-22 10:45:00 | 7475.15 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-10-22 10:30:00 | 7522.00 | 2024-10-22 11:10:00 | 7522.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:45:00 | 7208.80 | 2024-10-25 10:50:00 | 7247.38 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-11-18 11:15:00 | 6321.35 | 2024-11-18 11:35:00 | 6288.24 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-18 11:15:00 | 6321.35 | 2024-11-18 12:25:00 | 6321.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:10:00 | 6510.05 | 2024-11-22 11:00:00 | 6547.68 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-11-22 10:10:00 | 6510.05 | 2024-11-22 15:20:00 | 6659.00 | TARGET_HIT | 0.50 | 2.29% |
| BUY | retest1 | 2024-11-27 10:45:00 | 6751.55 | 2024-11-27 11:00:00 | 6732.06 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-03 09:30:00 | 6744.75 | 2024-12-03 09:40:00 | 6715.96 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-03 09:30:00 | 6744.75 | 2024-12-03 13:55:00 | 6744.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:55:00 | 6750.85 | 2024-12-05 12:05:00 | 6770.66 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-06 09:30:00 | 7010.00 | 2024-12-06 12:50:00 | 6941.67 | PARTIAL | 0.50 | 0.97% |
| SELL | retest1 | 2024-12-06 09:30:00 | 7010.00 | 2024-12-06 15:00:00 | 7010.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-09 11:00:00 | 6884.95 | 2024-12-09 11:05:00 | 6907.17 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-12 09:30:00 | 7067.10 | 2024-12-12 11:05:00 | 7036.85 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-13 11:00:00 | 6914.65 | 2024-12-13 11:25:00 | 6882.88 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-13 11:00:00 | 6914.65 | 2024-12-13 11:40:00 | 6914.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:50:00 | 6964.00 | 2024-12-17 11:40:00 | 6936.70 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-17 10:50:00 | 6964.00 | 2024-12-17 15:00:00 | 6954.90 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-12-18 10:00:00 | 6996.00 | 2024-12-18 10:45:00 | 6969.67 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-19 10:15:00 | 7140.00 | 2024-12-19 11:30:00 | 7108.28 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-02 11:05:00 | 7154.00 | 2025-01-02 11:15:00 | 7180.30 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-02 11:05:00 | 7154.00 | 2025-01-02 11:50:00 | 7154.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 11:10:00 | 7144.90 | 2025-01-06 11:40:00 | 7107.71 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-06 11:10:00 | 7144.90 | 2025-01-06 15:20:00 | 6974.35 | TARGET_HIT | 0.50 | 2.39% |
| SELL | retest1 | 2025-01-20 09:30:00 | 6135.75 | 2025-01-20 09:50:00 | 6107.17 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-20 09:30:00 | 6135.75 | 2025-01-20 10:30:00 | 6135.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:35:00 | 5672.05 | 2025-01-24 09:40:00 | 5695.13 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-28 11:10:00 | 5345.90 | 2025-01-28 11:15:00 | 5369.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-29 09:55:00 | 5558.65 | 2025-01-29 11:20:00 | 5526.83 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-02-11 10:45:00 | 5228.00 | 2025-02-11 11:00:00 | 5210.63 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-13 10:40:00 | 5348.05 | 2025-02-13 11:30:00 | 5326.27 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-02-18 11:10:00 | 4990.20 | 2025-02-18 11:30:00 | 4964.51 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-02-18 11:10:00 | 4990.20 | 2025-02-18 15:00:00 | 4990.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 11:00:00 | 5011.60 | 2025-02-20 11:10:00 | 5033.62 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-02-20 11:00:00 | 5011.60 | 2025-02-20 15:20:00 | 5099.90 | TARGET_HIT | 0.50 | 1.76% |
| SELL | retest1 | 2025-02-25 09:40:00 | 5065.75 | 2025-02-25 10:05:00 | 5033.52 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-02-25 09:40:00 | 5065.75 | 2025-02-25 15:20:00 | 4957.85 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2025-03-04 10:50:00 | 5038.60 | 2025-03-04 12:40:00 | 5015.27 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-03-10 10:50:00 | 4888.00 | 2025-03-10 12:05:00 | 4867.72 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-03-10 10:50:00 | 4888.00 | 2025-03-10 15:20:00 | 4797.90 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2025-03-12 09:40:00 | 5050.65 | 2025-03-12 09:50:00 | 5084.95 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-03-12 09:40:00 | 5050.65 | 2025-03-12 10:10:00 | 5050.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-13 10:30:00 | 5096.20 | 2025-03-13 10:35:00 | 5077.20 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-17 10:55:00 | 5130.00 | 2025-03-17 11:05:00 | 5114.86 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-18 09:50:00 | 5230.00 | 2025-03-18 10:20:00 | 5212.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-24 11:10:00 | 5099.80 | 2025-03-24 12:05:00 | 5081.86 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-03-24 11:10:00 | 5099.80 | 2025-03-24 15:20:00 | 5059.50 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2025-03-25 09:35:00 | 5137.00 | 2025-03-25 09:40:00 | 5120.78 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-26 10:25:00 | 5241.60 | 2025-03-26 10:30:00 | 5217.67 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-21 09:35:00 | 5201.50 | 2025-04-21 09:45:00 | 5232.24 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-21 09:35:00 | 5201.50 | 2025-04-21 15:20:00 | 5355.00 | TARGET_HIT | 0.50 | 2.95% |
| SELL | retest1 | 2025-04-23 09:30:00 | 5302.50 | 2025-04-23 09:55:00 | 5274.60 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-23 09:30:00 | 5302.50 | 2025-04-23 11:15:00 | 5294.00 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-04-24 10:35:00 | 5270.50 | 2025-04-24 10:50:00 | 5287.42 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-25 09:50:00 | 5190.50 | 2025-04-25 10:00:00 | 5159.78 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-04-25 09:50:00 | 5190.50 | 2025-04-25 10:10:00 | 5190.50 | STOP_HIT | 0.50 | 0.00% |

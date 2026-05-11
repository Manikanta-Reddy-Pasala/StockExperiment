# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-12-01 15:25:00 (27346 bars)
- **Last close:** 7040.50
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 9 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 47
- **Target hits / Stop hits / Partials:** 9 / 47 / 19
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 3.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 13 | 38.2% | 4 | 21 | 9 | 0.02% | 0.7% |
| BUY @ 2nd Alert (retest1) | 34 | 13 | 38.2% | 4 | 21 | 9 | 0.02% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 15 | 36.6% | 5 | 26 | 10 | 0.06% | 2.6% |
| SELL @ 2nd Alert (retest1) | 41 | 15 | 36.6% | 5 | 26 | 10 | 0.06% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 28 | 37.3% | 9 | 47 | 19 | 0.04% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:25:00 | 4374.10 | 4385.93 | 0.00 | ORB-short ORB[4389.20,4415.95] vol=2.7x ATR=10.89 |
| Stop hit — per-position SL triggered | 2024-05-16 10:35:00 | 4384.99 | 4385.22 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:15:00 | 4448.80 | 4420.00 | 0.00 | ORB-long ORB[4404.35,4430.00] vol=6.5x ATR=10.80 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 4438.00 | 4420.91 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:40:00 | 4464.70 | 4440.54 | 0.00 | ORB-long ORB[4400.05,4460.00] vol=2.2x ATR=17.18 |
| Stop hit — per-position SL triggered | 2024-05-21 10:45:00 | 4447.52 | 4457.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 4319.35 | 4350.66 | 0.00 | ORB-short ORB[4374.00,4432.05] vol=9.1x ATR=23.89 |
| Target hit | 2024-05-22 15:20:00 | 4301.85 | 4314.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:15:00 | 4301.10 | 4323.65 | 0.00 | ORB-short ORB[4312.55,4353.70] vol=2.3x ATR=11.22 |
| Stop hit — per-position SL triggered | 2024-05-23 10:20:00 | 4312.32 | 4322.01 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 4141.00 | 4161.15 | 0.00 | ORB-short ORB[4152.00,4212.00] vol=1.7x ATR=21.54 |
| Stop hit — per-position SL triggered | 2024-05-31 13:10:00 | 4162.54 | 4148.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:35:00 | 4270.00 | 4248.82 | 0.00 | ORB-long ORB[4192.00,4255.00] vol=5.6x ATR=23.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 09:45:00 | 4305.50 | 4257.87 | 0.00 | T1 1.5R @ 4305.50 |
| Target hit | 2024-06-06 13:45:00 | 4284.85 | 4287.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:40:00 | 4363.80 | 4328.33 | 0.00 | ORB-long ORB[4286.85,4335.75] vol=4.3x ATR=17.49 |
| Stop hit — per-position SL triggered | 2024-06-07 10:45:00 | 4346.31 | 4329.03 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 4442.40 | 4422.83 | 0.00 | ORB-long ORB[4370.05,4425.45] vol=5.3x ATR=18.07 |
| Stop hit — per-position SL triggered | 2024-06-10 12:55:00 | 4424.33 | 4433.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 4381.70 | 4392.58 | 0.00 | ORB-short ORB[4390.20,4415.00] vol=3.5x ATR=14.88 |
| Stop hit — per-position SL triggered | 2024-06-12 09:55:00 | 4396.58 | 4389.82 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 5527.60 | 5545.80 | 0.00 | ORB-short ORB[5538.45,5581.50] vol=1.7x ATR=18.29 |
| Stop hit — per-position SL triggered | 2024-07-02 09:35:00 | 5545.89 | 5541.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:30:00 | 5650.00 | 5716.56 | 0.00 | ORB-short ORB[5731.00,5799.00] vol=2.4x ATR=19.84 |
| Stop hit — per-position SL triggered | 2024-07-04 10:35:00 | 5669.84 | 5713.10 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 5308.40 | 5379.69 | 0.00 | ORB-short ORB[5376.05,5428.10] vol=2.0x ATR=21.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 5276.74 | 5361.45 | 0.00 | T1 1.5R @ 5276.74 |
| Stop hit — per-position SL triggered | 2024-07-10 11:05:00 | 5308.40 | 5306.83 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 11:00:00 | 5301.00 | 5328.42 | 0.00 | ORB-short ORB[5310.00,5379.35] vol=3.0x ATR=17.87 |
| Stop hit — per-position SL triggered | 2024-07-12 11:40:00 | 5318.87 | 5322.58 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 5281.00 | 5318.18 | 0.00 | ORB-short ORB[5313.20,5372.00] vol=2.0x ATR=24.99 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 5305.99 | 5309.07 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:15:00 | 5301.15 | 5259.98 | 0.00 | ORB-long ORB[5210.00,5284.00] vol=2.7x ATR=26.17 |
| Stop hit — per-position SL triggered | 2024-07-24 10:25:00 | 5274.98 | 5261.37 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:25:00 | 5405.00 | 5371.00 | 0.00 | ORB-long ORB[5348.75,5402.95] vol=2.2x ATR=15.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:40:00 | 5427.73 | 5393.28 | 0.00 | T1 1.5R @ 5427.73 |
| Stop hit — per-position SL triggered | 2024-07-30 13:15:00 | 5405.00 | 5405.44 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:00:00 | 5272.30 | 5261.82 | 0.00 | ORB-long ORB[5199.80,5251.60] vol=17.4x ATR=16.80 |
| Stop hit — per-position SL triggered | 2024-08-08 10:05:00 | 5255.50 | 5261.80 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:05:00 | 5353.90 | 5334.58 | 0.00 | ORB-long ORB[5275.00,5344.80] vol=1.7x ATR=21.99 |
| Stop hit — per-position SL triggered | 2024-08-09 11:45:00 | 5331.91 | 5341.98 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:55:00 | 5579.10 | 5550.28 | 0.00 | ORB-long ORB[5556.05,5577.00] vol=1.6x ATR=24.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:00:00 | 5613.68 | 5560.61 | 0.00 | T1 1.5R @ 5613.68 |
| Stop hit — per-position SL triggered | 2024-08-22 10:05:00 | 5579.10 | 5561.58 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:40:00 | 5665.05 | 5629.58 | 0.00 | ORB-long ORB[5568.65,5622.00] vol=2.2x ATR=28.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:45:00 | 5707.68 | 5656.53 | 0.00 | T1 1.5R @ 5707.68 |
| Target hit | 2024-08-23 10:30:00 | 5698.00 | 5709.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2024-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:50:00 | 5959.35 | 6059.37 | 0.00 | ORB-short ORB[6076.40,6152.90] vol=1.6x ATR=22.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:00:00 | 5924.88 | 6026.41 | 0.00 | T1 1.5R @ 5924.88 |
| Target hit | 2024-09-02 15:20:00 | 5861.55 | 5945.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:50:00 | 6035.00 | 6090.98 | 0.00 | ORB-short ORB[6041.65,6131.10] vol=2.4x ATR=28.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:40:00 | 5992.44 | 6081.87 | 0.00 | T1 1.5R @ 5992.44 |
| Target hit | 2024-09-10 15:20:00 | 5934.15 | 6003.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 11:15:00 | 6065.10 | 6044.77 | 0.00 | ORB-long ORB[5980.00,6044.40] vol=2.2x ATR=20.51 |
| Stop hit — per-position SL triggered | 2024-09-12 11:25:00 | 6044.59 | 6045.20 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:50:00 | 6780.35 | 6700.78 | 0.00 | ORB-long ORB[6550.00,6629.90] vol=1.7x ATR=46.42 |
| Stop hit — per-position SL triggered | 2024-09-26 11:30:00 | 6733.93 | 6720.21 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-10-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:50:00 | 6551.50 | 6493.21 | 0.00 | ORB-long ORB[6451.30,6530.00] vol=2.5x ATR=35.07 |
| Stop hit — per-position SL triggered | 2024-10-01 09:55:00 | 6516.43 | 6496.09 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 10:25:00 | 6600.15 | 6502.40 | 0.00 | ORB-long ORB[6400.00,6483.65] vol=6.7x ATR=31.03 |
| Stop hit — per-position SL triggered | 2024-10-03 10:30:00 | 6569.12 | 6507.22 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:35:00 | 6180.00 | 6244.21 | 0.00 | ORB-short ORB[6241.65,6329.00] vol=1.6x ATR=23.55 |
| Stop hit — per-position SL triggered | 2024-10-10 10:45:00 | 6203.55 | 6242.12 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:00:00 | 6190.00 | 6215.99 | 0.00 | ORB-short ORB[6204.05,6259.95] vol=1.8x ATR=12.75 |
| Stop hit — per-position SL triggered | 2024-10-11 11:05:00 | 6202.75 | 6215.02 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 11:00:00 | 6204.70 | 6253.60 | 0.00 | ORB-short ORB[6244.05,6310.00] vol=3.3x ATR=19.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 13:20:00 | 6175.80 | 6237.79 | 0.00 | T1 1.5R @ 6175.80 |
| Stop hit — per-position SL triggered | 2024-10-15 13:30:00 | 6204.70 | 6235.08 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 6055.05 | 6102.82 | 0.00 | ORB-short ORB[6132.05,6191.45] vol=2.6x ATR=18.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 13:05:00 | 6027.66 | 6080.19 | 0.00 | T1 1.5R @ 6027.66 |
| Stop hit — per-position SL triggered | 2024-10-17 13:40:00 | 6055.05 | 6078.20 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 4867.80 | 4933.24 | 0.00 | ORB-short ORB[4924.10,4990.95] vol=3.0x ATR=24.18 |
| Stop hit — per-position SL triggered | 2024-10-29 09:40:00 | 4891.98 | 4926.13 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:50:00 | 5091.25 | 5041.89 | 0.00 | ORB-long ORB[5015.95,5077.40] vol=1.6x ATR=25.51 |
| Stop hit — per-position SL triggered | 2024-10-31 10:20:00 | 5065.74 | 5056.08 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-11-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:45:00 | 4971.30 | 5003.67 | 0.00 | ORB-short ORB[4984.00,5048.00] vol=1.5x ATR=29.50 |
| Stop hit — per-position SL triggered | 2024-11-05 09:55:00 | 5000.80 | 4998.92 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 4885.50 | 4922.75 | 0.00 | ORB-short ORB[4910.05,4975.05] vol=2.5x ATR=18.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 4857.39 | 4875.23 | 0.00 | T1 1.5R @ 4857.39 |
| Target hit | 2024-11-13 10:30:00 | 4819.95 | 4819.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — SELL (started 2024-11-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:05:00 | 5102.55 | 5121.55 | 0.00 | ORB-short ORB[5158.90,5215.45] vol=14.5x ATR=22.50 |
| Target hit | 2024-11-26 15:20:00 | 5098.70 | 5114.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 5033.75 | 5048.04 | 0.00 | ORB-short ORB[5049.00,5092.00] vol=7.1x ATR=20.33 |
| Stop hit — per-position SL triggered | 2024-11-27 10:15:00 | 5054.08 | 5046.28 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:30:00 | 4925.05 | 4963.93 | 0.00 | ORB-short ORB[4984.40,5043.35] vol=11.4x ATR=28.12 |
| Stop hit — per-position SL triggered | 2024-12-03 11:05:00 | 4953.17 | 4959.59 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:55:00 | 4915.55 | 4944.71 | 0.00 | ORB-short ORB[4935.05,4978.50] vol=5.2x ATR=10.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:10:00 | 4899.69 | 4931.54 | 0.00 | T1 1.5R @ 4899.69 |
| Stop hit — per-position SL triggered | 2024-12-09 11:15:00 | 4915.55 | 4931.40 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:35:00 | 5006.00 | 4955.57 | 0.00 | ORB-long ORB[4910.25,4972.35] vol=1.8x ATR=20.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:45:00 | 5036.92 | 4985.23 | 0.00 | T1 1.5R @ 5036.92 |
| Stop hit — per-position SL triggered | 2024-12-10 09:50:00 | 5006.00 | 4987.74 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:05:00 | 5077.20 | 5066.97 | 0.00 | ORB-long ORB[5025.10,5074.70] vol=2.0x ATR=17.19 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 5060.01 | 5067.75 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:50:00 | 5205.45 | 5231.81 | 0.00 | ORB-short ORB[5210.00,5274.75] vol=3.1x ATR=16.27 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 5221.72 | 5224.44 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 5435.45 | 5404.79 | 0.00 | ORB-long ORB[5353.90,5405.00] vol=7.0x ATR=13.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:30:00 | 5456.34 | 5420.77 | 0.00 | T1 1.5R @ 5456.34 |
| Stop hit — per-position SL triggered | 2024-12-27 11:35:00 | 5435.45 | 5420.92 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 5311.45 | 5337.03 | 0.00 | ORB-short ORB[5319.00,5375.00] vol=2.1x ATR=18.11 |
| Stop hit — per-position SL triggered | 2025-01-02 10:10:00 | 5329.56 | 5335.99 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 09:30:00 | 5333.35 | 5304.93 | 0.00 | ORB-long ORB[5248.55,5321.85] vol=1.6x ATR=15.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 09:35:00 | 5357.31 | 5315.09 | 0.00 | T1 1.5R @ 5357.31 |
| Target hit | 2025-01-06 10:10:00 | 5355.30 | 5360.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-01-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:35:00 | 4668.90 | 4704.30 | 0.00 | ORB-short ORB[4709.80,4779.50] vol=2.3x ATR=21.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 13:10:00 | 4637.13 | 4690.10 | 0.00 | T1 1.5R @ 4637.13 |
| Stop hit — per-position SL triggered | 2025-01-21 15:10:00 | 4668.90 | 4670.85 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:50:00 | 4580.85 | 4608.03 | 0.00 | ORB-short ORB[4602.10,4649.00] vol=7.0x ATR=15.92 |
| Stop hit — per-position SL triggered | 2025-01-24 10:55:00 | 4596.77 | 4607.49 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-02-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:45:00 | 4085.75 | 4061.60 | 0.00 | ORB-long ORB[4026.45,4069.95] vol=1.7x ATR=16.30 |
| Stop hit — per-position SL triggered | 2025-02-21 10:00:00 | 4069.45 | 4073.80 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 11:00:00 | 3999.95 | 3995.60 | 0.00 | ORB-long ORB[3950.20,3998.95] vol=5.1x ATR=16.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 11:40:00 | 4025.20 | 3997.25 | 0.00 | T1 1.5R @ 4025.20 |
| Target hit | 2025-02-24 14:20:00 | 4032.55 | 4034.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:40:00 | 4673.85 | 4683.32 | 0.00 | ORB-short ORB[4681.00,4738.85] vol=3.5x ATR=35.77 |
| Stop hit — per-position SL triggered | 2025-03-17 10:05:00 | 4709.62 | 4682.73 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:45:00 | 4772.85 | 4752.84 | 0.00 | ORB-long ORB[4701.05,4757.80] vol=5.5x ATR=15.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:00:00 | 4795.88 | 4774.35 | 0.00 | T1 1.5R @ 4795.88 |
| Stop hit — per-position SL triggered | 2025-03-21 11:30:00 | 4772.85 | 4777.92 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 10:40:00 | 4802.75 | 4819.19 | 0.00 | ORB-short ORB[4816.10,4861.55] vol=1.8x ATR=16.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 10:45:00 | 4778.49 | 4818.43 | 0.00 | T1 1.5R @ 4778.49 |
| Stop hit — per-position SL triggered | 2025-03-24 10:50:00 | 4802.75 | 4810.98 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 09:30:00 | 4770.50 | 4790.89 | 0.00 | ORB-short ORB[4804.45,4851.50] vol=2.9x ATR=26.44 |
| Stop hit — per-position SL triggered | 2025-03-27 10:30:00 | 4796.94 | 4780.93 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 11:15:00 | 4712.95 | 4750.16 | 0.00 | ORB-short ORB[4726.15,4790.00] vol=5.7x ATR=13.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 12:40:00 | 4693.30 | 4732.14 | 0.00 | T1 1.5R @ 4693.30 |
| Stop hit — per-position SL triggered | 2025-04-03 14:50:00 | 4712.95 | 4716.53 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 4880.00 | 4822.13 | 0.00 | ORB-long ORB[4762.90,4820.00] vol=3.1x ATR=18.47 |
| Stop hit — per-position SL triggered | 2025-04-29 09:40:00 | 4861.53 | 4877.84 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:45:00 | 4709.90 | 4642.77 | 0.00 | ORB-long ORB[4586.00,4654.50] vol=1.7x ATR=18.37 |
| Stop hit — per-position SL triggered | 2025-05-05 12:25:00 | 4691.53 | 4665.83 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 10:25:00 | 4374.10 | 2024-05-16 10:35:00 | 4384.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-17 11:15:00 | 4448.80 | 2024-05-17 11:20:00 | 4438.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-21 09:40:00 | 4464.70 | 2024-05-21 10:45:00 | 4447.52 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-05-22 09:40:00 | 4319.35 | 2024-05-22 15:20:00 | 4301.85 | TARGET_HIT | 1.00 | 0.41% |
| SELL | retest1 | 2024-05-23 10:15:00 | 4301.10 | 2024-05-23 10:20:00 | 4312.32 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-31 09:40:00 | 4141.00 | 2024-05-31 13:10:00 | 4162.54 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-06-06 09:35:00 | 4270.00 | 2024-06-06 09:45:00 | 4305.50 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-06 09:35:00 | 4270.00 | 2024-06-06 13:45:00 | 4284.85 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-06-07 10:40:00 | 4363.80 | 2024-06-07 10:45:00 | 4346.31 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-10 09:30:00 | 4442.40 | 2024-06-10 12:55:00 | 4424.33 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-06-12 09:50:00 | 4381.70 | 2024-06-12 09:55:00 | 4396.58 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-02 09:30:00 | 5527.60 | 2024-07-02 09:35:00 | 5545.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-04 10:30:00 | 5650.00 | 2024-07-04 10:35:00 | 5669.84 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-10 10:10:00 | 5308.40 | 2024-07-10 10:20:00 | 5276.74 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-07-10 10:10:00 | 5308.40 | 2024-07-10 11:05:00 | 5308.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 11:00:00 | 5301.00 | 2024-07-12 11:40:00 | 5318.87 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-18 09:30:00 | 5281.00 | 2024-07-18 09:45:00 | 5305.99 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-07-24 10:15:00 | 5301.15 | 2024-07-24 10:25:00 | 5274.98 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-07-30 10:25:00 | 5405.00 | 2024-07-30 11:40:00 | 5427.73 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-30 10:25:00 | 5405.00 | 2024-07-30 13:15:00 | 5405.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:00:00 | 5272.30 | 2024-08-08 10:05:00 | 5255.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-09 10:05:00 | 5353.90 | 2024-08-09 11:45:00 | 5331.91 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-08-22 09:55:00 | 5579.10 | 2024-08-22 10:00:00 | 5613.68 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-08-22 09:55:00 | 5579.10 | 2024-08-22 10:05:00 | 5579.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-23 09:40:00 | 5665.05 | 2024-08-23 09:45:00 | 5707.68 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-08-23 09:40:00 | 5665.05 | 2024-08-23 10:30:00 | 5698.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-02 10:50:00 | 5959.35 | 2024-09-02 11:00:00 | 5924.88 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-02 10:50:00 | 5959.35 | 2024-09-02 15:20:00 | 5861.55 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2024-09-10 10:50:00 | 6035.00 | 2024-09-10 11:40:00 | 5992.44 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-09-10 10:50:00 | 6035.00 | 2024-09-10 15:20:00 | 5934.15 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2024-09-12 11:15:00 | 6065.10 | 2024-09-12 11:25:00 | 6044.59 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-26 10:50:00 | 6780.35 | 2024-09-26 11:30:00 | 6733.93 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-10-01 09:50:00 | 6551.50 | 2024-10-01 09:55:00 | 6516.43 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-10-03 10:25:00 | 6600.15 | 2024-10-03 10:30:00 | 6569.12 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-10-10 10:35:00 | 6180.00 | 2024-10-10 10:45:00 | 6203.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-11 11:00:00 | 6190.00 | 2024-10-11 11:05:00 | 6202.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-15 11:00:00 | 6204.70 | 2024-10-15 13:20:00 | 6175.80 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-15 11:00:00 | 6204.70 | 2024-10-15 13:30:00 | 6204.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:55:00 | 6055.05 | 2024-10-17 13:05:00 | 6027.66 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-17 10:55:00 | 6055.05 | 2024-10-17 13:40:00 | 6055.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 09:35:00 | 4867.80 | 2024-10-29 09:40:00 | 4891.98 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-31 09:50:00 | 5091.25 | 2024-10-31 10:20:00 | 5065.74 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-11-05 09:45:00 | 4971.30 | 2024-11-05 09:55:00 | 5000.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-11-13 09:30:00 | 4885.50 | 2024-11-13 09:40:00 | 4857.39 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-11-13 09:30:00 | 4885.50 | 2024-11-13 10:30:00 | 4819.95 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2024-11-26 11:05:00 | 5102.55 | 2024-11-26 15:20:00 | 5098.70 | TARGET_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2024-11-27 09:30:00 | 5033.75 | 2024-11-27 10:15:00 | 5054.08 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-03 10:30:00 | 4925.05 | 2024-12-03 11:05:00 | 4953.17 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-12-09 10:55:00 | 4915.55 | 2024-12-09 11:10:00 | 4899.69 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-09 10:55:00 | 4915.55 | 2024-12-09 11:15:00 | 4915.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:35:00 | 5006.00 | 2024-12-10 09:45:00 | 5036.92 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-12-10 09:35:00 | 5006.00 | 2024-12-10 09:50:00 | 5006.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 10:05:00 | 5077.20 | 2024-12-17 10:20:00 | 5060.01 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-26 09:50:00 | 5205.45 | 2024-12-26 10:15:00 | 5221.72 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-27 11:05:00 | 5435.45 | 2024-12-27 11:30:00 | 5456.34 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-27 11:05:00 | 5435.45 | 2024-12-27 11:35:00 | 5435.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 09:50:00 | 5311.45 | 2025-01-02 10:10:00 | 5329.56 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-06 09:30:00 | 5333.35 | 2025-01-06 09:35:00 | 5357.31 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-06 09:30:00 | 5333.35 | 2025-01-06 10:10:00 | 5355.30 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2025-01-21 10:35:00 | 4668.90 | 2025-01-21 13:10:00 | 4637.13 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-01-21 10:35:00 | 4668.90 | 2025-01-21 15:10:00 | 4668.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 10:50:00 | 4580.85 | 2025-01-24 10:55:00 | 4596.77 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-21 09:45:00 | 4085.75 | 2025-02-21 10:00:00 | 4069.45 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-24 11:00:00 | 3999.95 | 2025-02-24 11:40:00 | 4025.20 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-02-24 11:00:00 | 3999.95 | 2025-02-24 14:20:00 | 4032.55 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-03-17 09:40:00 | 4673.85 | 2025-03-17 10:05:00 | 4709.62 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest1 | 2025-03-21 10:45:00 | 4772.85 | 2025-03-21 11:00:00 | 4795.88 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-21 10:45:00 | 4772.85 | 2025-03-21 11:30:00 | 4772.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-24 10:40:00 | 4802.75 | 2025-03-24 10:45:00 | 4778.49 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-03-24 10:40:00 | 4802.75 | 2025-03-24 10:50:00 | 4802.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-27 09:30:00 | 4770.50 | 2025-03-27 10:30:00 | 4796.94 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-04-03 11:15:00 | 4712.95 | 2025-04-03 12:40:00 | 4693.30 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-04-03 11:15:00 | 4712.95 | 2025-04-03 14:50:00 | 4712.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-29 09:30:00 | 4880.00 | 2025-04-29 09:40:00 | 4861.53 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-05 10:45:00 | 4709.90 | 2025-05-05 12:25:00 | 4691.53 | STOP_HIT | 1.00 | -0.39% |

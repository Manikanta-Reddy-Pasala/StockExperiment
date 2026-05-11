# Pfizer Ltd. (PFIZER)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 4793.00
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 18 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 59
- **Target hits / Stop hits / Partials:** 18 / 59 / 37
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 24.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 17 | 40.5% | 5 | 25 | 12 | 0.17% | 7.0% |
| BUY @ 2nd Alert (retest1) | 42 | 17 | 40.5% | 5 | 25 | 12 | 0.17% | 7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 72 | 38 | 52.8% | 13 | 34 | 25 | 0.24% | 17.1% |
| SELL @ 2nd Alert (retest1) | 72 | 38 | 52.8% | 13 | 34 | 25 | 0.24% | 17.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 55 | 48.2% | 18 | 59 | 37 | 0.21% | 24.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:40:00 | 4309.80 | 4319.14 | 0.00 | ORB-short ORB[4327.00,4347.40] vol=3.3x ATR=15.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 12:35:00 | 4286.57 | 4310.36 | 0.00 | T1 1.5R @ 4286.57 |
| Target hit | 2024-05-15 13:10:00 | 4304.00 | 4296.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 4326.00 | 4319.92 | 0.00 | ORB-long ORB[4288.00,4325.90] vol=2.0x ATR=19.19 |
| Target hit | 2024-05-16 15:20:00 | 4332.00 | 4325.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:50:00 | 4384.00 | 4360.58 | 0.00 | ORB-long ORB[4338.05,4366.25] vol=2.8x ATR=10.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:10:00 | 4399.83 | 4372.87 | 0.00 | T1 1.5R @ 4399.83 |
| Stop hit — per-position SL triggered | 2024-05-17 11:05:00 | 4384.00 | 4383.23 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 4525.00 | 4504.18 | 0.00 | ORB-long ORB[4480.00,4509.55] vol=2.1x ATR=15.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:55:00 | 4548.33 | 4530.89 | 0.00 | T1 1.5R @ 4548.33 |
| Target hit | 2024-05-23 12:10:00 | 4589.00 | 4591.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2024-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 11:05:00 | 4545.00 | 4550.29 | 0.00 | ORB-short ORB[4550.00,4587.45] vol=1.5x ATR=12.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 12:00:00 | 4525.97 | 4546.84 | 0.00 | T1 1.5R @ 4525.97 |
| Target hit | 2024-05-24 13:35:00 | 4530.00 | 4528.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:15:00 | 4480.00 | 4490.69 | 0.00 | ORB-short ORB[4500.05,4532.35] vol=2.9x ATR=14.61 |
| Stop hit — per-position SL triggered | 2024-05-27 10:20:00 | 4494.61 | 4492.05 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:45:00 | 4959.00 | 4893.95 | 0.00 | ORB-long ORB[4864.10,4923.30] vol=2.2x ATR=19.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:50:00 | 4988.84 | 4921.09 | 0.00 | T1 1.5R @ 4988.84 |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 4959.00 | 4937.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:55:00 | 4918.00 | 4951.17 | 0.00 | ORB-short ORB[4926.75,4984.95] vol=2.0x ATR=17.77 |
| Stop hit — per-position SL triggered | 2024-06-10 10:00:00 | 4935.77 | 4948.69 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:35:00 | 4944.10 | 4930.49 | 0.00 | ORB-long ORB[4906.15,4936.95] vol=2.1x ATR=14.28 |
| Stop hit — per-position SL triggered | 2024-06-12 09:45:00 | 4929.82 | 4931.97 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:35:00 | 4882.00 | 4893.17 | 0.00 | ORB-short ORB[4902.75,4928.65] vol=1.5x ATR=11.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:20:00 | 4865.30 | 4887.82 | 0.00 | T1 1.5R @ 4865.30 |
| Stop hit — per-position SL triggered | 2024-06-13 11:50:00 | 4882.00 | 4885.52 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 10:55:00 | 4859.25 | 4864.95 | 0.00 | ORB-short ORB[4862.70,4899.95] vol=1.7x ATR=8.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:40:00 | 4846.30 | 4862.37 | 0.00 | T1 1.5R @ 4846.30 |
| Target hit | 2024-06-14 15:20:00 | 4829.25 | 4842.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:35:00 | 4765.35 | 4788.07 | 0.00 | ORB-short ORB[4775.00,4829.10] vol=2.2x ATR=14.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 13:30:00 | 4744.15 | 4766.90 | 0.00 | T1 1.5R @ 4744.15 |
| Stop hit — per-position SL triggered | 2024-06-18 14:15:00 | 4765.35 | 4764.95 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 4770.00 | 4777.17 | 0.00 | ORB-short ORB[4772.50,4798.00] vol=1.5x ATR=14.33 |
| Stop hit — per-position SL triggered | 2024-06-19 09:45:00 | 4784.33 | 4776.33 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 4649.75 | 4638.76 | 0.00 | ORB-long ORB[4620.00,4648.00] vol=3.5x ATR=11.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:40:00 | 4667.59 | 4642.31 | 0.00 | T1 1.5R @ 4667.59 |
| Stop hit — per-position SL triggered | 2024-06-25 10:00:00 | 4649.75 | 4651.79 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 4660.15 | 4638.04 | 0.00 | ORB-long ORB[4605.40,4649.00] vol=2.2x ATR=13.57 |
| Stop hit — per-position SL triggered | 2024-06-26 09:45:00 | 4646.58 | 4639.56 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-06-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 11:05:00 | 4541.30 | 4581.92 | 0.00 | ORB-short ORB[4562.20,4614.40] vol=7.9x ATR=16.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:30:00 | 4515.86 | 4565.38 | 0.00 | T1 1.5R @ 4515.86 |
| Stop hit — per-position SL triggered | 2024-06-28 14:30:00 | 4541.30 | 4529.86 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 4618.00 | 4626.87 | 0.00 | ORB-short ORB[4620.70,4646.00] vol=1.9x ATR=18.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 13:00:00 | 4590.04 | 4619.19 | 0.00 | T1 1.5R @ 4590.04 |
| Target hit | 2024-07-02 15:20:00 | 4579.60 | 4610.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:50:00 | 4605.00 | 4620.83 | 0.00 | ORB-short ORB[4611.00,4638.00] vol=1.6x ATR=13.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:00:00 | 4585.28 | 4601.61 | 0.00 | T1 1.5R @ 4585.28 |
| Stop hit — per-position SL triggered | 2024-07-03 11:05:00 | 4605.00 | 4600.63 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:40:00 | 4624.40 | 4599.51 | 0.00 | ORB-long ORB[4580.00,4619.00] vol=1.9x ATR=9.94 |
| Stop hit — per-position SL triggered | 2024-07-04 10:45:00 | 4614.46 | 4600.03 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 4588.05 | 4607.53 | 0.00 | ORB-short ORB[4606.40,4635.75] vol=1.8x ATR=8.37 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 4596.42 | 4607.09 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:40:00 | 5011.90 | 4996.45 | 0.00 | ORB-long ORB[4942.60,4998.00] vol=4.6x ATR=22.79 |
| Stop hit — per-position SL triggered | 2024-07-16 15:20:00 | 5005.10 | 5008.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-07-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:25:00 | 5078.20 | 5022.48 | 0.00 | ORB-long ORB[4970.05,5026.90] vol=4.9x ATR=16.52 |
| Stop hit — per-position SL triggered | 2024-07-22 14:10:00 | 5061.68 | 5051.65 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:40:00 | 5052.80 | 5036.10 | 0.00 | ORB-long ORB[4980.00,5050.00] vol=2.0x ATR=14.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:50:00 | 5074.81 | 5046.27 | 0.00 | T1 1.5R @ 5074.81 |
| Stop hit — per-position SL triggered | 2024-07-23 09:55:00 | 5052.80 | 5046.50 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:35:00 | 5301.40 | 5270.58 | 0.00 | ORB-long ORB[5250.00,5301.15] vol=7.2x ATR=22.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:35:00 | 5334.85 | 5291.47 | 0.00 | T1 1.5R @ 5334.85 |
| Target hit | 2024-07-25 14:15:00 | 5319.85 | 5323.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:15:00 | 5615.25 | 5593.04 | 0.00 | ORB-long ORB[5522.55,5581.00] vol=1.7x ATR=19.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:40:00 | 5644.69 | 5604.49 | 0.00 | T1 1.5R @ 5644.69 |
| Stop hit — per-position SL triggered | 2024-07-31 11:35:00 | 5615.25 | 5609.08 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 10:30:00 | 5638.15 | 5603.87 | 0.00 | ORB-long ORB[5551.85,5610.00] vol=2.5x ATR=15.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 10:50:00 | 5661.87 | 5610.94 | 0.00 | T1 1.5R @ 5661.87 |
| Target hit | 2024-08-02 15:20:00 | 5660.05 | 5631.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 09:30:00 | 5612.15 | 5596.80 | 0.00 | ORB-long ORB[5547.00,5595.50] vol=7.7x ATR=22.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 10:00:00 | 5646.32 | 5605.86 | 0.00 | T1 1.5R @ 5646.32 |
| Target hit | 2024-08-06 11:30:00 | 5840.00 | 5892.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2024-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 11:00:00 | 5936.80 | 5964.78 | 0.00 | ORB-short ORB[5971.10,6030.00] vol=2.3x ATR=18.16 |
| Stop hit — per-position SL triggered | 2024-08-22 11:10:00 | 5954.96 | 5965.15 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:50:00 | 5824.50 | 5859.32 | 0.00 | ORB-short ORB[5855.30,5899.95] vol=3.5x ATR=13.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 12:05:00 | 5803.74 | 5841.30 | 0.00 | T1 1.5R @ 5803.74 |
| Target hit | 2024-08-26 14:55:00 | 5801.40 | 5795.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 5855.85 | 5877.85 | 0.00 | ORB-short ORB[5871.50,5919.95] vol=1.6x ATR=16.53 |
| Stop hit — per-position SL triggered | 2024-08-28 10:05:00 | 5872.38 | 5867.54 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:50:00 | 5950.05 | 5965.54 | 0.00 | ORB-short ORB[5958.40,6004.95] vol=2.5x ATR=16.04 |
| Stop hit — per-position SL triggered | 2024-08-29 11:40:00 | 5966.09 | 5959.85 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:50:00 | 6240.40 | 6218.95 | 0.00 | ORB-long ORB[6180.00,6239.00] vol=1.8x ATR=28.36 |
| Stop hit — per-position SL triggered | 2024-09-05 10:55:00 | 6212.04 | 6226.32 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:50:00 | 6196.70 | 6170.51 | 0.00 | ORB-long ORB[6124.90,6184.05] vol=4.3x ATR=19.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:05:00 | 6226.03 | 6192.22 | 0.00 | T1 1.5R @ 6226.03 |
| Stop hit — per-position SL triggered | 2024-09-12 11:50:00 | 6196.70 | 6201.41 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:35:00 | 6179.90 | 6125.67 | 0.00 | ORB-long ORB[6101.05,6155.15] vol=2.3x ATR=17.03 |
| Stop hit — per-position SL triggered | 2024-09-13 10:45:00 | 6162.87 | 6127.09 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:50:00 | 5879.25 | 5916.35 | 0.00 | ORB-short ORB[5913.05,5983.70] vol=1.6x ATR=17.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:20:00 | 5852.45 | 5888.44 | 0.00 | T1 1.5R @ 5852.45 |
| Stop hit — per-position SL triggered | 2024-09-18 11:25:00 | 5879.25 | 5886.80 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:10:00 | 5443.15 | 5464.37 | 0.00 | ORB-short ORB[5446.00,5518.90] vol=2.6x ATR=15.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 12:00:00 | 5420.50 | 5451.95 | 0.00 | T1 1.5R @ 5420.50 |
| Stop hit — per-position SL triggered | 2024-09-25 14:45:00 | 5443.15 | 5445.36 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:05:00 | 5272.30 | 5291.68 | 0.00 | ORB-short ORB[5325.10,5399.00] vol=9.6x ATR=25.52 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 5297.82 | 5290.95 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:55:00 | 5305.05 | 5343.77 | 0.00 | ORB-short ORB[5319.25,5387.85] vol=2.2x ATR=16.90 |
| Stop hit — per-position SL triggered | 2024-11-07 10:20:00 | 5321.95 | 5335.21 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 10:15:00 | 5304.00 | 5317.68 | 0.00 | ORB-short ORB[5305.50,5385.00] vol=1.7x ATR=13.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 12:00:00 | 5283.14 | 5310.13 | 0.00 | T1 1.5R @ 5283.14 |
| Stop hit — per-position SL triggered | 2024-11-08 13:50:00 | 5304.00 | 5302.76 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 10:55:00 | 5263.95 | 5284.83 | 0.00 | ORB-short ORB[5280.05,5315.00] vol=1.7x ATR=10.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:15:00 | 5248.37 | 5279.30 | 0.00 | T1 1.5R @ 5248.37 |
| Target hit | 2024-11-11 15:20:00 | 5204.00 | 5226.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:40:00 | 5174.50 | 5224.53 | 0.00 | ORB-short ORB[5210.00,5285.30] vol=2.7x ATR=21.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:45:00 | 5142.63 | 5204.33 | 0.00 | T1 1.5R @ 5142.63 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 5174.50 | 5201.42 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 09:30:00 | 5269.55 | 5248.52 | 0.00 | ORB-long ORB[5205.00,5257.85] vol=1.8x ATR=15.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:35:00 | 5293.05 | 5255.30 | 0.00 | T1 1.5R @ 5293.05 |
| Stop hit — per-position SL triggered | 2024-11-14 09:40:00 | 5269.55 | 5256.09 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:40:00 | 5129.50 | 5146.33 | 0.00 | ORB-short ORB[5150.50,5210.65] vol=1.6x ATR=22.09 |
| Stop hit — per-position SL triggered | 2024-11-21 11:20:00 | 5151.59 | 5135.74 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 11:10:00 | 5289.20 | 5247.95 | 0.00 | ORB-long ORB[5230.60,5275.45] vol=4.1x ATR=16.25 |
| Stop hit — per-position SL triggered | 2024-11-26 11:15:00 | 5272.95 | 5248.61 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:45:00 | 5398.00 | 5383.26 | 0.00 | ORB-long ORB[5340.00,5390.00] vol=4.1x ATR=15.13 |
| Stop hit — per-position SL triggered | 2024-12-03 10:20:00 | 5382.87 | 5390.73 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 5337.00 | 5354.95 | 0.00 | ORB-short ORB[5350.00,5408.00] vol=3.6x ATR=10.61 |
| Stop hit — per-position SL triggered | 2024-12-04 09:35:00 | 5347.61 | 5353.28 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 5334.95 | 5346.20 | 0.00 | ORB-short ORB[5335.10,5389.90] vol=2.8x ATR=14.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:50:00 | 5313.90 | 5329.98 | 0.00 | T1 1.5R @ 5313.90 |
| Target hit | 2024-12-05 15:20:00 | 5289.15 | 5297.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-12-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:45:00 | 5230.50 | 5238.35 | 0.00 | ORB-short ORB[5233.95,5277.00] vol=1.7x ATR=10.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:55:00 | 5214.71 | 5236.56 | 0.00 | T1 1.5R @ 5214.71 |
| Target hit | 2024-12-06 10:55:00 | 5214.65 | 5214.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — SELL (started 2024-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:00:00 | 5091.40 | 5113.54 | 0.00 | ORB-short ORB[5095.95,5158.00] vol=1.9x ATR=14.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:05:00 | 5070.07 | 5108.80 | 0.00 | T1 1.5R @ 5070.07 |
| Stop hit — per-position SL triggered | 2024-12-12 14:20:00 | 5091.40 | 5092.92 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:40:00 | 5031.60 | 5053.23 | 0.00 | ORB-short ORB[5039.20,5098.00] vol=2.8x ATR=10.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:20:00 | 5016.46 | 5033.63 | 0.00 | T1 1.5R @ 5016.46 |
| Target hit | 2024-12-13 11:00:00 | 5028.00 | 5025.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 5006.55 | 5018.97 | 0.00 | ORB-short ORB[5016.10,5046.70] vol=6.2x ATR=14.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:25:00 | 4984.69 | 5012.30 | 0.00 | T1 1.5R @ 4984.69 |
| Stop hit — per-position SL triggered | 2024-12-17 12:05:00 | 5006.55 | 5003.31 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 4987.65 | 5015.57 | 0.00 | ORB-short ORB[4995.00,5055.50] vol=2.5x ATR=13.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:00:00 | 4967.85 | 4997.27 | 0.00 | T1 1.5R @ 4967.85 |
| Target hit | 2024-12-20 15:20:00 | 4633.00 | 4688.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2024-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:50:00 | 4914.70 | 4887.07 | 0.00 | ORB-long ORB[4846.60,4889.30] vol=2.1x ATR=16.59 |
| Stop hit — per-position SL triggered | 2024-12-27 11:15:00 | 4898.11 | 4898.99 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:00:00 | 5257.90 | 5296.30 | 0.00 | ORB-short ORB[5298.10,5371.90] vol=2.1x ATR=22.23 |
| Stop hit — per-position SL triggered | 2025-01-10 10:10:00 | 5280.13 | 5293.35 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 10:30:00 | 5022.00 | 5043.38 | 0.00 | ORB-short ORB[5060.00,5107.15] vol=1.8x ATR=16.47 |
| Stop hit — per-position SL triggered | 2025-01-15 10:45:00 | 5038.47 | 5042.62 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:45:00 | 4956.55 | 4964.19 | 0.00 | ORB-short ORB[4962.00,5029.45] vol=2.1x ATR=14.75 |
| Stop hit — per-position SL triggered | 2025-01-16 11:00:00 | 4971.30 | 4963.94 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:50:00 | 4931.40 | 4949.62 | 0.00 | ORB-short ORB[4939.35,5000.95] vol=2.4x ATR=10.53 |
| Stop hit — per-position SL triggered | 2025-01-17 11:10:00 | 4941.93 | 4946.23 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 4924.75 | 4953.06 | 0.00 | ORB-short ORB[4936.00,4999.45] vol=2.1x ATR=9.64 |
| Stop hit — per-position SL triggered | 2025-01-20 09:40:00 | 4934.39 | 4941.52 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:00:00 | 4900.05 | 4953.95 | 0.00 | ORB-short ORB[4921.00,4984.05] vol=1.9x ATR=15.65 |
| Stop hit — per-position SL triggered | 2025-01-21 11:10:00 | 4915.70 | 4953.08 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:30:00 | 4720.10 | 4748.19 | 0.00 | ORB-short ORB[4740.15,4811.10] vol=2.0x ATR=13.56 |
| Stop hit — per-position SL triggered | 2025-01-24 09:35:00 | 4733.66 | 4746.56 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:55:00 | 4557.50 | 4538.31 | 0.00 | ORB-long ORB[4494.60,4550.00] vol=1.6x ATR=16.62 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 4540.88 | 4540.34 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 4275.05 | 4307.96 | 0.00 | ORB-short ORB[4301.00,4347.00] vol=1.6x ATR=15.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 12:00:00 | 4252.15 | 4282.34 | 0.00 | T1 1.5R @ 4252.15 |
| Target hit | 2025-02-10 15:20:00 | 4238.05 | 4265.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-03-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:10:00 | 4148.15 | 4129.53 | 0.00 | ORB-long ORB[4097.70,4148.00] vol=2.6x ATR=16.69 |
| Stop hit — per-position SL triggered | 2025-03-07 12:00:00 | 4131.46 | 4135.74 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:25:00 | 4170.00 | 4121.87 | 0.00 | ORB-long ORB[4091.95,4150.00] vol=1.6x ATR=16.15 |
| Stop hit — per-position SL triggered | 2025-03-10 10:55:00 | 4153.85 | 4137.05 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:55:00 | 4119.50 | 4083.10 | 0.00 | ORB-long ORB[4050.95,4100.10] vol=2.3x ATR=10.86 |
| Stop hit — per-position SL triggered | 2025-03-17 11:00:00 | 4108.64 | 4084.49 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 11:05:00 | 4081.00 | 4101.40 | 0.00 | ORB-short ORB[4087.10,4121.00] vol=4.0x ATR=10.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 12:50:00 | 4065.88 | 4089.02 | 0.00 | T1 1.5R @ 4065.88 |
| Stop hit — per-position SL triggered | 2025-03-18 14:05:00 | 4081.00 | 4085.29 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 4125.60 | 4107.39 | 0.00 | ORB-long ORB[4065.05,4121.00] vol=2.2x ATR=9.06 |
| Stop hit — per-position SL triggered | 2025-03-20 09:35:00 | 4116.54 | 4108.23 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-03-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-21 10:45:00 | 4097.50 | 4123.19 | 0.00 | ORB-short ORB[4107.30,4159.60] vol=2.9x ATR=9.87 |
| Stop hit — per-position SL triggered | 2025-03-21 10:50:00 | 4107.37 | 4122.69 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 4066.30 | 4091.07 | 0.00 | ORB-short ORB[4077.55,4120.00] vol=1.9x ATR=10.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:40:00 | 4051.05 | 4080.96 | 0.00 | T1 1.5R @ 4051.05 |
| Stop hit — per-position SL triggered | 2025-03-26 10:10:00 | 4066.30 | 4073.49 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-03-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 09:35:00 | 3956.80 | 3978.39 | 0.00 | ORB-short ORB[3970.00,4011.85] vol=1.6x ATR=10.44 |
| Stop hit — per-position SL triggered | 2025-03-28 09:40:00 | 3967.24 | 3978.32 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 09:45:00 | 4001.40 | 4026.03 | 0.00 | ORB-short ORB[4014.45,4069.85] vol=2.4x ATR=17.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:15:00 | 3974.48 | 4005.11 | 0.00 | T1 1.5R @ 3974.48 |
| Target hit | 2025-04-04 10:50:00 | 3998.00 | 3994.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — SELL (started 2025-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:55:00 | 4189.80 | 4216.03 | 0.00 | ORB-short ORB[4220.20,4268.00] vol=2.4x ATR=11.52 |
| Stop hit — per-position SL triggered | 2025-04-23 11:00:00 | 4201.32 | 4215.81 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:00:00 | 4290.20 | 4279.25 | 0.00 | ORB-long ORB[4233.70,4289.50] vol=2.3x ATR=14.05 |
| Stop hit — per-position SL triggered | 2025-04-24 10:05:00 | 4276.15 | 4279.33 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:10:00 | 4232.90 | 4244.38 | 0.00 | ORB-short ORB[4255.30,4297.80] vol=1.6x ATR=16.11 |
| Stop hit — per-position SL triggered | 2025-04-29 12:00:00 | 4249.01 | 4239.17 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 09:40:00 | 4289.80 | 4297.48 | 0.00 | ORB-short ORB[4295.30,4325.90] vol=4.7x ATR=13.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 10:05:00 | 4269.33 | 4291.56 | 0.00 | T1 1.5R @ 4269.33 |
| Target hit | 2025-05-05 12:05:00 | 4281.00 | 4275.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — BUY (started 2025-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:45:00 | 4278.60 | 4256.03 | 0.00 | ORB-long ORB[4222.60,4272.20] vol=1.6x ATR=11.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:50:00 | 4296.55 | 4271.14 | 0.00 | T1 1.5R @ 4296.55 |
| Stop hit — per-position SL triggered | 2025-05-06 11:25:00 | 4278.60 | 4276.27 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 10:50:00 | 4230.00 | 4199.90 | 0.00 | ORB-long ORB[4190.00,4225.20] vol=1.8x ATR=11.06 |
| Stop hit — per-position SL triggered | 2025-05-07 10:55:00 | 4218.94 | 4201.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 09:40:00 | 4309.80 | 2024-05-15 12:35:00 | 4286.57 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-05-15 09:40:00 | 4309.80 | 2024-05-15 13:10:00 | 4304.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-05-16 09:35:00 | 4326.00 | 2024-05-16 15:20:00 | 4332.00 | TARGET_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2024-05-17 09:50:00 | 4384.00 | 2024-05-17 10:10:00 | 4399.83 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-05-17 09:50:00 | 4384.00 | 2024-05-17 11:05:00 | 4384.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-23 09:30:00 | 4525.00 | 2024-05-23 10:55:00 | 4548.33 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-23 09:30:00 | 4525.00 | 2024-05-23 12:10:00 | 4589.00 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2024-05-24 11:05:00 | 4545.00 | 2024-05-24 12:00:00 | 4525.97 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-24 11:05:00 | 4545.00 | 2024-05-24 13:35:00 | 4530.00 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-05-27 10:15:00 | 4480.00 | 2024-05-27 10:20:00 | 4494.61 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-07 09:45:00 | 4959.00 | 2024-06-07 10:50:00 | 4988.84 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-06-07 09:45:00 | 4959.00 | 2024-06-07 11:15:00 | 4959.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 09:55:00 | 4918.00 | 2024-06-10 10:00:00 | 4935.77 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-12 09:35:00 | 4944.10 | 2024-06-12 09:45:00 | 4929.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-13 10:35:00 | 4882.00 | 2024-06-13 11:20:00 | 4865.30 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-06-13 10:35:00 | 4882.00 | 2024-06-13 11:50:00 | 4882.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 10:55:00 | 4859.25 | 2024-06-14 11:40:00 | 4846.30 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-06-14 10:55:00 | 4859.25 | 2024-06-14 15:20:00 | 4829.25 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-06-18 09:35:00 | 4765.35 | 2024-06-18 13:30:00 | 4744.15 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-06-18 09:35:00 | 4765.35 | 2024-06-18 14:15:00 | 4765.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 09:30:00 | 4770.00 | 2024-06-19 09:45:00 | 4784.33 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-25 09:35:00 | 4649.75 | 2024-06-25 09:40:00 | 4667.59 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-06-25 09:35:00 | 4649.75 | 2024-06-25 10:00:00 | 4649.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 09:40:00 | 4660.15 | 2024-06-26 09:45:00 | 4646.58 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-28 11:05:00 | 4541.30 | 2024-06-28 11:30:00 | 4515.86 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-06-28 11:05:00 | 4541.30 | 2024-06-28 14:30:00 | 4541.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:40:00 | 4618.00 | 2024-07-02 13:00:00 | 4590.04 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-07-02 09:40:00 | 4618.00 | 2024-07-02 15:20:00 | 4579.60 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-07-03 09:50:00 | 4605.00 | 2024-07-03 11:00:00 | 4585.28 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-03 09:50:00 | 4605.00 | 2024-07-03 11:05:00 | 4605.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:40:00 | 4624.40 | 2024-07-04 10:45:00 | 4614.46 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-08 11:10:00 | 4588.05 | 2024-07-08 11:15:00 | 4596.42 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-07-16 09:40:00 | 5011.90 | 2024-07-16 15:20:00 | 5005.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-07-22 10:25:00 | 5078.20 | 2024-07-22 14:10:00 | 5061.68 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-23 09:40:00 | 5052.80 | 2024-07-23 09:50:00 | 5074.81 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-23 09:40:00 | 5052.80 | 2024-07-23 09:55:00 | 5052.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 09:35:00 | 5301.40 | 2024-07-25 10:35:00 | 5334.85 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-07-25 09:35:00 | 5301.40 | 2024-07-25 14:15:00 | 5319.85 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-31 10:15:00 | 5615.25 | 2024-07-31 10:40:00 | 5644.69 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-31 10:15:00 | 5615.25 | 2024-07-31 11:35:00 | 5615.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-02 10:30:00 | 5638.15 | 2024-08-02 10:50:00 | 5661.87 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-02 10:30:00 | 5638.15 | 2024-08-02 15:20:00 | 5660.05 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-06 09:30:00 | 5612.15 | 2024-08-06 10:00:00 | 5646.32 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-06 09:30:00 | 5612.15 | 2024-08-06 11:30:00 | 5840.00 | TARGET_HIT | 0.50 | 4.06% |
| SELL | retest1 | 2024-08-22 11:00:00 | 5936.80 | 2024-08-22 11:10:00 | 5954.96 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-26 10:50:00 | 5824.50 | 2024-08-26 12:05:00 | 5803.74 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-26 10:50:00 | 5824.50 | 2024-08-26 14:55:00 | 5801.40 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-28 09:30:00 | 5855.85 | 2024-08-28 10:05:00 | 5872.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-29 10:50:00 | 5950.05 | 2024-08-29 11:40:00 | 5966.09 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-05 09:50:00 | 6240.40 | 2024-09-05 10:55:00 | 6212.04 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-12 10:50:00 | 6196.70 | 2024-09-12 11:05:00 | 6226.03 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-12 10:50:00 | 6196.70 | 2024-09-12 11:50:00 | 6196.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:35:00 | 6179.90 | 2024-09-13 10:45:00 | 6162.87 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-18 09:50:00 | 5879.25 | 2024-09-18 11:20:00 | 5852.45 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-18 09:50:00 | 5879.25 | 2024-09-18 11:25:00 | 5879.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 10:10:00 | 5443.15 | 2024-09-25 12:00:00 | 5420.50 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-25 10:10:00 | 5443.15 | 2024-09-25 14:45:00 | 5443.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:05:00 | 5272.30 | 2024-10-25 10:20:00 | 5297.82 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-11-07 09:55:00 | 5305.05 | 2024-11-07 10:20:00 | 5321.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-08 10:15:00 | 5304.00 | 2024-11-08 12:00:00 | 5283.14 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-11-08 10:15:00 | 5304.00 | 2024-11-08 13:50:00 | 5304.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-11 10:55:00 | 5263.95 | 2024-11-11 11:15:00 | 5248.37 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-11-11 10:55:00 | 5263.95 | 2024-11-11 15:20:00 | 5204.00 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2024-11-13 09:40:00 | 5174.50 | 2024-11-13 09:45:00 | 5142.63 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-11-13 09:40:00 | 5174.50 | 2024-11-13 09:50:00 | 5174.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-14 09:30:00 | 5269.55 | 2024-11-14 09:35:00 | 5293.05 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-11-14 09:30:00 | 5269.55 | 2024-11-14 09:40:00 | 5269.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-21 09:40:00 | 5129.50 | 2024-11-21 11:20:00 | 5151.59 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-11-26 11:10:00 | 5289.20 | 2024-11-26 11:15:00 | 5272.95 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-03 09:45:00 | 5398.00 | 2024-12-03 10:20:00 | 5382.87 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-04 09:30:00 | 5337.00 | 2024-12-04 09:35:00 | 5347.61 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-05 09:30:00 | 5334.95 | 2024-12-05 09:50:00 | 5313.90 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-05 09:30:00 | 5334.95 | 2024-12-05 15:20:00 | 5289.15 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2024-12-06 09:45:00 | 5230.50 | 2024-12-06 09:55:00 | 5214.71 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-06 09:45:00 | 5230.50 | 2024-12-06 10:55:00 | 5214.65 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-12 10:00:00 | 5091.40 | 2024-12-12 10:05:00 | 5070.07 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-12 10:00:00 | 5091.40 | 2024-12-12 14:20:00 | 5091.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 09:40:00 | 5031.60 | 2024-12-13 10:20:00 | 5016.46 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-13 09:40:00 | 5031.60 | 2024-12-13 11:00:00 | 5028.00 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2024-12-17 09:35:00 | 5006.55 | 2024-12-17 10:25:00 | 4984.69 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-17 09:35:00 | 5006.55 | 2024-12-17 12:05:00 | 5006.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:30:00 | 4987.65 | 2024-12-20 10:00:00 | 4967.85 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-20 09:30:00 | 4987.65 | 2024-12-20 15:20:00 | 4633.00 | TARGET_HIT | 0.50 | 7.11% |
| BUY | retest1 | 2024-12-27 09:50:00 | 4914.70 | 2024-12-27 11:15:00 | 4898.11 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-10 10:00:00 | 5257.90 | 2025-01-10 10:10:00 | 5280.13 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-01-15 10:30:00 | 5022.00 | 2025-01-15 10:45:00 | 5038.47 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-16 10:45:00 | 4956.55 | 2025-01-16 11:00:00 | 4971.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-17 10:50:00 | 4931.40 | 2025-01-17 11:10:00 | 4941.93 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-20 09:30:00 | 4924.75 | 2025-01-20 09:40:00 | 4934.39 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-21 11:00:00 | 4900.05 | 2025-01-21 11:10:00 | 4915.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-24 09:30:00 | 4720.10 | 2025-01-24 09:35:00 | 4733.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-30 09:55:00 | 4557.50 | 2025-01-30 10:50:00 | 4540.88 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-10 09:30:00 | 4275.05 | 2025-02-10 12:00:00 | 4252.15 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-02-10 09:30:00 | 4275.05 | 2025-02-10 15:20:00 | 4238.05 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2025-03-07 10:10:00 | 4148.15 | 2025-03-07 12:00:00 | 4131.46 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-10 10:25:00 | 4170.00 | 2025-03-10 10:55:00 | 4153.85 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-17 10:55:00 | 4119.50 | 2025-03-17 11:00:00 | 4108.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-03-18 11:05:00 | 4081.00 | 2025-03-18 12:50:00 | 4065.88 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-03-18 11:05:00 | 4081.00 | 2025-03-18 14:05:00 | 4081.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 09:30:00 | 4125.60 | 2025-03-20 09:35:00 | 4116.54 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-03-21 10:45:00 | 4097.50 | 2025-03-21 10:50:00 | 4107.37 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-26 09:30:00 | 4066.30 | 2025-03-26 09:40:00 | 4051.05 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-03-26 09:30:00 | 4066.30 | 2025-03-26 10:10:00 | 4066.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-28 09:35:00 | 3956.80 | 2025-03-28 09:40:00 | 3967.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-04 09:45:00 | 4001.40 | 2025-04-04 10:15:00 | 3974.48 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-04-04 09:45:00 | 4001.40 | 2025-04-04 10:50:00 | 3998.00 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-04-23 10:55:00 | 4189.80 | 2025-04-23 11:00:00 | 4201.32 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-24 10:00:00 | 4290.20 | 2025-04-24 10:05:00 | 4276.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-29 10:10:00 | 4232.90 | 2025-04-29 12:00:00 | 4249.01 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-05-05 09:40:00 | 4289.80 | 2025-05-05 10:05:00 | 4269.33 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-05-05 09:40:00 | 4289.80 | 2025-05-05 12:05:00 | 4281.00 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-05-06 09:45:00 | 4278.60 | 2025-05-06 10:50:00 | 4296.55 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-05-06 09:45:00 | 4278.60 | 2025-05-06 11:25:00 | 4278.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-07 10:50:00 | 4230.00 | 2025-05-07 10:55:00 | 4218.94 | STOP_HIT | 1.00 | -0.26% |

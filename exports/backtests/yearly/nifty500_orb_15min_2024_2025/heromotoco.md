# Hero MotoCorp Ltd. (HEROMOTOCO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 5325.00
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 14 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 59
- **Target hits / Stop hits / Partials:** 14 / 59 / 25
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 16.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 27 | 42.2% | 10 | 37 | 17 | 0.19% | 12.2% |
| BUY @ 2nd Alert (retest1) | 64 | 27 | 42.2% | 10 | 37 | 17 | 0.19% | 12.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 12 | 35.3% | 4 | 22 | 8 | 0.13% | 4.5% |
| SELL @ 2nd Alert (retest1) | 34 | 12 | 35.3% | 4 | 22 | 8 | 0.13% | 4.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 39 | 39.8% | 14 | 59 | 25 | 0.17% | 16.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:50:00 | 5015.00 | 4990.77 | 0.00 | ORB-long ORB[4905.05,4980.55] vol=2.1x ATR=25.86 |
| Stop hit — per-position SL triggered | 2024-05-14 10:00:00 | 4989.14 | 4993.50 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 5045.50 | 5102.24 | 0.00 | ORB-short ORB[5064.95,5128.65] vol=2.7x ATR=16.46 |
| Stop hit — per-position SL triggered | 2024-05-16 11:30:00 | 5061.96 | 5100.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:50:00 | 5137.00 | 5087.65 | 0.00 | ORB-long ORB[5045.00,5093.75] vol=2.5x ATR=14.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:55:00 | 5158.86 | 5100.67 | 0.00 | T1 1.5R @ 5158.86 |
| Stop hit — per-position SL triggered | 2024-05-28 11:25:00 | 5137.00 | 5127.86 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 5185.90 | 5152.59 | 0.00 | ORB-long ORB[5119.05,5171.05] vol=3.2x ATR=14.04 |
| Stop hit — per-position SL triggered | 2024-05-29 10:55:00 | 5171.86 | 5157.11 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 5182.00 | 5155.16 | 0.00 | ORB-long ORB[5105.10,5180.60] vol=1.5x ATR=18.65 |
| Stop hit — per-position SL triggered | 2024-05-31 10:05:00 | 5163.35 | 5162.09 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-03 10:50:00 | 5217.25 | 5248.23 | 0.00 | ORB-short ORB[5240.00,5314.00] vol=1.6x ATR=27.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-03 12:35:00 | 5175.77 | 5242.93 | 0.00 | T1 1.5R @ 5175.77 |
| Target hit | 2024-06-03 15:20:00 | 5159.35 | 5221.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:05:00 | 5680.00 | 5626.26 | 0.00 | ORB-long ORB[5581.05,5630.00] vol=1.7x ATR=17.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:20:00 | 5706.51 | 5648.70 | 0.00 | T1 1.5R @ 5706.51 |
| Target hit | 2024-06-10 14:25:00 | 5716.70 | 5732.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:50:00 | 5803.35 | 5729.90 | 0.00 | ORB-long ORB[5706.00,5754.75] vol=1.7x ATR=16.67 |
| Stop hit — per-position SL triggered | 2024-06-11 10:55:00 | 5786.68 | 5736.29 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:40:00 | 5780.25 | 5803.74 | 0.00 | ORB-short ORB[5801.00,5840.00] vol=2.3x ATR=14.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:00:00 | 5758.83 | 5799.70 | 0.00 | T1 1.5R @ 5758.83 |
| Stop hit — per-position SL triggered | 2024-06-13 11:40:00 | 5780.25 | 5785.54 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 10:00:00 | 5786.45 | 5794.52 | 0.00 | ORB-short ORB[5792.05,5843.15] vol=1.5x ATR=16.03 |
| Stop hit — per-position SL triggered | 2024-06-14 10:25:00 | 5802.48 | 5794.17 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:35:00 | 5568.95 | 5548.76 | 0.00 | ORB-long ORB[5494.70,5566.80] vol=2.3x ATR=14.55 |
| Stop hit — per-position SL triggered | 2024-06-25 10:40:00 | 5554.40 | 5549.78 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:00:00 | 5562.05 | 5523.50 | 0.00 | ORB-long ORB[5452.10,5510.55] vol=3.0x ATR=17.36 |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 5544.69 | 5526.69 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:55:00 | 5588.45 | 5557.78 | 0.00 | ORB-long ORB[5514.75,5550.00] vol=1.9x ATR=12.55 |
| Stop hit — per-position SL triggered | 2024-07-12 10:10:00 | 5575.90 | 5567.63 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 5579.00 | 5560.54 | 0.00 | ORB-long ORB[5528.60,5568.60] vol=1.8x ATR=11.73 |
| Stop hit — per-position SL triggered | 2024-07-15 09:35:00 | 5567.27 | 5561.20 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:05:00 | 5624.75 | 5595.71 | 0.00 | ORB-long ORB[5570.05,5617.00] vol=3.0x ATR=10.70 |
| Stop hit — per-position SL triggered | 2024-07-16 11:20:00 | 5614.05 | 5600.45 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:55:00 | 5481.85 | 5455.13 | 0.00 | ORB-long ORB[5390.00,5459.60] vol=2.8x ATR=13.86 |
| Stop hit — per-position SL triggered | 2024-07-22 11:10:00 | 5467.99 | 5457.11 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:10:00 | 5509.45 | 5466.08 | 0.00 | ORB-long ORB[5438.05,5487.00] vol=4.3x ATR=16.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:20:00 | 5534.69 | 5486.59 | 0.00 | T1 1.5R @ 5534.69 |
| Target hit | 2024-07-23 12:10:00 | 5569.70 | 5576.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:15:00 | 5527.00 | 5474.61 | 0.00 | ORB-long ORB[5397.35,5444.20] vol=1.8x ATR=14.91 |
| Stop hit — per-position SL triggered | 2024-07-26 12:05:00 | 5512.09 | 5488.59 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:55:00 | 5427.40 | 5464.96 | 0.00 | ORB-short ORB[5448.60,5493.70] vol=2.3x ATR=14.91 |
| Stop hit — per-position SL triggered | 2024-07-31 11:20:00 | 5442.31 | 5454.42 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 11:15:00 | 5144.40 | 5187.45 | 0.00 | ORB-short ORB[5172.00,5247.00] vol=2.1x ATR=18.50 |
| Stop hit — per-position SL triggered | 2024-08-05 11:25:00 | 5162.90 | 5182.62 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 09:45:00 | 5200.00 | 5217.71 | 0.00 | ORB-short ORB[5200.25,5255.10] vol=5.6x ATR=18.99 |
| Stop hit — per-position SL triggered | 2024-08-06 09:50:00 | 5218.99 | 5217.33 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:45:00 | 5360.80 | 5328.67 | 0.00 | ORB-long ORB[5300.00,5350.00] vol=1.9x ATR=17.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:55:00 | 5387.37 | 5340.55 | 0.00 | T1 1.5R @ 5387.37 |
| Stop hit — per-position SL triggered | 2024-08-13 11:05:00 | 5360.80 | 5367.37 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:45:00 | 5322.00 | 5289.73 | 0.00 | ORB-long ORB[5263.45,5308.30] vol=2.3x ATR=11.78 |
| Stop hit — per-position SL triggered | 2024-08-22 10:55:00 | 5310.22 | 5293.05 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:35:00 | 5415.60 | 5375.90 | 0.00 | ORB-long ORB[5346.75,5413.00] vol=2.2x ATR=14.20 |
| Stop hit — per-position SL triggered | 2024-08-26 11:05:00 | 5401.40 | 5383.05 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:20:00 | 5387.00 | 5354.04 | 0.00 | ORB-long ORB[5335.55,5366.00] vol=2.0x ATR=12.27 |
| Stop hit — per-position SL triggered | 2024-08-27 11:30:00 | 5374.73 | 5364.48 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 5297.80 | 5325.08 | 0.00 | ORB-short ORB[5316.00,5383.20] vol=1.6x ATR=13.19 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 5310.99 | 5321.74 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:05:00 | 5626.20 | 5585.96 | 0.00 | ORB-long ORB[5549.90,5602.00] vol=3.1x ATR=13.65 |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 5612.55 | 5596.65 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:10:00 | 5704.75 | 5678.00 | 0.00 | ORB-long ORB[5650.00,5693.90] vol=3.1x ATR=12.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 12:10:00 | 5723.86 | 5691.49 | 0.00 | T1 1.5R @ 5723.86 |
| Target hit | 2024-09-05 15:00:00 | 5737.90 | 5740.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2024-09-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:50:00 | 5729.10 | 5697.71 | 0.00 | ORB-long ORB[5655.00,5724.85] vol=2.8x ATR=14.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:00:00 | 5751.31 | 5706.34 | 0.00 | T1 1.5R @ 5751.31 |
| Target hit | 2024-09-12 15:20:00 | 5800.00 | 5773.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2024-09-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:00:00 | 5837.20 | 5799.07 | 0.00 | ORB-long ORB[5763.60,5807.90] vol=2.9x ATR=13.30 |
| Stop hit — per-position SL triggered | 2024-09-17 10:05:00 | 5823.90 | 5805.83 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 6052.00 | 6023.14 | 0.00 | ORB-long ORB[5980.10,6042.00] vol=2.0x ATR=22.52 |
| Stop hit — per-position SL triggered | 2024-09-19 09:40:00 | 6029.48 | 6026.59 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:30:00 | 6080.00 | 6032.77 | 0.00 | ORB-long ORB[5985.75,6048.75] vol=2.4x ATR=16.33 |
| Stop hit — per-position SL triggered | 2024-09-20 10:45:00 | 6063.67 | 6038.79 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 6135.00 | 6177.00 | 0.00 | ORB-short ORB[6171.60,6246.25] vol=2.0x ATR=14.97 |
| Stop hit — per-position SL triggered | 2024-09-24 11:05:00 | 6149.97 | 6174.65 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 5979.65 | 6039.09 | 0.00 | ORB-short ORB[5995.00,6070.00] vol=1.8x ATR=18.30 |
| Stop hit — per-position SL triggered | 2024-09-27 11:50:00 | 5997.95 | 6030.50 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:50:00 | 4906.55 | 4935.34 | 0.00 | ORB-short ORB[4944.10,5015.85] vol=1.7x ATR=18.93 |
| Stop hit — per-position SL triggered | 2024-10-28 10:00:00 | 4925.48 | 4931.48 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:30:00 | 4872.85 | 4906.92 | 0.00 | ORB-short ORB[4887.70,4959.95] vol=1.8x ATR=17.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:40:00 | 4847.18 | 4890.33 | 0.00 | T1 1.5R @ 4847.18 |
| Target hit | 2024-10-29 15:20:00 | 4798.95 | 4798.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:45:00 | 4914.90 | 4875.23 | 0.00 | ORB-long ORB[4841.00,4888.55] vol=1.7x ATR=15.44 |
| Stop hit — per-position SL triggered | 2024-11-06 09:50:00 | 4899.46 | 4879.52 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:20:00 | 4537.85 | 4506.11 | 0.00 | ORB-long ORB[4475.00,4529.05] vol=1.6x ATR=19.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 10:30:00 | 4566.96 | 4511.66 | 0.00 | T1 1.5R @ 4566.96 |
| Target hit | 2024-11-14 15:20:00 | 4605.00 | 4572.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-11-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:30:00 | 4820.80 | 4780.75 | 0.00 | ORB-long ORB[4734.50,4766.40] vol=2.6x ATR=13.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 4841.45 | 4793.55 | 0.00 | T1 1.5R @ 4841.45 |
| Stop hit — per-position SL triggered | 2024-11-19 12:25:00 | 4820.80 | 4806.19 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:35:00 | 4800.50 | 4774.43 | 0.00 | ORB-long ORB[4741.20,4799.95] vol=1.5x ATR=18.18 |
| Stop hit — per-position SL triggered | 2024-11-21 11:25:00 | 4782.32 | 4783.82 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:00:00 | 4864.00 | 4838.31 | 0.00 | ORB-long ORB[4815.50,4859.90] vol=2.9x ATR=10.75 |
| Stop hit — per-position SL triggered | 2024-11-27 11:10:00 | 4853.25 | 4839.09 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 4860.70 | 4887.68 | 0.00 | ORB-short ORB[4861.15,4909.95] vol=2.4x ATR=11.43 |
| Stop hit — per-position SL triggered | 2024-11-28 10:50:00 | 4872.13 | 4882.94 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 4630.60 | 4642.00 | 0.00 | ORB-short ORB[4632.50,4660.15] vol=1.8x ATR=8.64 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 4639.24 | 4639.08 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 4524.50 | 4536.49 | 0.00 | ORB-short ORB[4545.00,4590.00] vol=1.6x ATR=8.60 |
| Stop hit — per-position SL triggered | 2024-12-16 11:05:00 | 4533.10 | 4535.78 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:50:00 | 4439.20 | 4432.97 | 0.00 | ORB-long ORB[4389.85,4435.95] vol=1.7x ATR=11.37 |
| Stop hit — per-position SL triggered | 2024-12-20 11:15:00 | 4427.83 | 4433.43 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 4228.00 | 4241.35 | 0.00 | ORB-short ORB[4237.75,4274.00] vol=3.4x ATR=10.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:55:00 | 4211.87 | 4233.70 | 0.00 | T1 1.5R @ 4211.87 |
| Target hit | 2024-12-30 15:20:00 | 4184.90 | 4198.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:40:00 | 4203.25 | 4174.94 | 0.00 | ORB-long ORB[4146.60,4199.70] vol=2.8x ATR=14.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:35:00 | 4224.25 | 4193.07 | 0.00 | T1 1.5R @ 4224.25 |
| Target hit | 2025-01-02 15:20:00 | 4305.05 | 4258.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:05:00 | 4051.00 | 4086.02 | 0.00 | ORB-short ORB[4082.05,4118.75] vol=2.4x ATR=12.57 |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 4063.57 | 4083.82 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 4124.20 | 4086.61 | 0.00 | ORB-long ORB[4060.10,4106.00] vol=2.1x ATR=12.88 |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 4111.32 | 4090.69 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 11:00:00 | 4101.90 | 4069.37 | 0.00 | ORB-long ORB[4025.15,4056.00] vol=1.5x ATR=8.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 12:20:00 | 4114.99 | 4083.78 | 0.00 | T1 1.5R @ 4114.99 |
| Stop hit — per-position SL triggered | 2025-01-23 13:20:00 | 4101.90 | 4087.00 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-27 09:35:00 | 4055.05 | 4032.43 | 0.00 | ORB-long ORB[4002.00,4049.95] vol=1.7x ATR=13.82 |
| Stop hit — per-position SL triggered | 2025-01-27 10:20:00 | 4041.23 | 4044.01 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:35:00 | 4099.75 | 4081.53 | 0.00 | ORB-long ORB[4060.05,4095.00] vol=1.6x ATR=11.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:50:00 | 4116.27 | 4090.90 | 0.00 | T1 1.5R @ 4116.27 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 4099.75 | 4102.41 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:40:00 | 4328.05 | 4241.26 | 0.00 | ORB-long ORB[4176.00,4225.00] vol=2.6x ATR=17.44 |
| Stop hit — per-position SL triggered | 2025-01-31 10:45:00 | 4310.61 | 4247.22 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:55:00 | 4239.85 | 4283.07 | 0.00 | ORB-short ORB[4294.00,4350.00] vol=3.1x ATR=13.54 |
| Stop hit — per-position SL triggered | 2025-02-04 11:50:00 | 4253.39 | 4261.14 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 11:15:00 | 4255.00 | 4236.46 | 0.00 | ORB-long ORB[4212.60,4249.95] vol=2.9x ATR=8.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:25:00 | 4268.04 | 4244.82 | 0.00 | T1 1.5R @ 4268.04 |
| Stop hit — per-position SL triggered | 2025-02-05 12:45:00 | 4255.00 | 4254.24 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:35:00 | 3936.00 | 3961.22 | 0.00 | ORB-short ORB[3950.00,3970.00] vol=1.8x ATR=11.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:45:00 | 3918.40 | 3954.67 | 0.00 | T1 1.5R @ 3918.40 |
| Target hit | 2025-02-14 15:20:00 | 3857.20 | 3897.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:15:00 | 3925.25 | 3890.69 | 0.00 | ORB-long ORB[3855.50,3885.20] vol=1.8x ATR=10.15 |
| Stop hit — per-position SL triggered | 2025-02-20 10:45:00 | 3915.10 | 3901.14 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 11:10:00 | 3847.65 | 3824.10 | 0.00 | ORB-long ORB[3815.00,3844.10] vol=1.6x ATR=9.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 11:40:00 | 3862.59 | 3829.40 | 0.00 | T1 1.5R @ 3862.59 |
| Target hit | 2025-02-24 15:20:00 | 3882.80 | 3856.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:15:00 | 3748.30 | 3770.03 | 0.00 | ORB-short ORB[3787.80,3810.30] vol=2.0x ATR=9.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 12:15:00 | 3734.33 | 3762.95 | 0.00 | T1 1.5R @ 3734.33 |
| Stop hit — per-position SL triggered | 2025-02-27 13:20:00 | 3748.30 | 3756.39 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:40:00 | 3616.50 | 3585.81 | 0.00 | ORB-long ORB[3552.75,3587.45] vol=2.2x ATR=12.70 |
| Stop hit — per-position SL triggered | 2025-03-11 11:25:00 | 3603.80 | 3593.46 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-03-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 11:00:00 | 3578.00 | 3594.40 | 0.00 | ORB-short ORB[3583.70,3625.55] vol=1.5x ATR=8.46 |
| Stop hit — per-position SL triggered | 2025-03-13 11:10:00 | 3586.46 | 3594.00 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 10:20:00 | 3540.30 | 3559.91 | 0.00 | ORB-short ORB[3542.80,3588.85] vol=2.1x ATR=10.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:30:00 | 3524.09 | 3553.87 | 0.00 | T1 1.5R @ 3524.09 |
| Stop hit — per-position SL triggered | 2025-03-19 11:30:00 | 3540.30 | 3545.75 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 3651.25 | 3630.08 | 0.00 | ORB-long ORB[3611.75,3630.00] vol=1.7x ATR=9.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:45:00 | 3666.02 | 3640.11 | 0.00 | T1 1.5R @ 3666.02 |
| Target hit | 2025-03-21 12:10:00 | 3666.75 | 3671.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2025-03-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 10:25:00 | 3673.10 | 3646.29 | 0.00 | ORB-long ORB[3620.65,3652.50] vol=1.5x ATR=11.01 |
| Stop hit — per-position SL triggered | 2025-03-25 10:30:00 | 3662.09 | 3648.86 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 3609.80 | 3634.22 | 0.00 | ORB-short ORB[3632.95,3657.25] vol=1.5x ATR=10.61 |
| Stop hit — per-position SL triggered | 2025-03-26 10:10:00 | 3620.41 | 3624.77 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:50:00 | 3694.60 | 3661.60 | 0.00 | ORB-long ORB[3610.35,3655.00] vol=1.7x ATR=13.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:10:00 | 3714.10 | 3676.76 | 0.00 | T1 1.5R @ 3714.10 |
| Target hit | 2025-03-27 15:20:00 | 3771.70 | 3744.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2025-04-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:50:00 | 3779.00 | 3761.15 | 0.00 | ORB-long ORB[3732.90,3774.60] vol=1.9x ATR=10.36 |
| Stop hit — per-position SL triggered | 2025-04-02 11:05:00 | 3768.64 | 3762.45 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 11:15:00 | 3615.15 | 3570.14 | 0.00 | ORB-long ORB[3551.15,3599.00] vol=1.9x ATR=12.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 11:20:00 | 3634.32 | 3574.85 | 0.00 | T1 1.5R @ 3634.32 |
| Stop hit — per-position SL triggered | 2025-04-09 15:00:00 | 3615.15 | 3615.84 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:40:00 | 3820.80 | 3800.89 | 0.00 | ORB-long ORB[3756.00,3796.40] vol=1.8x ATR=11.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:25:00 | 3838.46 | 3808.22 | 0.00 | T1 1.5R @ 3838.46 |
| Target hit | 2025-04-21 15:20:00 | 3916.20 | 3869.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-04-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:50:00 | 3841.30 | 3853.63 | 0.00 | ORB-short ORB[3847.80,3873.00] vol=1.8x ATR=11.19 |
| Stop hit — per-position SL triggered | 2025-04-23 11:00:00 | 3852.49 | 3853.23 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:50:00 | 3974.40 | 3965.49 | 0.00 | ORB-long ORB[3924.90,3960.00] vol=3.2x ATR=11.47 |
| Stop hit — per-position SL triggered | 2025-04-24 11:00:00 | 3962.93 | 3964.61 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-04-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 11:00:00 | 3883.40 | 3925.51 | 0.00 | ORB-short ORB[3951.40,3994.10] vol=1.5x ATR=14.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:15:00 | 3861.01 | 3921.03 | 0.00 | T1 1.5R @ 3861.01 |
| Stop hit — per-position SL triggered | 2025-04-25 12:10:00 | 3883.40 | 3909.82 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:50:00 | 3834.20 | 3843.22 | 0.00 | ORB-short ORB[3839.60,3875.00] vol=1.8x ATR=10.52 |
| Stop hit — per-position SL triggered | 2025-05-08 10:05:00 | 3844.72 | 3842.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:50:00 | 5015.00 | 2024-05-14 10:00:00 | 4989.14 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-05-16 11:15:00 | 5045.50 | 2024-05-16 11:30:00 | 5061.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-28 10:50:00 | 5137.00 | 2024-05-28 10:55:00 | 5158.86 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-05-28 10:50:00 | 5137.00 | 2024-05-28 11:25:00 | 5137.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 10:45:00 | 5185.90 | 2024-05-29 10:55:00 | 5171.86 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-31 09:45:00 | 5182.00 | 2024-05-31 10:05:00 | 5163.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-03 10:50:00 | 5217.25 | 2024-06-03 12:35:00 | 5175.77 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-06-03 10:50:00 | 5217.25 | 2024-06-03 15:20:00 | 5159.35 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2024-06-10 10:05:00 | 5680.00 | 2024-06-10 10:20:00 | 5706.51 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-10 10:05:00 | 5680.00 | 2024-06-10 14:25:00 | 5716.70 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2024-06-11 10:50:00 | 5803.35 | 2024-06-11 10:55:00 | 5786.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-13 10:40:00 | 5780.25 | 2024-06-13 11:00:00 | 5758.83 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-06-13 10:40:00 | 5780.25 | 2024-06-13 11:40:00 | 5780.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 10:00:00 | 5786.45 | 2024-06-14 10:25:00 | 5802.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-25 10:35:00 | 5568.95 | 2024-06-25 10:40:00 | 5554.40 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-28 11:00:00 | 5562.05 | 2024-06-28 11:15:00 | 5544.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-12 09:55:00 | 5588.45 | 2024-07-12 10:10:00 | 5575.90 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-15 09:30:00 | 5579.00 | 2024-07-15 09:35:00 | 5567.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-16 11:05:00 | 5624.75 | 2024-07-16 11:20:00 | 5614.05 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-22 10:55:00 | 5481.85 | 2024-07-22 11:10:00 | 5467.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-23 11:10:00 | 5509.45 | 2024-07-23 11:20:00 | 5534.69 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-23 11:10:00 | 5509.45 | 2024-07-23 12:10:00 | 5569.70 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2024-07-26 11:15:00 | 5527.00 | 2024-07-26 12:05:00 | 5512.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-31 10:55:00 | 5427.40 | 2024-07-31 11:20:00 | 5442.31 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-05 11:15:00 | 5144.40 | 2024-08-05 11:25:00 | 5162.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-06 09:45:00 | 5200.00 | 2024-08-06 09:50:00 | 5218.99 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-13 09:45:00 | 5360.80 | 2024-08-13 09:55:00 | 5387.37 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-13 09:45:00 | 5360.80 | 2024-08-13 11:05:00 | 5360.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:45:00 | 5322.00 | 2024-08-22 10:55:00 | 5310.22 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-26 10:35:00 | 5415.60 | 2024-08-26 11:05:00 | 5401.40 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-27 10:20:00 | 5387.00 | 2024-08-27 11:30:00 | 5374.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-28 09:30:00 | 5297.80 | 2024-08-28 09:35:00 | 5310.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-03 10:05:00 | 5626.20 | 2024-09-03 10:15:00 | 5612.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-05 11:10:00 | 5704.75 | 2024-09-05 12:10:00 | 5723.86 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-09-05 11:10:00 | 5704.75 | 2024-09-05 15:00:00 | 5737.90 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-12 10:50:00 | 5729.10 | 2024-09-12 11:00:00 | 5751.31 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-12 10:50:00 | 5729.10 | 2024-09-12 15:20:00 | 5800.00 | TARGET_HIT | 0.50 | 1.24% |
| BUY | retest1 | 2024-09-17 10:00:00 | 5837.20 | 2024-09-17 10:05:00 | 5823.90 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-19 09:30:00 | 6052.00 | 2024-09-19 09:40:00 | 6029.48 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-20 10:30:00 | 6080.00 | 2024-09-20 10:45:00 | 6063.67 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-24 11:00:00 | 6135.00 | 2024-09-24 11:05:00 | 6149.97 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-27 11:05:00 | 5979.65 | 2024-09-27 11:50:00 | 5997.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-28 09:50:00 | 4906.55 | 2024-10-28 10:00:00 | 4925.48 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-29 09:30:00 | 4872.85 | 2024-10-29 09:40:00 | 4847.18 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-29 09:30:00 | 4872.85 | 2024-10-29 15:20:00 | 4798.95 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-11-06 09:45:00 | 4914.90 | 2024-11-06 09:50:00 | 4899.46 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-14 10:20:00 | 4537.85 | 2024-11-14 10:30:00 | 4566.96 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-11-14 10:20:00 | 4537.85 | 2024-11-14 15:20:00 | 4605.00 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2024-11-19 10:30:00 | 4820.80 | 2024-11-19 11:15:00 | 4841.45 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-19 10:30:00 | 4820.80 | 2024-11-19 12:25:00 | 4820.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-21 10:35:00 | 4800.50 | 2024-11-21 11:25:00 | 4782.32 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-27 11:00:00 | 4864.00 | 2024-11-27 11:10:00 | 4853.25 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-28 10:35:00 | 4860.70 | 2024-11-28 10:50:00 | 4872.13 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-12 09:35:00 | 4630.60 | 2024-12-12 09:50:00 | 4639.24 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-16 11:00:00 | 4524.50 | 2024-12-16 11:05:00 | 4533.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-12-20 10:50:00 | 4439.20 | 2024-12-20 11:15:00 | 4427.83 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-30 09:40:00 | 4228.00 | 2024-12-30 09:55:00 | 4211.87 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-30 09:40:00 | 4228.00 | 2024-12-30 15:20:00 | 4184.90 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-01-02 09:40:00 | 4203.25 | 2025-01-02 10:35:00 | 4224.25 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-02 09:40:00 | 4203.25 | 2025-01-02 15:20:00 | 4305.05 | TARGET_HIT | 0.50 | 2.42% |
| SELL | retest1 | 2025-01-13 11:05:00 | 4051.00 | 2025-01-13 11:15:00 | 4063.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-14 11:00:00 | 4124.20 | 2025-01-14 11:15:00 | 4111.32 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-23 11:00:00 | 4101.90 | 2025-01-23 12:20:00 | 4114.99 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-01-23 11:00:00 | 4101.90 | 2025-01-23 13:20:00 | 4101.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-27 09:35:00 | 4055.05 | 2025-01-27 10:20:00 | 4041.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-30 09:35:00 | 4099.75 | 2025-01-30 09:50:00 | 4116.27 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-30 09:35:00 | 4099.75 | 2025-01-30 10:50:00 | 4099.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 10:40:00 | 4328.05 | 2025-01-31 10:45:00 | 4310.61 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-04 10:55:00 | 4239.85 | 2025-02-04 11:50:00 | 4253.39 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-05 11:15:00 | 4255.00 | 2025-02-05 11:25:00 | 4268.04 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-02-05 11:15:00 | 4255.00 | 2025-02-05 12:45:00 | 4255.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 10:35:00 | 3936.00 | 2025-02-14 10:45:00 | 3918.40 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-14 10:35:00 | 3936.00 | 2025-02-14 15:20:00 | 3857.20 | TARGET_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2025-02-20 10:15:00 | 3925.25 | 2025-02-20 10:45:00 | 3915.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-24 11:10:00 | 3847.65 | 2025-02-24 11:40:00 | 3862.59 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-02-24 11:10:00 | 3847.65 | 2025-02-24 15:20:00 | 3882.80 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2025-02-27 11:15:00 | 3748.30 | 2025-02-27 12:15:00 | 3734.33 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-02-27 11:15:00 | 3748.30 | 2025-02-27 13:20:00 | 3748.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-11 10:40:00 | 3616.50 | 2025-03-11 11:25:00 | 3603.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-03-13 11:00:00 | 3578.00 | 2025-03-13 11:10:00 | 3586.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-19 10:20:00 | 3540.30 | 2025-03-19 10:30:00 | 3524.09 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-03-19 10:20:00 | 3540.30 | 2025-03-19 11:30:00 | 3540.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:35:00 | 3651.25 | 2025-03-21 09:45:00 | 3666.02 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-21 09:35:00 | 3651.25 | 2025-03-21 12:10:00 | 3666.75 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2025-03-25 10:25:00 | 3673.10 | 2025-03-25 10:30:00 | 3662.09 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-26 09:40:00 | 3609.80 | 2025-03-26 10:10:00 | 3620.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-27 09:50:00 | 3694.60 | 2025-03-27 10:10:00 | 3714.10 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-27 09:50:00 | 3694.60 | 2025-03-27 15:20:00 | 3771.70 | TARGET_HIT | 0.50 | 2.09% |
| BUY | retest1 | 2025-04-02 10:50:00 | 3779.00 | 2025-04-02 11:05:00 | 3768.64 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-09 11:15:00 | 3615.15 | 2025-04-09 11:20:00 | 3634.32 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-09 11:15:00 | 3615.15 | 2025-04-09 15:00:00 | 3615.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 10:40:00 | 3820.80 | 2025-04-21 11:25:00 | 3838.46 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-21 10:40:00 | 3820.80 | 2025-04-21 15:20:00 | 3916.20 | TARGET_HIT | 0.50 | 2.50% |
| SELL | retest1 | 2025-04-23 10:50:00 | 3841.30 | 2025-04-23 11:00:00 | 3852.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-24 10:50:00 | 3974.40 | 2025-04-24 11:00:00 | 3962.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-25 11:00:00 | 3883.40 | 2025-04-25 11:15:00 | 3861.01 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-25 11:00:00 | 3883.40 | 2025-04-25 12:10:00 | 3883.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-08 09:50:00 | 3834.20 | 2025-05-08 10:05:00 | 3844.72 | STOP_HIT | 1.00 | -0.27% |

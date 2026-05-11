# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-09-05 15:25:00 (4575 bars)
- **Last close:** 5006.90
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
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 8
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 4.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.06% | 1.0% |
| BUY @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.06% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.35% | 3.5% |
| SELL @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.35% | 3.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 11 | 42.3% | 3 | 15 | 8 | 0.18% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:00:00 | 5170.80 | 5124.41 | 0.00 | ORB-long ORB[5085.50,5139.70] vol=3.4x ATR=18.16 |
| Stop hit — per-position SL triggered | 2025-05-15 10:05:00 | 5152.64 | 5130.39 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:05:00 | 5126.30 | 5111.50 | 0.00 | ORB-long ORB[5052.30,5108.40] vol=5.3x ATR=16.57 |
| Stop hit — per-position SL triggered | 2025-05-26 10:25:00 | 5109.73 | 5113.89 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:20:00 | 5223.00 | 5182.81 | 0.00 | ORB-long ORB[5116.80,5164.00] vol=10.0x ATR=17.22 |
| Stop hit — per-position SL triggered | 2025-05-27 10:25:00 | 5205.78 | 5186.98 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:35:00 | 5296.20 | 5282.34 | 0.00 | ORB-long ORB[5253.30,5287.00] vol=2.5x ATR=14.76 |
| Stop hit — per-position SL triggered | 2025-05-30 10:05:00 | 5281.44 | 5287.85 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:10:00 | 5194.00 | 5236.22 | 0.00 | ORB-short ORB[5225.00,5299.00] vol=3.0x ATR=18.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:55:00 | 5165.81 | 5221.45 | 0.00 | T1 1.5R @ 5165.81 |
| Stop hit — per-position SL triggered | 2025-06-02 12:30:00 | 5194.00 | 5216.02 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:45:00 | 5324.50 | 5303.51 | 0.00 | ORB-long ORB[5278.50,5310.00] vol=3.3x ATR=15.03 |
| Stop hit — per-position SL triggered | 2025-06-06 09:55:00 | 5309.47 | 5304.65 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 11:10:00 | 5357.50 | 5401.76 | 0.00 | ORB-short ORB[5362.00,5435.50] vol=3.3x ATR=14.73 |
| Stop hit — per-position SL triggered | 2025-06-09 11:30:00 | 5372.23 | 5400.64 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:00:00 | 5482.00 | 5428.78 | 0.00 | ORB-long ORB[5381.00,5425.00] vol=2.6x ATR=18.73 |
| Stop hit — per-position SL triggered | 2025-06-10 10:10:00 | 5463.27 | 5442.82 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 09:40:00 | 5395.00 | 5365.03 | 0.00 | ORB-long ORB[5291.00,5367.00] vol=3.5x ATR=18.91 |
| Stop hit — per-position SL triggered | 2025-06-16 09:45:00 | 5376.09 | 5365.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 11:00:00 | 5554.50 | 5596.89 | 0.00 | ORB-short ORB[5581.50,5652.00] vol=2.7x ATR=20.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 12:25:00 | 5523.14 | 5585.78 | 0.00 | T1 1.5R @ 5523.14 |
| Stop hit — per-position SL triggered | 2025-06-18 15:00:00 | 5554.50 | 5540.87 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:45:00 | 5671.00 | 5649.94 | 0.00 | ORB-long ORB[5575.00,5645.50] vol=1.5x ATR=30.68 |
| Stop hit — per-position SL triggered | 2025-06-19 09:50:00 | 5640.32 | 5650.11 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 6002.50 | 5976.01 | 0.00 | ORB-long ORB[5920.50,5990.00] vol=3.7x ATR=20.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:35:00 | 6033.31 | 5993.98 | 0.00 | T1 1.5R @ 6033.31 |
| Target hit | 2025-06-27 12:20:00 | 6047.50 | 6047.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2025-07-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:05:00 | 5957.50 | 5898.38 | 0.00 | ORB-long ORB[5862.00,5928.00] vol=1.7x ATR=19.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:10:00 | 5987.38 | 5920.46 | 0.00 | T1 1.5R @ 5987.38 |
| Target hit | 2025-07-03 15:10:00 | 6029.50 | 6031.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2025-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:30:00 | 5195.00 | 5217.28 | 0.00 | ORB-short ORB[5209.00,5248.00] vol=2.7x ATR=13.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:30:00 | 5174.12 | 5200.28 | 0.00 | T1 1.5R @ 5174.12 |
| Target hit | 2025-08-12 15:20:00 | 5078.00 | 5145.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:55:00 | 5396.50 | 5383.37 | 0.00 | ORB-long ORB[5354.00,5389.50] vol=1.6x ATR=13.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:20:00 | 5417.34 | 5387.64 | 0.00 | T1 1.5R @ 5417.34 |
| Stop hit — per-position SL triggered | 2025-08-21 10:25:00 | 5396.50 | 5388.91 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:20:00 | 5090.00 | 5131.86 | 0.00 | ORB-short ORB[5145.50,5221.50] vol=3.5x ATR=15.95 |
| Stop hit — per-position SL triggered | 2025-08-26 10:25:00 | 5105.95 | 5130.16 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 11:00:00 | 4990.00 | 4951.38 | 0.00 | ORB-long ORB[4897.50,4958.00] vol=1.8x ATR=15.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:10:00 | 5013.30 | 4955.89 | 0.00 | T1 1.5R @ 5013.30 |
| Stop hit — per-position SL triggered | 2025-08-29 11:55:00 | 4990.00 | 4971.70 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-09-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:55:00 | 5031.00 | 5055.38 | 0.00 | ORB-short ORB[5034.00,5080.10] vol=3.0x ATR=11.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:30:00 | 5013.27 | 5050.00 | 0.00 | T1 1.5R @ 5013.27 |
| Stop hit — per-position SL triggered | 2025-09-03 14:55:00 | 5031.00 | 5031.61 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:00:00 | 5170.80 | 2025-05-15 10:05:00 | 5152.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-26 10:05:00 | 5126.30 | 2025-05-26 10:25:00 | 5109.73 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-27 10:20:00 | 5223.00 | 2025-05-27 10:25:00 | 5205.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-30 09:35:00 | 5296.20 | 2025-05-30 10:05:00 | 5281.44 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-02 10:10:00 | 5194.00 | 2025-06-02 11:55:00 | 5165.81 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-06-02 10:10:00 | 5194.00 | 2025-06-02 12:30:00 | 5194.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 09:45:00 | 5324.50 | 2025-06-06 09:55:00 | 5309.47 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-09 11:10:00 | 5357.50 | 2025-06-09 11:30:00 | 5372.23 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-10 10:00:00 | 5482.00 | 2025-06-10 10:10:00 | 5463.27 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-16 09:40:00 | 5395.00 | 2025-06-16 09:45:00 | 5376.09 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-18 11:00:00 | 5554.50 | 2025-06-18 12:25:00 | 5523.14 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-06-18 11:00:00 | 5554.50 | 2025-06-18 15:00:00 | 5554.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-19 09:45:00 | 5671.00 | 2025-06-19 09:50:00 | 5640.32 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-06-27 09:30:00 | 6002.50 | 2025-06-27 09:35:00 | 6033.31 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-06-27 09:30:00 | 6002.50 | 2025-06-27 12:20:00 | 6047.50 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2025-07-03 10:05:00 | 5957.50 | 2025-07-03 10:10:00 | 5987.38 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-07-03 10:05:00 | 5957.50 | 2025-07-03 15:10:00 | 6029.50 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2025-08-12 09:30:00 | 5195.00 | 2025-08-12 10:30:00 | 5174.12 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-12 09:30:00 | 5195.00 | 2025-08-12 15:20:00 | 5078.00 | TARGET_HIT | 0.50 | 2.25% |
| BUY | retest1 | 2025-08-21 09:55:00 | 5396.50 | 2025-08-21 10:20:00 | 5417.34 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-21 09:55:00 | 5396.50 | 2025-08-21 10:25:00 | 5396.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 10:20:00 | 5090.00 | 2025-08-26 10:25:00 | 5105.95 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-29 11:00:00 | 4990.00 | 2025-08-29 11:10:00 | 5013.30 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-29 11:00:00 | 4990.00 | 2025-08-29 11:55:00 | 4990.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-03 10:55:00 | 5031.00 | 2025-09-03 11:30:00 | 5013.27 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-03 10:55:00 | 5031.00 | 2025-09-03 14:55:00 | 5031.00 | STOP_HIT | 0.50 | 0.00% |

# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5695.00
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
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 15
- **Target hits / Stop hits / Partials:** 5 / 15 / 8
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 6.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | -0.05% | -0.5% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | -0.05% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.39% | 7.4% |
| SELL @ 2nd Alert (retest1) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.39% | 7.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 13 | 46.4% | 5 | 15 | 8 | 0.25% | 6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:10:00 | 5723.50 | 5748.70 | 0.00 | ORB-short ORB[5750.00,5811.50] vol=1.9x ATR=11.37 |
| Stop hit — per-position SL triggered | 2026-02-16 11:50:00 | 5734.87 | 5744.42 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 5788.00 | 5813.82 | 0.00 | ORB-short ORB[5817.50,5858.00] vol=3.0x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:35:00 | 5774.76 | 5809.21 | 0.00 | T1 1.5R @ 5774.76 |
| Target hit | 2026-02-19 15:20:00 | 5645.50 | 5752.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 5410.00 | 5446.16 | 0.00 | ORB-short ORB[5423.50,5460.00] vol=2.8x ATR=13.71 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 5423.71 | 5445.04 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:05:00 | 5224.50 | 5245.75 | 0.00 | ORB-short ORB[5234.50,5276.00] vol=5.2x ATR=18.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:45:00 | 5196.19 | 5230.63 | 0.00 | T1 1.5R @ 5196.19 |
| Stop hit — per-position SL triggered | 2026-03-13 13:25:00 | 5224.50 | 5227.42 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:35:00 | 5130.50 | 5143.32 | 0.00 | ORB-short ORB[5173.00,5222.50] vol=12.4x ATR=18.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:10:00 | 5102.94 | 5133.80 | 0.00 | T1 1.5R @ 5102.94 |
| Target hit | 2026-03-19 15:20:00 | 5071.50 | 5117.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:55:00 | 4967.00 | 4993.71 | 0.00 | ORB-short ORB[4970.00,5027.00] vol=2.1x ATR=20.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:00:00 | 4936.50 | 4973.93 | 0.00 | T1 1.5R @ 4936.50 |
| Target hit | 2026-03-27 15:20:00 | 4850.00 | 4900.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-04-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 09:50:00 | 4849.90 | 4873.01 | 0.00 | ORB-short ORB[4859.40,4924.10] vol=2.0x ATR=20.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:10:00 | 4819.54 | 4845.65 | 0.00 | T1 1.5R @ 4819.54 |
| Stop hit — per-position SL triggered | 2026-04-06 10:45:00 | 4849.90 | 4842.94 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 5143.90 | 5118.85 | 0.00 | ORB-long ORB[5055.60,5118.00] vol=2.1x ATR=21.17 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 5122.73 | 5122.01 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:35:00 | 5164.00 | 5141.61 | 0.00 | ORB-long ORB[5101.00,5160.00] vol=2.7x ATR=15.99 |
| Target hit | 2026-04-15 15:20:00 | 5169.70 | 5151.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 5169.20 | 5199.40 | 0.00 | ORB-short ORB[5187.40,5230.00] vol=1.8x ATR=14.16 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 5183.36 | 5196.25 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 5331.50 | 5313.06 | 0.00 | ORB-long ORB[5256.20,5308.00] vol=2.5x ATR=21.28 |
| Stop hit — per-position SL triggered | 2026-04-17 10:00:00 | 5310.22 | 5327.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 5573.50 | 5529.85 | 0.00 | ORB-long ORB[5471.70,5511.70] vol=2.0x ATR=15.78 |
| Stop hit — per-position SL triggered | 2026-04-21 09:55:00 | 5557.72 | 5530.82 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 5380.00 | 5422.41 | 0.00 | ORB-short ORB[5428.70,5500.00] vol=3.9x ATR=19.09 |
| Stop hit — per-position SL triggered | 2026-04-22 10:30:00 | 5399.09 | 5415.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 5413.00 | 5450.49 | 0.00 | ORB-short ORB[5424.00,5492.90] vol=3.3x ATR=11.59 |
| Stop hit — per-position SL triggered | 2026-04-23 11:20:00 | 5424.59 | 5450.25 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 5299.60 | 5340.24 | 0.00 | ORB-short ORB[5333.90,5384.60] vol=5.0x ATR=13.00 |
| Stop hit — per-position SL triggered | 2026-04-24 11:40:00 | 5312.60 | 5336.84 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 5490.10 | 5440.55 | 0.00 | ORB-long ORB[5351.00,5420.90] vol=4.6x ATR=21.72 |
| Stop hit — per-position SL triggered | 2026-04-27 10:00:00 | 5468.38 | 5448.36 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 5460.40 | 5452.95 | 0.00 | ORB-long ORB[5413.50,5449.80] vol=7.0x ATR=12.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:55:00 | 5479.18 | 5458.04 | 0.00 | T1 1.5R @ 5479.18 |
| Target hit | 2026-04-28 10:45:00 | 5472.30 | 5472.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2026-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:35:00 | 5446.00 | 5436.32 | 0.00 | ORB-long ORB[5413.20,5445.50] vol=1.7x ATR=12.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:45:00 | 5464.90 | 5442.66 | 0.00 | T1 1.5R @ 5464.90 |
| Stop hit — per-position SL triggered | 2026-04-30 10:05:00 | 5446.00 | 5449.55 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 5533.50 | 5550.36 | 0.00 | ORB-short ORB[5544.00,5588.00] vol=2.0x ATR=12.15 |
| Stop hit — per-position SL triggered | 2026-05-06 11:50:00 | 5545.65 | 5549.72 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-05-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:00:00 | 5652.00 | 5683.37 | 0.00 | ORB-short ORB[5671.00,5753.00] vol=3.0x ATR=22.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:40:00 | 5617.53 | 5674.14 | 0.00 | T1 1.5R @ 5617.53 |
| Stop hit — per-position SL triggered | 2026-05-07 11:10:00 | 5652.00 | 5670.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 11:10:00 | 5723.50 | 2026-02-16 11:50:00 | 5734.87 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-19 11:10:00 | 5788.00 | 2026-02-19 11:35:00 | 5774.76 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2026-02-19 11:10:00 | 5788.00 | 2026-02-19 15:20:00 | 5645.50 | TARGET_HIT | 0.50 | 2.46% |
| SELL | retest1 | 2026-03-06 10:45:00 | 5410.00 | 2026-03-06 10:50:00 | 5423.71 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-13 10:05:00 | 5224.50 | 2026-03-13 12:45:00 | 5196.19 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 10:05:00 | 5224.50 | 2026-03-13 13:25:00 | 5224.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 09:35:00 | 5130.50 | 2026-03-19 11:10:00 | 5102.94 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-19 09:35:00 | 5130.50 | 2026-03-19 15:20:00 | 5071.50 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2026-03-27 09:55:00 | 4967.00 | 2026-03-27 10:00:00 | 4936.50 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-27 09:55:00 | 4967.00 | 2026-03-27 15:20:00 | 4850.00 | TARGET_HIT | 0.50 | 2.36% |
| SELL | retest1 | 2026-04-06 09:50:00 | 4849.90 | 2026-04-06 10:10:00 | 4819.54 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-04-06 09:50:00 | 4849.90 | 2026-04-06 10:45:00 | 4849.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:40:00 | 5143.90 | 2026-04-10 10:00:00 | 5122.73 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-15 10:35:00 | 5164.00 | 2026-04-15 15:20:00 | 5169.70 | TARGET_HIT | 1.00 | 0.11% |
| SELL | retest1 | 2026-04-16 09:55:00 | 5169.20 | 2026-04-16 10:25:00 | 5183.36 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-17 09:35:00 | 5331.50 | 2026-04-17 10:00:00 | 5310.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-21 09:50:00 | 5573.50 | 2026-04-21 09:55:00 | 5557.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-22 10:10:00 | 5380.00 | 2026-04-22 10:30:00 | 5399.09 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-23 11:10:00 | 5413.00 | 2026-04-23 11:20:00 | 5424.59 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-24 11:10:00 | 5299.60 | 2026-04-24 11:40:00 | 5312.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-27 09:50:00 | 5490.10 | 2026-04-27 10:00:00 | 5468.38 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-28 09:45:00 | 5460.40 | 2026-04-28 09:55:00 | 5479.18 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-04-28 09:45:00 | 5460.40 | 2026-04-28 10:45:00 | 5472.30 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2026-04-30 09:35:00 | 5446.00 | 2026-04-30 09:45:00 | 5464.90 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-30 09:35:00 | 5446.00 | 2026-04-30 10:05:00 | 5446.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:10:00 | 5533.50 | 2026-05-06 11:50:00 | 5545.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-07 10:00:00 | 5652.00 | 2026-05-07 10:40:00 | 5617.53 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-05-07 10:00:00 | 5652.00 | 2026-05-07 11:10:00 | 5652.00 | STOP_HIT | 0.50 | 0.00% |

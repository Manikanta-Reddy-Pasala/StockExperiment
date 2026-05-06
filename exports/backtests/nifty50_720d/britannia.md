# BRITANNIA (BRITANNIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4997 bars)
- **Last close:** 5783.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 6 |
| ALERT3 | 12 |
| PENDING | 50 |
| PENDING_CANCEL | 19 |
| ENTRY1 | 12 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 25
- **Target hits / Stop hits / Partials:** 0 / 31 / 0
- **Avg / median % per leg:** -1.14% / -1.74%
- **Sum % (uncompounded):** -35.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 0 | 22 | 0 | -0.76% | -16.7% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 0 | 11 | 0 | 0.16% | 1.8% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.68% | -18.4% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.07% | -18.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.56% | -4.6% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.76% | -14.1% |
| retest1 (combined) | 12 | 6 | 50.0% | 0 | 12 | 0 | -0.23% | -2.8% |
| retest2 (combined) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.71% | -32.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 15:15:00 | 4718.35 | 4606.79 | 4606.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 4771.90 | 4630.96 | 4619.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 12:15:00 | 5059.65 | 5069.27 | 4927.88 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-10 14:15:00 | 5084.95 | 5069.42 | 4929.37 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 15:15:00 | 5096.85 | 5069.69 | 4930.20 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-19 10:15:00 | 5102.50 | 5087.10 | 4967.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 11:15:00 | 5113.60 | 5087.37 | 4968.63 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 4961.00 | 5085.52 | 4972.95 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-23 13:15:00 | 4972.95 | 5085.52 | 4972.95 | SL hit qty=1.00 sl=4972.95 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-23 13:15:00 | 4972.95 | 5085.52 | 4972.95 | SL hit qty=1.00 sl=4972.95 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-23 14:15:00 | 5043.40 | 5085.10 | 4973.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 15:15:00 | 5048.10 | 5084.73 | 4973.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-07 10:15:00 | 4926.15 | 5102.57 | 5015.34 | SL hit qty=1.00 sl=4926.15 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-12 13:15:00 | 4994.00 | 5074.29 | 5010.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-12 14:15:00 | 4972.10 | 5073.27 | 5010.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-13 09:15:00 | 5020.45 | 5071.64 | 5009.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 10:15:00 | 5011.25 | 5071.04 | 5009.96 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-13 14:15:00 | 4985.45 | 5068.17 | 5009.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-13 15:15:00 | 4981.60 | 5067.31 | 5009.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-14 14:15:00 | 5002.90 | 5061.66 | 5008.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 15:15:00 | 5010.65 | 5061.15 | 5008.43 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-16 10:15:00 | 4926.15 | 5051.57 | 5005.86 | SL hit qty=1.00 sl=4926.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-16 10:15:00 | 4926.15 | 5051.57 | 5005.86 | SL hit qty=1.00 sl=4926.15 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 14:15:00 | 4837.60 | 4975.33 | 4975.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 4833.80 | 4972.59 | 4974.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 10:15:00 | 4947.05 | 4945.14 | 4958.69 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 10:15:00 | 4947.05 | 4945.14 | 4958.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 4947.05 | 4945.14 | 4958.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-03-13 11:15:00 | 4894.70 | 4944.63 | 4958.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 12:15:00 | 4868.90 | 4943.88 | 4957.93 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-15 09:15:00 | 4975.00 | 4939.39 | 4954.84 | SL hit qty=1.00 sl=4975.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-19 09:15:00 | 4918.00 | 4944.53 | 4956.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:15:00 | 4892.10 | 4944.01 | 4956.15 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-26 14:15:00 | 4975.00 | 4916.78 | 4939.43 | SL hit qty=1.00 sl=4975.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-27 10:15:00 | 4907.30 | 4917.42 | 4939.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 11:15:00 | 4885.15 | 4917.10 | 4939.15 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-01 10:15:00 | 4917.20 | 4916.39 | 4937.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-01 11:15:00 | 4889.70 | 4916.12 | 4937.15 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 4906.05 | 4810.37 | 4855.05 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-06 10:15:00 | 4975.00 | 4813.84 | 4856.56 | SL hit qty=1.00 sl=4975.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-06 10:15:00 | 4975.00 | 4813.84 | 4856.56 | SL hit qty=1.00 sl=4975.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 5067.30 | 4894.82 | 4894.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 09:15:00 | 5108.30 | 4904.93 | 4899.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 5669.20 | 5694.08 | 5512.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-08-05 09:15:00 | 5780.55 | 5695.93 | 5519.59 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-05 10:15:00 | 5733.35 | 5696.30 | 5520.66 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-05 11:15:00 | 5755.30 | 5696.89 | 5521.83 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-05 12:15:00 | 5742.90 | 5697.35 | 5522.93 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-05 13:15:00 | 5770.90 | 5698.08 | 5524.17 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-05 14:15:00 | 5708.65 | 5698.19 | 5525.09 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-06 09:15:00 | 5824.00 | 5699.44 | 5527.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:15:00 | 5813.35 | 5700.57 | 5528.86 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-08 13:15:00 | 5756.20 | 5721.68 | 5553.94 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-08 14:15:00 | 5744.45 | 5721.91 | 5554.89 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-09 09:15:00 | 5760.00 | 5722.51 | 5556.86 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:15:00 | 5753.00 | 5722.82 | 5557.84 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-09 12:15:00 | 5749.30 | 5723.32 | 5559.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 13:15:00 | 5749.20 | 5723.57 | 5560.68 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-19 12:15:00 | 5753.00 | 5713.04 | 5580.33 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-19 13:15:00 | 5744.70 | 5713.36 | 5581.15 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-20 09:15:00 | 5763.25 | 5714.19 | 5583.54 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-20 10:15:00 | 5742.15 | 5714.47 | 5584.33 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-20 12:15:00 | 5751.25 | 5715.03 | 5585.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 13:15:00 | 5761.90 | 5715.50 | 5586.79 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 5952.35 | 6090.39 | 5950.86 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 5950.86 | 6090.39 | 5950.86 | SL hit qty=1.00 sl=5950.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 5950.86 | 6090.39 | 5950.86 | SL hit qty=1.00 sl=5950.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 5950.86 | 6090.39 | 5950.86 | SL hit qty=1.00 sl=5950.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 5950.86 | 6090.39 | 5950.86 | SL hit qty=1.00 sl=5950.86 alert=retest1 |
| Cross detected — sustain check pending | 2024-10-15 10:15:00 | 6025.00 | 6081.83 | 5951.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:15:00 | 6044.20 | 6081.46 | 5952.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 5930.15 | 6073.99 | 5960.28 | SL hit qty=1.00 sl=5930.15 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 5625.20 | 5883.74 | 5884.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 5593.65 | 5876.30 | 5880.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 4928.45 | 4879.19 | 5088.01 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-15 09:15:00 | 4822.30 | 4889.42 | 5066.34 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-15 10:15:00 | 4862.55 | 4889.15 | 5065.33 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-16 09:15:00 | 4824.10 | 4887.71 | 5059.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:15:00 | 4811.80 | 4886.96 | 5058.16 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-17 10:15:00 | 4840.00 | 4883.33 | 5050.42 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-17 11:15:00 | 4849.70 | 4883.00 | 5049.42 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 5025.15 | 4891.03 | 5031.03 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-23 12:15:00 | 5031.03 | 4891.03 | 5031.03 | SL hit qty=1.00 sl=5031.03 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-04 12:15:00 | 5002.30 | 4975.03 | 5044.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-04 13:15:00 | 5012.55 | 4975.40 | 5044.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-05 09:15:00 | 4961.25 | 4976.37 | 5044.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 10:15:00 | 4970.00 | 4976.31 | 5043.75 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 5042.95 | 4973.52 | 5038.07 | SL hit qty=1.00 sl=5042.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-03 09:15:00 | 5004.75 | 4834.98 | 4876.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-03 10:15:00 | 5032.70 | 4836.95 | 4877.73 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-04 12:15:00 | 4980.05 | 4856.40 | 4885.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-04 13:15:00 | 5015.00 | 4857.98 | 4886.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 4949.10 | 4862.12 | 4888.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 4966.40 | 4863.16 | 4888.58 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 5042.95 | 4867.13 | 4890.20 | SL hit qty=1.00 sl=5042.95 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 5343.60 | 4913.10 | 4912.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 5362.85 | 4930.01 | 4920.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 10:15:00 | 5520.00 | 5524.70 | 5407.32 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-20 11:15:00 | 5545.00 | 5524.90 | 5408.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 12:15:00 | 5555.00 | 5525.20 | 5408.74 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 12:15:00 | 5562.50 | 5526.87 | 5413.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 13:15:00 | 5580.50 | 5527.40 | 5414.44 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 5614.50 | 5715.82 | 5612.70 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 5612.70 | 5715.82 | 5612.70 | SL hit qty=1.00 sl=5612.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 5612.70 | 5715.82 | 5612.70 | SL hit qty=1.00 sl=5612.70 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-28 10:15:00 | 5643.50 | 5709.93 | 5612.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 5648.00 | 5709.32 | 5612.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 5600.50 | 5707.41 | 5612.94 | SL hit qty=1.00 sl=5600.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 5684.00 | 5699.12 | 5613.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 5707.00 | 5699.20 | 5613.75 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-05 15:15:00 | 5640.00 | 5715.42 | 5635.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-06 09:15:00 | 5530.50 | 5713.58 | 5635.25 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 5600.50 | 5713.58 | 5635.25 | SL hit qty=1.00 sl=5600.50 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5412.50 | 5574.88 | 5575.51 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 5662.50 | 5576.54 | 5576.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5690.00 | 5577.66 | 5576.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 5931.00 | 5948.59 | 5817.31 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-09-24 12:15:00 | 6015.00 | 5948.74 | 5820.63 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 13:15:00 | 6009.50 | 5949.35 | 5821.57 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-29 15:15:00 | 5965.00 | 5948.80 | 5835.27 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-30 09:15:00 | 5929.00 | 5948.61 | 5835.73 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-30 11:15:00 | 6009.00 | 5949.20 | 5837.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 12:15:00 | 6019.50 | 5949.90 | 5838.07 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-01 15:15:00 | 5967.00 | 5950.56 | 5843.86 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:15:00 | 5977.00 | 5950.82 | 5844.53 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 2520m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 5844.00 | 5953.40 | 5856.58 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5856.58 | 5953.40 | 5856.58 | SL hit qty=1.00 sl=5856.58 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5856.58 | 5953.40 | 5856.58 | SL hit qty=1.00 sl=5856.58 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5856.58 | 5953.40 | 5856.58 | SL hit qty=1.00 sl=5856.58 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 5904.50 | 5916.90 | 5854.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-16 10:15:00 | 5896.50 | 5916.70 | 5855.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 5944.50 | 5916.97 | 5855.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 5970.00 | 5917.50 | 5856.02 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 5841.00 | 5961.18 | 5890.91 | SL hit qty=1.00 sl=5841.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-04 14:15:00 | 5902.50 | 5922.05 | 5882.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 5912.00 | 5921.95 | 5883.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 5841.00 | 5956.83 | 5905.41 | SL hit qty=1.00 sl=5841.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 13:15:00 | 5925.50 | 5954.22 | 5905.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:15:00 | 5951.50 | 5954.19 | 5905.34 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-12 13:15:00 | 5907.00 | 5952.39 | 5905.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 5841.00 | 5952.39 | 5905.88 | SL hit qty=1.00 sl=5841.00 alert=retest2 |
| Sustain check cancelled (price retraced) | 2025-11-12 14:15:00 | 5886.00 | 5951.73 | 5905.78 | ENTRY2 sustain failed after 60m |

### Cycle 8 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 5816.50 | 5877.87 | 5877.91 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5877.98 | 5877.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.80 | 5878.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA400 retest candle locked |

### Cycle 10 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 5849.00 | 5877.80 | 5877.88 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5877.99 | 5877.97 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5825.50 | 5877.53 | 5877.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.00 | 5876.96 | 5877.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5875.03 | 5876.46 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5891.00 | 5875.03 | 5876.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5891.00 | 5875.03 | 5876.46 | EMA400 retest candle locked |

### Cycle 13 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 6010.00 | 5878.35 | 5878.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.47 | 5879.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5978.09 | 5940.33 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 5992.00 | 5978.32 | 5941.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.32 | 5941.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 6024.50 | 5978.77 | 5941.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 6000.00 | 5978.99 | 5941.90 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 6026.00 | 5981.77 | 5944.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 6047.00 | 5982.42 | 5945.11 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 5938.50 | 6002.80 | 5960.20 | SL hit qty=1.00 sl=5938.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 5938.50 | 6002.80 | 5960.20 | SL hit qty=1.00 sl=5938.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-09 15:15:00 | 6000.00 | 6002.77 | 5960.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-12 09:15:00 | 5963.50 | 6002.38 | 5960.42 | ENTRY2 sustain failed after 3960m |

### Cycle 14 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5710.00 | 5935.13 | 5935.29 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 5995.50 | 5927.02 | 5926.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 6042.50 | 5928.17 | 5927.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5993.00 | 6021.23 | 5982.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 5991.00 | 6020.93 | 5982.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5991.00 | 6020.93 | 5982.27 | EMA400 retest candle locked |

### Cycle 16 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5956.32 | 5957.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.72 | 5948.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5670.05 | 5770.21 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 5830.00 | 5678.29 | 5767.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 5830.00 | 5678.29 | 5767.22 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-22 14:15:00 | 5722.50 | 5685.06 | 5767.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:15:00 | 5720.00 | 5685.41 | 5767.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-27 12:15:00 | 5737.50 | 5688.50 | 5761.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 5734.50 | 5688.96 | 5761.77 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-04 11:15:00 | 5738.00 | 5692.69 | 5754.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-04 12:15:00 | 5765.50 | 5693.41 | 5754.83 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 5830.00 | 5702.49 | 5756.76 | SL hit qty=1.00 sl=5830.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 5830.00 | 5702.49 | 5756.76 | SL hit qty=1.00 sl=5830.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-10 15:15:00 | 5096.85 | 2024-01-23 13:15:00 | 4972.95 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest1 | 2024-01-19 11:15:00 | 5113.60 | 2024-01-23 13:15:00 | 4972.95 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-01-23 15:15:00 | 5048.10 | 2024-02-07 10:15:00 | 4926.15 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-02-13 10:15:00 | 5011.25 | 2024-02-16 10:15:00 | 4926.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-02-14 15:15:00 | 5010.65 | 2024-02-16 10:15:00 | 4926.15 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-03-13 12:15:00 | 4868.90 | 2024-03-15 09:15:00 | 4975.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-03-19 10:15:00 | 4892.10 | 2024-03-26 14:15:00 | 4975.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-03-27 11:15:00 | 4885.15 | 2024-05-06 10:15:00 | 4975.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-04-01 11:15:00 | 4889.70 | 2024-05-06 10:15:00 | 4975.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest1 | 2024-08-06 10:15:00 | 5813.35 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest1 | 2024-08-09 10:15:00 | 5753.00 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 3.44% |
| BUY | retest1 | 2024-08-09 13:15:00 | 5749.20 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 3.51% |
| BUY | retest1 | 2024-08-20 13:15:00 | 5761.90 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 3.28% |
| BUY | retest2 | 2024-10-15 11:15:00 | 6044.20 | 2024-10-18 09:15:00 | 5930.15 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest1 | 2025-01-16 10:15:00 | 4811.80 | 2025-01-23 12:15:00 | 5031.03 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-02-05 10:15:00 | 4970.00 | 2025-02-07 09:15:00 | 5042.95 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-04-07 10:15:00 | 4966.40 | 2025-04-07 13:15:00 | 5042.95 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest1 | 2025-06-20 12:15:00 | 5555.00 | 2025-07-25 11:15:00 | 5612.70 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest1 | 2025-06-23 13:15:00 | 5580.50 | 2025-07-25 11:15:00 | 5612.70 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-07-28 11:15:00 | 5648.00 | 2025-07-28 13:15:00 | 5600.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-07-30 10:15:00 | 5707.00 | 2025-08-06 09:15:00 | 5600.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest1 | 2025-09-24 13:15:00 | 6009.50 | 2025-10-08 09:15:00 | 5856.58 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2025-09-30 12:15:00 | 6019.50 | 2025-10-08 09:15:00 | 5856.58 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest1 | 2025-10-03 09:15:00 | 5977.00 | 2025-10-08 09:15:00 | 5856.58 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-16 12:15:00 | 5970.00 | 2025-10-27 13:15:00 | 5841.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-11-04 15:15:00 | 5912.00 | 2025-11-11 09:15:00 | 5841.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-11 14:15:00 | 5951.50 | 2025-11-12 13:15:00 | 5841.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-01-05 11:15:00 | 6000.00 | 2026-01-09 14:15:00 | 5938.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-01-06 12:15:00 | 6047.00 | 2026-01-09 14:15:00 | 5938.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-04-22 15:15:00 | 5720.00 | 2026-05-05 14:15:00 | 5830.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-27 13:15:00 | 5734.50 | 2026-05-05 14:15:00 | 5830.00 | STOP_HIT | 1.00 | -1.67% |

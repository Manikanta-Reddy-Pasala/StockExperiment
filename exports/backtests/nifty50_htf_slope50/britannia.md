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
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 7 |
| PENDING | 37 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 9 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 19
- **Target hits / Stop hits / Partials:** 0 / 24 / 0
- **Avg / median % per leg:** -0.84% / -1.48%
- **Sum % (uncompounded):** -20.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 5 | 20.8% | 0 | 24 | 0 | -0.84% | -20.2% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 9 | 0 | 0.02% | 0.1% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 0 | 15 | 0 | -1.36% | -20.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 9 | 0 | 0.02% | 0.1% |
| retest2 (combined) | 15 | 1 | 6.7% | 0 | 15 | 0 | -1.36% | -20.4% |

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
| CROSSOVER_SKIP | 2024-03-04 14:15:00 | 4837.60 | 4975.33 | 4975.53 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-03-15 10:15:00 | 4997.35 | 4939.97 | 4955.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 11:15:00 | 4999.90 | 4940.57 | 4955.28 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 4965.95 | 4941.91 | 4955.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-03-18 09:15:00 | 4986.25 | 4942.35 | 4955.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-18 10:15:00 | 4977.00 | 4942.69 | 4955.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-18 11:15:00 | 4992.00 | 4943.18 | 4956.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 12:15:00 | 4991.00 | 4943.66 | 4956.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-18 14:15:00 | 4989.95 | 4944.45 | 4956.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 15:15:00 | 4980.00 | 4944.80 | 4956.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-19 09:15:00 | 4926.15 | 4944.53 | 4956.47 | SL hit qty=1.00 sl=4926.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-19 09:15:00 | 4950.00 | 4944.53 | 4956.47 | SL hit qty=1.00 sl=4950.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-19 09:15:00 | 4950.00 | 4944.53 | 4956.47 | SL hit qty=1.00 sl=4950.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-06 10:15:00 | 5158.80 | 4813.84 | 4856.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:15:00 | 5107.60 | 4816.76 | 4857.81 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-10 10:15:00 | 5067.30 | 4894.82 | 4894.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-10 10:15:00)

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
| CROSSOVER_SKIP | 2024-11-04 15:15:00 | 5625.20 | 5883.74 | 5884.72 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2025-04-09 14:15:00 | 5343.60 | 4913.10 | 4912.23 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-08-19 09:15:00 | 5412.50 | 5574.88 | 5575.51 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2025-08-21 11:15:00)

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
| CROSSOVER_SKIP | 2025-12-03 14:15:00 | 5816.50 | 5877.87 | 5877.91 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-12-05 11:15:00 | 5907.00 | 5876.96 | 5877.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 5915.00 | 5877.34 | 5877.62 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 5941.50 | 5877.98 | 5877.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5877.98 | 5877.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.80 | 5878.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-12-09 10:15:00 | 5849.00 | 5877.80 | 5877.88 | HTF filter: close above htf_sma |

### Cycle 5 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5877.99 | 5877.97 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-12-10 11:15:00 | 5825.50 | 5877.53 | 5877.74 | HTF filter: close above htf_sma |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 5995.50 | 5875.56 | 5876.61 | Break + close above crossover candle high |

### Cycle 6 — BUY (started 2025-12-15 11:15:00)

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
| CROSSOVER_SKIP | 2026-01-29 09:15:00 | 5710.00 | 5935.13 | 5935.29 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2026-02-11 09:15:00 | 6057.50 | 5898.48 | 5913.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 6082.00 | 5900.30 | 5914.19 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 5995.50 | 5927.02 | 5926.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 5995.50 | 5927.02 | 5926.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 6042.50 | 5928.17 | 5927.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5993.00 | 6021.23 | 5982.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 5991.00 | 6020.93 | 5982.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5991.00 | 6020.93 | 5982.27 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2026-03-16 13:15:00 | 5806.50 | 5956.32 | 5957.05 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-10 15:15:00 | 5096.85 | 2024-01-23 13:15:00 | 4972.95 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest1 | 2024-01-19 11:15:00 | 5113.60 | 2024-01-23 13:15:00 | 4972.95 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-01-23 15:15:00 | 5048.10 | 2024-02-07 10:15:00 | 4926.15 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-02-13 10:15:00 | 5011.25 | 2024-02-16 10:15:00 | 4926.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-02-14 15:15:00 | 5010.65 | 2024-02-16 10:15:00 | 4926.15 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-03-15 11:15:00 | 4999.90 | 2024-03-19 09:15:00 | 4926.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-03-18 12:15:00 | 4991.00 | 2024-03-19 09:15:00 | 4950.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-03-18 15:15:00 | 4980.00 | 2024-03-19 09:15:00 | 4950.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-05-06 11:15:00 | 5107.60 | 2024-05-10 10:15:00 | 5067.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest1 | 2024-08-06 10:15:00 | 5813.35 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest1 | 2024-08-09 10:15:00 | 5753.00 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 3.44% |
| BUY | retest1 | 2024-08-09 13:15:00 | 5749.20 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 3.51% |
| BUY | retest1 | 2024-08-20 13:15:00 | 5761.90 | 2024-10-14 09:15:00 | 5950.86 | STOP_HIT | 1.00 | 3.28% |
| BUY | retest2 | 2024-10-15 11:15:00 | 6044.20 | 2024-10-18 09:15:00 | 5930.15 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2025-09-24 13:15:00 | 6009.50 | 2025-10-08 09:15:00 | 5856.58 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2025-09-30 12:15:00 | 6019.50 | 2025-10-08 09:15:00 | 5856.58 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest1 | 2025-10-03 09:15:00 | 5977.00 | 2025-10-08 09:15:00 | 5856.58 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-16 12:15:00 | 5970.00 | 2025-10-27 13:15:00 | 5841.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-11-04 15:15:00 | 5912.00 | 2025-11-11 09:15:00 | 5841.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-11 14:15:00 | 5951.50 | 2025-11-12 13:15:00 | 5841.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-12-05 12:15:00 | 5915.00 | 2025-12-05 13:15:00 | 5941.50 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2026-01-05 11:15:00 | 6000.00 | 2026-01-09 14:15:00 | 5938.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-01-06 12:15:00 | 6047.00 | 2026-01-09 14:15:00 | 5938.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-02-11 10:15:00 | 6082.00 | 2026-02-16 09:15:00 | 5995.50 | STOP_HIT | 1.00 | -1.42% |

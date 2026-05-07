# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 5885.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 21 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 12
- **Target hits / Stop hits / Partials:** 0 / 15 / 0
- **Avg / median % per leg:** -1.38% / -1.64%
- **Sum % (uncompounded):** -20.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 15 | 0 | -1.38% | -20.7% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 5 | 0 | -1.27% | -6.4% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 0 | 10 | 0 | -1.43% | -14.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -1.27% | -6.4% |
| retest2 (combined) | 10 | 1 | 10.0% | 0 | 10 | 0 | -1.43% | -14.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 5343.60 | 4914.34 | 4913.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 5361.05 | 4931.18 | 4922.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 10:15:00 | 5520.00 | 5524.42 | 5407.45 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-20 11:15:00 | 5547.00 | 5524.65 | 5408.14 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 12:15:00 | 5555.00 | 5524.95 | 5408.87 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 12:15:00 | 5562.50 | 5526.40 | 5413.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 13:15:00 | 5580.50 | 5526.94 | 5414.44 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 5615.00 | 5715.84 | 5612.77 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 5610.50 | 5714.79 | 5612.75 | SL hit (close<ema400) qty=1.00 sl=5612.75 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 5610.50 | 5714.79 | 5612.75 | SL hit (close<ema400) qty=1.00 sl=5612.75 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-28 10:15:00 | 5643.50 | 5709.95 | 5612.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 5648.00 | 5709.33 | 5612.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 5586.50 | 5704.54 | 5612.95 | SL hit (close<static) qty=1.00 sl=5600.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-29 15:15:00 | 5650.00 | 5699.42 | 5613.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 5684.00 | 5699.27 | 5613.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-08-05 15:15:00 | 5640.00 | 5715.31 | 5635.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-06 09:15:00 | 5530.50 | 5713.47 | 5635.24 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 5530.50 | 5713.47 | 5635.24 | SL hit (close<static) qty=1.00 sl=5600.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-20 12:15:00 | 5640.00 | 5570.25 | 5573.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:15:00 | 5702.50 | 5571.57 | 5573.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 5662.50 | 5576.69 | 5576.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 5662.50 | 5576.69 | 5576.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5690.00 | 5577.90 | 5576.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 5931.00 | 5948.54 | 5817.33 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-09-24 12:15:00 | 6015.00 | 5948.70 | 5820.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 13:15:00 | 6010.50 | 5949.32 | 5821.60 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-29 15:15:00 | 5965.00 | 5948.99 | 5835.40 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-30 09:15:00 | 5929.00 | 5948.79 | 5835.86 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-30 11:15:00 | 6009.00 | 5949.38 | 5837.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 12:15:00 | 6019.50 | 5950.08 | 5838.19 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-01 15:15:00 | 5973.00 | 5950.92 | 5844.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:15:00 | 5977.00 | 5951.18 | 5844.75 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 2520m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | SL hit (close<ema400) qty=1.00 sl=5856.80 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | SL hit (close<ema400) qty=1.00 sl=5856.80 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | SL hit (close<ema400) qty=1.00 sl=5856.80 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 5904.50 | 5916.99 | 5854.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-16 10:15:00 | 5896.50 | 5916.79 | 5855.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 5944.50 | 5917.07 | 5855.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 5970.00 | 5917.59 | 5856.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 5824.00 | 5955.73 | 5890.29 | SL hit (close<static) qty=1.00 sl=5825.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-04 14:15:00 | 5902.50 | 5922.32 | 5883.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 5912.00 | 5922.22 | 5883.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 5815.00 | 5957.01 | 5905.59 | SL hit (close<static) qty=1.00 sl=5825.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 13:15:00 | 5926.00 | 5954.41 | 5905.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:15:00 | 5952.50 | 5954.39 | 5905.53 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-12 13:15:00 | 5907.00 | 5952.58 | 5906.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-12 14:15:00 | 5886.00 | 5951.91 | 5905.96 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 5800.00 | 5943.90 | 5903.89 | SL hit (close<static) qty=1.00 sl=5825.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-05 11:15:00 | 5907.00 | 5877.09 | 5877.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 5915.00 | 5877.47 | 5877.75 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 5941.50 | 5878.11 | 5878.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5878.11 | 5878.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.92 | 5878.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA400 retest candle locked |

### Cycle 4 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5878.32 | 5878.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 5995.50 | 5875.79 | 5876.80 | Break + close above crossover candle high |

### Cycle 5 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 6010.00 | 5878.55 | 5878.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.66 | 5879.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5977.79 | 5940.20 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 6023.00 | 5978.51 | 5941.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 6000.00 | 5978.72 | 5941.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 6028.00 | 5981.83 | 5944.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 6049.00 | 5982.50 | 5945.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 15:15:00 | 6000.00 | 6002.64 | 5960.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-12 09:15:00 | 5963.50 | 6002.25 | 5960.36 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.13 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.13 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-11 09:15:00 | 6057.50 | 5893.66 | 5909.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 6082.00 | 5895.54 | 5910.82 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-16 10:15:00 | 6042.50 | 5924.46 | 5924.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 6120.00 | 5929.32 | 5927.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5992.00 | 6019.06 | 5979.96 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.06 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-20 12:15:00 | 5555.00 | 2025-07-25 12:15:00 | 5610.50 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest1 | 2025-06-23 13:15:00 | 5580.50 | 2025-07-25 12:15:00 | 5610.50 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-07-28 11:15:00 | 5648.00 | 2025-07-29 09:15:00 | 5586.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-30 09:15:00 | 5684.00 | 2025-08-06 09:15:00 | 5530.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-08-20 13:15:00 | 5702.50 | 2025-08-21 11:15:00 | 5662.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2025-09-24 13:15:00 | 6010.50 | 2025-10-08 09:15:00 | 5844.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2025-09-30 12:15:00 | 6019.50 | 2025-10-08 09:15:00 | 5844.00 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2025-10-03 09:15:00 | 5977.00 | 2025-10-08 09:15:00 | 5844.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-10-16 12:15:00 | 5970.00 | 2025-10-28 12:15:00 | 5824.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-11-04 15:15:00 | 5912.00 | 2025-11-11 09:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-11-11 14:15:00 | 5952.50 | 2025-11-14 09:15:00 | 5800.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-12-05 12:15:00 | 5915.00 | 2025-12-05 13:15:00 | 5941.50 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2026-01-05 11:15:00 | 6000.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-06 12:15:00 | 6049.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-02-11 10:15:00 | 6082.00 | 2026-02-16 11:15:00 | 6071.00 | STOP_HIT | 1.00 | -0.18% |

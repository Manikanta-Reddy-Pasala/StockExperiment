# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 5516.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 5 |
| ALERT3 | 7 |
| PENDING | 25 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 3 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 17
- **Target hits / Stop hits / Partials:** 0 / 17 / 0
- **Avg / median % per leg:** -1.88% / -2.00%
- **Sum % (uncompounded):** -31.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.25% | -18.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.64% | -7.9% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.02% | -10.1% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.54% | -13.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.54% | -13.9% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.64% | -7.9% |
| retest2 (combined) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.71% | -24.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 5371.50 | 5605.30 | 5605.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 5336.00 | 5598.10 | 5602.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 5592.00 | 5569.88 | 5586.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 10:15:00 | 5592.00 | 5569.88 | 5586.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 5592.00 | 5569.88 | 5586.77 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-22 11:15:00 | 5550.00 | 5578.46 | 5590.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 5547.50 | 5578.15 | 5589.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 5544.50 | 5577.57 | 5589.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-22 15:15:00 | 5568.00 | 5577.48 | 5589.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-25 14:15:00 | 5549.50 | 5577.70 | 5589.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 5549.50 | 5577.42 | 5588.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 5690.00 | 5578.54 | 5589.40 | SL hit (close>static) qty=1.00 sl=5609.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 5690.00 | 5578.54 | 5589.40 | SL hit (close>static) qty=1.00 sl=5609.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 5786.00 | 5600.45 | 5600.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 15:15:00 | 5895.00 | 5645.49 | 5623.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 5931.00 | 5948.70 | 5823.58 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-24 12:15:00 | 6015.00 | 5948.86 | 5826.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 13:15:00 | 6010.50 | 5949.47 | 5827.66 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-29 15:15:00 | 5965.00 | 5949.11 | 5840.80 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-30 09:15:00 | 5929.00 | 5948.91 | 5841.24 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-30 11:15:00 | 6009.00 | 5949.49 | 5842.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 12:15:00 | 6019.50 | 5950.19 | 5843.49 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-01 15:15:00 | 5973.00 | 5951.02 | 5849.13 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:15:00 | 5977.00 | 5951.28 | 5849.76 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 2520m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 5844.00 | 5953.81 | 5861.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5844.00 | 5953.81 | 5861.32 | SL hit (close<ema400) qty=1.00 sl=5861.32 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5844.00 | 5953.81 | 5861.32 | SL hit (close<ema400) qty=1.00 sl=5861.32 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 5844.00 | 5953.81 | 5861.32 | SL hit (close<ema400) qty=1.00 sl=5861.32 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 5904.50 | 5917.05 | 5858.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-16 10:15:00 | 5896.50 | 5916.84 | 5858.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 5944.50 | 5917.12 | 5859.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 5970.00 | 5917.65 | 5859.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 5824.00 | 5955.76 | 5893.19 | SL hit (close<static) qty=1.00 sl=5825.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-04 14:15:00 | 5902.50 | 5922.34 | 5885.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 5912.00 | 5922.24 | 5885.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 5815.00 | 5957.03 | 5907.73 | SL hit (close<static) qty=1.00 sl=5825.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 13:15:00 | 5926.00 | 5954.43 | 5907.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:15:00 | 5952.50 | 5954.41 | 5907.62 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-12 13:15:00 | 5907.00 | 5952.59 | 5908.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-12 14:15:00 | 5886.00 | 5951.93 | 5907.97 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 5800.00 | 5943.91 | 5905.82 | SL hit (close<static) qty=1.00 sl=5825.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 5831.00 | 5880.06 | 5880.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 5816.50 | 5877.86 | 5879.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 5878.50 | 5877.12 | 5878.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 5878.50 | 5877.12 | 5878.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 5878.50 | 5877.12 | 5878.77 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-04 13:15:00 | 5860.00 | 5876.83 | 5878.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-04 14:15:00 | 5882.00 | 5876.88 | 5878.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-04 15:15:00 | 5862.00 | 5876.73 | 5878.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-05 09:15:00 | 5877.50 | 5876.74 | 5878.53 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-12-08 12:15:00 | 5844.00 | 5879.67 | 5879.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 13:15:00 | 5846.50 | 5879.34 | 5879.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-09 09:15:00 | 5830.50 | 5878.41 | 5879.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:15:00 | 5849.00 | 5878.12 | 5879.17 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 5882.00 | 5877.46 | 5878.81 | SL hit (close>static) qty=1.00 sl=5879.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 5882.00 | 5877.46 | 5878.81 | SL hit (close>static) qty=1.00 sl=5879.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-10 11:15:00 | 5826.00 | 5877.87 | 5878.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 5820.50 | 5877.30 | 5878.70 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5877.72 | SL hit (close>static) qty=1.00 sl=5879.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-11 13:15:00 | 5844.50 | 5875.05 | 5877.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 5843.50 | 5874.73 | 5877.32 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 5875.00 | 5873.62 | 5876.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5900.00 | 5873.89 | 5876.81 | SL hit (close>static) qty=1.00 sl=5879.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 6032.00 | 5880.08 | 5879.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.66 | 5880.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5977.79 | 5940.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.65 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 6023.00 | 5978.51 | 5942.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 6000.00 | 5978.72 | 5942.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 6028.00 | 5981.83 | 5945.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 6049.00 | 5982.50 | 5945.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 15:15:00 | 6000.00 | 6002.64 | 5960.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-12 09:15:00 | 5963.50 | 6002.25 | 5960.83 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.60 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.60 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.44 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 6120.00 | 5929.32 | 5927.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5992.00 | 6019.06 | 5980.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.20 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5955.73 | 5955.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.33 | 5947.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5669.91 | 5769.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.70 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-21 11:15:00 | 5729.50 | 5676.77 | 5765.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 12:15:00 | 5715.50 | 5677.16 | 5765.27 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-21 14:15:00 | 5830.00 | 5679.28 | 5765.46 | SL hit (close>static) qty=1.00 sl=5767.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 14:15:00 | 5722.50 | 5686.11 | 5765.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:15:00 | 5720.00 | 5686.45 | 5765.75 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-27 13:15:00 | 5733.50 | 5689.59 | 5760.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 5719.50 | 5689.89 | 5759.96 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.42 | SL hit (close>static) qty=1.00 sl=5767.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.42 | SL hit (close>static) qty=1.00 sl=5767.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 5615.50 | 5716.09 | 5758.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 5562.50 | 5714.56 | 5757.27 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-22 12:15:00 | 5547.50 | 2025-08-26 09:15:00 | 5690.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-08-25 15:15:00 | 5549.50 | 2025-08-26 09:15:00 | 5690.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest1 | 2025-09-24 13:15:00 | 6010.50 | 2025-10-08 09:15:00 | 5844.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2025-09-30 12:15:00 | 6019.50 | 2025-10-08 09:15:00 | 5844.00 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2025-10-03 09:15:00 | 5977.00 | 2025-10-08 09:15:00 | 5844.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-10-16 12:15:00 | 5970.00 | 2025-10-28 12:15:00 | 5824.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-11-04 15:15:00 | 5912.00 | 2025-11-11 09:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-11-11 14:15:00 | 5952.50 | 2025-11-14 09:15:00 | 5800.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-12-08 13:15:00 | 5846.50 | 2025-12-09 14:15:00 | 5882.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-09 10:15:00 | 5849.00 | 2025-12-09 14:15:00 | 5882.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-10 12:15:00 | 5820.50 | 2025-12-11 11:15:00 | 5891.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-11 14:15:00 | 5843.50 | 2025-12-12 13:15:00 | 5900.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-05 11:15:00 | 6000.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-06 12:15:00 | 6049.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-04-21 12:15:00 | 5715.50 | 2026-04-21 14:15:00 | 5830.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-04-22 15:15:00 | 5720.00 | 2026-05-04 10:15:00 | 5818.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-27 14:15:00 | 5719.50 | 2026-05-04 10:15:00 | 5818.00 | STOP_HIT | 1.00 | -1.72% |

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
| CROSSOVER | 7 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 16 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 1 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -1.88% / -1.69%
- **Sum % (uncompounded):** -13.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.88% | -13.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.60% | -5.6% |
| SELL @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.26% | -7.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.60% | -5.6% |
| retest2 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.26% | -7.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 5602.10 | 5890.78 | 5890.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 5518.55 | 5823.29 | 5853.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 4928.45 | 4878.91 | 5088.12 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-15 09:15:00 | 4822.95 | 4889.06 | 5066.36 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-15 10:15:00 | 4862.55 | 4888.80 | 5065.34 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-16 09:15:00 | 4824.10 | 4887.35 | 5059.39 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:15:00 | 4800.00 | 4885.74 | 5056.87 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-17 10:15:00 | 4840.00 | 4883.00 | 5050.42 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-17 11:15:00 | 4849.70 | 4882.67 | 5049.42 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 5024.50 | 4890.78 | 5031.02 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-24 09:15:00 | 5068.70 | 4896.35 | 5031.06 | SL hit (close>ema400) qty=1.00 sl=5031.06 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-04 12:15:00 | 5002.60 | 4990.13 | 5050.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-04 13:15:00 | 5012.55 | 4990.36 | 5050.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-05 09:15:00 | 4961.25 | 4990.90 | 5049.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 5004.30 | 4990.83 | 5048.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-03 09:15:00 | 5004.75 | 4836.60 | 4878.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-03 10:15:00 | 5033.15 | 4838.56 | 4879.54 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 5054.15 | 4840.71 | 4880.41 | SL hit (close>static) qty=1.00 sl=5042.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-04 12:15:00 | 4980.30 | 4857.88 | 4887.62 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-04 13:15:00 | 5015.05 | 4859.44 | 4888.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 4950.50 | 4863.60 | 4889.92 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 4973.55 | 4865.71 | 4890.72 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 5057.75 | 4868.55 | 4891.90 | SL hit (close>static) qty=1.00 sl=5042.95 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5412.50 | 5574.88 | 5575.53 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 5816.50 | 5877.86 | 5877.98 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 5833.00 | 5877.67 | 5877.88 | EMA200 below EMA400 |

### Cycle 5 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5826.00 | 5877.86 | 5877.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.50 | 5877.29 | 5877.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-11 15:15:00 | 5833.00 | 5874.31 | 5876.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-12 09:15:00 | 5845.50 | 5874.03 | 5875.99 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-01-21 10:15:00 | 5820.50 | 5971.59 | 5951.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-21 11:15:00 | 5853.00 | 5970.41 | 5951.10 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-21 13:15:00 | 5810.50 | 5967.83 | 5949.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 15:15:00 | 5800.00 | 5964.50 | 5948.50 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 5896.00 | 5963.82 | 5948.24 | SL hit (close>static) qty=1.00 sl=5895.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-23 14:15:00 | 5825.00 | 5956.86 | 5945.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-23 15:15:00 | 5850.00 | 5955.80 | 5945.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 5780.00 | 5949.15 | 5942.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 5733.50 | 5944.83 | 5940.05 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.12 | EMA200 below EMA400 |

### Cycle 7 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5955.73 | 5955.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.33 | 5947.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5669.91 | 5769.58 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-21 11:15:00 | 5729.50 | 5676.77 | 5765.48 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-21 13:15:00 | 5738.00 | 5677.76 | 5765.09 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-22 14:15:00 | 5722.50 | 5686.11 | 5765.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 5709.50 | 5686.68 | 5765.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2026-04-27 13:15:00 | 5733.50 | 5689.59 | 5760.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 5720.00 | 5690.19 | 5759.72 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.38 | SL hit (close>static) qty=1.00 sl=5767.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.38 | SL hit (close>static) qty=1.00 sl=5767.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-16 11:15:00 | 4800.00 | 2025-01-24 09:15:00 | 5068.70 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2025-02-05 11:15:00 | 5004.30 | 2025-04-03 11:15:00 | 5054.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-07 11:15:00 | 4973.55 | 2025-04-07 13:15:00 | 5057.75 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-01-21 15:15:00 | 5800.00 | 2026-01-22 09:15:00 | 5896.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-28 11:15:00 | 5733.50 | 2026-01-29 09:15:00 | 5712.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-04-23 09:15:00 | 5709.50 | 2026-05-04 10:15:00 | 5818.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-04-27 15:15:00 | 5720.00 | 2026-05-04 10:15:00 | 5818.00 | STOP_HIT | 1.00 | -1.71% |

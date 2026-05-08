# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 5520.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 5 |
| PENDING | 13 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -2.36% / -1.90%
- **Sum % (uncompounded):** -16.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.36% | -16.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.36% | -16.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.36% | -16.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 5822.00 | 5908.96 | 5909.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 5804.50 | 5905.26 | 5907.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 5878.50 | 5875.27 | 5889.39 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-04 13:15:00 | 5860.00 | 5875.02 | 5889.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-04 14:15:00 | 5882.00 | 5875.09 | 5889.09 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-04 15:15:00 | 5862.00 | 5874.96 | 5888.95 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-05 09:15:00 | 5877.50 | 5874.98 | 5888.89 | ENTRY1 sustain failed after 1080m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 11:15:00 | 5907.00 | 5875.38 | 5888.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 5907.00 | 5875.38 | 5888.95 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-08 11:15:00 | 5863.00 | 5878.38 | 5890.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:15:00 | 5844.00 | 5878.03 | 5889.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 5955.00 | 5876.68 | 5888.44 | SL hit (close>static) qty=1.00 sl=5911.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-10 11:15:00 | 5825.50 | 5876.25 | 5888.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 5820.00 | 5875.69 | 5887.77 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-11 12:15:00 | 5866.50 | 5873.76 | 5886.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:15:00 | 5844.50 | 5873.47 | 5886.16 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 5913.50 | 5872.96 | 5885.39 | SL hit (close>static) qty=1.00 sl=5911.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 5913.50 | 5872.96 | 5885.39 | SL hit (close>static) qty=1.00 sl=5911.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 6098.50 | 5898.81 | 5897.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 6132.50 | 5984.62 | 5952.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 5986.50 | 6003.76 | 5965.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 12:15:00 | 5964.50 | 6003.37 | 5965.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 5964.50 | 6003.37 | 5965.15 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-09 15:15:00 | 6000.00 | 6002.49 | 5965.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-12 09:15:00 | 5963.50 | 6002.10 | 5965.27 | ENTRY2 sustain failed after 3960m |

### Cycle 3 — SELL (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 13:15:00 | 5753.00 | 5941.10 | 5941.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 15:15:00 | 5742.00 | 5937.26 | 5939.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 5951.00 | 5908.49 | 5923.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 5951.00 | 5908.49 | 5923.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 5951.00 | 5908.49 | 5923.67 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 5822.50 | 5905.09 | 5920.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 5835.50 | 5904.39 | 5920.48 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-09 09:15:00 | 5824.50 | 5903.08 | 5919.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 5812.50 | 5902.17 | 5918.80 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-10 13:15:00 | 5866.50 | 5897.30 | 5915.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-10 14:15:00 | 5871.50 | 5897.04 | 5915.26 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 6057.50 | 5898.41 | 5915.76 | SL hit (close>static) qty=1.00 sl=5965.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 6057.50 | 5898.41 | 5915.76 | SL hit (close>static) qty=1.00 sl=5965.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 6120.00 | 5932.85 | 5932.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 6146.00 | 5946.85 | 5939.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5993.00 | 6021.21 | 5983.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 5991.00 | 6020.91 | 5983.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5991.00 | 6020.91 | 5983.81 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 5798.00 | 5959.56 | 5959.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 5784.00 | 5957.81 | 5958.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5670.05 | 5770.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 5830.00 | 5678.29 | 5767.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 5830.00 | 5678.29 | 5767.74 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-22 14:15:00 | 5722.50 | 5685.06 | 5768.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:15:00 | 5720.00 | 5685.41 | 5767.87 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-27 12:15:00 | 5737.50 | 5688.50 | 5762.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 5734.50 | 5688.96 | 5762.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-04 11:15:00 | 5738.00 | 5692.69 | 5755.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-04 12:15:00 | 5765.50 | 5693.41 | 5755.22 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 5836.50 | 5702.49 | 5757.14 | SL hit (close>static) qty=1.00 sl=5830.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 5836.50 | 5702.49 | 5757.14 | SL hit (close>static) qty=1.00 sl=5830.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 5615.50 | 5715.65 | 5759.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 5562.50 | 5714.12 | 5758.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-08 12:15:00 | 5844.00 | 2025-12-10 09:15:00 | 5955.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-10 12:15:00 | 5820.00 | 2025-12-12 14:15:00 | 5913.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-12-11 13:15:00 | 5844.50 | 2025-12-12 14:15:00 | 5913.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-02-06 10:15:00 | 5835.50 | 2026-02-11 09:15:00 | 6057.50 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2026-02-09 10:15:00 | 5812.50 | 2026-02-11 09:15:00 | 6057.50 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2026-04-22 15:15:00 | 5720.00 | 2026-05-05 14:15:00 | 5836.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-04-27 13:15:00 | 5734.50 | 2026-05-05 14:15:00 | 5836.50 | STOP_HIT | 1.00 | -1.78% |

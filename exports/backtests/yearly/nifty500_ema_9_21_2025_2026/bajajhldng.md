# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1528 bars)
- **Last close:** 10678.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 37 |
| ALERT2 | 37 |
| ALERT2_SKIP | 22 |
| ALERT3 | 80 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 35 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 17
- **Target hits / Stop hits / Partials:** 0 / 37 / 1
- **Avg / median % per leg:** 0.70% / 0.70%
- **Sum % (uncompounded):** 26.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 10 | 66.7% | 0 | 15 | 0 | 1.65% | 24.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.90% | -1.9% |
| BUY @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 0 | 14 | 0 | 1.90% | 26.7% |
| SELL (all) | 23 | 11 | 47.8% | 0 | 22 | 1 | 0.08% | 1.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.97% | -1.0% |
| SELL @ 3rd Alert (retest2) | 22 | 11 | 50.0% | 0 | 21 | 1 | 0.13% | 2.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.44% | -2.9% |
| retest2 (combined) | 36 | 21 | 58.3% | 0 | 35 | 1 | 0.82% | 29.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 12019.00 | 11713.67 | 11676.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 12114.00 | 11793.74 | 11716.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 11915.00 | 11953.38 | 11856.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 11979.00 | 11949.97 | 11870.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 11979.00 | 11949.97 | 11870.82 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 13561.00 | 13709.83 | 13729.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 13475.00 | 13605.50 | 13668.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 13928.00 | 13674.51 | 13657.19 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 13500.00 | 13634.82 | 13645.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 13394.00 | 13586.66 | 13622.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 13324.00 | 13257.78 | 13346.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 13240.00 | 13254.23 | 13336.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 13240.00 | 13254.23 | 13336.91 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 13450.00 | 13367.65 | 13364.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 13485.00 | 13410.68 | 13386.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 14364.00 | 14373.31 | 14251.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 14364.00 | 14373.31 | 14251.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 14229.00 | 14358.51 | 14299.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 14229.00 | 14358.51 | 14299.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 14253.00 | 14337.40 | 14295.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 14253.00 | 14337.40 | 14295.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 14084.00 | 14286.72 | 14276.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 14084.00 | 14286.72 | 14276.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 14081.00 | 14245.58 | 14258.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 14026.00 | 14193.25 | 14231.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 14124.00 | 14068.77 | 14142.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 14124.00 | 14068.77 | 14142.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 14124.00 | 14068.77 | 14142.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 14124.00 | 14068.77 | 14142.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 14020.00 | 14059.01 | 14131.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 13808.00 | 14059.01 | 14131.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:30:00 | 13992.00 | 14013.59 | 14073.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 13980.00 | 14004.87 | 14064.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 13862.00 | 14003.90 | 14058.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 13879.00 | 13978.92 | 14042.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 13811.00 | 13927.39 | 14006.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 13840.00 | 13876.28 | 13960.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:00:00 | 13709.00 | 13750.63 | 13849.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 13781.00 | 13622.01 | 13597.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 13605.00 | 13673.63 | 13635.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 11:15:00 | 13605.00 | 13673.63 | 13635.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 13605.00 | 13673.63 | 13635.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 13605.00 | 13673.63 | 13635.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 13648.00 | 13668.50 | 13637.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:00:00 | 13713.00 | 13679.40 | 13647.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 14191.00 | 14298.75 | 14301.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 14191.00 | 14298.75 | 14301.32 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 14325.00 | 14304.00 | 14303.47 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 14236.00 | 14290.40 | 14297.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 14214.00 | 14275.12 | 14289.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 14262.00 | 14251.73 | 14272.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 14107.00 | 14251.73 | 14272.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 14194.00 | 14240.18 | 14264.93 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 14338.00 | 14265.86 | 14265.56 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 14138.00 | 14240.29 | 14253.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 14017.00 | 14195.63 | 14232.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 13830.00 | 13821.14 | 13918.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 13830.00 | 13821.14 | 13918.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 13814.00 | 13836.85 | 13895.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 13777.00 | 13836.85 | 13895.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 13788.00 | 13832.08 | 13888.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 13781.00 | 13826.86 | 13880.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:00:00 | 13795.00 | 13820.49 | 13873.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 13898.00 | 13810.51 | 13848.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 13876.00 | 13810.51 | 13848.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 13847.00 | 13817.81 | 13848.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:15:00 | 13958.00 | 13817.81 | 13848.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 14070.00 | 13868.24 | 13868.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 14070.00 | 13868.24 | 13868.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 14060.00 | 13906.60 | 13885.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 14060.00 | 13906.60 | 13885.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 14130.00 | 13980.62 | 13925.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 15:15:00 | 14140.00 | 14169.15 | 14076.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:15:00 | 14098.00 | 14169.15 | 14076.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 14096.00 | 14143.78 | 14080.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 14088.00 | 14143.78 | 14080.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 14116.00 | 14138.22 | 14083.90 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 14016.00 | 14077.97 | 14080.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 13975.00 | 14029.00 | 14053.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 13:15:00 | 13980.00 | 13896.19 | 13954.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 13:15:00 | 13980.00 | 13896.19 | 13954.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 13980.00 | 13896.19 | 13954.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:15:00 | 14031.00 | 13896.19 | 13954.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 14052.00 | 13927.35 | 13963.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 14050.00 | 13927.35 | 13963.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 14001.00 | 13942.08 | 13966.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 13984.00 | 13942.08 | 13966.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 13993.00 | 13954.01 | 13967.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 13993.00 | 13954.01 | 13967.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 14071.00 | 13977.41 | 13977.28 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 13950.00 | 13994.48 | 13996.18 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 11:15:00 | 14035.00 | 13999.48 | 13996.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 14185.00 | 14052.29 | 14022.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 14015.00 | 14062.22 | 14033.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 14015.00 | 14062.22 | 14033.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 14015.00 | 14062.22 | 14033.50 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 13820.00 | 14013.78 | 14014.09 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 14091.00 | 13995.40 | 13986.12 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 13779.00 | 13972.23 | 13980.49 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 14001.00 | 13941.57 | 13938.86 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 13881.00 | 13942.64 | 13947.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 13845.00 | 13916.45 | 13934.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 14:15:00 | 13953.00 | 13923.76 | 13936.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 13953.00 | 13923.76 | 13936.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 13953.00 | 13923.76 | 13936.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 13953.00 | 13923.76 | 13936.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 14059.00 | 13950.81 | 13947.39 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 13898.00 | 13936.60 | 13941.30 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 13980.00 | 13945.28 | 13944.82 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 13828.00 | 13924.78 | 13936.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 13800.00 | 13899.83 | 13923.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 13960.00 | 13879.89 | 13908.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 10:15:00 | 13960.00 | 13879.89 | 13908.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 13960.00 | 13879.89 | 13908.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 13960.00 | 13879.89 | 13908.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 14003.00 | 13904.51 | 13917.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 14188.00 | 13904.51 | 13917.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 13903.00 | 13910.11 | 13917.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 13903.00 | 13910.11 | 13917.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 13883.00 | 13904.67 | 13913.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:15:00 | 13843.00 | 13904.95 | 13912.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:30:00 | 13848.00 | 13876.69 | 13897.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 13797.00 | 13855.34 | 13882.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 14171.00 | 13802.57 | 13821.71 | SL hit (close>static) qty=1.00 sl=13976.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 14080.00 | 13858.06 | 13845.19 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 13792.00 | 13827.88 | 13832.66 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 13997.00 | 13847.25 | 13837.30 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 13718.00 | 13843.59 | 13844.50 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 13975.00 | 13846.24 | 13839.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 14059.00 | 13907.96 | 13870.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 12:15:00 | 13888.00 | 13935.15 | 13902.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 12:15:00 | 13888.00 | 13935.15 | 13902.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 13888.00 | 13935.15 | 13902.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 13889.00 | 13935.15 | 13902.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 13855.00 | 13919.12 | 13898.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 13855.00 | 13919.12 | 13898.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 13824.00 | 13884.24 | 13885.13 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 13900.00 | 13887.39 | 13886.48 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 13882.00 | 13885.29 | 13885.62 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 13890.00 | 13886.19 | 13885.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 13934.00 | 13895.75 | 13890.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 14292.00 | 14471.12 | 14303.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 14292.00 | 14471.12 | 14303.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 14292.00 | 14471.12 | 14303.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 14292.00 | 14471.12 | 14303.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 14176.00 | 14412.10 | 14292.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 14200.00 | 14412.10 | 14292.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 14003.00 | 14330.28 | 14265.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 13998.00 | 14330.28 | 14265.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 13795.00 | 14150.54 | 14190.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 13527.00 | 13925.61 | 14069.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 13744.00 | 13738.06 | 13881.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 12:30:00 | 13657.00 | 13704.17 | 13828.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 13790.00 | 13709.60 | 13789.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 13790.00 | 13709.60 | 13789.29 | SL hit (close>ema400) qty=1.00 sl=13789.29 alert=retest1 |

### Cycle 37 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 13056.00 | 12805.37 | 12779.09 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 12875.00 | 12889.70 | 12890.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 12736.00 | 12858.96 | 12876.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 12903.00 | 12847.77 | 12867.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 12903.00 | 12847.77 | 12867.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 12903.00 | 12847.77 | 12867.23 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 12938.00 | 12887.55 | 12882.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 13035.00 | 12917.04 | 12896.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 12950.00 | 12951.94 | 12917.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 13:15:00 | 12972.00 | 13029.61 | 12980.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 12972.00 | 13029.61 | 12980.70 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 13273.00 | 13433.92 | 13450.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 13235.00 | 13374.15 | 13419.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 14:15:00 | 13266.00 | 13265.34 | 13318.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 13314.00 | 13279.81 | 13316.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 13314.00 | 13279.81 | 13316.39 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 15:15:00 | 12145.00 | 11977.23 | 11968.30 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 11900.00 | 12005.90 | 12011.43 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 12112.00 | 12009.89 | 12001.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 12179.00 | 12088.72 | 12053.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 12132.00 | 12203.95 | 12136.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 12132.00 | 12203.95 | 12136.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 12132.00 | 12203.95 | 12136.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 12132.00 | 12203.95 | 12136.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 12139.00 | 12190.96 | 12137.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:15:00 | 12269.00 | 12181.57 | 12137.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 12193.00 | 12170.08 | 12140.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:45:00 | 12174.00 | 12168.47 | 12142.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 12179.00 | 12168.47 | 12142.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 12178.00 | 12170.37 | 12145.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:00:00 | 12197.00 | 12175.70 | 12150.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 14:15:00 | 12258.00 | 12176.45 | 12155.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 12330.00 | 12180.37 | 12160.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 12750.00 | 12883.02 | 12925.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 12:15:00 | 12164.00 | 12132.80 | 12239.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:45:00 | 12151.00 | 12132.80 | 12239.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 12343.00 | 12167.23 | 12221.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 12343.00 | 12167.23 | 12221.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 12675.00 | 12268.78 | 12262.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 11:15:00 | 12882.00 | 12391.43 | 12318.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 12623.00 | 12782.44 | 12578.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 12623.00 | 12782.44 | 12578.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 12708.00 | 12767.55 | 12590.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 12699.00 | 12767.55 | 12590.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 12560.00 | 12689.78 | 12606.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 12560.00 | 12689.78 | 12606.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 12650.00 | 12681.83 | 12610.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 13022.00 | 12681.83 | 12610.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 12736.00 | 12690.21 | 12638.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 12742.00 | 12709.17 | 12652.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 12250.00 | 12583.05 | 12605.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 12250.00 | 12583.05 | 12605.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 12127.00 | 12307.47 | 12443.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 12081.00 | 12071.49 | 12210.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 12076.00 | 12071.49 | 12210.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 12227.00 | 12060.65 | 12129.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 12227.00 | 12060.65 | 12129.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 12410.00 | 12130.52 | 12155.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 12410.00 | 12130.52 | 12155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 12434.00 | 12191.21 | 12180.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 12:15:00 | 12539.00 | 12260.77 | 12213.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 12312.00 | 12383.99 | 12298.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:00:00 | 12312.00 | 12383.99 | 12298.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 12322.00 | 12371.59 | 12301.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 12319.00 | 12371.59 | 12301.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 12250.00 | 12347.27 | 12296.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 12254.00 | 12347.27 | 12296.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 12190.00 | 12315.82 | 12286.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 12190.00 | 12315.82 | 12286.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 12105.00 | 12254.16 | 12262.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 12045.00 | 12193.42 | 12232.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 15:15:00 | 11737.00 | 11727.02 | 11813.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:15:00 | 11755.00 | 11727.02 | 11813.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 11696.00 | 11720.82 | 11802.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 11650.00 | 11720.82 | 11802.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:00:00 | 11649.00 | 11675.17 | 11721.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:30:00 | 11655.00 | 11634.13 | 11691.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 11463.00 | 11634.13 | 11691.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 11576.00 | 11539.68 | 11611.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 11651.00 | 11539.68 | 11611.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 11643.00 | 11569.84 | 11613.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 11643.00 | 11569.84 | 11613.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 11635.00 | 11582.87 | 11615.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 11639.00 | 11582.87 | 11615.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 11651.00 | 11596.50 | 11618.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 11725.00 | 11639.69 | 11628.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 11725.00 | 11639.69 | 11628.11 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 11520.00 | 11615.76 | 11618.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 11436.00 | 11579.80 | 11601.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 11515.00 | 11343.75 | 11426.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 12:15:00 | 11515.00 | 11343.75 | 11426.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 11515.00 | 11343.75 | 11426.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 11536.00 | 11343.75 | 11426.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 11572.00 | 11389.40 | 11440.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:45:00 | 11570.00 | 11389.40 | 11440.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 11502.00 | 11433.45 | 11452.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 11743.00 | 11433.45 | 11452.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 11640.00 | 11474.76 | 11469.63 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 11351.00 | 11451.74 | 11462.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 11325.00 | 11418.76 | 11444.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 11132.00 | 11098.61 | 11194.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 11132.00 | 11098.61 | 11194.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 11167.00 | 11116.19 | 11185.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 11167.00 | 11116.19 | 11185.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 11300.00 | 11152.95 | 11196.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 11223.00 | 11152.95 | 11196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 11353.00 | 11192.96 | 11210.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 11365.00 | 11192.96 | 11210.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 11323.00 | 11218.97 | 11220.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 11347.00 | 11218.97 | 11220.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 11270.00 | 11229.18 | 11225.18 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 11180.00 | 11219.34 | 11221.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 11144.00 | 11204.27 | 11214.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 11178.00 | 11135.42 | 11170.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 11:15:00 | 11178.00 | 11135.42 | 11170.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 11178.00 | 11135.42 | 11170.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 11178.00 | 11135.42 | 11170.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 11195.00 | 11147.33 | 11172.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 11243.00 | 11147.33 | 11172.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 11188.00 | 11155.47 | 11173.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 11140.00 | 11161.77 | 11175.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 11115.00 | 11164.42 | 11175.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 11062.00 | 11008.73 | 11003.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 11062.00 | 11008.73 | 11003.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 11114.00 | 11031.27 | 11014.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 11065.00 | 11071.65 | 11045.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 11065.00 | 11071.65 | 11045.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 11065.00 | 11071.65 | 11045.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 11101.00 | 11045.12 | 11041.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 11015.00 | 11047.09 | 11044.72 | SL hit (close<static) qty=1.00 sl=11031.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 11027.00 | 11043.07 | 11043.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 10980.00 | 11030.46 | 11037.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 11022.00 | 10998.49 | 11018.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 11022.00 | 10998.49 | 11018.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 11022.00 | 10998.49 | 11018.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 11027.00 | 10998.49 | 11018.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 11169.00 | 11032.59 | 11032.22 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 9288.00 | 10753.76 | 10919.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 9270.00 | 10223.13 | 10634.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 12:15:00 | 8945.50 | 8941.71 | 9326.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:45:00 | 8951.00 | 8941.71 | 9326.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 9061.50 | 8941.15 | 9054.62 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 9134.50 | 9085.34 | 9081.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 9169.50 | 9111.64 | 9094.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 9901.00 | 9910.16 | 9730.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 10039.50 | 9910.16 | 9730.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 9960.00 | 9979.98 | 9872.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 9849.00 | 9925.75 | 9885.56 | SL hit (close<ema400) qty=1.00 sl=9885.56 alert=retest1 |

### Cycle 60 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 10275.00 | 10358.47 | 10369.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 10266.00 | 10339.97 | 10359.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 10366.00 | 10326.29 | 10346.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 10408.50 | 10342.73 | 10352.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 10408.50 | 10342.73 | 10352.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 10399.00 | 10359.23 | 10358.42 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 10315.00 | 10353.63 | 10356.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 10279.50 | 10333.34 | 10345.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:15:00 | 10427.00 | 10311.61 | 10328.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 10382.50 | 10325.79 | 10333.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 10401.00 | 10325.79 | 10333.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 10399.00 | 10340.43 | 10339.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 10434.50 | 10359.24 | 10348.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 10305.50 | 10348.49 | 10344.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 10301.00 | 10339.00 | 10340.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 10170.00 | 10305.36 | 10324.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 10260.50 | 10250.12 | 10287.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:45:00 | 10275.00 | 10250.12 | 10287.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 10236.00 | 10247.30 | 10282.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 10287.50 | 10247.30 | 10282.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 10241.00 | 10246.04 | 10278.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 10532.00 | 10246.04 | 10278.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 10646.00 | 10326.03 | 10312.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 10759.00 | 10486.46 | 10417.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 10539.00 | 10539.20 | 10463.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 13:00:00 | 10539.00 | 10539.20 | 10463.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 10525.00 | 10583.50 | 10548.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 10517.00 | 10583.50 | 10548.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 10625.00 | 10591.80 | 10555.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 10535.00 | 10591.80 | 10555.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 10772.00 | 10639.22 | 10587.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:30:00 | 10703.00 | 10639.22 | 10587.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 10607.00 | 10632.78 | 10589.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 10607.00 | 10632.78 | 10589.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 10678.00 | 10641.82 | 10597.34 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-16 09:15:00 | 13808.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-06-16 13:30:00 | 13992.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-06-16 14:45:00 | 13980.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 2.15% |
| SELL | retest2 | 2025-06-17 09:15:00 | 13862.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2025-06-17 11:45:00 | 13811.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-06-17 14:30:00 | 13840.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-06-18 14:00:00 | 13709.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-06-25 15:00:00 | 13713.00 | 2025-07-03 15:15:00 | 14191.00 | STOP_HIT | 1.00 | 3.49% |
| SELL | retest2 | 2025-07-11 10:15:00 | 13777.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-11 11:15:00 | 13788.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-11 12:15:00 | 13781.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-07-11 13:00:00 | 13795.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-05 12:15:00 | 13843.00 | 2025-08-07 09:15:00 | 14171.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-05 13:30:00 | 13848.00 | 2025-08-07 09:15:00 | 14171.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-06 10:00:00 | 13797.00 | 2025-08-07 09:15:00 | 14171.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest1 | 2025-08-21 12:30:00 | 13657.00 | 2025-08-22 09:15:00 | 13790.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-22 11:15:00 | 13682.00 | 2025-08-26 14:15:00 | 12997.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 11:15:00 | 13682.00 | 2025-09-02 13:15:00 | 12680.00 | STOP_HIT | 0.50 | 7.32% |
| BUY | retest2 | 2025-10-14 14:15:00 | 12269.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 4.41% |
| BUY | retest2 | 2025-10-15 09:15:00 | 12193.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.06% |
| BUY | retest2 | 2025-10-15 09:45:00 | 12174.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.22% |
| BUY | retest2 | 2025-10-15 10:15:00 | 12179.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.18% |
| BUY | retest2 | 2025-10-15 12:00:00 | 12197.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.03% |
| BUY | retest2 | 2025-10-15 14:15:00 | 12258.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 4.50% |
| BUY | retest2 | 2025-10-16 09:15:00 | 12330.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2025-11-10 09:15:00 | 13022.00 | 2025-11-11 09:15:00 | 12250.00 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest2 | 2025-11-10 12:45:00 | 12736.00 | 2025-11-11 09:15:00 | 12250.00 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2025-11-10 13:45:00 | 12742.00 | 2025-11-11 09:15:00 | 12250.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-11-21 10:15:00 | 11650.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-24 12:00:00 | 11649.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-11-24 14:30:00 | 11655.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-24 15:00:00 | 11463.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-12-05 14:30:00 | 11140.00 | 2025-12-15 14:15:00 | 11062.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-12-08 09:15:00 | 11115.00 | 2025-12-15 14:15:00 | 11062.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-12-17 15:15:00 | 11101.00 | 2025-12-18 11:15:00 | 11015.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest1 | 2026-04-10 09:15:00 | 10039.50 | 2026-04-13 14:15:00 | 9849.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-15 09:15:00 | 10062.50 | 2026-04-24 12:15:00 | 10275.00 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2026-04-15 14:30:00 | 10058.50 | 2026-04-24 12:15:00 | 10275.00 | STOP_HIT | 1.00 | 2.15% |

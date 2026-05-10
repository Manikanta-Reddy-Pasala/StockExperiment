# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 10678.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 89 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 29 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 62 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 31
- **Target hits / Stop hits / Partials:** 8 / 58 / 13
- **Avg / median % per leg:** 2.15% / 2.11%
- **Sum % (uncompounded):** 169.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 13 | 59.1% | 2 | 20 | 0 | 1.65% | 36.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.70% | -3.4% |
| BUY @ 3rd Alert (retest2) | 20 | 13 | 65.0% | 2 | 18 | 0 | 1.99% | 39.8% |
| SELL (all) | 57 | 35 | 61.4% | 6 | 38 | 13 | 2.34% | 133.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.96% | -1.9% |
| SELL @ 3rd Alert (retest2) | 55 | 35 | 63.6% | 6 | 36 | 13 | 2.46% | 135.3% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.33% | -5.3% |
| retest2 (combined) | 75 | 48 | 64.0% | 8 | 54 | 13 | 2.33% | 175.0% |

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
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 11915.00 | 11953.38 | 11856.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 11979.00 | 11949.97 | 11870.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 12055.00 | 11971.14 | 11894.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 12069.00 | 11990.61 | 11923.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-16 13:15:00 | 13260.50 | 12906.30 | 12561.41 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-16 13:15:00 | 13275.90 | 12906.30 | 12561.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 13561.00 | 13709.83 | 13729.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 13475.00 | 13605.50 | 13668.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 13740.00 | 13521.42 | 13596.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 13970.00 | 13611.14 | 13630.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 13970.00 | 13611.14 | 13630.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

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
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 13324.00 | 13257.78 | 13346.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 13240.00 | 13254.23 | 13336.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 13302.00 | 13254.23 | 13336.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 13272.00 | 13257.78 | 13331.01 | EMA400 retest candle locked (from downside) |

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
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
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
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 14060.00 | 13906.60 | 13885.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 14060.00 | 13906.60 | 13885.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
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
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 14171.00 | 13802.57 | 13821.71 | SL hit (close>static) qty=1.00 sl=13976.00 alert=retest2 |
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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 13682.00 | 13711.08 | 13782.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 14:15:00 | 12997.90 | 13206.04 | 13359.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 12680.00 | 12645.97 | 12747.73 | SL hit (close>ema200) qty=0.50 sl=12645.97 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 12945.00 | 12847.77 | 12867.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 12921.00 | 12862.42 | 12872.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 12921.00 | 12862.42 | 12872.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 12925.00 | 12874.93 | 12876.92 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 12938.00 | 12887.55 | 12882.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 13035.00 | 12917.04 | 12896.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 12950.00 | 12951.94 | 12917.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 13170.00 | 12951.94 | 12917.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 12972.00 | 13029.61 | 12980.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 12972.00 | 13029.61 | 12980.70 | SL hit (close<ema400) qty=1.00 sl=12980.70 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 13099.00 | 13027.69 | 12984.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 13273.00 | 13433.92 | 13450.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 13273.00 | 13433.92 | 13450.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 13235.00 | 13374.15 | 13419.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 14:15:00 | 13266.00 | 13265.34 | 13318.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 13265.00 | 13265.34 | 13318.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 13314.00 | 13279.81 | 13316.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 13314.00 | 13279.81 | 13316.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 13253.00 | 13274.45 | 13310.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 13230.00 | 13274.45 | 13310.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 13242.00 | 13273.20 | 13301.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 13236.00 | 13269.36 | 13297.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 13242.00 | 13269.36 | 13297.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 13233.00 | 13257.71 | 13286.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 13183.00 | 13241.69 | 13274.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 13149.00 | 13194.46 | 13238.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12568.50 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12579.90 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12574.20 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12579.90 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 12857.00 | 12790.15 | 12910.64 | SL hit (close>ema200) qty=0.50 sl=12790.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 12857.00 | 12790.15 | 12910.64 | SL hit (close>ema200) qty=0.50 sl=12790.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 12857.00 | 12790.15 | 12910.64 | SL hit (close>ema200) qty=0.50 sl=12790.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 12857.00 | 12790.15 | 12910.64 | SL hit (close>ema200) qty=0.50 sl=12790.15 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 12523.85 | 12700.45 | 12847.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 12491.55 | 12700.45 | 12847.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-03 09:15:00 | 11864.70 | 12041.97 | 12249.43 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-03 09:15:00 | 11834.10 | 12041.97 | 12249.43 | Target hit (10%) qty=0.50 alert=retest2 |

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
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
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
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 12250.00 | 12583.05 | 12605.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
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
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 11725.00 | 11639.69 | 11628.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 11725.00 | 11639.69 | 11628.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
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
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 11300.00 | 11194.56 | 11132.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 11223.00 | 11237.57 | 11181.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 13:15:00 | 11227.00 | 11239.05 | 11218.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 11227.00 | 11239.05 | 11218.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 11215.00 | 11239.05 | 11218.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 11228.00 | 11236.84 | 11219.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 11220.00 | 11236.84 | 11219.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 11220.00 | 11233.47 | 11219.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 11234.00 | 11233.47 | 11219.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 11255.00 | 11237.78 | 11222.65 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 11160.00 | 11205.27 | 11210.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 11152.00 | 11194.61 | 11205.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 11277.00 | 11207.15 | 11209.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 11277.00 | 11207.15 | 11209.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 11277.00 | 11207.15 | 11209.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 11284.00 | 11207.15 | 11209.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 11205.00 | 11206.72 | 11208.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 11190.00 | 11206.72 | 11208.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 11233.00 | 11211.98 | 11210.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 11233.00 | 11211.98 | 11210.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 11254.00 | 11224.13 | 11216.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 09:15:00 | 11190.00 | 11219.20 | 11216.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 11190.00 | 11219.20 | 11216.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 11190.00 | 11219.20 | 11216.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 11211.00 | 11219.20 | 11216.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 11159.00 | 11207.16 | 11210.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 11104.00 | 11170.58 | 11191.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 11295.00 | 11141.13 | 11163.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 11295.00 | 11141.13 | 11163.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 11295.00 | 11141.13 | 11163.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 11295.00 | 11141.13 | 11163.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 11369.00 | 11186.71 | 11182.26 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 11090.00 | 11224.94 | 11237.88 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 11266.00 | 11221.82 | 11218.11 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 11161.00 | 11215.40 | 11216.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 11128.00 | 11189.22 | 11203.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 11214.00 | 11194.17 | 11204.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 11214.00 | 11194.17 | 11204.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 11214.00 | 11194.17 | 11204.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 11214.00 | 11194.17 | 11204.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 11180.00 | 11191.34 | 11202.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 11075.00 | 11191.34 | 11202.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 11145.00 | 11182.07 | 11197.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 11037.00 | 11128.06 | 11166.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 11516.00 | 11228.83 | 11202.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 11516.00 | 11228.83 | 11202.32 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 11161.00 | 11199.52 | 11204.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 11043.00 | 11133.60 | 11168.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 10797.00 | 10749.62 | 10846.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 10649.00 | 10723.90 | 10825.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 10755.00 | 10730.12 | 10819.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 10709.00 | 10730.12 | 10819.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 10707.00 | 10746.79 | 10791.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 10676.00 | 10724.25 | 10769.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 10665.00 | 10708.68 | 10753.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 10523.00 | 10704.74 | 10748.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 10750.00 | 10703.12 | 10739.13 | SL hit (close>ema400) qty=1.00 sl=10739.13 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 10678.00 | 10701.31 | 10732.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 10600.00 | 10583.73 | 10632.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 10681.00 | 10637.16 | 10636.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 10681.00 | 10637.16 | 10636.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 10681.00 | 10637.16 | 10636.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 10681.00 | 10637.16 | 10636.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 10681.00 | 10637.16 | 10636.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 10727.00 | 10666.14 | 10650.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 10628.00 | 10686.30 | 10667.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 10628.00 | 10686.30 | 10667.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 10628.00 | 10686.30 | 10667.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 10628.00 | 10686.30 | 10667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 10688.00 | 10686.64 | 10669.39 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 10525.00 | 10644.55 | 10654.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 10685.00 | 10631.65 | 10631.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 10840.00 | 10732.33 | 10693.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 10749.00 | 10767.36 | 10734.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 10749.00 | 10767.36 | 10734.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 10749.00 | 10767.36 | 10734.56 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 10608.00 | 10708.78 | 10715.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 10555.00 | 10678.02 | 10700.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 10562.00 | 10561.23 | 10619.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 10562.00 | 10561.23 | 10619.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 10660.00 | 10580.98 | 10623.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 10660.00 | 10580.98 | 10623.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 10640.00 | 10592.79 | 10625.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 10811.00 | 10592.79 | 10625.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 10927.00 | 10695.50 | 10668.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 10974.00 | 10817.07 | 10736.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 10889.00 | 10952.49 | 10867.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 14:15:00 | 10889.00 | 10952.49 | 10867.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 10889.00 | 10952.49 | 10867.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 11125.00 | 10936.99 | 10868.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 10825.00 | 10953.77 | 10924.77 | SL hit (close<static) qty=1.00 sl=10827.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 10696.00 | 10902.22 | 10903.98 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 10999.00 | 10901.13 | 10892.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 11044.00 | 10929.70 | 10906.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 11040.00 | 11050.60 | 11004.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:00:00 | 11040.00 | 11050.60 | 11004.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 11060.00 | 11061.28 | 11025.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:30:00 | 11158.00 | 11082.03 | 11037.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 11165.00 | 11082.03 | 11037.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 10852.00 | 11023.58 | 11046.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 10852.00 | 11023.58 | 11046.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 10852.00 | 11023.58 | 11046.96 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 11109.00 | 11014.88 | 11009.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 11208.00 | 11053.51 | 11027.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 11300.00 | 11320.25 | 11238.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:30:00 | 11280.00 | 11320.25 | 11238.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 11241.00 | 11296.41 | 11252.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 11241.00 | 11296.41 | 11252.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 11167.00 | 11270.53 | 11244.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 11167.00 | 11270.53 | 11244.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 11121.00 | 11240.62 | 11233.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 11000.00 | 11240.62 | 11233.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 11065.00 | 11205.50 | 11218.06 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 11299.00 | 11203.92 | 11203.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 11321.00 | 11227.34 | 11214.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 11358.00 | 11426.08 | 11356.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 12:15:00 | 11358.00 | 11426.08 | 11356.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 11358.00 | 11426.08 | 11356.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 11358.00 | 11426.08 | 11356.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 11326.00 | 11406.07 | 11353.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 11290.00 | 11406.07 | 11353.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 11321.00 | 11389.05 | 11350.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 11348.00 | 11389.05 | 11350.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 11347.00 | 11380.64 | 11350.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 11319.00 | 11360.51 | 11343.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 11229.00 | 11334.21 | 11333.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 11250.00 | 11334.21 | 11333.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 11205.00 | 11308.37 | 11321.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 11174.00 | 11281.50 | 11308.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 10685.00 | 10675.18 | 10817.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:45:00 | 10665.00 | 10675.18 | 10817.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 10630.00 | 10632.49 | 10707.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 10626.00 | 10635.79 | 10702.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 14:00:00 | 10598.00 | 10636.00 | 10686.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 10620.00 | 10644.25 | 10675.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 10615.00 | 10638.40 | 10669.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 10347.00 | 10241.52 | 10328.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 10360.00 | 10241.52 | 10328.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 10282.00 | 10249.62 | 10324.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 10262.00 | 10249.62 | 10324.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 10215.00 | 10237.29 | 10311.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10094.70 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10068.10 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10089.00 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10084.25 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 9748.90 | 9885.97 | 10045.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 9704.25 | 9885.97 | 10045.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 13:15:00 | 9563.40 | 9751.36 | 9926.43 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-13 13:15:00 | 9558.00 | 9751.36 | 9926.43 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-13 13:15:00 | 9553.50 | 9751.36 | 9926.43 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-13 14:15:00 | 9538.20 | 9705.09 | 9889.48 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 13:15:00 | 9538.00 | 9525.40 | 9702.02 | SL hit (close>ema200) qty=0.50 sl=9525.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 13:15:00 | 9538.00 | 9525.40 | 9702.02 | SL hit (close>ema200) qty=0.50 sl=9525.40 alert=retest2 |

### Cycle 79 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 9870.00 | 9711.15 | 9700.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 9895.00 | 9772.86 | 9732.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 9707.00 | 9796.10 | 9759.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 9707.00 | 9796.10 | 9759.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 9707.00 | 9796.10 | 9759.97 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 9658.00 | 9732.44 | 9739.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 9606.00 | 9686.87 | 9712.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 9276.00 | 9270.25 | 9410.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 9294.00 | 9270.25 | 9410.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 9392.00 | 9302.24 | 9401.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 9413.00 | 9302.24 | 9401.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 9363.00 | 9314.39 | 9398.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 9403.00 | 9314.39 | 9398.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 9650.00 | 9392.44 | 9413.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 9650.00 | 9392.44 | 9413.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 9625.00 | 9438.95 | 9432.90 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 9292.00 | 9434.83 | 9447.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 9220.00 | 9349.67 | 9401.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 8942.00 | 8910.37 | 9080.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 8942.00 | 8910.37 | 9080.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 8942.00 | 8910.37 | 9080.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 8860.00 | 8898.80 | 9059.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 8850.00 | 8924.54 | 9022.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 8854.50 | 8907.95 | 8942.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 9041.50 | 8973.60 | 8967.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 9041.50 | 8973.60 | 8967.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 9041.50 | 8973.60 | 8967.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 9041.50 | 8973.60 | 8967.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 9059.50 | 8990.78 | 8976.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 9901.00 | 9910.15 | 9722.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 10039.50 | 9910.15 | 9722.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 9960.00 | 9979.97 | 9869.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 9849.00 | 9925.75 | 9883.33 | SL hit (close<ema400) qty=1.00 sl=9883.33 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 10062.50 | 9909.40 | 9879.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 14:30:00 | 10058.50 | 10011.65 | 9954.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 10275.00 | 10358.47 | 10369.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 10275.00 | 10358.47 | 10369.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 10275.00 | 10358.47 | 10369.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 10266.00 | 10339.97 | 10359.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 10366.00 | 10326.29 | 10346.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 10408.50 | 10342.73 | 10352.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 10408.50 | 10342.73 | 10352.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 10399.00 | 10359.23 | 10358.42 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 10315.00 | 10353.63 | 10356.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 10279.50 | 10333.34 | 10345.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:15:00 | 10427.00 | 10311.61 | 10328.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 10382.50 | 10325.79 | 10333.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 10401.00 | 10325.79 | 10333.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 10399.00 | 10340.43 | 10339.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 10434.50 | 10359.24 | 10348.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 10305.50 | 10348.49 | 10344.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2026-04-29 14:15:00)

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

### Cycle 89 — BUY (started 2026-05-04 09:15:00)

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
| BUY | retest2 | 2025-05-14 12:00:00 | 12055.00 | 2025-05-16 13:15:00 | 13260.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 15:15:00 | 12069.00 | 2025-05-16 13:15:00 | 13275.90 | TARGET_HIT | 1.00 | 10.00% |
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
| BUY | retest1 | 2025-09-10 09:15:00 | 13170.00 | 2025-09-10 13:15:00 | 12972.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-09-10 15:15:00 | 13099.00 | 2025-09-19 10:15:00 | 13273.00 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2025-09-23 11:15:00 | 13230.00 | 2025-09-26 15:15:00 | 12568.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:15:00 | 13242.00 | 2025-09-26 15:15:00 | 12579.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13236.00 | 2025-09-26 15:15:00 | 12574.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:15:00 | 13242.00 | 2025-09-26 15:15:00 | 12579.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:15:00 | 13230.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-09-23 14:15:00 | 13242.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13236.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-09-23 15:15:00 | 13242.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-09-24 12:00:00 | 13183.00 | 2025-09-29 11:15:00 | 12523.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 13149.00 | 2025-09-29 11:15:00 | 12491.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 12:00:00 | 13183.00 | 2025-10-03 09:15:00 | 11864.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 13149.00 | 2025-10-03 09:15:00 | 11834.10 | TARGET_HIT | 0.50 | 10.00% |
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
| SELL | retest2 | 2025-12-29 11:15:00 | 11190.00 | 2025-12-29 11:15:00 | 11233.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-07 12:30:00 | 11037.00 | 2026-01-08 09:15:00 | 11516.00 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest1 | 2026-01-14 09:45:00 | 10649.00 | 2026-01-19 10:15:00 | 10750.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-16 13:00:00 | 10676.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-01-16 15:00:00 | 10665.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-01-19 09:15:00 | 10523.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-19 13:15:00 | 10678.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-02-05 09:15:00 | 11125.00 | 2026-02-05 15:15:00 | 10825.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-02-11 10:30:00 | 11158.00 | 2026-02-13 09:15:00 | 10852.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-02-11 11:00:00 | 11165.00 | 2026-02-13 09:15:00 | 10852.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-03-05 10:30:00 | 10626.00 | 2026-03-12 09:15:00 | 10094.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 14:00:00 | 10598.00 | 2026-03-12 09:15:00 | 10068.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:30:00 | 10620.00 | 2026-03-12 09:15:00 | 10089.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:00:00 | 10615.00 | 2026-03-12 09:15:00 | 10084.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 10262.00 | 2026-03-13 09:15:00 | 9748.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:45:00 | 10215.00 | 2026-03-13 09:15:00 | 9704.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:30:00 | 10626.00 | 2026-03-13 13:15:00 | 9563.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-05 14:00:00 | 10598.00 | 2026-03-13 13:15:00 | 9558.00 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2026-03-06 10:30:00 | 10620.00 | 2026-03-13 13:15:00 | 9553.50 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2026-03-06 12:00:00 | 10615.00 | 2026-03-13 14:15:00 | 9538.20 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2026-03-11 10:15:00 | 10262.00 | 2026-03-16 13:15:00 | 9538.00 | STOP_HIT | 0.50 | 7.06% |
| SELL | retest2 | 2026-03-11 10:45:00 | 10215.00 | 2026-03-16 13:15:00 | 9538.00 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-04-01 10:30:00 | 8860.00 | 2026-04-06 11:15:00 | 9041.50 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-01 15:15:00 | 8850.00 | 2026-04-06 11:15:00 | 9041.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-04-06 09:15:00 | 8854.50 | 2026-04-06 11:15:00 | 9041.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2026-04-10 09:15:00 | 10039.50 | 2026-04-13 14:15:00 | 9849.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-15 09:15:00 | 10062.50 | 2026-04-24 12:15:00 | 10275.00 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2026-04-15 14:30:00 | 10058.50 | 2026-04-24 12:15:00 | 10275.00 | STOP_HIT | 1.00 | 2.15% |

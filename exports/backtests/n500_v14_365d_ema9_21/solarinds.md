# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 16101.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 44 |
| ALERT2 | 45 |
| ALERT2_SKIP | 29 |
| ALERT3 | 137 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 51 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 22 / 35
- **Target hits / Stop hits / Partials:** 2 / 47 / 8
- **Avg / median % per leg:** 1.19% / -0.63%
- **Sum % (uncompounded):** 67.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 5 | 23.8% | 1 | 20 | 0 | 0.23% | 4.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 5 | 23.8% | 1 | 20 | 0 | 0.23% | 4.9% |
| SELL (all) | 36 | 17 | 47.2% | 1 | 27 | 8 | 1.75% | 62.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 17 | 47.2% | 1 | 27 | 8 | 1.75% | 62.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 22 | 38.6% | 2 | 47 | 8 | 1.19% | 67.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 13705.00 | 13847.28 | 13852.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 13644.00 | 13784.66 | 13821.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 13743.00 | 13681.46 | 13740.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 13743.00 | 13681.46 | 13740.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 13743.00 | 13681.46 | 13740.77 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 14293.00 | 13852.56 | 13807.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 14652.00 | 14152.43 | 13975.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 15699.00 | 15761.81 | 15426.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 12:00:00 | 15699.00 | 15761.81 | 15426.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 16013.00 | 16143.60 | 16004.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 16013.00 | 16143.60 | 16004.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 16085.00 | 16131.88 | 16011.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 16001.00 | 16131.88 | 16011.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 16079.00 | 16121.30 | 16017.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 16079.00 | 16121.30 | 16017.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 16015.00 | 16100.04 | 16017.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:45:00 | 16014.00 | 16100.04 | 16017.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 15970.00 | 16074.03 | 16013.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 16019.00 | 16074.03 | 16013.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 16048.00 | 16085.83 | 16024.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 16515.00 | 16667.78 | 16681.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 16515.00 | 16667.78 | 16681.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 16515.00 | 16667.78 | 16681.87 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 16926.00 | 16691.12 | 16682.49 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 16616.00 | 16780.37 | 16783.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 16599.00 | 16744.10 | 16766.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 16747.00 | 16677.19 | 16719.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 16747.00 | 16677.19 | 16719.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 16747.00 | 16677.19 | 16719.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 16790.00 | 16677.19 | 16719.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 16653.00 | 16672.35 | 16713.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 16612.00 | 16658.28 | 16703.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:45:00 | 16625.00 | 16654.63 | 16697.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:15:00 | 16612.00 | 16654.63 | 16697.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:00:00 | 16601.00 | 16643.90 | 16689.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | SL hit (close>static) qty=1.00 sl=16752.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | SL hit (close>static) qty=1.00 sl=16752.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | SL hit (close>static) qty=1.00 sl=16752.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | SL hit (close>static) qty=1.00 sl=16752.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 16776.00 | 16673.29 | 16691.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 16810.00 | 16700.63 | 16702.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 16795.00 | 16700.63 | 16702.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 11:15:00 | 16875.00 | 16735.51 | 16717.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 16920.00 | 16814.64 | 16765.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 17063.00 | 17115.55 | 17036.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 11:15:00 | 17063.00 | 17115.55 | 17036.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 17063.00 | 17115.55 | 17036.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 17095.00 | 17115.55 | 17036.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 17061.00 | 17104.64 | 17038.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 17000.00 | 17104.64 | 17038.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 17101.00 | 17103.91 | 17044.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 17174.00 | 17096.31 | 17051.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 16945.00 | 17067.81 | 17050.46 | SL hit (close<static) qty=1.00 sl=17032.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 16820.00 | 17018.25 | 17029.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 16798.00 | 16916.01 | 16972.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 16929.00 | 16918.61 | 16968.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 16929.00 | 16918.61 | 16968.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 16929.00 | 16918.61 | 16968.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 16929.00 | 16918.61 | 16968.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 16958.00 | 16926.49 | 16967.66 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 17098.00 | 16991.54 | 16980.15 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 16876.00 | 16997.15 | 17010.81 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 17094.00 | 17022.75 | 17013.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 17136.00 | 17045.40 | 17024.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 17399.00 | 17441.83 | 17277.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 17399.00 | 17441.83 | 17277.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 17416.00 | 17540.36 | 17455.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 17416.00 | 17540.36 | 17455.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 17347.00 | 17501.69 | 17445.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 17347.00 | 17501.69 | 17445.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 17268.00 | 17407.16 | 17409.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 16980.00 | 17261.17 | 17336.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 16934.00 | 16904.75 | 17014.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 09:45:00 | 16930.00 | 16904.75 | 17014.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 16631.00 | 16819.37 | 16915.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 16618.00 | 16770.68 | 16874.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 16594.00 | 16708.82 | 16805.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 16567.00 | 16571.90 | 16627.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 15787.10 | 16043.12 | 16282.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 15764.30 | 16043.12 | 16282.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 15738.65 | 16043.12 | 16282.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 15392.00 | 15321.73 | 15570.83 | SL hit (close>ema200) qty=0.50 sl=15321.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 15392.00 | 15321.73 | 15570.83 | SL hit (close>ema200) qty=0.50 sl=15321.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 15392.00 | 15321.73 | 15570.83 | SL hit (close>ema200) qty=0.50 sl=15321.73 alert=retest2 |

### Cycle 12 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 14495.00 | 14240.45 | 14208.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 14687.00 | 14329.76 | 14251.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 14832.00 | 14948.08 | 14756.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:30:00 | 14810.00 | 14948.08 | 14756.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 14726.00 | 14903.67 | 14754.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 14726.00 | 14903.67 | 14754.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 14714.00 | 14865.73 | 14750.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 14697.00 | 14865.73 | 14750.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 14677.00 | 14827.99 | 14743.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 14677.00 | 14827.99 | 14743.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 14699.00 | 14802.19 | 14739.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 14648.00 | 14802.19 | 14739.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 14713.00 | 14815.56 | 14764.09 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 14445.00 | 14713.52 | 14724.91 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 14883.00 | 14720.62 | 14699.45 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 14625.00 | 14713.30 | 14720.35 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 14955.00 | 14706.73 | 14706.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 15066.00 | 14815.59 | 14758.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 14973.00 | 15072.31 | 14995.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 11:15:00 | 14973.00 | 15072.31 | 14995.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 14973.00 | 15072.31 | 14995.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 14973.00 | 15072.31 | 14995.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 15028.00 | 15063.45 | 14998.48 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 14793.00 | 14972.31 | 14973.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 10:15:00 | 14774.00 | 14932.65 | 14954.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 14641.00 | 14600.25 | 14705.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:00:00 | 14641.00 | 14600.25 | 14705.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 14837.00 | 14647.60 | 14717.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 14837.00 | 14647.60 | 14717.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 14724.00 | 14662.88 | 14718.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 14843.00 | 14662.88 | 14718.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 14965.00 | 14695.00 | 14708.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 14965.00 | 14695.00 | 14708.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 14959.00 | 14747.80 | 14731.22 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 14712.00 | 14760.24 | 14761.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 14646.00 | 14728.55 | 14745.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 13948.00 | 13929.77 | 14142.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 13948.00 | 13929.77 | 14142.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 14103.00 | 13944.13 | 14027.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 14103.00 | 13944.13 | 14027.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 14025.00 | 13960.31 | 14027.55 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 14126.00 | 14048.50 | 14046.84 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 14024.00 | 14043.60 | 14044.77 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 14255.00 | 14078.09 | 14059.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 14297.00 | 14190.03 | 14127.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 14129.00 | 14192.38 | 14140.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 14129.00 | 14192.38 | 14140.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 14129.00 | 14192.38 | 14140.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 14130.00 | 14192.38 | 14140.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 13891.00 | 14132.10 | 14117.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 13891.00 | 14132.10 | 14117.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 13878.00 | 14081.28 | 14095.64 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 14135.00 | 13973.81 | 13957.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 14170.00 | 14035.00 | 13989.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 14008.00 | 14029.60 | 13990.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 14008.00 | 14029.60 | 13990.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 14008.00 | 14029.60 | 13990.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 14009.00 | 14029.60 | 13990.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 13958.00 | 14015.28 | 13987.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 13976.00 | 14015.28 | 13987.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 14015.00 | 14015.22 | 13990.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 14097.00 | 14005.98 | 13990.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 14537.00 | 14640.43 | 14641.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 14537.00 | 14640.43 | 14641.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 14465.00 | 14578.60 | 14609.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 14130.00 | 14119.76 | 14243.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 14236.00 | 14174.21 | 14232.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 14236.00 | 14174.21 | 14232.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:30:00 | 14233.00 | 14174.21 | 14232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 14108.00 | 14160.97 | 14221.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 14225.00 | 14160.97 | 14221.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 13972.00 | 14115.02 | 14189.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:45:00 | 13940.00 | 14079.02 | 14166.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 13887.00 | 14028.97 | 14127.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 14:15:00 | 13243.00 | 13442.76 | 13629.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 13388.00 | 13380.74 | 13499.25 | SL hit (close>ema200) qty=0.50 sl=13380.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 13859.00 | 13555.22 | 13554.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 13859.00 | 13555.22 | 13554.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 14064.00 | 13801.25 | 13692.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 14090.00 | 14119.70 | 14018.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 14090.00 | 14119.70 | 14018.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 13970.00 | 14089.76 | 14013.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 13970.00 | 14089.76 | 14013.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 13975.00 | 14066.81 | 14010.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:00:00 | 14010.00 | 14055.45 | 14010.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 14089.00 | 14028.41 | 14005.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 14009.00 | 14021.72 | 14004.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 14094.00 | 14029.38 | 14009.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 14097.00 | 14146.73 | 14110.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 14113.00 | 14146.73 | 14110.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 14120.00 | 14141.39 | 14110.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 14001.00 | 14141.39 | 14110.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 14097.00 | 14132.51 | 14109.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 14059.00 | 14132.51 | 14109.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 13992.00 | 14104.41 | 14098.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 13992.00 | 14104.41 | 14098.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 13984.00 | 14037.82 | 14057.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 13997.00 | 13993.94 | 14025.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 13997.00 | 13993.94 | 14025.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 13997.00 | 13993.94 | 14025.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 13987.00 | 13993.94 | 14025.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 14034.00 | 14001.96 | 14025.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 14023.00 | 14001.96 | 14025.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 14063.00 | 14014.16 | 14029.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 14077.00 | 14014.16 | 14029.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 14056.00 | 14028.74 | 14033.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 14071.00 | 14028.74 | 14033.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 14020.00 | 14028.86 | 14032.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 13992.00 | 14028.86 | 14032.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 13979.00 | 14016.91 | 14026.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 14080.00 | 14034.02 | 14032.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 14080.00 | 14034.02 | 14032.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 14080.00 | 14034.02 | 14032.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 14095.00 | 14047.43 | 14039.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 13972.00 | 14040.54 | 14038.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 13972.00 | 14040.54 | 14038.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 13972.00 | 14040.54 | 14038.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 13972.00 | 14040.54 | 14038.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 14051.00 | 14042.63 | 14039.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 14011.00 | 14042.63 | 14039.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 14086.00 | 14051.31 | 14044.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 14088.00 | 14051.31 | 14044.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 13982.00 | 14037.44 | 14038.46 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 14096.00 | 14049.16 | 14043.69 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 14056.00 | 14071.08 | 14072.88 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 14097.00 | 14057.23 | 14055.98 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 14021.00 | 14050.46 | 14053.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 13986.00 | 14027.26 | 14041.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 13921.00 | 13903.84 | 13951.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:00:00 | 13921.00 | 13903.84 | 13951.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 13924.00 | 13904.70 | 13943.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 13935.00 | 13904.70 | 13943.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 13936.00 | 13910.96 | 13942.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 13970.00 | 13910.96 | 13942.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 13970.00 | 13922.77 | 13945.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 13928.00 | 13922.77 | 13945.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 13854.00 | 13909.01 | 13936.89 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 13958.00 | 13923.26 | 13918.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 14101.00 | 13971.89 | 13942.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 13951.00 | 14037.18 | 13995.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 13951.00 | 14037.18 | 13995.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 13951.00 | 14037.18 | 13995.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 13970.00 | 14037.18 | 13995.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 13879.00 | 14005.54 | 13984.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 13846.00 | 14005.54 | 13984.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 13748.00 | 13954.03 | 13963.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 13623.00 | 13845.26 | 13909.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 13449.00 | 13436.27 | 13578.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 13449.00 | 13436.27 | 13578.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 13566.00 | 13468.26 | 13568.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 13582.00 | 13468.26 | 13568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 13595.00 | 13493.60 | 13571.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 13560.00 | 13493.60 | 13571.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 13648.00 | 13524.48 | 13578.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:15:00 | 13674.00 | 13524.48 | 13578.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 13635.00 | 13546.59 | 13583.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 13665.00 | 13546.59 | 13583.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 13521.00 | 13554.98 | 13581.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 13498.00 | 13554.98 | 13581.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 13450.00 | 13524.26 | 13562.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 13707.00 | 13537.89 | 13560.79 | SL hit (close>static) qty=1.00 sl=13648.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 13707.00 | 13537.89 | 13560.79 | SL hit (close>static) qty=1.00 sl=13648.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 13972.00 | 13624.71 | 13598.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 14116.00 | 13828.33 | 13723.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 13871.00 | 13990.60 | 13884.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 13871.00 | 13990.60 | 13884.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 13871.00 | 13990.60 | 13884.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 13871.00 | 13990.60 | 13884.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 13934.00 | 13979.28 | 13888.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 13812.00 | 13979.28 | 13888.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 13905.00 | 13960.70 | 13895.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 13894.00 | 13960.70 | 13895.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 13841.00 | 13936.76 | 13890.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 13841.00 | 13936.76 | 13890.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 13741.00 | 13897.61 | 13877.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 13741.00 | 13897.61 | 13877.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 13946.00 | 13887.24 | 13875.24 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 13736.00 | 13868.31 | 13877.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 11:15:00 | 13675.00 | 13810.72 | 13848.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 13847.00 | 13817.97 | 13848.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 12:15:00 | 13847.00 | 13817.97 | 13848.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 13847.00 | 13817.97 | 13848.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 13866.00 | 13817.97 | 13848.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 13892.00 | 13832.78 | 13852.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 13892.00 | 13832.78 | 13852.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 13767.00 | 13819.62 | 13844.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 13741.00 | 13792.97 | 13825.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:30:00 | 13727.00 | 13787.17 | 13819.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 13745.00 | 13778.34 | 13812.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:30:00 | 13749.00 | 13779.78 | 13806.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 13792.00 | 13782.22 | 13805.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 13827.00 | 13782.22 | 13805.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 13884.00 | 13802.58 | 13812.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:15:00 | 13926.00 | 13802.58 | 13812.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 13875.00 | 13817.06 | 13818.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 13908.00 | 13817.06 | 13818.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 12:15:00 | 13958.00 | 13864.44 | 13840.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 13791.00 | 13892.15 | 13866.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 13791.00 | 13892.15 | 13866.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 13791.00 | 13892.15 | 13866.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 13791.00 | 13892.15 | 13866.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 13756.00 | 13864.92 | 13856.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 13756.00 | 13864.92 | 13856.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 13820.00 | 13846.19 | 13849.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 13756.00 | 13829.08 | 13840.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 13467.00 | 13367.03 | 13475.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 13467.00 | 13367.03 | 13475.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 13467.00 | 13367.03 | 13475.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 13467.00 | 13367.03 | 13475.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 13475.00 | 13388.62 | 13475.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 13481.00 | 13388.62 | 13475.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 13476.00 | 13406.10 | 13475.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 13483.00 | 13406.10 | 13475.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 13436.00 | 13412.08 | 13472.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:30:00 | 13422.00 | 13429.33 | 13470.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 13418.00 | 13435.11 | 13463.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:15:00 | 12750.90 | 12964.21 | 13096.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:15:00 | 12747.10 | 12964.21 | 13096.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 12971.00 | 12952.57 | 13067.25 | SL hit (close>ema200) qty=0.50 sl=12952.57 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 12971.00 | 12952.57 | 13067.25 | SL hit (close>ema200) qty=0.50 sl=12952.57 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 12198.00 | 11912.24 | 11891.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 12345.00 | 11998.79 | 11932.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 12502.00 | 12513.54 | 12355.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 12584.00 | 12513.54 | 12355.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 12475.00 | 12561.05 | 12456.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 12475.00 | 12561.05 | 12456.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 12472.00 | 12543.24 | 12458.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 12679.00 | 12543.24 | 12458.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 12426.00 | 12520.67 | 12486.88 | SL hit (close<static) qty=1.00 sl=12453.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 12433.00 | 12474.56 | 12475.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 12337.00 | 12440.08 | 12459.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 12290.00 | 12141.54 | 12255.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 12290.00 | 12141.54 | 12255.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 12290.00 | 12141.54 | 12255.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 12247.00 | 12141.54 | 12255.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 12255.00 | 12164.23 | 12255.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 12263.00 | 12164.23 | 12255.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 12305.00 | 12192.39 | 12260.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 12325.00 | 12192.39 | 12260.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 12228.00 | 12199.51 | 12257.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 12300.00 | 12199.51 | 12257.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 12202.00 | 12200.01 | 12252.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 12165.00 | 12209.27 | 12242.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 12192.00 | 12197.29 | 12230.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 12183.00 | 12186.15 | 12210.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 13:15:00 | 12290.00 | 12234.07 | 12228.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 13:15:00 | 12290.00 | 12234.07 | 12228.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 13:15:00 | 12290.00 | 12234.07 | 12228.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 12290.00 | 12234.07 | 12228.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 12351.00 | 12257.45 | 12239.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 12614.00 | 12653.23 | 12523.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 12614.00 | 12653.23 | 12523.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 13202.00 | 13392.90 | 13258.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 13202.00 | 13392.90 | 13258.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 13147.00 | 13343.72 | 13248.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 13147.00 | 13343.72 | 13248.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 13195.00 | 13289.54 | 13238.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 13027.00 | 13289.54 | 13238.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 13072.00 | 13226.59 | 13217.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:00:00 | 13072.00 | 13226.59 | 13217.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 13063.00 | 13193.87 | 13203.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 11:15:00 | 12769.00 | 13004.86 | 13089.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 12828.00 | 12807.64 | 12942.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:45:00 | 12805.00 | 12807.64 | 12942.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 12900.00 | 12826.49 | 12927.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 12920.00 | 12826.49 | 12927.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 12935.00 | 12848.19 | 12928.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 12935.00 | 12848.19 | 12928.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 12845.00 | 12847.55 | 12920.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 12812.00 | 12887.93 | 12921.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 13005.00 | 12917.12 | 12929.48 | SL hit (close>static) qty=1.00 sl=12943.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 12764.00 | 12901.69 | 12921.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 12824.00 | 12853.56 | 12894.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 12956.00 | 12863.88 | 12891.07 | SL hit (close>static) qty=1.00 sl=12943.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 12956.00 | 12863.88 | 12891.07 | SL hit (close>static) qty=1.00 sl=12943.00 alert=retest2 |

### Cycle 44 — BUY (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 09:15:00 | 12972.00 | 12908.80 | 12902.65 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 12801.00 | 12881.25 | 12891.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 12638.00 | 12832.60 | 12868.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 12783.00 | 12607.91 | 12682.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 12783.00 | 12607.91 | 12682.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 12783.00 | 12607.91 | 12682.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 12783.00 | 12607.91 | 12682.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 12663.00 | 12618.93 | 12680.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 12630.00 | 12636.54 | 12682.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 12814.00 | 12692.59 | 12701.53 | SL hit (close>static) qty=1.00 sl=12791.00 alert=retest2 |

### Cycle 46 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 12889.00 | 12731.87 | 12718.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 12931.00 | 12771.70 | 12737.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 12779.00 | 12789.02 | 12755.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 12810.00 | 12789.02 | 12755.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 12701.00 | 12769.97 | 12752.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 12701.00 | 12769.97 | 12752.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 12658.00 | 12747.58 | 12744.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 12658.00 | 12747.58 | 12744.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 12660.00 | 12730.06 | 12736.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 12596.00 | 12671.81 | 12703.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 12737.00 | 12684.85 | 12706.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 12737.00 | 12684.85 | 12706.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 12737.00 | 12684.85 | 12706.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 12737.00 | 12684.85 | 12706.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 12832.00 | 12714.28 | 12717.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 13133.00 | 12714.28 | 12717.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 13147.00 | 12800.82 | 12756.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 13299.00 | 12900.46 | 12806.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 13386.00 | 13496.89 | 13230.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 11:00:00 | 13386.00 | 13496.89 | 13230.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 13221.00 | 13387.66 | 13241.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 13221.00 | 13387.66 | 13241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 13320.00 | 13374.13 | 13248.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 13371.00 | 13351.32 | 13268.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:30:00 | 13365.00 | 13358.06 | 13279.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:30:00 | 13370.00 | 13341.85 | 13278.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 13369.00 | 13346.28 | 13286.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 13348.00 | 13487.88 | 13390.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 13348.00 | 13487.88 | 13390.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 13321.00 | 13454.50 | 13383.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:45:00 | 13431.00 | 13440.60 | 13383.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 13202.00 | 13392.88 | 13367.43 | SL hit (close<static) qty=1.00 sl=13210.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 13202.00 | 13392.88 | 13367.43 | SL hit (close<static) qty=1.00 sl=13210.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 13202.00 | 13392.88 | 13367.43 | SL hit (close<static) qty=1.00 sl=13210.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 13202.00 | 13392.88 | 13367.43 | SL hit (close<static) qty=1.00 sl=13210.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 13120.00 | 13338.30 | 13344.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 13120.00 | 13338.30 | 13344.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 12935.00 | 13213.28 | 13284.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 13205.00 | 13137.57 | 13218.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 13205.00 | 13137.57 | 13218.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 13274.00 | 13164.86 | 13223.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 13418.00 | 13164.86 | 13223.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 13377.00 | 13207.29 | 13237.19 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 13512.00 | 13268.23 | 13262.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 14028.00 | 13490.55 | 13374.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 11:15:00 | 13585.00 | 13639.18 | 13497.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 11:15:00 | 13585.00 | 13639.18 | 13497.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 13585.00 | 13639.18 | 13497.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 13548.00 | 13639.18 | 13497.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 13548.00 | 13620.94 | 13502.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 13451.00 | 13620.94 | 13502.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 13352.00 | 13553.83 | 13507.81 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 13325.00 | 13462.01 | 13471.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 13110.00 | 13337.79 | 13402.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 13086.00 | 13085.60 | 13212.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 13086.00 | 13085.60 | 13212.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 13086.00 | 13085.60 | 13212.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 13193.00 | 13085.60 | 13212.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 13213.00 | 13111.08 | 13212.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 13213.00 | 13111.08 | 13212.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 13298.00 | 13148.47 | 13220.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 13298.00 | 13148.47 | 13220.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 13328.00 | 13184.37 | 13229.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 13328.00 | 13184.37 | 13229.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 13375.00 | 13276.22 | 13265.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 12:15:00 | 13443.00 | 13363.70 | 13315.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 13376.00 | 13385.91 | 13343.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 13376.00 | 13385.91 | 13343.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 13376.00 | 13385.91 | 13343.27 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 13230.00 | 13341.37 | 13351.64 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 13425.00 | 13360.27 | 13358.58 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 13110.00 | 13320.28 | 13341.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 13062.00 | 13185.23 | 13260.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 13126.00 | 13069.99 | 13146.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 13126.00 | 13069.99 | 13146.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 13249.00 | 13112.84 | 13153.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 13252.00 | 13112.84 | 13153.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 13262.00 | 13142.67 | 13163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 13240.00 | 13142.67 | 13163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 13272.00 | 13184.11 | 13179.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 13284.00 | 13228.78 | 13204.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 13309.00 | 13353.11 | 13315.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 13309.00 | 13353.11 | 13315.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 13309.00 | 13353.11 | 13315.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 13408.00 | 13353.11 | 13315.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 13396.00 | 13361.69 | 13322.82 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 13211.00 | 13329.69 | 13331.90 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 13385.00 | 13337.23 | 13333.51 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 13206.00 | 13315.43 | 13324.52 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 13474.00 | 13340.06 | 13323.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 13563.00 | 13431.48 | 13380.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 13528.00 | 13647.00 | 13544.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 13528.00 | 13647.00 | 13544.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 13528.00 | 13647.00 | 13544.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 13528.00 | 13647.00 | 13544.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 13522.00 | 13622.00 | 13542.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 13482.00 | 13622.00 | 13542.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 13522.00 | 13602.00 | 13540.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 13541.00 | 13602.00 | 13540.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 13576.00 | 13596.80 | 13543.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 13602.00 | 13575.55 | 13543.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 13491.00 | 13558.64 | 13538.44 | SL hit (close<static) qty=1.00 sl=13516.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 13752.00 | 13558.64 | 13538.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-06 09:15:00 | 15127.20 | 14720.22 | 14436.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 14720.00 | 14798.90 | 14804.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 14512.00 | 14741.52 | 14777.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 14640.00 | 14566.88 | 14644.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 14640.00 | 14566.88 | 14644.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 14640.00 | 14566.88 | 14644.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 14640.00 | 14566.88 | 14644.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 14679.00 | 14589.30 | 14647.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 14741.00 | 14589.30 | 14647.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 14566.00 | 14584.64 | 14639.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 14474.00 | 14558.85 | 14612.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:45:00 | 14442.00 | 14521.28 | 14590.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:15:00 | 13750.30 | 14028.14 | 14197.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 14042.00 | 14003.93 | 14128.65 | SL hit (close>ema200) qty=0.50 sl=14003.93 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 12:15:00 | 13719.90 | 13896.94 | 14029.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-20 14:15:00 | 12997.80 | 13161.61 | 13391.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 13025.00 | 12819.61 | 12816.90 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 12550.00 | 12815.73 | 12825.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 12315.00 | 12715.59 | 12778.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 12718.00 | 12321.07 | 12439.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 12718.00 | 12321.07 | 12439.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 12718.00 | 12321.07 | 12439.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 12718.00 | 12321.07 | 12439.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 12529.00 | 12362.65 | 12447.81 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 12970.00 | 12578.10 | 12536.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 13014.00 | 12758.93 | 12665.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 14966.00 | 15012.30 | 14883.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 14850.00 | 14955.24 | 14909.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 14850.00 | 14955.24 | 14909.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 15084.00 | 14991.59 | 14930.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 14969.00 | 15002.79 | 14970.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 14:15:00 | 15045.00 | 15303.85 | 15308.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 14:15:00 | 15045.00 | 15303.85 | 15308.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 15045.00 | 15303.85 | 15308.95 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 15394.00 | 15270.29 | 15269.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 15541.00 | 15366.28 | 15323.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 15342.00 | 15396.54 | 15351.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 15342.00 | 15396.54 | 15351.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 15342.00 | 15396.54 | 15351.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 15342.00 | 15396.54 | 15351.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 15400.00 | 15397.24 | 15355.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 15400.00 | 15397.24 | 15355.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 15410.00 | 15399.79 | 15360.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 15284.00 | 15399.79 | 15360.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 15225.00 | 15364.83 | 15348.37 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 15210.00 | 15333.86 | 15335.79 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 15376.00 | 15342.29 | 15339.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 15440.00 | 15361.83 | 15348.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 13:15:00 | 15412.00 | 15442.07 | 15407.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 13:15:00 | 15412.00 | 15442.07 | 15407.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 15412.00 | 15442.07 | 15407.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 15389.00 | 15442.07 | 15407.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 15467.00 | 15447.06 | 15413.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:30:00 | 15464.00 | 15447.06 | 15413.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 15560.00 | 15477.32 | 15433.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 15613.00 | 15524.48 | 15463.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 15603.00 | 15700.95 | 15647.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:30:00 | 13770.00 | 2025-05-19 14:15:00 | 13705.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-15 09:15:00 | 13781.00 | 2025-05-19 14:15:00 | 13705.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-05-15 11:15:00 | 13780.00 | 2025-05-19 14:15:00 | 13705.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-30 14:15:00 | 16019.00 | 2025-06-06 13:15:00 | 16515.00 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2025-05-30 14:45:00 | 16048.00 | 2025-06-06 13:15:00 | 16515.00 | STOP_HIT | 1.00 | 2.91% |
| SELL | retest2 | 2025-06-12 11:30:00 | 16612.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-12 12:45:00 | 16625.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-12 13:15:00 | 16612.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-12 14:00:00 | 16601.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-06-19 09:15:00 | 17174.00 | 2025-06-19 11:15:00 | 16945.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-07 11:30:00 | 16618.00 | 2025-07-11 09:15:00 | 15787.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:30:00 | 16594.00 | 2025-07-11 09:15:00 | 15764.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 14:15:00 | 16567.00 | 2025-07-11 09:15:00 | 15738.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 11:30:00 | 16618.00 | 2025-07-15 09:15:00 | 15392.00 | STOP_HIT | 0.50 | 7.38% |
| SELL | retest2 | 2025-07-08 09:30:00 | 16594.00 | 2025-07-15 09:15:00 | 15392.00 | STOP_HIT | 0.50 | 7.24% |
| SELL | retest2 | 2025-07-09 14:15:00 | 16567.00 | 2025-07-15 09:15:00 | 15392.00 | STOP_HIT | 0.50 | 7.09% |
| BUY | retest2 | 2025-09-12 09:15:00 | 14097.00 | 2025-09-22 10:15:00 | 14537.00 | STOP_HIT | 1.00 | 3.12% |
| SELL | retest2 | 2025-09-26 10:45:00 | 13940.00 | 2025-09-30 14:15:00 | 13243.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 10:45:00 | 13940.00 | 2025-10-01 14:15:00 | 13388.00 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-09-26 13:15:00 | 13887.00 | 2025-10-03 10:15:00 | 13859.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-10-08 13:00:00 | 14010.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-10-08 14:30:00 | 14089.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-09 09:15:00 | 14009.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-10-09 09:45:00 | 14094.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-16 10:15:00 | 13992.00 | 2025-10-16 13:15:00 | 14080.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-16 11:45:00 | 13979.00 | 2025-10-16 13:15:00 | 14080.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-10 13:15:00 | 13498.00 | 2025-11-11 09:15:00 | 13707.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-10 14:45:00 | 13450.00 | 2025-11-11 09:15:00 | 13707.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-19 11:00:00 | 13741.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-19 11:30:00 | 13727.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-19 12:30:00 | 13745.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-19 14:30:00 | 13749.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-26 14:30:00 | 13422.00 | 2025-12-04 09:15:00 | 12750.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:45:00 | 13418.00 | 2025-12-04 09:15:00 | 12747.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 14:30:00 | 13422.00 | 2025-12-04 11:15:00 | 12971.00 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-11-27 10:45:00 | 13418.00 | 2025-12-04 11:15:00 | 12971.00 | STOP_HIT | 0.50 | 3.33% |
| BUY | retest2 | 2025-12-26 09:15:00 | 12679.00 | 2025-12-26 14:15:00 | 12426.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 12165.00 | 2026-01-02 13:15:00 | 12290.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-01 11:45:00 | 12192.00 | 2026-01-02 13:15:00 | 12290.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-02 09:45:00 | 12183.00 | 2026-01-02 13:15:00 | 12290.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-01-16 09:30:00 | 12812.00 | 2026-01-16 11:15:00 | 13005.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-16 13:15:00 | 12764.00 | 2026-01-19 09:15:00 | 12956.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-16 15:00:00 | 12824.00 | 2026-01-19 09:15:00 | 12956.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-22 11:30:00 | 12630.00 | 2026-01-22 13:15:00 | 12814.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-30 10:30:00 | 13371.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-30 11:30:00 | 13365.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-30 12:30:00 | 13370.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-30 13:30:00 | 13369.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-01 13:45:00 | 13431.00 | 2026-02-01 15:15:00 | 13120.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-02-27 14:45:00 | 13602.00 | 2026-02-27 15:15:00 | 13491.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-03-02 09:15:00 | 13752.00 | 2026-03-06 09:15:00 | 15127.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 14474.00 | 2026-03-17 10:15:00 | 13750.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 14474.00 | 2026-03-17 14:15:00 | 14042.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-13 09:45:00 | 14442.00 | 2026-03-18 12:15:00 | 13719.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:45:00 | 14442.00 | 2026-03-20 14:15:00 | 12997.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 09:30:00 | 15084.00 | 2026-04-24 14:15:00 | 15045.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-22 09:15:00 | 14969.00 | 2026-04-24 14:15:00 | 15045.00 | STOP_HIT | 1.00 | 0.51% |

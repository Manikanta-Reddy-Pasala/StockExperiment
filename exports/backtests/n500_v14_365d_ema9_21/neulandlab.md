# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 17713.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 49 |
| ALERT2 | 49 |
| ALERT2_SKIP | 26 |
| ALERT3 | 117 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 56 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 40
- **Target hits / Stop hits / Partials:** 8 / 48 / 7
- **Avg / median % per leg:** 0.93% / -1.58%
- **Sum % (uncompounded):** 58.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 9 | 37.5% | 8 | 16 | 0 | 1.66% | 39.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 9 | 37.5% | 8 | 16 | 0 | 1.66% | 39.8% |
| SELL (all) | 39 | 14 | 35.9% | 0 | 32 | 7 | 0.47% | 18.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 14 | 35.9% | 0 | 32 | 7 | 0.47% | 18.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 23 | 36.5% | 8 | 48 | 7 | 0.93% | 58.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 12071.00 | 12453.12 | 12500.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 11:15:00 | 11902.00 | 12342.89 | 12445.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 12:15:00 | 11255.00 | 11246.32 | 11546.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:30:00 | 11228.00 | 11246.32 | 11546.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 11161.00 | 11186.87 | 11316.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 11180.00 | 11186.87 | 11316.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 11194.00 | 11185.92 | 11253.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 11110.00 | 11199.83 | 11219.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 11315.00 | 11241.04 | 11232.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 11315.00 | 11241.04 | 11232.63 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 11173.00 | 11246.22 | 11247.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 11103.00 | 11217.58 | 11234.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 10:15:00 | 11356.00 | 11150.11 | 11164.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 10:15:00 | 11356.00 | 11150.11 | 11164.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 11356.00 | 11150.11 | 11164.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 11356.00 | 11150.11 | 11164.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 11206.00 | 11161.29 | 11168.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 11190.00 | 11163.43 | 11168.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 11368.00 | 11204.35 | 11186.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 11368.00 | 11204.35 | 11186.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 11485.00 | 11325.46 | 11255.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 11565.00 | 11622.33 | 11509.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 11557.00 | 11622.33 | 11509.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 11640.00 | 11714.94 | 11633.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 11640.00 | 11714.94 | 11633.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 11598.00 | 11691.55 | 11630.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 11598.00 | 11691.55 | 11630.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 11666.00 | 11686.44 | 11633.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 11728.00 | 11686.44 | 11633.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 15:00:00 | 11740.00 | 11776.98 | 11728.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 09:15:00 | 12900.80 | 11993.39 | 11835.74 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-06 09:15:00 | 12914.00 | 11993.39 | 11835.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 10:15:00 | 12967.00 | 13147.61 | 13154.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 13:15:00 | 12891.00 | 13041.30 | 13099.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 15:15:00 | 12220.00 | 12197.23 | 12382.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 12228.00 | 12197.23 | 12382.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 12610.00 | 12279.78 | 12403.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 12610.00 | 12279.78 | 12403.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 12496.00 | 12323.03 | 12411.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 12315.00 | 12347.42 | 12414.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 12315.00 | 12337.11 | 12365.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 11699.25 | 11867.18 | 11914.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 11699.25 | 11867.18 | 11914.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 11918.00 | 11767.18 | 11816.37 | SL hit (close>ema200) qty=0.50 sl=11767.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 11918.00 | 11767.18 | 11816.37 | SL hit (close>ema200) qty=0.50 sl=11767.18 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 11980.00 | 11846.04 | 11845.92 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 11760.00 | 11840.60 | 11848.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 11685.00 | 11772.61 | 11808.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 12037.00 | 11788.24 | 11793.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 12037.00 | 11788.24 | 11793.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 12037.00 | 11788.24 | 11793.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 12365.00 | 11788.24 | 11793.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 11965.00 | 11823.59 | 11809.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 12090.00 | 12003.76 | 11946.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 14535.00 | 14647.08 | 14044.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:00:00 | 14535.00 | 14647.08 | 14044.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 14340.00 | 14491.87 | 14189.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 14459.00 | 14491.87 | 14189.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:45:00 | 14530.00 | 14460.40 | 14224.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 14496.00 | 14481.92 | 14339.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 14160.00 | 14313.42 | 14318.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 14160.00 | 14313.42 | 14318.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 14160.00 | 14313.42 | 14318.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 14160.00 | 14313.42 | 14318.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 14087.00 | 14244.55 | 14284.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 13897.00 | 13843.29 | 13948.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 13897.00 | 13843.29 | 13948.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 13952.00 | 13865.03 | 13949.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 13969.00 | 13865.03 | 13949.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 14090.00 | 13910.02 | 13961.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 14090.00 | 13910.02 | 13961.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 14120.00 | 13952.02 | 13976.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 13970.00 | 13952.02 | 13976.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 14020.00 | 13889.62 | 13888.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 14020.00 | 13889.62 | 13888.48 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 13811.00 | 13887.36 | 13890.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 09:15:00 | 13725.00 | 13854.89 | 13875.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 13906.00 | 13865.11 | 13878.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 13906.00 | 13865.11 | 13878.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 13906.00 | 13865.11 | 13878.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 13906.00 | 13865.11 | 13878.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 13856.00 | 13863.29 | 13876.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 13801.00 | 13852.17 | 13864.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 13922.00 | 13873.19 | 13871.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 13922.00 | 13873.19 | 13871.97 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 13768.00 | 13871.48 | 13874.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 11:15:00 | 13658.00 | 13828.79 | 13854.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 13023.00 | 13016.23 | 13230.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 13023.00 | 13016.23 | 13230.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 13024.00 | 13036.69 | 13173.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 13003.00 | 13036.69 | 13173.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 13000.00 | 13007.41 | 13124.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 13418.00 | 12889.54 | 12871.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 13418.00 | 12889.54 | 12871.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 13418.00 | 12889.54 | 12871.35 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 13050.00 | 13099.40 | 13103.25 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 13188.00 | 13104.62 | 13095.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 13276.00 | 13138.89 | 13111.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 13239.00 | 13329.26 | 13247.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 13239.00 | 13329.26 | 13247.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 13239.00 | 13329.26 | 13247.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 13239.00 | 13329.26 | 13247.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 13170.00 | 13297.40 | 13240.67 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 13124.00 | 13208.91 | 13209.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 14:15:00 | 13055.00 | 13178.13 | 13195.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 13232.00 | 13156.64 | 13178.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 10:15:00 | 13232.00 | 13156.64 | 13178.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 13232.00 | 13156.64 | 13178.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 13232.00 | 13156.64 | 13178.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 13849.00 | 13295.11 | 13239.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 13937.00 | 13638.63 | 13454.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 13740.00 | 13768.74 | 13616.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 13649.00 | 13768.74 | 13616.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 13640.00 | 13742.99 | 13618.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 13622.00 | 13742.99 | 13618.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 13727.00 | 13739.79 | 13628.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:15:00 | 13870.00 | 13758.19 | 13656.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 13849.00 | 13784.08 | 13686.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 13850.00 | 13784.08 | 13686.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 13918.00 | 13807.81 | 13714.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 13749.00 | 13795.02 | 13732.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 13749.00 | 13795.02 | 13732.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 13714.00 | 13778.81 | 13730.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 13714.00 | 13778.81 | 13730.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | SL hit (close<static) qty=1.00 sl=13622.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | SL hit (close<static) qty=1.00 sl=13622.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | SL hit (close<static) qty=1.00 sl=13622.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | SL hit (close<static) qty=1.00 sl=13622.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 13561.00 | 13735.25 | 13715.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 13506.00 | 13689.40 | 13696.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 13356.00 | 13622.72 | 13665.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 13190.00 | 13183.24 | 13321.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 13286.00 | 13183.24 | 13321.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 13239.00 | 13194.39 | 13313.88 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 13492.00 | 13366.14 | 13361.99 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 13342.00 | 13364.42 | 13366.21 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 13452.00 | 13381.94 | 13374.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 13468.00 | 13400.61 | 13384.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 14:15:00 | 14435.00 | 14636.96 | 14419.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 14435.00 | 14636.96 | 14419.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 14435.00 | 14636.96 | 14419.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 14435.00 | 14636.96 | 14419.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 14549.00 | 14619.37 | 14431.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 14348.00 | 14619.37 | 14431.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 14262.00 | 14547.89 | 14416.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 14479.00 | 14534.12 | 14421.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:30:00 | 14471.00 | 14500.29 | 14416.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 14485.00 | 14444.39 | 14407.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 14448.00 | 14436.97 | 14413.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 14387.00 | 14426.97 | 14411.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 14387.00 | 14426.97 | 14411.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 14378.00 | 14417.18 | 14408.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 14395.00 | 14417.18 | 14408.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 14418.00 | 14415.50 | 14409.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 14564.00 | 14415.50 | 14409.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 14543.00 | 14441.00 | 14421.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 14749.00 | 14558.41 | 14488.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 09:15:00 | 15926.90 | 15423.26 | 15050.06 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-12 09:15:00 | 15918.10 | 15423.26 | 15050.06 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-12 09:15:00 | 15933.50 | 15423.26 | 15050.06 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-12 09:15:00 | 15892.80 | 15423.26 | 15050.06 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-16 09:15:00 | 16223.90 | 15807.41 | 15632.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 15822.00 | 15943.92 | 15952.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 15449.00 | 15830.52 | 15892.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 15023.00 | 14955.06 | 15159.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:45:00 | 15080.00 | 14955.06 | 15159.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 14412.00 | 14247.34 | 14534.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 14490.00 | 14247.34 | 14534.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 14483.00 | 14294.48 | 14529.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 14483.00 | 14294.48 | 14529.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 14405.00 | 14316.58 | 14518.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 14372.00 | 14406.18 | 14501.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 14618.00 | 14457.84 | 14480.89 | SL hit (close>static) qty=1.00 sl=14533.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 14923.00 | 14573.62 | 14530.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 15110.00 | 14680.89 | 14583.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 15328.00 | 15371.99 | 15114.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:45:00 | 15284.00 | 15371.99 | 15114.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 15195.00 | 15315.47 | 15131.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 15049.00 | 15315.47 | 15131.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 15183.00 | 15288.98 | 15136.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 15466.00 | 15191.58 | 15149.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 16147.00 | 16178.17 | 16179.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 16147.00 | 16178.17 | 16179.57 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 12:15:00 | 16310.00 | 16196.33 | 16186.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 16393.00 | 16292.75 | 16242.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 16209.00 | 16350.43 | 16309.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 16209.00 | 16350.43 | 16309.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 16209.00 | 16350.43 | 16309.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 16209.00 | 16350.43 | 16309.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 16240.00 | 16328.35 | 16302.78 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 16151.00 | 16272.98 | 16280.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 16143.00 | 16194.90 | 16224.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 15914.00 | 15882.90 | 16003.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 15914.00 | 15882.90 | 16003.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 15914.00 | 15882.90 | 16003.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 15713.00 | 15843.03 | 15955.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 15757.00 | 15803.73 | 15906.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 15740.00 | 15636.59 | 15724.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 16056.00 | 15802.03 | 15784.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 16056.00 | 15802.03 | 15784.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 16056.00 | 15802.03 | 15784.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 13:15:00 | 16056.00 | 15802.03 | 15784.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 16177.00 | 15877.03 | 15820.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 17408.00 | 17488.29 | 17184.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:45:00 | 17406.00 | 17488.29 | 17184.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 17570.00 | 17510.31 | 17308.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 18721.00 | 17697.87 | 17502.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 18160.00 | 17872.24 | 17620.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:00:00 | 18190.00 | 17935.79 | 17672.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 18102.00 | 18013.50 | 17777.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 17740.00 | 17960.80 | 17795.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 17740.00 | 17960.80 | 17795.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 17341.00 | 17836.84 | 17753.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 17341.00 | 17836.84 | 17753.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 17295.00 | 17728.47 | 17712.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 17201.00 | 17497.59 | 17599.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 17522.00 | 17398.08 | 17519.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 17522.00 | 17398.08 | 17519.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 17522.00 | 17398.08 | 17519.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 17522.00 | 17398.08 | 17519.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 17421.00 | 17402.66 | 17510.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 17335.00 | 17439.57 | 17485.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 17297.00 | 17411.06 | 17468.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 17333.00 | 17351.86 | 17422.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 17315.00 | 17388.09 | 17432.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 17360.00 | 17490.58 | 17496.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 17125.00 | 17417.46 | 17462.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 16931.00 | 16844.92 | 16991.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 16931.00 | 16844.92 | 16991.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 16931.00 | 16844.92 | 16991.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 17025.00 | 16844.92 | 16991.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 16609.00 | 16391.62 | 16516.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 16609.00 | 16391.62 | 16516.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 16584.00 | 16430.09 | 16522.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 16627.00 | 16430.09 | 16522.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 16909.00 | 16602.02 | 16589.46 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 16489.00 | 16610.29 | 16623.29 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 16723.00 | 16643.51 | 16636.79 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 16564.00 | 16632.80 | 16633.41 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 16716.00 | 16639.32 | 16635.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 16823.00 | 16676.06 | 16652.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 16883.00 | 16956.49 | 16822.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 16883.00 | 16956.49 | 16822.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 16866.00 | 16934.47 | 16835.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 16889.00 | 16934.47 | 16835.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 16853.00 | 16918.18 | 16836.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 16828.00 | 16918.18 | 16836.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 16772.00 | 16888.94 | 16831.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 16766.00 | 16888.94 | 16831.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 16667.00 | 16844.55 | 16816.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 16667.00 | 16844.55 | 16816.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 16530.00 | 16781.64 | 16790.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 16325.00 | 16690.31 | 16747.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 16656.00 | 16482.70 | 16579.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 16656.00 | 16482.70 | 16579.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 16656.00 | 16482.70 | 16579.37 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 17411.00 | 16713.21 | 16670.15 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 16611.00 | 16824.20 | 16850.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 16500.00 | 16759.36 | 16819.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 16431.00 | 16374.38 | 16564.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 16431.00 | 16374.38 | 16564.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 16431.00 | 16374.38 | 16564.47 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 16900.00 | 16670.97 | 16660.06 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 16550.00 | 16638.14 | 16646.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 16460.00 | 16575.69 | 16613.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 16285.00 | 16237.27 | 16358.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:15:00 | 16314.00 | 16237.27 | 16358.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 16320.00 | 16253.82 | 16355.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:15:00 | 16171.00 | 16267.08 | 16345.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 16426.00 | 16291.38 | 16294.37 | SL hit (close>static) qty=1.00 sl=16410.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 16450.00 | 16323.11 | 16308.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 16525.00 | 16363.49 | 16328.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 16177.00 | 16326.19 | 16314.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 16177.00 | 16326.19 | 16314.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 16177.00 | 16326.19 | 16314.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:15:00 | 16150.00 | 16326.19 | 16314.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 16080.00 | 16276.95 | 16293.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 15990.00 | 16125.85 | 16197.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 15767.00 | 15747.98 | 15911.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:30:00 | 15645.00 | 15747.98 | 15911.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 16262.00 | 15872.09 | 15918.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 15890.00 | 15933.83 | 15939.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 15996.00 | 15946.26 | 15944.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 15996.00 | 15946.26 | 15944.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 16089.00 | 15974.81 | 15957.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 15970.00 | 15973.85 | 15958.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 15:15:00 | 15970.00 | 15973.85 | 15958.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 15970.00 | 15973.85 | 15958.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 15873.00 | 15973.85 | 15958.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 15862.00 | 15951.48 | 15949.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:15:00 | 15835.00 | 15951.48 | 15949.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 10:15:00 | 15850.00 | 15931.18 | 15940.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 15770.00 | 15877.96 | 15913.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 15272.00 | 15186.08 | 15326.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 15272.00 | 15186.08 | 15326.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 15200.00 | 15201.81 | 15310.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 15295.00 | 15201.81 | 15310.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 14901.00 | 14967.02 | 15044.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 14865.00 | 14952.01 | 15030.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:45:00 | 14867.00 | 14936.21 | 15016.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 15210.00 | 15001.69 | 15026.87 | SL hit (close>static) qty=1.00 sl=15098.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 15210.00 | 15001.69 | 15026.87 | SL hit (close>static) qty=1.00 sl=15098.00 alert=retest2 |

### Cycle 46 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 15207.00 | 15076.08 | 15058.38 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 14646.00 | 15027.84 | 15078.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 14556.00 | 14933.47 | 15030.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 14958.00 | 14742.93 | 14870.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 14958.00 | 14742.93 | 14870.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 14958.00 | 14742.93 | 14870.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 14958.00 | 14742.93 | 14870.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 14928.00 | 14779.95 | 14875.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 14928.00 | 14779.95 | 14875.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 14998.00 | 14823.56 | 14886.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 14959.00 | 14823.56 | 14886.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 15039.00 | 14866.65 | 14900.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 15039.00 | 14866.65 | 14900.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 15149.00 | 14957.41 | 14937.95 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 14797.00 | 14942.19 | 14953.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 14711.00 | 14857.48 | 14906.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 14656.00 | 14579.34 | 14711.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 14656.00 | 14579.34 | 14711.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 14796.00 | 14622.67 | 14718.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 14790.00 | 14622.67 | 14718.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 14620.00 | 14622.14 | 14709.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 14467.00 | 14622.14 | 14709.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 14370.00 | 14571.71 | 14679.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 14350.00 | 14520.57 | 14646.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 14333.00 | 14417.75 | 14562.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 14345.00 | 14422.36 | 14496.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 14351.00 | 14343.26 | 14429.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13632.50 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13616.35 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13627.75 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13633.45 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | SL hit (close>ema200) qty=0.50 sl=13340.02 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | SL hit (close>ema200) qty=0.50 sl=13340.02 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | SL hit (close>ema200) qty=0.50 sl=13340.02 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | SL hit (close>ema200) qty=0.50 sl=13340.02 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 13579.00 | 13340.02 | 13527.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 13691.00 | 13410.22 | 13542.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 13664.00 | 13410.22 | 13542.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 13598.00 | 13447.77 | 13547.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 13521.00 | 13558.52 | 13579.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 11:15:00 | 12844.95 | 13225.32 | 13371.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 12761.00 | 12688.54 | 12822.62 | SL hit (close>ema200) qty=0.50 sl=12688.54 alert=retest2 |

### Cycle 50 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 13275.00 | 12946.21 | 12912.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 13549.00 | 13145.52 | 13018.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 13450.00 | 13537.90 | 13320.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 13172.00 | 13537.90 | 13320.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 13022.00 | 13434.72 | 13293.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 13022.00 | 13434.72 | 13293.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 12931.00 | 13333.98 | 13260.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 12931.00 | 13333.98 | 13260.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 13060.00 | 13187.19 | 13203.16 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 13526.00 | 13244.11 | 13223.86 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 13120.00 | 13295.13 | 13314.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 13040.00 | 13167.74 | 13234.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 13322.00 | 13126.70 | 13174.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 13322.00 | 13126.70 | 13174.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 13322.00 | 13126.70 | 13174.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 13284.00 | 13126.70 | 13174.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 13556.00 | 13212.56 | 13209.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 14123.00 | 13535.92 | 13373.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 13519.00 | 13635.47 | 13468.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 10:15:00 | 13519.00 | 13635.47 | 13468.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 13519.00 | 13635.47 | 13468.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 13519.00 | 13635.47 | 13468.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 13361.00 | 13580.58 | 13459.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 13361.00 | 13580.58 | 13459.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 13532.00 | 13570.86 | 13465.79 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 13228.00 | 13396.66 | 13419.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 13109.00 | 13339.13 | 13391.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 13148.00 | 13055.41 | 13137.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 13:15:00 | 13148.00 | 13055.41 | 13137.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 13148.00 | 13055.41 | 13137.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 13148.00 | 13055.41 | 13137.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 13063.00 | 13056.93 | 13131.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 13045.00 | 13056.93 | 13131.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 13329.00 | 13109.44 | 13142.03 | SL hit (close>static) qty=1.00 sl=13159.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 13267.00 | 13178.64 | 13170.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 13953.00 | 13376.59 | 13269.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 13328.00 | 13557.86 | 13447.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 13328.00 | 13557.86 | 13447.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 13328.00 | 13557.86 | 13447.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 13328.00 | 13557.86 | 13447.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 13529.00 | 13552.09 | 13455.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 13325.00 | 13552.09 | 13455.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 13461.00 | 13582.50 | 13520.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 13520.00 | 13582.50 | 13520.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 13409.00 | 13547.80 | 13510.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 13428.00 | 13547.80 | 13510.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 13391.00 | 13486.71 | 13487.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 13171.00 | 13423.57 | 13458.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 13115.00 | 12890.84 | 13068.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 13115.00 | 12890.84 | 13068.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 13115.00 | 12890.84 | 13068.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 13199.00 | 12890.84 | 13068.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 12896.00 | 12891.87 | 13052.55 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 13519.00 | 13141.72 | 13138.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 13670.00 | 13247.37 | 13186.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 13321.00 | 13369.09 | 13321.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 13321.00 | 13369.09 | 13321.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 13321.00 | 13369.09 | 13321.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 13321.00 | 13369.09 | 13321.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 13241.00 | 13343.47 | 13314.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 13241.00 | 13343.47 | 13314.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 13342.00 | 13343.18 | 13316.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 13407.00 | 13347.74 | 13321.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 13555.00 | 13373.39 | 13335.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 12:15:00 | 13211.00 | 13320.90 | 13319.26 | SL hit (close<static) qty=1.00 sl=13230.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 12:15:00 | 13211.00 | 13320.90 | 13319.26 | SL hit (close<static) qty=1.00 sl=13230.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 13217.00 | 13300.12 | 13309.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 12870.00 | 13186.80 | 13253.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 09:15:00 | 12933.00 | 12921.52 | 13051.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:45:00 | 12955.00 | 12921.52 | 13051.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 13017.00 | 12940.61 | 13048.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:30:00 | 13062.00 | 12940.61 | 13048.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 13070.00 | 12982.86 | 13035.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 13070.00 | 12982.86 | 13035.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 13109.00 | 13008.09 | 13042.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 12833.00 | 13008.09 | 13042.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 13067.00 | 12968.76 | 12957.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 13067.00 | 12968.76 | 12957.95 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 12637.00 | 12938.25 | 12957.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 12592.00 | 12869.00 | 12924.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 12710.00 | 12692.03 | 12792.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 12710.00 | 12692.03 | 12792.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 12710.00 | 12692.03 | 12792.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:30:00 | 12665.00 | 12686.54 | 12772.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:00:00 | 12659.00 | 12686.54 | 12772.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 13017.00 | 12716.54 | 12748.75 | SL hit (close>static) qty=1.00 sl=12895.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 13017.00 | 12716.54 | 12748.75 | SL hit (close>static) qty=1.00 sl=12895.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 13024.00 | 12778.03 | 12773.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 13091.00 | 12840.62 | 12802.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 12795.00 | 12856.50 | 12821.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 12795.00 | 12856.50 | 12821.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 12795.00 | 12856.50 | 12821.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 12795.00 | 12856.50 | 12821.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 12838.00 | 12852.80 | 12823.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 12560.00 | 12852.80 | 12823.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 12635.00 | 12809.24 | 12806.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 12506.00 | 12809.24 | 12806.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 12550.00 | 12757.39 | 12782.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 12295.00 | 12586.09 | 12680.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 12192.00 | 12191.27 | 12333.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 12192.00 | 12191.27 | 12333.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 12330.00 | 12225.21 | 12324.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 12340.00 | 12225.21 | 12324.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 12284.00 | 12236.97 | 12320.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 12368.00 | 12236.97 | 12320.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 12351.00 | 12259.78 | 12323.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 12366.00 | 12259.78 | 12323.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 12309.00 | 12269.62 | 12322.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:15:00 | 12333.00 | 12269.62 | 12322.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 12293.00 | 12274.30 | 12319.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 12247.00 | 12265.44 | 12311.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 12466.00 | 12322.04 | 12324.07 | SL hit (close>static) qty=1.00 sl=12390.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 12375.00 | 12332.64 | 12328.70 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 12125.00 | 12303.21 | 12325.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 11961.00 | 12178.13 | 12257.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 11810.00 | 11677.02 | 11809.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 11810.00 | 11677.02 | 11809.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 11810.00 | 11677.02 | 11809.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 11810.00 | 11677.02 | 11809.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 11985.00 | 11738.62 | 11825.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 12015.00 | 11738.62 | 11825.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 12038.00 | 11798.49 | 11845.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 12038.00 | 11798.49 | 11845.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 12002.00 | 11882.16 | 11877.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 12336.00 | 11972.92 | 11919.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 12090.00 | 12137.80 | 12039.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 12090.00 | 12137.80 | 12039.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 11890.00 | 12082.04 | 12030.60 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 11900.00 | 12031.64 | 12033.04 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 15:15:00 | 12060.00 | 12032.36 | 12031.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 12722.00 | 12170.28 | 12094.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 12178.00 | 12443.88 | 12314.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 12178.00 | 12443.88 | 12314.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 12178.00 | 12443.88 | 12314.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 12178.00 | 12443.88 | 12314.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 12266.00 | 12408.31 | 12310.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 12273.00 | 12408.31 | 12310.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 09:15:00 | 13500.30 | 13034.83 | 12753.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 14900.00 | 15119.10 | 15126.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 14776.00 | 15050.48 | 15094.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 15043.00 | 15007.31 | 15064.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 12:00:00 | 15043.00 | 15007.31 | 15064.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 15266.00 | 15059.04 | 15083.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 15266.00 | 15059.04 | 15083.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 15115.00 | 15070.24 | 15086.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:30:00 | 15318.00 | 15070.24 | 15086.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 15042.00 | 15066.15 | 15081.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 15120.00 | 15066.15 | 15081.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 15368.00 | 15126.52 | 15107.62 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 14950.00 | 15092.35 | 15106.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 14870.00 | 15047.88 | 15085.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 14748.00 | 14655.67 | 14825.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 14748.00 | 14655.67 | 14825.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 14748.00 | 14655.67 | 14825.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 14782.00 | 14655.67 | 14825.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 14809.00 | 14686.34 | 14823.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 14943.00 | 14686.34 | 14823.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 14871.00 | 14723.27 | 14827.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 14871.00 | 14723.27 | 14827.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 14887.00 | 14756.02 | 14833.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 14887.00 | 14756.02 | 14833.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 14888.00 | 14782.41 | 14838.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 14840.00 | 14782.41 | 14838.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:45:00 | 14836.00 | 14770.93 | 14827.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 14847.00 | 14709.64 | 14707.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 14847.00 | 14709.64 | 14707.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 14847.00 | 14709.64 | 14707.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 14980.00 | 14799.61 | 14751.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 17153.00 | 17182.90 | 16810.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:15:00 | 17563.00 | 17182.90 | 16810.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 10:15:00 | 12665.00 | 2025-05-14 14:15:00 | 12480.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-14 12:00:00 | 12672.00 | 2025-05-14 14:15:00 | 12480.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-23 15:15:00 | 11110.00 | 2025-05-26 11:15:00 | 11315.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-05-29 12:30:00 | 11190.00 | 2025-05-29 13:15:00 | 11368.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-06-04 13:15:00 | 11728.00 | 2025-06-06 09:15:00 | 12900.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-05 15:00:00 | 11740.00 | 2025-06-06 09:15:00 | 12914.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-20 12:15:00 | 12315.00 | 2025-07-03 10:15:00 | 11699.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-23 13:00:00 | 12315.00 | 2025-07-03 10:15:00 | 11699.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-20 12:15:00 | 12315.00 | 2025-07-04 11:15:00 | 11918.00 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-06-23 13:00:00 | 12315.00 | 2025-07-04 11:15:00 | 11918.00 | STOP_HIT | 0.50 | 3.22% |
| BUY | retest2 | 2025-07-17 09:15:00 | 14459.00 | 2025-07-21 09:15:00 | 14160.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-17 10:45:00 | 14530.00 | 2025-07-21 09:15:00 | 14160.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-07-18 10:00:00 | 14496.00 | 2025-07-21 09:15:00 | 14160.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-07-24 13:15:00 | 13970.00 | 2025-07-28 11:15:00 | 14020.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-30 10:45:00 | 13801.00 | 2025-07-30 13:15:00 | 13922.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-05 10:15:00 | 13003.00 | 2025-08-08 09:15:00 | 13418.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-08-05 12:45:00 | 13000.00 | 2025-08-08 09:15:00 | 13418.00 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-08-22 13:15:00 | 13870.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-08-22 14:30:00 | 13849.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-22 15:15:00 | 13850.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-25 09:30:00 | 13918.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-09-08 11:00:00 | 14479.00 | 2025-09-12 09:15:00 | 15926.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-08 11:30:00 | 14471.00 | 2025-09-12 09:15:00 | 15918.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-08 14:30:00 | 14485.00 | 2025-09-12 09:15:00 | 15933.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 10:30:00 | 14448.00 | 2025-09-12 09:15:00 | 15892.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 14:00:00 | 14749.00 | 2025-09-16 09:15:00 | 16223.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 14372.00 | 2025-09-30 14:15:00 | 14618.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-07 10:15:00 | 15466.00 | 2025-10-17 09:15:00 | 16147.00 | STOP_HIT | 1.00 | 4.40% |
| SELL | retest2 | 2025-10-28 12:45:00 | 15713.00 | 2025-10-30 13:15:00 | 16056.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-29 09:15:00 | 15757.00 | 2025-10-30 13:15:00 | 16056.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-10-30 11:15:00 | 15740.00 | 2025-10-30 13:15:00 | 16056.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-11-10 09:15:00 | 18721.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -7.61% |
| BUY | retest2 | 2025-11-10 10:30:00 | 18160.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-11-10 12:00:00 | 18190.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2025-11-10 14:30:00 | 18102.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-11-13 13:00:00 | 17335.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-11-13 14:00:00 | 17297.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-11-14 09:30:00 | 17333.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-14 10:30:00 | 17315.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-12-12 12:15:00 | 16171.00 | 2025-12-15 13:15:00 | 16426.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-19 13:15:00 | 15890.00 | 2025-12-19 13:15:00 | 15996.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-31 11:15:00 | 14865.00 | 2025-12-31 14:15:00 | 15210.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-12-31 11:45:00 | 14867.00 | 2025-12-31 14:15:00 | 15210.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-01-13 10:30:00 | 14350.00 | 2026-01-20 10:15:00 | 13632.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 14333.00 | 2026-01-20 10:15:00 | 13616.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 14345.00 | 2026-01-20 10:15:00 | 13627.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:00:00 | 14351.00 | 2026-01-20 10:15:00 | 13633.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:30:00 | 14350.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2026-01-13 13:30:00 | 14333.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2026-01-14 13:30:00 | 14345.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.34% |
| SELL | retest2 | 2026-01-16 11:00:00 | 14351.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2026-01-22 15:15:00 | 13521.00 | 2026-01-27 11:15:00 | 12844.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 15:15:00 | 13521.00 | 2026-01-30 09:15:00 | 12761.00 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2026-02-13 15:15:00 | 13045.00 | 2026-02-16 09:15:00 | 13329.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-26 09:15:00 | 13407.00 | 2026-02-26 12:15:00 | 13211.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-02-26 09:45:00 | 13555.00 | 2026-02-26 12:15:00 | 13211.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-03-04 09:15:00 | 12833.00 | 2026-03-06 10:15:00 | 13067.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-03-10 11:30:00 | 12665.00 | 2026-03-11 09:15:00 | 13017.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-03-10 12:00:00 | 12659.00 | 2026-03-11 09:15:00 | 13017.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2026-03-17 13:30:00 | 12247.00 | 2026-03-18 10:15:00 | 12466.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-04-02 11:15:00 | 12273.00 | 2026-04-07 09:15:00 | 13500.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 14:15:00 | 14840.00 | 2026-04-30 10:15:00 | 14847.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-04-27 14:45:00 | 14836.00 | 2026-04-30 10:15:00 | 14847.00 | STOP_HIT | 1.00 | -0.07% |

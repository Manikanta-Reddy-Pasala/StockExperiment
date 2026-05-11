# Honeywell Automation India Ltd. (HONAUT)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 30210.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 92 |
| ALERT2 | 90 |
| ALERT2_SKIP | 46 |
| ALERT3 | 231 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 127 |
| PARTIAL | 13 |
| TARGET_HIT | 2 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 97
- **Target hits / Stop hits / Partials:** 2 / 127 / 13
- **Avg / median % per leg:** 0.32% / -0.57%
- **Sum % (uncompounded):** 46.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 4 | 9.3% | 2 | 41 | 0 | -0.86% | -36.9% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.03% | 0.1% |
| BUY @ 3rd Alert (retest2) | 41 | 3 | 7.3% | 2 | 39 | 0 | -0.90% | -36.9% |
| SELL (all) | 99 | 41 | 41.4% | 0 | 86 | 13 | 0.84% | 83.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 99 | 41 | 41.4% | 0 | 86 | 13 | 0.84% | 83.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.03% | 0.1% |
| retest2 (combined) | 140 | 44 | 31.4% | 2 | 125 | 13 | 0.33% | 46.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 52300.00 | 53027.43 | 53125.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 11:15:00 | 51599.90 | 52363.10 | 52765.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 51569.45 | 51548.82 | 52080.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 51999.50 | 51647.14 | 52033.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 51999.50 | 51647.14 | 52033.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 52364.80 | 51647.14 | 52033.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 51858.05 | 51689.32 | 52017.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 15:15:00 | 51600.00 | 51744.04 | 52012.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 52430.30 | 51858.24 | 52016.65 | SL hit (close>static) qty=1.00 sl=52170.05 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 52485.20 | 52088.66 | 52088.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 51969.95 | 52256.40 | 52262.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 12:15:00 | 51575.00 | 52050.30 | 52163.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 52898.70 | 51963.87 | 52057.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 52898.70 | 51963.87 | 52057.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 52898.70 | 51963.87 | 52057.79 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 52694.20 | 52193.76 | 52151.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 52865.00 | 52328.01 | 52216.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 52300.40 | 52622.48 | 52423.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 52300.40 | 52622.48 | 52423.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 52300.40 | 52622.48 | 52423.51 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 50455.00 | 52188.99 | 52244.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 49385.95 | 51628.38 | 51984.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 50957.60 | 50878.27 | 51363.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 50957.60 | 50878.27 | 51363.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 51066.35 | 50777.35 | 51185.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 51066.35 | 50777.35 | 51185.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 50594.35 | 50740.75 | 51131.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 50727.65 | 50740.75 | 51131.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 52118.40 | 51048.96 | 51205.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 52118.40 | 51048.96 | 51205.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 52523.75 | 51343.92 | 51325.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 53450.00 | 52136.18 | 51751.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 52299.05 | 52444.37 | 52082.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 14:15:00 | 52299.05 | 52444.37 | 52082.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 52299.05 | 52444.37 | 52082.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 14:45:00 | 52371.00 | 52444.37 | 52082.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 52650.00 | 52485.50 | 52134.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:30:00 | 52670.00 | 52499.40 | 52172.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:15:00 | 53632.80 | 52499.40 | 52172.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-14 15:15:00 | 57937.00 | 56284.64 | 55349.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 55237.30 | 55985.72 | 56033.60 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 56375.00 | 56029.32 | 56022.37 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 13:15:00 | 55834.85 | 55985.10 | 56003.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 14:15:00 | 55330.10 | 55854.10 | 55941.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 55932.35 | 55765.33 | 55879.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 55932.35 | 55765.33 | 55879.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 55932.35 | 55765.33 | 55879.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 55587.00 | 55878.29 | 55908.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 55500.00 | 55802.63 | 55871.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 56521.65 | 55915.61 | 55909.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 56521.65 | 55915.61 | 55909.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 14:15:00 | 57080.00 | 56596.19 | 56288.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 57224.55 | 57304.81 | 56848.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 14:00:00 | 57224.55 | 57304.81 | 56848.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 57301.00 | 58038.18 | 57587.15 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 10:15:00 | 57508.35 | 57581.81 | 57583.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 13:15:00 | 57299.95 | 57501.38 | 57544.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 15:15:00 | 57000.00 | 56815.27 | 57046.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:15:00 | 56886.05 | 56815.27 | 57046.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 56499.95 | 56752.21 | 56996.85 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 57094.20 | 56872.07 | 56865.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 12:15:00 | 58680.00 | 57233.66 | 57030.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 11:15:00 | 57700.00 | 57987.35 | 57590.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 12:00:00 | 57700.00 | 57987.35 | 57590.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 57364.10 | 57862.70 | 57569.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 57364.10 | 57862.70 | 57569.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 57663.50 | 57822.86 | 57578.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 57799.95 | 57775.49 | 57578.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:45:00 | 57772.05 | 57763.09 | 57607.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 56588.70 | 57528.21 | 57515.05 | SL hit (close<static) qty=1.00 sl=57285.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 57400.35 | 57502.64 | 57504.63 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 57875.00 | 57563.63 | 57525.21 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 57069.95 | 57484.64 | 57497.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 12:15:00 | 56900.00 | 57367.71 | 57442.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 56950.00 | 56698.90 | 56954.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 56950.00 | 56698.90 | 56954.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 56950.00 | 56698.90 | 56954.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 56950.00 | 56698.90 | 56954.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 57036.00 | 56766.32 | 56961.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 57127.90 | 56766.32 | 56961.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 56800.00 | 56773.06 | 56947.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 56521.00 | 56843.37 | 56911.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 56249.85 | 56841.57 | 56897.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 56499.95 | 56852.86 | 56897.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 53694.95 | 54939.28 | 55588.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 53437.36 | 54939.28 | 55588.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 53674.95 | 54939.28 | 55588.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 12:15:00 | 53981.50 | 53950.77 | 54540.69 | SL hit (close>ema200) qty=0.50 sl=53950.77 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 55116.70 | 54503.81 | 54471.68 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 53899.10 | 54360.41 | 54417.56 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 54925.10 | 54389.72 | 54359.30 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 54228.95 | 54398.13 | 54400.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 53900.00 | 54208.61 | 54306.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 54300.00 | 54226.88 | 54305.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 54300.00 | 54226.88 | 54305.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 54300.00 | 54226.88 | 54305.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 12:45:00 | 53860.10 | 54108.83 | 54231.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 56499.95 | 54483.38 | 54350.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 56499.95 | 54483.38 | 54350.49 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 54139.15 | 54407.68 | 54429.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 15:15:00 | 54100.00 | 54292.22 | 54367.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 15:15:00 | 51915.00 | 51665.45 | 52118.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 09:15:00 | 51898.95 | 51665.45 | 52118.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 52154.15 | 51763.19 | 52122.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:30:00 | 52134.35 | 51763.19 | 52122.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 51950.00 | 51800.55 | 52106.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:00:00 | 51623.00 | 51765.04 | 52062.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 51500.00 | 51545.42 | 51680.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 10:15:00 | 52050.00 | 51435.63 | 51364.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 52050.00 | 51435.63 | 51364.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 52276.05 | 51678.81 | 51519.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 15:15:00 | 51865.05 | 51921.80 | 51734.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:15:00 | 52316.65 | 51921.80 | 51734.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 52001.15 | 51937.67 | 51759.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 51993.95 | 51937.67 | 51759.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 52100.00 | 52143.05 | 51950.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 53017.10 | 52117.64 | 51956.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 52700.40 | 52448.60 | 52260.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:45:00 | 52499.05 | 52419.48 | 52291.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 52223.95 | 52380.37 | 52285.53 | SL hit (close<ema400) qty=1.00 sl=52285.53 alert=retest1 |

### Cycle 23 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 52111.30 | 52274.74 | 52288.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 52007.30 | 52185.30 | 52243.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 13:15:00 | 52206.05 | 52189.45 | 52239.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 14:00:00 | 52206.05 | 52189.45 | 52239.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 52179.80 | 52187.52 | 52234.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 52200.00 | 52187.52 | 52234.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 52136.95 | 52177.40 | 52225.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 52590.00 | 52177.40 | 52225.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 52344.95 | 52210.91 | 52236.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 52485.80 | 52210.91 | 52236.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 52579.75 | 52284.68 | 52267.49 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 51911.05 | 52256.04 | 52279.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 51687.60 | 52097.36 | 52200.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 51044.15 | 51003.86 | 51346.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 51044.15 | 51003.86 | 51346.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 51044.15 | 51003.86 | 51346.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:45:00 | 50698.25 | 50935.05 | 51104.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 50688.30 | 50762.17 | 50969.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 14:15:00 | 50721.20 | 50850.49 | 50953.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 15:00:00 | 50709.00 | 50822.19 | 50931.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 50612.55 | 50792.71 | 50899.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:30:00 | 50855.75 | 50792.71 | 50899.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 50775.45 | 50566.23 | 50711.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 50787.75 | 50566.23 | 50711.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 50471.25 | 50547.24 | 50690.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 50122.85 | 50521.20 | 50665.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 50618.40 | 49872.45 | 49826.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 50618.40 | 49872.45 | 49826.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 51153.50 | 50373.22 | 50140.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 10:15:00 | 51100.10 | 51164.59 | 50864.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 11:00:00 | 51100.10 | 51164.59 | 50864.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 50928.00 | 51117.27 | 50869.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:45:00 | 50862.30 | 51117.27 | 50869.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 50850.00 | 51063.82 | 50868.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 50850.00 | 51063.82 | 50868.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 50611.80 | 50973.41 | 50844.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:45:00 | 50522.80 | 50973.41 | 50844.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 50400.15 | 50858.76 | 50804.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 50400.15 | 50858.76 | 50804.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 49893.10 | 50598.62 | 50691.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 49719.00 | 50304.20 | 50534.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 49950.00 | 49886.84 | 50175.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 10:15:00 | 49950.00 | 49886.84 | 50175.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 49950.00 | 49886.84 | 50175.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 50073.85 | 49886.84 | 50175.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 49700.00 | 49905.30 | 50069.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:30:00 | 49618.75 | 49829.91 | 50004.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:30:00 | 49612.15 | 49765.78 | 49943.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:45:00 | 49450.05 | 49644.41 | 49844.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 49627.90 | 49567.72 | 49695.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 49347.75 | 49523.73 | 49663.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:15:00 | 49204.65 | 49455.40 | 49595.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 49755.00 | 49619.37 | 49618.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 49755.00 | 49619.37 | 49618.64 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 14:15:00 | 49405.65 | 49603.98 | 49615.11 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 49683.75 | 49622.80 | 49621.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 12:15:00 | 49860.95 | 49685.18 | 49650.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 49684.25 | 49742.35 | 49694.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 49684.25 | 49742.35 | 49694.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 49684.25 | 49742.35 | 49694.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:45:00 | 49580.35 | 49742.35 | 49694.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 49584.10 | 49710.70 | 49684.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 49584.10 | 49710.70 | 49684.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 49425.00 | 49653.56 | 49661.09 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 49925.00 | 49648.99 | 49642.40 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 49300.00 | 49579.19 | 49611.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 49080.05 | 49203.96 | 49362.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 11:15:00 | 49115.15 | 48941.23 | 49133.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:00:00 | 49115.15 | 48941.23 | 49133.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 49193.35 | 48991.65 | 49138.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 49193.35 | 48991.65 | 49138.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 49244.90 | 49042.30 | 49148.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 49215.05 | 49042.30 | 49148.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 49000.00 | 49033.84 | 49134.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:15:00 | 48739.00 | 49033.84 | 49134.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 48739.00 | 48974.87 | 49098.84 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 49357.95 | 49119.45 | 49116.03 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 48948.40 | 49099.23 | 49114.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 48700.00 | 49019.38 | 49076.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 48574.20 | 48563.88 | 48767.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 48574.20 | 48563.88 | 48767.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 47700.00 | 48203.82 | 48497.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 47247.05 | 47991.11 | 48373.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 14:30:00 | 47544.55 | 47828.38 | 48170.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 48948.00 | 48239.36 | 48179.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 48948.00 | 48239.36 | 48179.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 49349.90 | 48810.97 | 48501.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 10:15:00 | 49907.25 | 50172.30 | 49870.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 11:00:00 | 49907.25 | 50172.30 | 49870.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 50009.20 | 50139.68 | 49882.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 50067.10 | 50024.93 | 49885.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 15:15:00 | 49735.25 | 49967.00 | 49871.70 | SL hit (close<static) qty=1.00 sl=49802.65 alert=retest2 |

### Cycle 37 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 49670.00 | 49844.48 | 49861.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 49630.00 | 49777.32 | 49826.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 49763.05 | 49737.42 | 49792.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 13:45:00 | 49700.15 | 49737.42 | 49792.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 49652.00 | 49720.34 | 49779.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:30:00 | 49749.15 | 49720.34 | 49779.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 49719.95 | 49720.26 | 49774.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 49550.20 | 49720.26 | 49774.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 49255.50 | 49627.31 | 49727.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:00:00 | 49050.00 | 49511.85 | 49665.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 12:00:00 | 49100.00 | 49429.48 | 49614.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 49936.05 | 49612.34 | 49640.54 | SL hit (close>static) qty=1.00 sl=49850.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 50388.25 | 49767.52 | 49708.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 50481.45 | 49910.31 | 49778.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 51299.95 | 51393.48 | 51005.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 12:00:00 | 51299.95 | 51393.48 | 51005.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 51056.40 | 51272.06 | 51090.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 50909.95 | 51272.06 | 51090.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 50600.25 | 51137.70 | 51046.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 50600.25 | 51137.70 | 51046.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 50400.00 | 50990.16 | 50987.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 50400.00 | 50990.16 | 50987.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 12:15:00 | 50700.00 | 50932.13 | 50961.33 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 10:15:00 | 51111.15 | 50949.32 | 50944.81 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 50596.90 | 50883.13 | 50915.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 50063.30 | 50576.17 | 50748.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 13:15:00 | 48849.50 | 48672.03 | 49166.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 13:30:00 | 48734.20 | 48672.03 | 49166.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 49100.00 | 48789.28 | 49136.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 45497.10 | 48789.28 | 49136.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 14:15:00 | 43222.24 | 43467.14 | 43768.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 42700.45 | 42458.61 | 42927.20 | SL hit (close>ema200) qty=0.50 sl=42458.61 alert=retest2 |

### Cycle 42 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 41890.00 | 41552.83 | 41527.46 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 15:15:00 | 41377.00 | 41521.97 | 41527.99 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 42050.00 | 41627.58 | 41575.45 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 41350.00 | 41510.66 | 41532.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 41000.00 | 41408.53 | 41483.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 15:15:00 | 40880.00 | 40861.42 | 41089.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 09:15:00 | 40908.15 | 40861.42 | 41089.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 40966.05 | 40882.34 | 41077.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 11:15:00 | 40468.55 | 40890.81 | 41063.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 41500.05 | 41006.84 | 41016.61 | SL hit (close>static) qty=1.00 sl=41496.50 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 41110.30 | 41027.53 | 41025.13 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 40901.15 | 41001.81 | 41013.82 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 41130.00 | 41039.87 | 41028.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 41420.05 | 41115.91 | 41063.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 40928.45 | 41078.41 | 41051.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 10:15:00 | 40928.45 | 41078.41 | 41051.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 40928.45 | 41078.41 | 41051.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 40928.45 | 41078.41 | 41051.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 40761.80 | 41015.09 | 41025.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 40579.40 | 40862.15 | 40941.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 40798.00 | 40793.28 | 40893.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 12:00:00 | 40798.00 | 40793.28 | 40893.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 41051.90 | 40845.00 | 40907.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:00:00 | 41051.90 | 40845.00 | 40907.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 40950.15 | 40866.03 | 40911.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 40865.05 | 40918.42 | 40931.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:45:00 | 40827.10 | 40903.43 | 40923.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 12:15:00 | 40894.15 | 40894.24 | 40914.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 14:30:00 | 40742.55 | 40853.38 | 40891.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 40529.65 | 40736.89 | 40829.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 41100.00 | 40841.26 | 40840.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 41100.00 | 40841.26 | 40840.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 11:15:00 | 41626.15 | 41070.81 | 40949.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 41157.05 | 41495.37 | 41347.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 41157.05 | 41495.37 | 41347.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 41157.05 | 41495.37 | 41347.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 41175.25 | 41495.37 | 41347.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 40971.50 | 41390.59 | 41313.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 40971.50 | 41390.59 | 41313.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 40864.95 | 41219.38 | 41244.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 40520.30 | 40940.32 | 41092.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 11:15:00 | 40849.35 | 40734.64 | 40860.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 11:45:00 | 40804.20 | 40734.64 | 40860.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 40850.05 | 40778.29 | 40850.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:30:00 | 40879.00 | 40778.29 | 40850.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 40850.10 | 40792.65 | 40850.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 41300.00 | 40792.65 | 40850.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 40606.60 | 40755.44 | 40828.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:00:00 | 40407.75 | 40695.98 | 40784.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 41330.00 | 40513.87 | 40487.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 41330.00 | 40513.87 | 40487.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 09:15:00 | 42215.65 | 41215.68 | 41040.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 41600.00 | 41933.38 | 41595.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 41600.00 | 41933.38 | 41595.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 41600.00 | 41933.38 | 41595.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 41600.00 | 41933.38 | 41595.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 41503.25 | 41847.36 | 41586.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 41474.80 | 41847.36 | 41586.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 41450.00 | 41767.88 | 41574.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 41307.15 | 41767.88 | 41574.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 41546.20 | 41704.41 | 41577.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 41546.20 | 41704.41 | 41577.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 41391.40 | 41641.81 | 41560.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 41391.40 | 41641.81 | 41560.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 41151.00 | 41543.65 | 41523.71 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 41176.55 | 41470.23 | 41492.15 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 42174.50 | 41601.14 | 41543.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 42310.10 | 41962.10 | 41797.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 15:15:00 | 42100.00 | 42149.26 | 41987.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:15:00 | 42399.85 | 42149.26 | 41987.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 42500.65 | 42939.66 | 42705.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 42500.65 | 42939.66 | 42705.19 | SL hit (close<ema400) qty=1.00 sl=42705.19 alert=retest1 |

### Cycle 55 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 42084.30 | 42492.27 | 42537.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 41641.10 | 42322.04 | 42456.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 42719.05 | 42317.91 | 42426.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 42719.05 | 42317.91 | 42426.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 42719.05 | 42317.91 | 42426.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 42944.00 | 42317.91 | 42426.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 42599.45 | 42374.22 | 42441.83 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 42840.30 | 42538.62 | 42506.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 09:15:00 | 42868.30 | 42624.39 | 42555.76 | Break + close above crossover candle high |

### Cycle 57 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 42044.75 | 42508.47 | 42509.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 14:15:00 | 41976.20 | 42228.45 | 42360.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 42278.85 | 42177.18 | 42310.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 42278.85 | 42177.18 | 42310.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 42278.85 | 42177.18 | 42310.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:30:00 | 41807.90 | 42052.44 | 42197.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 41179.95 | 40561.27 | 40532.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 41179.95 | 40561.27 | 40532.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 41524.75 | 40753.96 | 40622.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 40918.90 | 41428.53 | 41253.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 40918.90 | 41428.53 | 41253.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 40918.90 | 41428.53 | 41253.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:15:00 | 40602.15 | 41428.53 | 41253.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 39528.30 | 41048.48 | 41096.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 39169.75 | 39985.20 | 40466.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 40708.75 | 39518.35 | 39887.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 40708.75 | 39518.35 | 39887.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 40708.75 | 39518.35 | 39887.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 40708.75 | 39518.35 | 39887.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 40619.95 | 39738.67 | 39953.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 40639.65 | 39738.67 | 39953.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 40375.00 | 40082.41 | 40076.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 40640.20 | 40202.46 | 40134.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 40121.00 | 40499.50 | 40372.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 15:15:00 | 40121.00 | 40499.50 | 40372.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 40121.00 | 40499.50 | 40372.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 40278.70 | 40465.74 | 40368.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 40215.05 | 40415.60 | 40354.75 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 40101.35 | 40285.15 | 40303.88 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 40496.10 | 40327.34 | 40321.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 40920.10 | 40454.02 | 40380.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 39650.05 | 40395.54 | 40383.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 39650.05 | 40395.54 | 40383.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 39650.05 | 40395.54 | 40383.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 39650.05 | 40395.54 | 40383.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 39550.00 | 40226.43 | 40307.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 39050.05 | 39685.53 | 39997.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 38202.40 | 38195.22 | 38753.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 38202.40 | 38195.22 | 38753.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 38654.60 | 38279.98 | 38615.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:45:00 | 38699.25 | 38279.98 | 38615.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 38298.80 | 38283.74 | 38586.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 13:30:00 | 38200.00 | 38254.99 | 38546.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 38248.20 | 38358.77 | 38481.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:30:00 | 38224.80 | 38345.08 | 38463.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:15:00 | 38240.80 | 38345.08 | 38463.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 38073.00 | 38251.77 | 38388.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 37759.15 | 38287.47 | 38341.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 36290.00 | 37194.01 | 37635.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 36335.79 | 37194.01 | 37635.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 36313.56 | 37194.01 | 37635.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 36328.76 | 37194.01 | 37635.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 35871.19 | 36420.53 | 36971.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 09:15:00 | 35321.45 | 35205.65 | 35637.77 | SL hit (close>ema200) qty=0.50 sl=35205.65 alert=retest2 |

### Cycle 64 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 33782.80 | 33626.17 | 33613.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 14:15:00 | 33899.00 | 33697.14 | 33654.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 33743.70 | 33751.20 | 33693.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-28 11:00:00 | 33743.70 | 33751.20 | 33693.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 33761.10 | 33753.18 | 33700.08 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 33555.55 | 33670.00 | 33676.16 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-03-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 11:15:00 | 33760.50 | 33696.10 | 33687.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 13:15:00 | 33985.50 | 33753.01 | 33714.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 15:15:00 | 33700.05 | 33765.91 | 33728.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 15:15:00 | 33700.05 | 33765.91 | 33728.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 33700.05 | 33765.91 | 33728.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 33830.00 | 33765.91 | 33728.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 33779.25 | 33768.58 | 33733.31 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 33578.90 | 33706.87 | 33709.91 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 34128.60 | 33753.28 | 33721.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 34378.20 | 33878.26 | 33781.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 35540.15 | 35593.05 | 35162.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 35540.15 | 35593.05 | 35162.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 35540.15 | 35593.05 | 35162.92 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 34470.00 | 35025.86 | 35048.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 34112.65 | 34843.22 | 34963.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 34325.00 | 34153.56 | 34368.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 34325.00 | 34153.56 | 34368.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 34325.00 | 34153.56 | 34368.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 34325.00 | 34153.56 | 34368.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 34311.50 | 34201.54 | 34353.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:30:00 | 33947.90 | 34099.42 | 34255.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 15:00:00 | 33930.95 | 33820.93 | 33890.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 34375.00 | 33938.00 | 33932.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 34375.00 | 33938.00 | 33932.41 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 09:15:00 | 34062.00 | 34198.77 | 34206.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 10:15:00 | 33885.70 | 34136.15 | 34177.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 34100.00 | 34022.70 | 34089.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 09:15:00 | 34100.00 | 34022.70 | 34089.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 34100.00 | 34022.70 | 34089.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 14:45:00 | 33243.35 | 33844.98 | 33974.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:45:00 | 33759.15 | 33789.05 | 33911.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 14:15:00 | 31581.18 | 33570.43 | 33763.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 14:15:00 | 32071.19 | 33570.43 | 33763.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 33710.60 | 33478.68 | 33631.52 | SL hit (close>ema200) qty=0.50 sl=33478.68 alert=retest2 |

### Cycle 72 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 33900.00 | 33705.86 | 33680.90 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 33633.20 | 33736.58 | 33737.81 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 34108.80 | 33803.12 | 33767.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 34225.95 | 33983.29 | 33867.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 33626.55 | 34160.64 | 34079.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 33626.55 | 34160.64 | 34079.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 33626.55 | 34160.64 | 34079.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 33626.55 | 34160.64 | 34079.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 33604.40 | 34049.39 | 34036.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:45:00 | 33602.25 | 34049.39 | 34036.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 33411.80 | 33921.88 | 33979.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 32762.35 | 33535.42 | 33753.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 33056.05 | 33031.36 | 33312.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 33056.05 | 33031.36 | 33312.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 33261.50 | 33077.39 | 33307.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:30:00 | 33211.45 | 33077.39 | 33307.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 33606.05 | 33183.12 | 33334.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 33606.05 | 33183.12 | 33334.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 33543.10 | 33255.12 | 33353.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:00:00 | 33543.10 | 33255.12 | 33353.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 33441.00 | 33369.02 | 33388.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:30:00 | 33499.70 | 33369.02 | 33388.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 33700.00 | 33435.21 | 33416.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 33797.00 | 33551.22 | 33475.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 15:15:00 | 34000.00 | 34098.03 | 33970.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:15:00 | 34250.00 | 34098.03 | 33970.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 34020.00 | 34082.43 | 33975.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:15:00 | 34040.00 | 34082.43 | 33975.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 34090.00 | 34083.94 | 33985.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:30:00 | 34145.00 | 34094.12 | 34007.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:00:00 | 34190.00 | 34113.30 | 34024.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:30:00 | 34270.00 | 34153.53 | 34067.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 34100.00 | 34434.16 | 34478.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 34100.00 | 34434.16 | 34478.14 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 34900.00 | 34483.07 | 34451.00 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 33830.00 | 34436.91 | 34455.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 33620.00 | 34273.53 | 34379.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 34305.00 | 34156.09 | 34287.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 34305.00 | 34156.09 | 34287.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 34305.00 | 34156.09 | 34287.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 34305.00 | 34156.09 | 34287.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 34480.00 | 34220.87 | 34304.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:45:00 | 34560.00 | 34220.87 | 34304.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 34485.00 | 34132.12 | 34190.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:00:00 | 34485.00 | 34132.12 | 34190.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 34400.00 | 34185.69 | 34209.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:30:00 | 34300.00 | 34194.56 | 34211.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 10:15:00 | 34435.00 | 34238.00 | 34222.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 34435.00 | 34238.00 | 34222.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 13:15:00 | 34570.00 | 34371.86 | 34293.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 34400.00 | 34409.99 | 34326.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 09:15:00 | 34430.00 | 34409.99 | 34326.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 34565.00 | 34440.99 | 34348.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 11:00:00 | 34865.00 | 34525.79 | 34395.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:30:00 | 34850.00 | 34648.69 | 34487.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 35060.00 | 34701.56 | 34540.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 11:00:00 | 34885.00 | 34771.60 | 34602.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 34905.00 | 34932.82 | 34770.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 34670.00 | 34932.82 | 34770.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 34845.00 | 34896.88 | 34804.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 34845.00 | 34896.88 | 34804.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 34935.00 | 34904.50 | 34815.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:30:00 | 34840.00 | 34904.50 | 34815.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 35005.00 | 34918.28 | 34837.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 09:30:00 | 35635.00 | 35277.58 | 35078.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:00:00 | 35355.00 | 35293.07 | 35103.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 34205.00 | 34905.37 | 34980.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 34205.00 | 34905.37 | 34980.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 12:15:00 | 33945.00 | 34497.99 | 34757.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 35400.00 | 34550.47 | 34682.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 35400.00 | 34550.47 | 34682.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 35400.00 | 34550.47 | 34682.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 35400.00 | 34550.47 | 34682.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 35160.00 | 34672.38 | 34725.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:30:00 | 34935.00 | 34731.90 | 34748.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 35095.00 | 34804.52 | 34779.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 35095.00 | 34804.52 | 34779.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 35250.00 | 34952.09 | 34854.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 38150.00 | 38360.84 | 37638.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 37990.00 | 38360.84 | 37638.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 37690.00 | 38069.46 | 37824.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 37625.00 | 38069.46 | 37824.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 37725.00 | 38000.57 | 37815.05 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 37070.00 | 37652.31 | 37698.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 37010.00 | 37523.85 | 37635.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 37790.00 | 37577.08 | 37649.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 37790.00 | 37577.08 | 37649.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 37790.00 | 37577.08 | 37649.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 37790.00 | 37577.08 | 37649.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 37715.00 | 37604.66 | 37655.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 37780.00 | 37604.66 | 37655.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 37625.00 | 37607.99 | 37648.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 37650.00 | 37607.99 | 37648.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 37680.00 | 37622.39 | 37651.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 37680.00 | 37622.39 | 37651.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 37830.00 | 37663.91 | 37667.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 37830.00 | 37663.91 | 37667.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 37785.00 | 37688.13 | 37678.14 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 37475.00 | 37645.88 | 37661.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 37295.00 | 37575.71 | 37627.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 37375.00 | 37359.75 | 37483.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 37375.00 | 37359.75 | 37483.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 37395.00 | 37366.80 | 37475.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 37430.00 | 37366.80 | 37475.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 37210.00 | 37335.44 | 37451.33 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 37500.00 | 37462.84 | 37460.38 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 37205.00 | 37422.02 | 37442.92 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 37735.00 | 37466.75 | 37455.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 37975.00 | 37568.40 | 37502.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 38005.00 | 38105.66 | 37912.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 38005.00 | 38105.66 | 37912.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 37900.00 | 38064.53 | 37911.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 38240.00 | 38064.53 | 37911.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 38235.00 | 38815.53 | 38835.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 38235.00 | 38815.53 | 38835.24 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 39440.00 | 38888.21 | 38830.71 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 09:15:00 | 38990.00 | 39058.13 | 39062.69 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 39120.00 | 39070.51 | 39067.90 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 38950.00 | 39055.12 | 39061.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 38900.00 | 39011.42 | 39039.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 10:15:00 | 38980.00 | 38893.76 | 38944.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 10:15:00 | 38980.00 | 38893.76 | 38944.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 38980.00 | 38893.76 | 38944.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 38985.00 | 38893.76 | 38944.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 38810.00 | 38877.01 | 38932.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 39015.00 | 38877.01 | 38932.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 38125.00 | 38119.66 | 38377.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 37855.00 | 38087.73 | 38339.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 37995.00 | 38097.18 | 38321.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 38005.00 | 38078.20 | 38272.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 38020.00 | 38097.60 | 38203.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 37925.00 | 38004.43 | 38119.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 38110.00 | 38004.43 | 38119.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 37905.00 | 37984.55 | 38100.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 37780.00 | 37887.09 | 38011.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 37685.00 | 37846.67 | 37981.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 37750.00 | 37796.67 | 37933.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 37795.00 | 37680.10 | 37774.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 37705.00 | 37709.07 | 37772.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:15:00 | 37635.00 | 37709.07 | 37772.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 37875.00 | 37690.30 | 37727.51 | SL hit (close>static) qty=1.00 sl=37835.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 38430.00 | 37905.85 | 37822.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 38630.00 | 38647.81 | 38465.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:00:00 | 38630.00 | 38647.81 | 38465.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 38925.00 | 39287.98 | 39121.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 38925.00 | 39287.98 | 39121.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 38615.00 | 39153.38 | 39075.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 38560.00 | 39153.38 | 39075.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 38825.00 | 38997.13 | 39013.84 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 40330.00 | 39151.43 | 39039.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 40835.00 | 39488.14 | 39202.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 40700.00 | 40731.38 | 40200.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 13:00:00 | 40700.00 | 40731.38 | 40200.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 40685.00 | 40823.52 | 40715.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 40720.00 | 40823.52 | 40715.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 40650.00 | 40788.82 | 40709.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 40615.00 | 40788.82 | 40709.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 40625.00 | 40756.06 | 40701.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 40625.00 | 40756.06 | 40701.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 40470.00 | 40698.84 | 40680.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 40355.00 | 40698.84 | 40680.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 40485.00 | 40656.08 | 40663.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 40280.00 | 40517.69 | 40590.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 40670.00 | 40516.52 | 40575.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 12:15:00 | 40670.00 | 40516.52 | 40575.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 40670.00 | 40516.52 | 40575.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 40670.00 | 40516.52 | 40575.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 40425.00 | 40498.22 | 40561.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:30:00 | 40605.00 | 40498.22 | 40561.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 40635.00 | 40525.57 | 40568.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 40725.00 | 40525.57 | 40568.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 40740.00 | 40568.46 | 40583.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 40435.00 | 40568.46 | 40583.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 40505.00 | 40555.77 | 40576.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 40295.00 | 40480.49 | 40536.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 40295.00 | 40423.41 | 40493.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 40600.00 | 40517.22 | 40510.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 40600.00 | 40517.22 | 40510.63 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 40460.00 | 40505.78 | 40506.03 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 40965.00 | 40597.62 | 40547.75 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 40455.00 | 40597.01 | 40604.61 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 40705.00 | 40616.38 | 40608.42 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 40480.00 | 40592.57 | 40601.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 40300.00 | 40524.84 | 40568.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 40545.00 | 40528.87 | 40566.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 40545.00 | 40528.87 | 40566.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 40545.00 | 40528.87 | 40566.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 40240.00 | 40505.78 | 40545.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 14:15:00 | 40575.00 | 40547.30 | 40544.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 40575.00 | 40547.30 | 40544.24 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 40480.00 | 40533.84 | 40538.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 40320.00 | 40491.07 | 40518.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 40425.00 | 40419.30 | 40468.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 40425.00 | 40419.30 | 40468.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 40395.00 | 40414.44 | 40462.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 40430.00 | 40414.44 | 40462.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 40880.00 | 40507.55 | 40500.17 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 40440.00 | 40486.43 | 40491.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 40310.00 | 40451.15 | 40474.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 40540.00 | 40415.63 | 40447.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 40540.00 | 40415.63 | 40447.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 40540.00 | 40415.63 | 40447.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 40540.00 | 40415.63 | 40447.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 40500.00 | 40432.50 | 40452.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 40590.00 | 40432.50 | 40452.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 40540.00 | 40454.00 | 40460.28 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 40515.00 | 40466.20 | 40465.26 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 40140.00 | 40406.37 | 40438.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 40000.00 | 40325.09 | 40398.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 38375.00 | 38312.56 | 38742.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 38375.00 | 38312.56 | 38742.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 38375.00 | 38312.56 | 38742.24 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 38790.00 | 38711.57 | 38711.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 15:15:00 | 39235.00 | 38829.67 | 38768.10 | Break + close above crossover candle high |

### Cycle 111 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 37450.00 | 38553.74 | 38648.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 36930.00 | 37418.38 | 37745.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 36885.00 | 36853.98 | 37156.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:30:00 | 36905.00 | 36853.98 | 37156.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 36075.00 | 35885.34 | 36042.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 36075.00 | 35885.34 | 36042.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 36155.00 | 35939.27 | 36053.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 36155.00 | 35939.27 | 36053.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 36775.00 | 36182.53 | 36148.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 37260.00 | 36501.62 | 36306.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 37990.00 | 38155.87 | 37931.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 37990.00 | 38155.87 | 37931.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 37990.00 | 38155.87 | 37931.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 37990.00 | 38155.87 | 37931.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 37725.00 | 38069.70 | 37912.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 37725.00 | 38069.70 | 37912.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 37785.00 | 38012.76 | 37900.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:15:00 | 37905.00 | 38012.76 | 37900.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 37905.00 | 37965.77 | 37898.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 37905.00 | 38077.88 | 38100.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 37905.00 | 38077.88 | 38100.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 37875.00 | 38037.30 | 38080.04 | Break + close below crossover candle low |

### Cycle 114 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 39075.00 | 38234.87 | 38161.81 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 38120.00 | 38517.74 | 38543.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 37885.00 | 38330.75 | 38450.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 35950.00 | 35871.84 | 36329.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 35950.00 | 35871.84 | 36329.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 35865.00 | 35666.61 | 36033.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:45:00 | 36260.00 | 35666.61 | 36033.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 35660.00 | 35665.29 | 35999.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 35960.00 | 35665.29 | 35999.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 36230.00 | 35778.23 | 36020.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 36200.00 | 35778.23 | 36020.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 36140.00 | 35850.58 | 36031.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 36045.00 | 35883.47 | 36029.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 36505.00 | 35918.05 | 35977.89 | SL hit (close>static) qty=1.00 sl=36265.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 36190.00 | 36019.15 | 36016.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 36360.00 | 36130.66 | 36070.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 36500.00 | 36563.46 | 36374.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 36500.00 | 36563.46 | 36374.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 36890.00 | 37082.10 | 36913.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 36890.00 | 37082.10 | 36913.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 36795.00 | 37024.68 | 36902.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:45:00 | 36705.00 | 37024.68 | 36902.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 36740.00 | 36967.74 | 36888.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 36885.00 | 36967.74 | 36888.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 36915.00 | 36986.96 | 36920.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 36935.00 | 36986.96 | 36920.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 36945.00 | 36978.57 | 36922.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 37225.00 | 36928.07 | 36910.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 37215.00 | 37350.39 | 37351.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 37215.00 | 37350.39 | 37351.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 36985.00 | 37153.32 | 37242.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 36900.00 | 36843.11 | 36987.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 36900.00 | 36843.11 | 36987.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 36900.00 | 36843.11 | 36987.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 36885.00 | 36843.11 | 36987.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 36795.00 | 36837.79 | 36960.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 36635.00 | 36797.23 | 36930.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 37015.00 | 36746.63 | 36854.77 | SL hit (close>static) qty=1.00 sl=37000.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 36605.00 | 35940.29 | 35856.17 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 36215.00 | 36265.05 | 36267.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 36045.00 | 36221.04 | 36247.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 36190.00 | 36129.46 | 36185.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 36190.00 | 36129.46 | 36185.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 36190.00 | 36129.46 | 36185.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 36190.00 | 36129.46 | 36185.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 36080.00 | 36119.57 | 36176.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 36125.00 | 36119.57 | 36176.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 36115.00 | 36118.66 | 36170.75 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 36250.00 | 36152.58 | 36147.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 13:15:00 | 36320.00 | 36203.56 | 36173.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 36270.00 | 36271.04 | 36223.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 11:15:00 | 36270.00 | 36271.04 | 36223.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 36270.00 | 36271.04 | 36223.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 36330.00 | 36271.04 | 36223.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 36295.00 | 36275.83 | 36230.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 36295.00 | 36275.83 | 36230.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 36270.00 | 36274.66 | 36233.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:15:00 | 36210.00 | 36274.66 | 36233.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 36330.00 | 36285.73 | 36242.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 36255.00 | 36285.73 | 36242.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 36350.00 | 36298.59 | 36252.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 36480.00 | 36324.87 | 36268.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 36155.00 | 36291.57 | 36268.27 | SL hit (close<static) qty=1.00 sl=36175.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 36175.00 | 36252.81 | 36253.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 36160.00 | 36234.25 | 36245.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 14:15:00 | 36315.00 | 36250.40 | 36251.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 36315.00 | 36250.40 | 36251.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 36315.00 | 36250.40 | 36251.56 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 36300.00 | 36252.25 | 36251.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 36645.00 | 36341.64 | 36293.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 36870.00 | 36893.29 | 36707.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 36870.00 | 36893.29 | 36707.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 36700.00 | 36859.09 | 36771.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 36980.00 | 36896.28 | 36796.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 37100.00 | 37112.63 | 37017.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 37030.00 | 37112.63 | 37017.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 36920.00 | 37074.10 | 37008.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 36920.00 | 37074.10 | 37008.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 36900.00 | 37039.28 | 36998.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 37165.00 | 37039.28 | 36998.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 36690.00 | 36976.74 | 36977.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 36690.00 | 36976.74 | 36977.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 36520.00 | 36835.51 | 36910.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 36915.00 | 36715.31 | 36816.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 36915.00 | 36715.31 | 36816.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 36915.00 | 36715.31 | 36816.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 37025.00 | 36715.31 | 36816.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 36750.00 | 36722.25 | 36810.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 36700.00 | 36742.24 | 36804.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:30:00 | 36705.00 | 36719.23 | 36783.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 37225.00 | 36818.91 | 36817.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 37225.00 | 36818.91 | 36817.83 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 36150.00 | 36712.86 | 36788.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 35475.00 | 35833.49 | 36045.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 35640.00 | 35633.93 | 35831.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 35640.00 | 35633.93 | 35831.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 35640.00 | 35633.93 | 35831.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 35530.00 | 35613.14 | 35803.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 35580.00 | 35583.21 | 35756.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 35530.00 | 35515.21 | 35600.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 35585.00 | 35532.17 | 35600.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 35580.00 | 35541.73 | 35598.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 35565.00 | 35541.73 | 35598.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 35160.00 | 35408.03 | 35433.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 34950.00 | 35098.58 | 35226.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 35265.00 | 35116.89 | 35211.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 35265.00 | 35116.89 | 35211.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 35265.00 | 35116.89 | 35211.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 35265.00 | 35116.89 | 35211.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 35115.00 | 35116.51 | 35202.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:45:00 | 34950.00 | 35091.21 | 35183.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:00:00 | 34825.00 | 35017.77 | 35132.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 34085.00 | 34047.55 | 34046.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 34085.00 | 34047.55 | 34046.35 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 33880.00 | 34014.04 | 34031.23 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 34135.00 | 34051.99 | 34046.35 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 33950.00 | 34024.87 | 34034.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 33760.00 | 33950.33 | 33996.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 33860.00 | 33808.24 | 33895.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 33860.00 | 33808.24 | 33895.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 33860.00 | 33808.24 | 33895.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 33860.00 | 33808.24 | 33895.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 33740.00 | 33794.60 | 33880.96 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 33930.00 | 33795.96 | 33794.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 33990.00 | 33834.77 | 33811.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 14:15:00 | 33755.00 | 33818.82 | 33806.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 33755.00 | 33818.82 | 33806.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 33755.00 | 33818.82 | 33806.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 33755.00 | 33818.82 | 33806.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 33860.00 | 33827.05 | 33811.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 33845.00 | 33827.05 | 33811.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 09:15:00 | 33695.00 | 33800.64 | 33800.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 33605.00 | 33723.89 | 33762.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 32995.00 | 32976.78 | 33057.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:45:00 | 33000.00 | 32976.78 | 33057.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 32985.00 | 32982.14 | 33046.25 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 33430.00 | 33065.65 | 33021.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 33665.00 | 33335.99 | 33176.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 34155.00 | 34159.36 | 33852.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 34155.00 | 34159.36 | 33852.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 34090.00 | 34152.46 | 34000.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 34090.00 | 34152.46 | 34000.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 34090.00 | 34139.97 | 34008.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 34220.00 | 34109.01 | 34032.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 33965.00 | 34097.17 | 34041.60 | SL hit (close<static) qty=1.00 sl=34005.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 33795.00 | 33994.79 | 34002.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 33685.00 | 33867.93 | 33936.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 33300.00 | 33256.09 | 33442.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 33300.00 | 33256.09 | 33442.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 33300.00 | 33256.09 | 33442.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 33100.00 | 33321.30 | 33412.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 33185.00 | 33258.63 | 33365.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 33455.00 | 33414.29 | 33408.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 33455.00 | 33414.29 | 33408.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 14:15:00 | 34000.00 | 33586.36 | 33499.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 33675.00 | 33710.27 | 33577.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 33675.00 | 33710.27 | 33577.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 33675.00 | 33710.27 | 33577.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 33705.00 | 33710.27 | 33577.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 33705.00 | 33709.22 | 33588.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 33815.00 | 33730.37 | 33609.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:45:00 | 33790.00 | 33742.30 | 33625.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:45:00 | 33790.00 | 33733.84 | 33632.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 33815.00 | 33737.07 | 33643.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 33850.00 | 33759.66 | 33662.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 33540.00 | 33759.66 | 33662.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 33885.00 | 33784.73 | 33682.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 33545.00 | 33784.73 | 33682.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 33635.00 | 33801.26 | 33730.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 33635.00 | 33801.26 | 33730.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | SL hit (close<static) qty=1.00 sl=33575.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 33500.00 | 33718.60 | 33722.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 09:15:00 | 33295.00 | 33545.47 | 33630.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 31155.00 | 31080.75 | 31604.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:45:00 | 31405.00 | 31080.75 | 31604.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 31470.00 | 31158.60 | 31592.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 31570.00 | 31158.60 | 31592.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 31545.00 | 31290.50 | 31580.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 31545.00 | 31290.50 | 31580.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 31475.00 | 31327.40 | 31570.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 31445.00 | 31488.58 | 31582.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 32100.00 | 31610.87 | 31629.61 | SL hit (close>static) qty=1.00 sl=31600.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 32100.00 | 31708.69 | 31672.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 32300.00 | 31888.76 | 31764.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 33070.00 | 33178.10 | 32823.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 33070.00 | 33178.10 | 32823.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 32850.00 | 33112.48 | 32825.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 32850.00 | 33112.48 | 32825.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 33195.00 | 33128.98 | 32859.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 32580.00 | 33128.98 | 32859.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 32605.00 | 33024.19 | 32836.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:45:00 | 33375.00 | 32975.28 | 32846.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 33200.00 | 33457.60 | 33358.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 32985.00 | 33296.26 | 33298.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 32985.00 | 33296.26 | 33298.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 32710.00 | 33131.61 | 33220.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 32865.00 | 32755.05 | 32929.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 32865.00 | 32755.05 | 32929.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 32415.00 | 32687.04 | 32883.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 13:00:00 | 32300.00 | 32609.63 | 32830.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 32300.00 | 32460.74 | 32577.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 32105.00 | 32440.90 | 32537.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 32270.00 | 32276.42 | 32382.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 32155.00 | 32252.14 | 32361.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 32335.00 | 32252.14 | 32361.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 31370.00 | 31320.79 | 31465.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 31540.00 | 31320.79 | 31465.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 31435.00 | 31343.63 | 31462.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 31495.00 | 31343.63 | 31462.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 31360.00 | 31365.33 | 31436.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 31300.00 | 31368.06 | 31404.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 31255.00 | 31341.56 | 31386.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 31560.00 | 31300.85 | 31232.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 31430.00 | 31557.88 | 31455.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 10:15:00 | 31430.00 | 31557.88 | 31455.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 31430.00 | 31557.88 | 31455.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 31390.00 | 31557.88 | 31455.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 31380.00 | 31522.31 | 31448.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:30:00 | 31310.00 | 31522.31 | 31448.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 31375.00 | 31454.88 | 31428.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 31345.00 | 31454.88 | 31428.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 31355.00 | 31432.52 | 31422.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 31305.00 | 31432.52 | 31422.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 09:15:00 | 31265.00 | 31399.02 | 31408.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 31130.00 | 31270.70 | 31328.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 30365.00 | 30211.80 | 30482.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 30365.00 | 30211.80 | 30482.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 30200.00 | 30156.35 | 30406.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 30015.00 | 30144.08 | 30378.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:00:00 | 30000.00 | 30115.27 | 30343.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 30025.00 | 30108.21 | 30319.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:30:00 | 29985.00 | 30079.57 | 30287.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 30535.00 | 29875.52 | 29932.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 30405.00 | 29875.52 | 29932.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 30645.00 | 30270.46 | 30124.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 30720.00 | 30721.94 | 30477.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:45:00 | 30725.00 | 30721.94 | 30477.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 30630.00 | 30660.92 | 30506.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 30690.00 | 30666.73 | 30523.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 30780.00 | 30609.93 | 30529.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 30285.00 | 30471.02 | 30488.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 30285.00 | 30471.02 | 30488.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 30120.00 | 30400.81 | 30455.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 29500.00 | 29479.09 | 29784.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 11:00:00 | 29500.00 | 29479.09 | 29784.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 29480.00 | 29403.25 | 29606.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 29530.00 | 29403.25 | 29606.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 29470.00 | 29427.50 | 29568.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 29470.00 | 29427.50 | 29568.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 29510.00 | 29444.00 | 29562.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:30:00 | 29575.00 | 29444.00 | 29562.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 28920.00 | 28847.31 | 29074.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 28895.00 | 28847.31 | 29074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 27840.00 | 27528.70 | 27896.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 27840.00 | 27528.70 | 27896.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 27745.00 | 27571.96 | 27882.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 27860.00 | 27571.96 | 27882.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 28255.00 | 27717.76 | 27872.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 28205.00 | 27717.76 | 27872.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 28220.00 | 27818.21 | 27904.42 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 28220.00 | 27958.85 | 27957.12 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 27655.00 | 27923.92 | 27953.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 27420.00 | 27771.31 | 27876.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 27595.00 | 26843.12 | 27139.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 27595.00 | 26843.12 | 27139.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 27595.00 | 26843.12 | 27139.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 27595.00 | 26843.12 | 27139.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 27445.00 | 26963.50 | 27167.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 27295.00 | 27089.80 | 27206.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 27300.00 | 27231.67 | 27255.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 27300.00 | 27231.67 | 27255.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 27360.00 | 27082.43 | 27063.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 27360.00 | 27082.43 | 27063.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 27995.00 | 27334.03 | 27224.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 27510.00 | 27582.63 | 27399.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 13:45:00 | 27525.00 | 27582.63 | 27399.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 27090.00 | 27484.10 | 27371.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 27090.00 | 27484.10 | 27371.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 27500.00 | 27487.28 | 27383.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 27965.00 | 27487.28 | 27383.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 30761.50 | 30202.76 | 29470.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 31830.00 | 32388.51 | 32451.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 31650.00 | 32240.81 | 32378.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 31825.00 | 31731.79 | 31983.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 31825.00 | 31731.79 | 31983.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 31780.00 | 31782.46 | 31931.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:00:00 | 31600.00 | 31745.97 | 31901.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 31640.00 | 31724.78 | 31877.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:15:00 | 30020.00 | 30522.71 | 30803.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:15:00 | 30058.00 | 30522.71 | 30803.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 30445.00 | 30410.71 | 30673.70 | SL hit (close>ema200) qty=0.50 sl=30410.71 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 14:00:00 | 54260.00 | 2024-05-24 15:15:00 | 52300.00 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2024-05-16 15:00:00 | 57650.00 | 2024-05-24 15:15:00 | 52300.00 | STOP_HIT | 1.00 | -9.28% |
| BUY | retest2 | 2024-05-17 09:15:00 | 54550.20 | 2024-05-24 15:15:00 | 52300.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-05-17 11:00:00 | 53520.00 | 2024-05-24 15:15:00 | 52300.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-05-28 15:15:00 | 51600.00 | 2024-05-29 09:15:00 | 52430.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-06-10 09:30:00 | 52670.00 | 2024-06-14 15:15:00 | 57937.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 10:15:00 | 53632.80 | 2024-06-19 14:15:00 | 55237.30 | STOP_HIT | 1.00 | 2.99% |
| SELL | retest2 | 2024-06-21 14:15:00 | 55587.00 | 2024-06-24 09:15:00 | 56521.65 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-06-21 15:00:00 | 55500.00 | 2024-06-24 09:15:00 | 56521.65 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-05 15:15:00 | 57799.95 | 2024-07-08 10:15:00 | 56588.70 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-07-08 09:45:00 | 57772.05 | 2024-07-08 10:15:00 | 56588.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-07-12 10:30:00 | 56521.00 | 2024-07-19 10:15:00 | 53694.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 14:30:00 | 56249.85 | 2024-07-19 10:15:00 | 53437.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 09:15:00 | 56499.95 | 2024-07-19 10:15:00 | 53674.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 10:30:00 | 56521.00 | 2024-07-22 12:15:00 | 53981.50 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2024-07-12 14:30:00 | 56249.85 | 2024-07-22 12:15:00 | 53981.50 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2024-07-15 09:15:00 | 56499.95 | 2024-07-22 12:15:00 | 53981.50 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2024-07-30 12:45:00 | 53860.10 | 2024-07-31 09:15:00 | 56499.95 | STOP_HIT | 1.00 | -4.90% |
| SELL | retest2 | 2024-08-08 12:00:00 | 51623.00 | 2024-08-16 10:15:00 | 52050.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-08-12 09:15:00 | 51500.00 | 2024-08-16 10:15:00 | 52050.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2024-08-20 09:15:00 | 52316.65 | 2024-08-22 12:15:00 | 52223.95 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-08-21 09:15:00 | 53017.10 | 2024-08-26 10:15:00 | 52111.30 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-08-22 09:15:00 | 52700.40 | 2024-08-26 10:15:00 | 52111.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-08-22 11:45:00 | 52499.05 | 2024-08-26 10:15:00 | 52111.30 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-08-22 15:15:00 | 52800.00 | 2024-08-26 10:15:00 | 52111.30 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-26 09:15:00 | 52522.90 | 2024-08-26 10:15:00 | 52111.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-09-02 13:45:00 | 50698.25 | 2024-09-10 10:15:00 | 50618.40 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-09-03 09:30:00 | 50688.30 | 2024-09-10 10:15:00 | 50618.40 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-09-03 14:15:00 | 50721.20 | 2024-09-10 10:15:00 | 50618.40 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-09-03 15:00:00 | 50709.00 | 2024-09-10 10:15:00 | 50618.40 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2024-09-05 12:15:00 | 50122.85 | 2024-09-10 10:15:00 | 50618.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-09-18 11:30:00 | 49618.75 | 2024-09-23 11:15:00 | 49755.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-09-18 13:30:00 | 49612.15 | 2024-09-23 11:15:00 | 49755.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-09-19 09:45:00 | 49450.05 | 2024-09-23 11:15:00 | 49755.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-09-20 09:15:00 | 49627.90 | 2024-09-23 11:15:00 | 49755.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-09-20 13:15:00 | 49204.65 | 2024-09-23 11:15:00 | 49755.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-10-07 10:30:00 | 47247.05 | 2024-10-09 09:15:00 | 48948.00 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-10-07 14:30:00 | 47544.55 | 2024-10-09 09:15:00 | 48948.00 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-10-14 15:00:00 | 50067.10 | 2024-10-14 15:15:00 | 49735.25 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-10-15 09:15:00 | 50116.15 | 2024-10-15 12:15:00 | 49777.65 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-10-15 10:15:00 | 50150.75 | 2024-10-15 12:15:00 | 49777.65 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-17 11:00:00 | 49050.00 | 2024-10-18 09:15:00 | 49936.05 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-10-17 12:00:00 | 49100.00 | 2024-10-18 09:15:00 | 49936.05 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-10-30 09:15:00 | 45497.10 | 2024-11-12 14:15:00 | 43222.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-30 09:15:00 | 45497.10 | 2024-11-14 09:15:00 | 42700.45 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2024-12-02 11:15:00 | 40468.55 | 2024-12-03 09:15:00 | 41500.05 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-12-05 14:45:00 | 40865.05 | 2024-12-10 09:15:00 | 41100.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-12-06 09:45:00 | 40827.10 | 2024-12-10 09:15:00 | 41100.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-06 12:15:00 | 40894.15 | 2024-12-10 09:15:00 | 41100.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-12-06 14:30:00 | 40742.55 | 2024-12-10 09:15:00 | 41100.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-17 13:00:00 | 40407.75 | 2024-12-19 11:15:00 | 41330.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2025-01-02 09:15:00 | 42399.85 | 2025-01-06 09:15:00 | 42500.65 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-01-10 09:30:00 | 41807.90 | 2025-01-23 09:15:00 | 41179.95 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-02-05 13:30:00 | 38200.00 | 2025-02-11 09:15:00 | 36290.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 12:30:00 | 38248.20 | 2025-02-11 09:15:00 | 36335.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 13:30:00 | 38224.80 | 2025-02-11 09:15:00 | 36313.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 14:15:00 | 38240.80 | 2025-02-11 09:15:00 | 36328.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 37759.15 | 2025-02-12 09:15:00 | 35871.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 13:30:00 | 38200.00 | 2025-02-14 09:15:00 | 35321.45 | STOP_HIT | 0.50 | 7.54% |
| SELL | retest2 | 2025-02-06 12:30:00 | 38248.20 | 2025-02-14 09:15:00 | 35321.45 | STOP_HIT | 0.50 | 7.65% |
| SELL | retest2 | 2025-02-06 13:30:00 | 38224.80 | 2025-02-14 09:15:00 | 35321.45 | STOP_HIT | 0.50 | 7.60% |
| SELL | retest2 | 2025-02-06 14:15:00 | 38240.80 | 2025-02-14 09:15:00 | 35321.45 | STOP_HIT | 0.50 | 7.63% |
| SELL | retest2 | 2025-02-10 09:15:00 | 37759.15 | 2025-02-14 09:15:00 | 35321.45 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2025-03-13 13:30:00 | 33947.90 | 2025-03-19 09:15:00 | 34375.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-03-18 15:00:00 | 33930.95 | 2025-03-19 09:15:00 | 34375.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-03-24 14:45:00 | 33243.35 | 2025-03-25 14:15:00 | 31581.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:45:00 | 33759.15 | 2025-03-25 14:15:00 | 32071.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 14:45:00 | 33243.35 | 2025-03-26 12:15:00 | 33710.60 | STOP_HIT | 0.50 | -1.41% |
| SELL | retest2 | 2025-03-25 10:45:00 | 33759.15 | 2025-03-26 12:15:00 | 33710.60 | STOP_HIT | 0.50 | 0.14% |
| SELL | retest2 | 2025-03-26 14:15:00 | 33795.35 | 2025-03-28 09:15:00 | 33900.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-04-16 12:30:00 | 34145.00 | 2025-04-23 10:15:00 | 34100.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-04-16 14:00:00 | 34190.00 | 2025-04-23 10:15:00 | 34100.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-04-17 09:30:00 | 34270.00 | 2025-04-23 10:15:00 | 34100.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-04-29 11:30:00 | 34300.00 | 2025-04-30 10:15:00 | 34435.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-02 11:00:00 | 34865.00 | 2025-05-09 09:15:00 | 34205.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-05-02 13:30:00 | 34850.00 | 2025-05-09 09:15:00 | 34205.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-05-05 09:15:00 | 35060.00 | 2025-05-09 09:15:00 | 34205.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-05-05 11:00:00 | 34885.00 | 2025-05-09 09:15:00 | 34205.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-05-08 09:30:00 | 35635.00 | 2025-05-09 09:15:00 | 34205.00 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-05-08 11:00:00 | 35355.00 | 2025-05-09 09:15:00 | 34205.00 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-05-12 11:30:00 | 34935.00 | 2025-05-12 12:15:00 | 35095.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-05-29 09:15:00 | 38240.00 | 2025-06-04 09:15:00 | 38235.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-06-16 09:30:00 | 37855.00 | 2025-06-23 10:15:00 | 37875.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-06-16 11:15:00 | 37995.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-06-16 13:15:00 | 38005.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-06-17 12:00:00 | 38020.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-06-18 14:15:00 | 37780.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-06-18 15:00:00 | 37685.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-19 10:00:00 | 37750.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-06-20 10:45:00 | 37795.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-06-20 13:15:00 | 37635.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-11 12:15:00 | 40295.00 | 2025-07-14 15:15:00 | 40600.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-11 15:15:00 | 40295.00 | 2025-07-14 15:15:00 | 40600.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-18 15:15:00 | 40240.00 | 2025-07-21 14:15:00 | 40575.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-21 14:15:00 | 37905.00 | 2025-08-26 13:15:00 | 37905.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-08-22 09:15:00 | 37905.00 | 2025-08-26 13:15:00 | 37905.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-09 11:45:00 | 36045.00 | 2025-09-10 09:15:00 | 36505.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-17 09:15:00 | 37225.00 | 2025-09-22 09:15:00 | 37215.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-09-24 13:00:00 | 36635.00 | 2025-09-25 09:15:00 | 37015.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-25 11:15:00 | 36680.00 | 2025-10-08 09:15:00 | 36605.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-09-25 14:45:00 | 36675.00 | 2025-10-08 09:15:00 | 36605.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-10-21 13:45:00 | 36480.00 | 2025-10-23 10:15:00 | 36155.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-31 09:15:00 | 37165.00 | 2025-10-31 10:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-03 12:30:00 | 36700.00 | 2025-11-04 09:15:00 | 37225.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-03 14:30:00 | 36705.00 | 2025-11-04 09:15:00 | 37225.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-12 12:00:00 | 35530.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-11-12 13:45:00 | 35580.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-14 10:15:00 | 35530.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-11-14 11:15:00 | 35585.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-14 12:15:00 | 35565.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-04 12:45:00 | 34950.00 | 2025-12-15 15:15:00 | 34085.00 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-12-04 15:00:00 | 34825.00 | 2025-12-15 15:15:00 | 34085.00 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2026-01-08 09:15:00 | 34220.00 | 2026-01-08 10:15:00 | 33965.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-13 15:15:00 | 33100.00 | 2026-01-14 15:15:00 | 33455.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-14 10:00:00 | 33185.00 | 2026-01-14 15:15:00 | 33455.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-19 12:00:00 | 33815.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-19 12:45:00 | 33790.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-19 13:45:00 | 33790.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-19 14:45:00 | 33815.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-29 10:30:00 | 31445.00 | 2026-01-29 11:15:00 | 32100.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-02-02 13:45:00 | 33375.00 | 2026-02-04 14:15:00 | 32985.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-02-04 13:00:00 | 33200.00 | 2026-02-04 14:15:00 | 32985.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-06 13:00:00 | 32300.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.88% |
| SELL | retest2 | 2026-02-09 15:15:00 | 32300.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.88% |
| SELL | retest2 | 2026-02-10 11:15:00 | 32105.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2026-02-11 11:15:00 | 32270.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.79% |
| SELL | retest2 | 2026-02-19 09:45:00 | 31300.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-02-19 11:45:00 | 31255.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-03-05 11:15:00 | 30015.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-05 12:00:00 | 30000.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-03-05 12:30:00 | 30025.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-03-05 13:30:00 | 29985.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-03-12 11:00:00 | 30690.00 | 2026-03-13 11:15:00 | 30285.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-03-12 14:15:00 | 30780.00 | 2026-03-13 11:15:00 | 30285.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-01 11:30:00 | 27295.00 | 2026-04-06 13:15:00 | 27360.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-04-01 13:45:00 | 27300.00 | 2026-04-06 13:15:00 | 27360.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-04-01 14:15:00 | 27300.00 | 2026-04-06 13:15:00 | 27360.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-04-09 09:15:00 | 27965.00 | 2026-04-16 09:15:00 | 30761.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:00:00 | 31600.00 | 2026-05-06 13:15:00 | 30020.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 12:00:00 | 31640.00 | 2026-05-06 13:15:00 | 30058.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 11:00:00 | 31600.00 | 2026-05-07 09:15:00 | 30445.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2026-04-28 12:00:00 | 31640.00 | 2026-05-07 09:15:00 | 30445.00 | STOP_HIT | 0.50 | 3.78% |

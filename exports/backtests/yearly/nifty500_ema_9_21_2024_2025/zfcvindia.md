# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 14532.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 158 |
| ALERT1 | 91 |
| ALERT2 | 89 |
| ALERT2_SKIP | 46 |
| ALERT3 | 266 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 128 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 38 / 99
- **Target hits / Stop hits / Partials:** 5 / 122 / 10
- **Avg / median % per leg:** 0.11% / -0.71%
- **Sum % (uncompounded):** 14.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 11 | 15.5% | 5 | 66 | 0 | -0.30% | -21.4% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.58% | -7.9% |
| BUY @ 3rd Alert (retest2) | 66 | 11 | 16.7% | 5 | 61 | 0 | -0.20% | -13.5% |
| SELL (all) | 66 | 27 | 40.9% | 0 | 56 | 10 | 0.54% | 35.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 66 | 27 | 40.9% | 0 | 56 | 10 | 0.54% | 35.9% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.58% | -7.9% |
| retest2 (combined) | 132 | 38 | 28.8% | 5 | 117 | 10 | 0.17% | 22.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 12:15:00 | 13300.00 | 13175.31 | 13162.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 13725.00 | 13365.42 | 13262.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 13600.00 | 13600.62 | 13437.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 13600.00 | 13600.62 | 13437.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 13561.00 | 13628.59 | 13564.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 13561.00 | 13628.59 | 13564.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 13500.00 | 13602.87 | 13558.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 13500.00 | 13602.87 | 13558.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 13468.05 | 13575.91 | 13550.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 13470.80 | 13556.03 | 13543.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 13617.95 | 13568.42 | 13550.38 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 13300.00 | 13519.77 | 13535.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 09:15:00 | 13254.00 | 13429.86 | 13489.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 13278.85 | 13253.34 | 13346.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 13278.85 | 13253.34 | 13346.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 13278.85 | 13253.34 | 13346.55 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 13515.00 | 13396.89 | 13390.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 15:15:00 | 13600.00 | 13437.51 | 13409.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 16390.00 | 16417.28 | 15659.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 11:00:00 | 16390.00 | 16417.28 | 15659.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 15844.65 | 16309.74 | 15942.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 15844.65 | 16309.74 | 15942.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 15900.00 | 16227.79 | 15938.65 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 15567.40 | 15851.04 | 15865.09 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 16499.80 | 15945.82 | 15903.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 17457.30 | 16276.42 | 16062.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 16556.90 | 17061.21 | 16776.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 16556.90 | 17061.21 | 16776.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 16556.90 | 17061.21 | 16776.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 16556.90 | 17061.21 | 16776.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 16301.90 | 16909.35 | 16733.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 16301.90 | 16909.35 | 16733.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 16145.00 | 16756.48 | 16679.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 16678.50 | 16756.48 | 16679.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 15:15:00 | 16120.10 | 16562.35 | 16605.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 16120.10 | 16562.35 | 16605.41 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 17221.10 | 16678.04 | 16641.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 14:15:00 | 17669.60 | 17292.69 | 17167.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 17500.25 | 17572.74 | 17429.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 17489.00 | 17572.74 | 17429.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 17340.10 | 17526.21 | 17421.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 17340.10 | 17526.21 | 17421.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 17302.65 | 17481.50 | 17410.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:15:00 | 17355.15 | 17410.39 | 17389.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:00:00 | 17381.45 | 17404.60 | 17388.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:30:00 | 17358.15 | 17380.17 | 17379.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:15:00 | 17474.95 | 17380.17 | 17379.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 17708.50 | 17445.83 | 17409.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 13:00:00 | 17759.05 | 17541.14 | 17460.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 14:15:00 | 17300.00 | 17483.30 | 17447.90 | SL hit (close<static) qty=1.00 sl=17350.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 17339.95 | 17414.04 | 17421.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 16249.95 | 17116.80 | 17274.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 14:15:00 | 15859.40 | 15814.94 | 16127.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 15859.40 | 15814.94 | 16127.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 15973.10 | 15809.56 | 15955.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 15973.10 | 15809.56 | 15955.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 16138.00 | 15875.25 | 15971.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 15909.20 | 15875.25 | 15971.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:00:00 | 15901.55 | 15880.51 | 15965.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 15845.00 | 15706.78 | 15706.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 15845.00 | 15706.78 | 15706.69 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 15445.80 | 15671.33 | 15692.46 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 15805.75 | 15711.75 | 15707.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 15949.85 | 15759.37 | 15729.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 15875.75 | 15934.10 | 15863.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 15875.75 | 15934.10 | 15863.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 15875.75 | 15934.10 | 15863.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 15844.05 | 15934.10 | 15863.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 15894.75 | 15926.23 | 15865.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:45:00 | 15991.35 | 15940.98 | 15878.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 14:00:00 | 15976.00 | 16008.00 | 15946.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 14:45:00 | 15994.30 | 15995.61 | 15946.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:15:00 | 15979.00 | 15995.61 | 15946.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 15979.00 | 15992.29 | 15949.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 15907.90 | 15992.29 | 15949.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 15885.90 | 15971.01 | 15943.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 15885.90 | 15971.01 | 15943.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 15918.30 | 15960.47 | 15941.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 15851.50 | 15960.47 | 15941.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 15929.50 | 15954.28 | 15940.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:45:00 | 15947.00 | 15954.28 | 15940.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 15981.20 | 15959.66 | 15944.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:30:00 | 15952.25 | 15959.66 | 15944.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 15925.85 | 15952.90 | 15942.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 15925.85 | 15952.90 | 15942.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 15966.60 | 15955.64 | 15944.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 15:15:00 | 15995.30 | 15955.64 | 15944.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:45:00 | 16016.05 | 15970.24 | 15953.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 12:15:00 | 15740.05 | 15912.07 | 15930.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 12:15:00 | 15740.05 | 15912.07 | 15930.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 13:15:00 | 15686.60 | 15866.98 | 15908.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 15737.20 | 15695.41 | 15779.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 14:45:00 | 15755.10 | 15695.41 | 15779.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 15750.00 | 15706.33 | 15776.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 15838.25 | 15706.33 | 15776.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 15764.40 | 15717.94 | 15775.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 15867.80 | 15717.94 | 15775.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 15787.80 | 15731.91 | 15776.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 15787.80 | 15731.91 | 15776.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 15757.80 | 15737.09 | 15774.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:45:00 | 15796.35 | 15737.09 | 15774.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 15920.00 | 15773.67 | 15788.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 15920.00 | 15773.67 | 15788.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 16088.05 | 15836.55 | 15815.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 11:15:00 | 16257.00 | 15995.47 | 15909.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 13:15:00 | 16000.00 | 16040.30 | 15947.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 13:15:00 | 16000.00 | 16040.30 | 15947.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 16000.00 | 16040.30 | 15947.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:45:00 | 16150.10 | 16040.30 | 15947.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 15947.95 | 16021.83 | 15947.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 15947.95 | 16021.83 | 15947.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 15940.10 | 16005.48 | 15946.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 15938.10 | 15992.01 | 15946.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 15875.30 | 15968.67 | 15939.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 15899.75 | 15968.67 | 15939.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 15859.35 | 15934.21 | 15928.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:00:00 | 15859.35 | 15934.21 | 15928.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 15890.10 | 15931.85 | 15928.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 15890.10 | 15931.85 | 15928.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 15:15:00 | 15871.00 | 15919.68 | 15923.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 10:15:00 | 15805.00 | 15895.19 | 15911.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 14:15:00 | 15900.00 | 15854.87 | 15882.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 14:15:00 | 15900.00 | 15854.87 | 15882.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 15900.00 | 15854.87 | 15882.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 15900.00 | 15854.87 | 15882.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 15900.00 | 15863.89 | 15884.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 15861.35 | 15863.89 | 15884.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 15771.00 | 15825.82 | 15856.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 14:30:00 | 15760.00 | 15816.65 | 15849.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 15:15:00 | 15880.00 | 15829.32 | 15852.28 | SL hit (close>static) qty=1.00 sl=15865.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 15581.00 | 15403.36 | 15399.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 15664.25 | 15455.54 | 15423.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 15418.95 | 15478.56 | 15444.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 14:15:00 | 15418.95 | 15478.56 | 15444.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 15418.95 | 15478.56 | 15444.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:00:00 | 15418.95 | 15478.56 | 15444.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 15325.05 | 15447.86 | 15433.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 15314.05 | 15447.86 | 15433.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 15280.70 | 15414.42 | 15419.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 13:15:00 | 15194.65 | 15306.64 | 15361.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 15388.85 | 15306.07 | 15346.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 15388.85 | 15306.07 | 15346.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 15388.85 | 15306.07 | 15346.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 15388.85 | 15306.07 | 15346.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 15501.20 | 15345.10 | 15360.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 15545.95 | 15345.10 | 15360.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 15513.20 | 15378.72 | 15374.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 15592.45 | 15421.47 | 15394.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 15820.10 | 15897.09 | 15801.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 15847.60 | 15887.19 | 15806.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 15847.60 | 15887.19 | 15806.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 15838.80 | 15887.19 | 15806.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 15827.70 | 15875.29 | 15808.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 15877.35 | 15828.80 | 15807.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:00:00 | 15880.20 | 15828.80 | 15807.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 15766.75 | 15851.45 | 15829.90 | SL hit (close<static) qty=1.00 sl=15800.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 15640.00 | 15872.16 | 15895.67 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 16232.70 | 15932.85 | 15913.08 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 15807.80 | 15943.62 | 15944.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 11:15:00 | 15770.40 | 15908.97 | 15928.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 11:15:00 | 15471.15 | 15402.04 | 15508.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 12:00:00 | 15471.15 | 15402.04 | 15508.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 15496.50 | 15439.56 | 15500.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 15496.50 | 15439.56 | 15500.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 15494.40 | 15450.53 | 15499.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 15341.55 | 15450.53 | 15499.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 15:15:00 | 15999.90 | 15441.64 | 15417.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 15999.90 | 15441.64 | 15417.52 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 15643.90 | 15691.23 | 15691.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 15556.10 | 15646.76 | 15669.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 10:15:00 | 15520.00 | 15482.32 | 15545.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 10:15:00 | 15520.00 | 15482.32 | 15545.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 15520.00 | 15482.32 | 15545.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:30:00 | 15521.20 | 15482.32 | 15545.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 15540.00 | 15499.82 | 15542.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 15540.00 | 15499.82 | 15542.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 15527.25 | 15505.31 | 15541.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 15487.55 | 15533.22 | 15547.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:00:00 | 15457.50 | 15518.08 | 15539.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 15464.60 | 15337.81 | 15374.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 13:15:00 | 15475.00 | 15386.79 | 15378.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 15475.00 | 15386.79 | 15378.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 15880.00 | 15504.39 | 15434.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 15500.05 | 15503.52 | 15440.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 10:00:00 | 15500.05 | 15503.52 | 15440.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 15375.00 | 15477.82 | 15434.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 15380.15 | 15477.82 | 15434.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 15245.00 | 15431.25 | 15417.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 15245.00 | 15431.25 | 15417.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 15198.70 | 15384.74 | 15397.67 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 15454.85 | 15397.29 | 15391.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 10:15:00 | 15510.10 | 15419.86 | 15402.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 14:15:00 | 15934.85 | 15966.64 | 15826.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 15:00:00 | 15934.85 | 15966.64 | 15826.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 15777.05 | 15925.85 | 15831.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 12:45:00 | 16161.75 | 16000.24 | 15890.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 10:15:00 | 16622.55 | 16861.04 | 16882.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 16622.55 | 16861.04 | 16882.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 16599.70 | 16808.77 | 16856.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 10:15:00 | 16332.65 | 16330.08 | 16468.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 11:00:00 | 16332.65 | 16330.08 | 16468.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 16452.75 | 16369.96 | 16463.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:45:00 | 16443.05 | 16369.96 | 16463.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 16456.40 | 16387.25 | 16462.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:30:00 | 16460.35 | 16387.25 | 16462.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 16560.00 | 16421.80 | 16471.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 16560.00 | 16421.80 | 16471.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 16573.50 | 16452.14 | 16481.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 16376.00 | 16452.14 | 16481.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 16515.55 | 16467.20 | 16483.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 16521.70 | 16467.20 | 16483.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 16425.55 | 16458.87 | 16477.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 16598.95 | 16458.87 | 16477.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 16420.05 | 16451.11 | 16472.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:45:00 | 16349.70 | 16421.37 | 16454.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 16365.00 | 16326.44 | 16371.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 16479.30 | 16377.48 | 16375.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 16479.30 | 16377.48 | 16375.64 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 15950.00 | 16311.11 | 16346.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 15852.30 | 16010.73 | 16091.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 16337.10 | 16035.11 | 16086.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 16337.10 | 16035.11 | 16086.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 16337.10 | 16035.11 | 16086.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 16337.10 | 16035.11 | 16086.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 16590.00 | 16146.08 | 16132.20 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 15950.00 | 16099.14 | 16114.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 15929.35 | 16065.18 | 16097.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 16037.85 | 15986.67 | 16042.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 16037.85 | 15986.67 | 16042.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 16037.85 | 15986.67 | 16042.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 15667.80 | 15925.39 | 15972.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 14884.41 | 15182.10 | 15418.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 12:15:00 | 15132.75 | 15126.48 | 15330.07 | SL hit (close>ema200) qty=0.50 sl=15126.48 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 15:15:00 | 15280.00 | 15171.88 | 15159.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 15369.30 | 15241.52 | 15207.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 15309.85 | 15457.28 | 15359.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 15309.85 | 15457.28 | 15359.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 15309.85 | 15457.28 | 15359.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 15376.85 | 15457.28 | 15359.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 15289.80 | 15423.79 | 15353.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:30:00 | 15292.40 | 15423.79 | 15353.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 15290.00 | 15374.03 | 15341.65 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 15024.85 | 15278.59 | 15302.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 14900.00 | 15202.87 | 15265.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 14:15:00 | 14220.00 | 14214.30 | 14451.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 15:00:00 | 14220.00 | 14214.30 | 14451.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 14147.45 | 14041.03 | 14227.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 14147.45 | 14041.03 | 14227.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 13724.00 | 13868.11 | 14011.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:30:00 | 13851.05 | 13868.11 | 14011.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 14025.00 | 13797.96 | 13908.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 14025.00 | 13797.96 | 13908.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 13898.95 | 13818.16 | 13908.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 13745.45 | 13772.52 | 13879.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 15:15:00 | 13751.00 | 13711.06 | 13797.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 14:15:00 | 13998.90 | 13824.62 | 13813.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 13998.90 | 13824.62 | 13813.72 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 13683.35 | 13812.60 | 13816.11 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 14061.00 | 13843.32 | 13826.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 14255.05 | 14021.61 | 13939.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 15:15:00 | 14470.00 | 14548.03 | 14452.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 15:15:00 | 14470.00 | 14548.03 | 14452.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 14470.00 | 14548.03 | 14452.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 12:00:00 | 14588.20 | 14552.96 | 14478.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 12:15:00 | 14419.10 | 14526.19 | 14472.71 | SL hit (close<static) qty=1.00 sl=14425.05 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 14800.00 | 14862.13 | 14862.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 13:15:00 | 14516.00 | 14764.37 | 14811.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 14974.20 | 14703.76 | 14725.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 14974.20 | 14703.76 | 14725.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 14974.20 | 14703.76 | 14725.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 14902.35 | 14703.76 | 14725.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 14991.50 | 14761.31 | 14749.84 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 11:15:00 | 14671.45 | 14813.24 | 14814.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 12:15:00 | 14577.15 | 14766.02 | 14792.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 14667.25 | 14636.38 | 14712.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 14667.25 | 14636.38 | 14712.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 14667.25 | 14636.38 | 14712.69 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 14935.00 | 14768.22 | 14752.97 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 14599.25 | 14747.01 | 14751.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 14494.75 | 14696.56 | 14728.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 12494.00 | 12464.96 | 12811.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 14:00:00 | 12494.00 | 12464.96 | 12811.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 12496.80 | 12418.15 | 12491.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 12488.00 | 12418.15 | 12491.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 12500.05 | 12434.53 | 12491.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 12518.85 | 12434.53 | 12491.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 12497.80 | 12447.19 | 12492.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:30:00 | 12518.95 | 12447.19 | 12492.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 12508.70 | 12459.49 | 12493.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 12486.60 | 12459.49 | 12493.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 12500.00 | 12467.59 | 12494.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:30:00 | 12500.10 | 12467.59 | 12494.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 12553.00 | 12484.67 | 12499.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 12553.00 | 12484.67 | 12499.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 12588.00 | 12505.34 | 12507.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 12429.75 | 12505.34 | 12507.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 12412.60 | 12427.83 | 12460.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:00:00 | 12412.60 | 12427.83 | 12460.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 12355.00 | 12402.41 | 12442.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 15:15:00 | 12325.00 | 12375.54 | 12412.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 12328.40 | 12351.16 | 12394.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 14:15:00 | 11708.75 | 11883.09 | 12038.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 14:15:00 | 11711.98 | 11883.09 | 12038.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 11524.40 | 11518.18 | 11629.95 | SL hit (close>ema200) qty=0.50 sl=11518.18 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 11649.65 | 11593.78 | 11592.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 11662.15 | 11616.45 | 11603.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 11585.00 | 11631.34 | 11615.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 11585.00 | 11631.34 | 11615.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 11585.00 | 11631.34 | 11615.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:45:00 | 11590.00 | 11631.34 | 11615.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 11588.90 | 11622.85 | 11613.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 11588.90 | 11622.85 | 11613.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 11637.15 | 11625.71 | 11615.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 11723.55 | 11630.56 | 11618.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 09:15:00 | 11764.35 | 11622.67 | 11620.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 11550.00 | 11621.51 | 11622.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 11:15:00 | 11550.00 | 11621.51 | 11622.34 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 09:15:00 | 11694.95 | 11628.25 | 11623.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 11769.40 | 11656.48 | 11636.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 15:15:00 | 11690.50 | 11713.30 | 11678.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 09:15:00 | 11707.00 | 11713.30 | 11678.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 11904.30 | 11971.14 | 11946.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 11988.30 | 11971.14 | 11946.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 11897.80 | 11943.99 | 11941.35 | SL hit (close<static) qty=1.00 sl=11904.30 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 11854.75 | 11926.14 | 11933.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 15:15:00 | 11816.00 | 11904.11 | 11922.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 11608.20 | 11590.13 | 11687.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 11631.60 | 11590.13 | 11687.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 11500.00 | 11517.77 | 11595.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 11344.90 | 11472.85 | 11508.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 10:45:00 | 11400.25 | 11443.16 | 11487.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:15:00 | 11423.65 | 11442.19 | 11483.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 13:15:00 | 11384.85 | 11326.10 | 11319.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 11384.85 | 11326.10 | 11319.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 11449.35 | 11350.75 | 11331.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 11466.90 | 11484.68 | 11431.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 11466.90 | 11484.68 | 11431.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 11466.90 | 11484.68 | 11431.30 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 11303.95 | 11395.72 | 11404.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 11160.95 | 11328.65 | 11371.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 10785.05 | 10773.92 | 10911.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 10785.05 | 10773.92 | 10911.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 10136.00 | 9921.68 | 10088.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 10136.00 | 9921.68 | 10088.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 10148.70 | 9967.09 | 10094.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 10148.70 | 9967.09 | 10094.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 10127.05 | 9999.08 | 10097.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 10187.10 | 9999.08 | 10097.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 10357.15 | 10181.11 | 10158.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 10521.50 | 10249.19 | 10191.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 10696.00 | 10707.22 | 10504.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 10696.00 | 10707.22 | 10504.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 10600.10 | 10714.21 | 10575.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 10600.10 | 10714.21 | 10575.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 10550.00 | 10681.37 | 10572.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 10550.00 | 10681.37 | 10572.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 10586.30 | 10662.35 | 10574.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:45:00 | 10546.90 | 10662.35 | 10574.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 10919.65 | 10713.81 | 10605.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:30:00 | 11001.00 | 10780.65 | 10645.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:30:00 | 10987.10 | 10930.04 | 10794.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 11001.55 | 10936.99 | 10810.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 11074.90 | 10931.99 | 10819.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 10922.10 | 11013.38 | 10951.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:30:00 | 10950.20 | 11013.38 | 10951.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 10920.95 | 10994.89 | 10948.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:45:00 | 10931.70 | 10994.89 | 10948.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 10875.15 | 10969.68 | 10944.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 10875.15 | 10969.68 | 10944.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 10950.00 | 10965.74 | 10945.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 10833.00 | 10915.08 | 10924.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 10833.00 | 10915.08 | 10924.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 14:15:00 | 10800.00 | 10880.34 | 10906.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 10830.00 | 10810.39 | 10850.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 10830.00 | 10810.39 | 10850.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 10830.00 | 10810.39 | 10850.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 10830.00 | 10810.39 | 10850.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 10795.00 | 10807.31 | 10845.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 11271.50 | 10807.31 | 10845.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 11229.75 | 10891.80 | 10880.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 11:15:00 | 11779.95 | 11151.44 | 11005.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 10:15:00 | 11554.25 | 11558.20 | 11318.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 11:00:00 | 11554.25 | 11558.20 | 11318.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 11285.95 | 11544.66 | 11488.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 11285.95 | 11544.66 | 11488.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 11100.00 | 11455.73 | 11452.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 11341.25 | 11455.73 | 11452.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 10903.00 | 11345.18 | 11402.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 10903.00 | 11345.18 | 11402.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 10:15:00 | 10763.85 | 10993.98 | 11157.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 10786.00 | 10723.14 | 10869.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 15:00:00 | 10786.00 | 10723.14 | 10869.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 10653.45 | 10719.42 | 10842.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 10477.20 | 10661.54 | 10805.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:45:00 | 10487.00 | 10579.73 | 10727.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 10986.00 | 10660.98 | 10751.16 | SL hit (close>static) qty=1.00 sl=10867.95 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 10761.30 | 10609.87 | 10603.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 10985.35 | 10738.57 | 10671.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 11:15:00 | 10865.75 | 10888.87 | 10804.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 14:15:00 | 10787.05 | 10852.80 | 10807.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 10787.05 | 10852.80 | 10807.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 10787.05 | 10852.80 | 10807.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 10793.25 | 10840.89 | 10806.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 10693.40 | 10840.89 | 10806.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 10655.70 | 10803.85 | 10792.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 10872.00 | 10830.12 | 10806.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:45:00 | 10825.85 | 10849.74 | 10824.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 10845.00 | 10849.74 | 10824.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 11:15:00 | 10845.60 | 10869.69 | 10858.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 10837.50 | 10863.25 | 10856.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:15:00 | 10765.80 | 10863.25 | 10856.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-27 12:15:00 | 10776.55 | 10845.91 | 10849.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 10776.55 | 10845.91 | 10849.68 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 10890.30 | 10853.57 | 10852.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 15:15:00 | 10978.00 | 10878.46 | 10863.85 | Break + close above crossover candle high |

### Cycle 54 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 10656.80 | 10834.12 | 10845.03 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 14:15:00 | 10965.40 | 10839.35 | 10838.39 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 10549.00 | 10783.05 | 10813.07 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 10900.15 | 10768.84 | 10753.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 11076.40 | 10847.34 | 10793.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 11210.50 | 11302.09 | 11167.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:30:00 | 11456.20 | 11361.22 | 11206.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 11250.00 | 11396.07 | 11290.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 11250.00 | 11396.07 | 11290.73 | SL hit (close<ema400) qty=1.00 sl=11290.73 alert=retest1 |

### Cycle 58 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 11173.55 | 11246.52 | 11251.71 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 11280.00 | 11257.71 | 11256.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 09:15:00 | 11309.90 | 11268.15 | 11261.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 10:15:00 | 12180.05 | 12195.82 | 12032.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 11:00:00 | 12180.05 | 12195.82 | 12032.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 12083.55 | 12161.70 | 12067.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 12083.55 | 12161.70 | 12067.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 12095.00 | 12148.36 | 12070.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 12230.00 | 12148.36 | 12070.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 11994.65 | 12117.62 | 12063.37 | SL hit (close<static) qty=1.00 sl=12001.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 12:15:00 | 11885.00 | 12010.18 | 12022.35 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 12169.70 | 12017.65 | 12005.39 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 11994.65 | 12024.70 | 12025.30 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 12736.65 | 12154.81 | 12082.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 13467.40 | 12820.51 | 12608.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 13006.15 | 13074.87 | 12842.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 15:00:00 | 13006.15 | 13074.87 | 12842.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 12816.35 | 13023.17 | 12840.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 13164.10 | 13064.61 | 12875.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 13077.00 | 13029.07 | 12982.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 13160.00 | 13029.07 | 12982.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 13139.05 | 13069.05 | 13010.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 13199.30 | 13095.10 | 13027.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 10:15:00 | 12904.50 | 13024.60 | 13034.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 12904.50 | 13024.60 | 13034.99 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 12:15:00 | 13141.35 | 13054.23 | 13047.04 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 12300.00 | 12930.93 | 12996.11 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 12807.00 | 12544.24 | 12542.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 13016.00 | 12638.59 | 12585.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 13274.00 | 13443.62 | 13265.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 14:15:00 | 13274.00 | 13443.62 | 13265.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 13274.00 | 13443.62 | 13265.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 13274.00 | 13443.62 | 13265.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 13350.00 | 13424.89 | 13273.21 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 15:15:00 | 12996.00 | 13208.83 | 13226.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 09:15:00 | 12770.00 | 13035.03 | 13123.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 15:15:00 | 12980.00 | 12905.79 | 13000.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 09:15:00 | 13041.00 | 12905.79 | 13000.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 13150.00 | 12954.63 | 13014.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:30:00 | 13115.00 | 12954.63 | 13014.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 13258.00 | 13015.30 | 13036.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 13250.00 | 13015.30 | 13036.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 13080.00 | 13053.63 | 13051.86 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 12793.00 | 13027.61 | 13043.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 12592.00 | 12940.49 | 13002.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 12761.00 | 12736.43 | 12849.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:30:00 | 12709.00 | 12736.43 | 12849.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 12525.00 | 12511.41 | 12586.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 12408.00 | 12514.05 | 12563.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:00:00 | 12413.00 | 12493.84 | 12549.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 12406.00 | 12248.91 | 12231.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 12406.00 | 12248.91 | 12231.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 12496.00 | 12377.41 | 12304.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 12283.00 | 12460.71 | 12386.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 12283.00 | 12460.71 | 12386.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 12283.00 | 12460.71 | 12386.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 12283.00 | 12460.71 | 12386.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 12424.00 | 12453.37 | 12389.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 12266.00 | 12453.37 | 12389.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 13095.00 | 13365.49 | 13257.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 13095.00 | 13365.49 | 13257.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 13021.00 | 13296.59 | 13236.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 13021.00 | 13296.59 | 13236.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 13074.00 | 13221.90 | 13210.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:30:00 | 13068.00 | 13221.90 | 13210.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 13204.00 | 13204.41 | 13204.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 13200.00 | 13204.41 | 13204.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 13280.00 | 13219.53 | 13210.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 13400.00 | 13219.53 | 13210.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 13315.00 | 13215.62 | 13209.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 13410.00 | 13305.75 | 13263.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:00:00 | 13291.00 | 13340.92 | 13319.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 13213.00 | 13315.34 | 13309.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 13213.00 | 13315.34 | 13309.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 13205.00 | 13293.27 | 13300.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 13205.00 | 13293.27 | 13300.02 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 13359.00 | 13306.42 | 13305.39 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 13199.00 | 13288.72 | 13299.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 13156.00 | 13262.17 | 13286.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 13000.00 | 12985.49 | 13086.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 13000.00 | 12985.49 | 13086.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 13000.00 | 12985.49 | 13086.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 12875.00 | 12988.14 | 13024.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 12843.00 | 12947.13 | 12998.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 13252.00 | 12946.31 | 12919.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 13252.00 | 12946.31 | 12919.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 13391.00 | 13083.36 | 12989.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 13412.00 | 13527.40 | 13394.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:15:00 | 13594.00 | 13527.40 | 13394.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:45:00 | 13620.00 | 13545.01 | 13425.33 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 13442.00 | 13524.41 | 13426.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 13427.00 | 13524.41 | 13426.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 13375.00 | 13494.53 | 13422.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 13375.00 | 13494.53 | 13422.13 | SL hit (close<ema400) qty=1.00 sl=13422.13 alert=retest1 |

### Cycle 76 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 13536.00 | 13733.36 | 13754.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 13507.00 | 13655.39 | 13713.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 13455.00 | 13348.00 | 13413.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 13455.00 | 13348.00 | 13413.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 13455.00 | 13348.00 | 13413.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 13455.00 | 13348.00 | 13413.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 13302.00 | 13338.80 | 13403.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 13176.00 | 13338.80 | 13403.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 13135.00 | 13298.04 | 13379.01 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 13341.00 | 13297.53 | 13293.24 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 13137.00 | 13267.00 | 13280.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 13120.00 | 13237.60 | 13266.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 13310.00 | 13220.73 | 13240.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 13310.00 | 13220.73 | 13240.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 13310.00 | 13220.73 | 13240.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 13110.00 | 13210.51 | 13228.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 13126.00 | 13132.36 | 13143.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 13268.00 | 13154.23 | 13150.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 13268.00 | 13154.23 | 13150.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 13359.00 | 13274.44 | 13228.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 13401.00 | 13434.24 | 13369.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 13424.00 | 13432.19 | 13374.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 13301.00 | 13405.95 | 13367.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 13301.00 | 13405.95 | 13367.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 13347.00 | 13394.16 | 13365.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:30:00 | 13369.00 | 13369.13 | 13358.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 13280.00 | 13349.84 | 13351.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 13280.00 | 13349.84 | 13351.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 13232.00 | 13326.28 | 13340.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 13379.00 | 13331.98 | 13340.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 13379.00 | 13331.98 | 13340.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 13379.00 | 13331.98 | 13340.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 13379.00 | 13331.98 | 13340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 13366.00 | 13338.78 | 13342.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 13395.00 | 13338.78 | 13342.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 13337.00 | 13340.22 | 13342.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 13292.00 | 13340.22 | 13342.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 13229.00 | 13317.98 | 13332.47 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 13465.00 | 13350.26 | 13344.81 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 13238.00 | 13327.66 | 13338.37 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 13418.00 | 13354.37 | 13348.41 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 13321.00 | 13352.47 | 13352.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 14:15:00 | 13287.00 | 13339.38 | 13346.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 10:15:00 | 13172.00 | 13141.76 | 13200.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 10:15:00 | 13172.00 | 13141.76 | 13200.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 13172.00 | 13141.76 | 13200.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 13172.00 | 13141.76 | 13200.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 13200.00 | 13153.41 | 13200.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 13200.00 | 13153.41 | 13200.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 13178.00 | 13158.33 | 13198.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 13223.00 | 13158.33 | 13198.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 13240.00 | 13179.89 | 13201.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 13249.00 | 13179.89 | 13201.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 13179.00 | 13179.71 | 13199.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 13100.00 | 13164.37 | 13190.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 13112.00 | 13158.71 | 13176.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 13112.00 | 13133.77 | 13163.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 13143.00 | 13048.45 | 13044.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 13143.00 | 13048.45 | 13044.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 13171.00 | 13072.96 | 13055.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 13082.00 | 13119.07 | 13090.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 13082.00 | 13119.07 | 13090.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 13082.00 | 13119.07 | 13090.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 13093.00 | 13119.07 | 13090.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 13057.00 | 13106.66 | 13087.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 13057.00 | 13106.66 | 13087.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 13123.00 | 13109.93 | 13090.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 13053.00 | 13109.93 | 13090.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 13068.00 | 13101.54 | 13088.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 13068.00 | 13101.54 | 13088.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 13070.00 | 13095.23 | 13087.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 13043.00 | 13095.23 | 13087.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 13105.00 | 13089.95 | 13085.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 13070.00 | 13089.95 | 13085.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 13130.00 | 13097.96 | 13089.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 13242.00 | 13121.37 | 13101.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 13189.00 | 13200.79 | 13179.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 13192.00 | 13186.90 | 13177.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:00:00 | 13250.00 | 13186.90 | 13177.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 13154.00 | 13180.32 | 13175.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 13154.00 | 13180.32 | 13175.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 13218.00 | 13187.85 | 13179.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 13146.00 | 13187.85 | 13179.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 13220.00 | 13199.27 | 13186.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 13261.00 | 13188.45 | 13184.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 13229.00 | 13186.96 | 13183.19 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 13115.00 | 13167.93 | 13174.91 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 13292.00 | 13191.49 | 13183.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 13391.00 | 13250.83 | 13212.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 13463.00 | 13506.02 | 13393.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:45:00 | 13441.00 | 13506.02 | 13393.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 13565.00 | 13517.82 | 13409.44 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 13220.00 | 13409.57 | 13426.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 13093.00 | 13346.25 | 13396.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 13256.00 | 13214.93 | 13282.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 13256.00 | 13214.93 | 13282.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 13240.00 | 13219.94 | 13279.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 13295.00 | 13219.94 | 13279.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 13246.00 | 13225.15 | 13276.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 13192.00 | 13225.15 | 13276.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 13420.00 | 13290.99 | 13292.53 | SL hit (close>static) qty=1.00 sl=13364.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 13393.00 | 13311.39 | 13301.67 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 13199.00 | 13289.74 | 13298.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 13166.00 | 13249.99 | 13277.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 13234.00 | 13077.80 | 13135.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 13234.00 | 13077.80 | 13135.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 13234.00 | 13077.80 | 13135.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 13234.00 | 13077.80 | 13135.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 13326.00 | 13127.44 | 13152.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 13377.00 | 13127.44 | 13152.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 13350.00 | 13171.95 | 13170.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 13473.00 | 13256.65 | 13211.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 13534.00 | 13634.12 | 13525.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 13534.00 | 13634.12 | 13525.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 13534.00 | 13634.12 | 13525.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 13534.00 | 13634.12 | 13525.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 13452.00 | 13597.69 | 13518.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 13452.00 | 13597.69 | 13518.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 13448.00 | 13567.76 | 13512.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 13418.00 | 13567.76 | 13512.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 13256.00 | 13446.98 | 13466.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 13155.00 | 13369.87 | 13426.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 14:15:00 | 13320.00 | 13276.39 | 13347.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 13320.00 | 13276.39 | 13347.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 13175.00 | 13263.25 | 13329.64 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 13611.00 | 13367.77 | 13362.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 13702.00 | 13434.61 | 13393.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 13362.00 | 13660.64 | 13594.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 13362.00 | 13660.64 | 13594.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 13362.00 | 13660.64 | 13594.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 13362.00 | 13660.64 | 13594.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 13402.00 | 13608.91 | 13576.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 13331.00 | 13608.91 | 13576.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 13500.00 | 13587.30 | 13572.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 13512.00 | 13587.30 | 13572.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 13612.00 | 13592.24 | 13576.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 13650.00 | 13593.79 | 13578.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 13662.00 | 13610.23 | 13589.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 14084.00 | 13704.98 | 13634.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-25 11:15:00 | 15015.00 | 14752.22 | 14598.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 14322.00 | 14606.18 | 14611.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 14241.00 | 14533.14 | 14578.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 14147.00 | 14134.71 | 14265.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 14147.00 | 14134.71 | 14265.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 14207.00 | 14149.17 | 14259.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 14308.00 | 14149.17 | 14259.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 14100.00 | 14122.08 | 14203.68 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 14239.00 | 14204.23 | 14200.89 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 14111.00 | 14187.47 | 14194.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 09:15:00 | 13841.00 | 14118.18 | 14162.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 13647.00 | 13544.39 | 13646.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 13647.00 | 13544.39 | 13646.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 13647.00 | 13544.39 | 13646.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 13647.00 | 13544.39 | 13646.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 13696.00 | 13574.71 | 13650.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 13696.00 | 13574.71 | 13650.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 13724.00 | 13604.57 | 13657.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 13722.00 | 13604.57 | 13657.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 13672.00 | 13649.94 | 13665.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 13672.00 | 13649.94 | 13665.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 13695.00 | 13658.95 | 13668.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 13714.00 | 13658.95 | 13668.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 13665.00 | 13660.16 | 13668.01 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 13702.00 | 13674.58 | 13673.58 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 13607.00 | 13677.87 | 13678.32 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 13721.00 | 13686.50 | 13682.20 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 13611.00 | 13677.75 | 13679.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 13545.00 | 13626.36 | 13652.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 13658.00 | 13544.76 | 13582.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 13658.00 | 13544.76 | 13582.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 13658.00 | 13544.76 | 13582.36 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 13638.00 | 13604.84 | 13602.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 13703.00 | 13632.87 | 13617.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 13716.00 | 13725.53 | 13689.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:45:00 | 13711.00 | 13725.53 | 13689.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 13658.00 | 13712.02 | 13686.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 13658.00 | 13712.02 | 13686.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 13623.00 | 13694.22 | 13680.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 13623.00 | 13694.22 | 13680.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 13550.00 | 13665.38 | 13669.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 13465.00 | 13600.67 | 13635.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 13:15:00 | 13341.00 | 13277.75 | 13364.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 13:15:00 | 13341.00 | 13277.75 | 13364.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 13341.00 | 13277.75 | 13364.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 13341.00 | 13277.75 | 13364.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 13322.00 | 13286.60 | 13360.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 13322.00 | 13286.60 | 13360.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 13052.00 | 13246.62 | 13329.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 13040.00 | 13185.90 | 13294.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 13010.00 | 13062.49 | 13193.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 13039.00 | 13049.43 | 13153.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 12388.00 | 12589.30 | 12730.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 12387.05 | 12589.30 | 12730.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 15:15:00 | 12359.50 | 12534.04 | 12657.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 12554.00 | 12538.03 | 12648.06 | SL hit (close>ema200) qty=0.50 sl=12538.03 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 12906.00 | 12729.51 | 12710.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 13008.00 | 12817.29 | 12755.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 13415.00 | 13426.25 | 13238.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:00:00 | 13415.00 | 13426.25 | 13238.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 13412.00 | 13495.82 | 13421.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 13412.00 | 13495.82 | 13421.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 13250.00 | 13446.66 | 13405.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 13250.00 | 13446.66 | 13405.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 13238.00 | 13404.93 | 13390.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 13231.00 | 13404.93 | 13390.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 13209.00 | 13365.74 | 13374.02 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 13374.00 | 13359.10 | 13357.10 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 13281.00 | 13343.48 | 13350.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 11:15:00 | 13225.00 | 13319.79 | 13338.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 13296.00 | 13293.62 | 13320.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 13296.00 | 13293.62 | 13320.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 13296.00 | 13293.62 | 13320.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 13296.00 | 13293.62 | 13320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 13299.00 | 13294.69 | 13318.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 13181.00 | 13294.69 | 13318.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 13139.00 | 13263.56 | 13302.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 13096.00 | 13230.24 | 13283.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 13049.00 | 13055.38 | 13102.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 13091.00 | 13048.65 | 13083.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 13114.00 | 13077.27 | 13087.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 13089.00 | 13079.62 | 13087.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 12976.00 | 13079.62 | 13087.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 13011.00 | 13052.12 | 13072.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:45:00 | 13042.00 | 13050.48 | 13067.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:00:00 | 13025.00 | 13006.83 | 13025.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 13060.00 | 13017.46 | 13028.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 13010.00 | 13034.78 | 13035.77 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 13083.00 | 13044.42 | 13040.06 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 13019.00 | 13033.99 | 13035.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 12926.00 | 12996.79 | 13015.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 12911.00 | 12909.58 | 12950.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 12911.00 | 12909.58 | 12950.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 12911.00 | 12909.58 | 12950.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 12911.00 | 12909.58 | 12950.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 12902.00 | 12898.20 | 12928.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 12902.00 | 12898.20 | 12928.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 12900.00 | 12898.56 | 12925.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 13040.00 | 12898.56 | 12925.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 13006.00 | 12920.05 | 12933.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 13040.00 | 12920.05 | 12933.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 13090.00 | 12954.04 | 12947.39 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 12886.00 | 12943.24 | 12946.84 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 13088.00 | 12975.48 | 12960.71 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 12896.00 | 12964.80 | 12968.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 12875.00 | 12946.84 | 12960.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 12537.00 | 12536.94 | 12678.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:15:00 | 12570.00 | 12536.94 | 12678.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 12670.00 | 12594.35 | 12672.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 12670.00 | 12594.35 | 12672.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 12641.00 | 12603.68 | 12669.42 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 12796.00 | 12697.83 | 12697.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 13:15:00 | 12864.00 | 12748.05 | 12721.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 12750.00 | 12789.45 | 12753.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 12750.00 | 12789.45 | 12753.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 12750.00 | 12789.45 | 12753.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 12705.00 | 12789.45 | 12753.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 12730.00 | 12777.56 | 12751.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 12723.00 | 12777.56 | 12751.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 12814.00 | 12784.85 | 12757.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:15:00 | 12702.00 | 12784.85 | 12757.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 12502.00 | 12728.28 | 12733.91 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 12816.00 | 12745.82 | 12741.37 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 12684.00 | 12738.77 | 12739.26 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 12800.00 | 12749.86 | 12743.96 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 12600.00 | 12716.87 | 12729.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 12584.00 | 12679.76 | 12710.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 12660.00 | 12653.80 | 12685.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:00:00 | 12660.00 | 12653.80 | 12685.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 12649.00 | 12652.84 | 12682.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 12696.00 | 12652.84 | 12682.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 12698.00 | 12661.87 | 12683.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 12629.00 | 12661.87 | 12683.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 12616.00 | 12613.43 | 12638.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 12867.00 | 12668.27 | 12654.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 12867.00 | 12668.27 | 12654.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 12887.00 | 12788.93 | 12722.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 12789.00 | 12822.31 | 12757.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 12789.00 | 12822.31 | 12757.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 12789.00 | 12822.31 | 12757.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 12789.00 | 12822.31 | 12757.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 12884.00 | 12834.65 | 12768.67 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 12599.00 | 12738.87 | 12743.76 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 13064.00 | 12727.62 | 12708.11 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 12864.00 | 12878.01 | 12878.75 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 12918.00 | 12886.01 | 12882.32 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 12861.00 | 12880.04 | 12880.19 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 12900.00 | 12883.38 | 12881.36 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 12818.00 | 12870.31 | 12875.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 12749.00 | 12846.05 | 12864.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 12839.00 | 12815.36 | 12836.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 12839.00 | 12815.36 | 12836.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 12839.00 | 12815.36 | 12836.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 12852.00 | 12815.36 | 12836.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 12924.00 | 12837.08 | 12844.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 12924.00 | 12837.08 | 12844.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 12904.00 | 12850.47 | 12850.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 12969.00 | 12885.46 | 12866.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 12801.00 | 12873.93 | 12865.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 15:15:00 | 12801.00 | 12873.93 | 12865.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 12801.00 | 12873.93 | 12865.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 12977.00 | 12873.93 | 12865.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 12886.00 | 12876.35 | 12866.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 13050.00 | 12911.08 | 12883.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:45:00 | 13063.00 | 12964.41 | 12914.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 13052.00 | 13009.68 | 12964.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-01 13:15:00 | 14355.00 | 13730.33 | 13379.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 14416.00 | 14706.18 | 14709.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 14270.00 | 14451.31 | 14522.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 14246.00 | 14211.15 | 14285.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:00:00 | 14246.00 | 14211.15 | 14285.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 14266.00 | 14222.12 | 14283.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 14266.00 | 14222.12 | 14283.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 14335.00 | 14244.70 | 14288.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 14302.00 | 14244.70 | 14288.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 14490.00 | 14293.76 | 14306.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 14490.00 | 14293.76 | 14306.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 14565.00 | 14348.01 | 14329.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 14622.00 | 14402.81 | 14356.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 14:15:00 | 14750.00 | 14972.51 | 14801.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 14750.00 | 14972.51 | 14801.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 14750.00 | 14972.51 | 14801.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 14750.00 | 14972.51 | 14801.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 14703.00 | 14918.61 | 14792.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:15:00 | 14991.00 | 14880.09 | 14786.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 14850.00 | 14869.90 | 14799.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 14982.00 | 15138.42 | 15159.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 14982.00 | 15138.42 | 15159.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 14802.00 | 15030.47 | 15101.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 14900.00 | 14856.41 | 14954.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 14900.00 | 14856.41 | 14954.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 14900.00 | 14856.41 | 14954.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 14877.00 | 14856.41 | 14954.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 14717.00 | 14776.16 | 14858.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 14670.00 | 14747.37 | 14800.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 14953.00 | 14802.06 | 14789.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 14953.00 | 14802.06 | 14789.30 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 14713.00 | 14798.08 | 14803.88 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 14966.00 | 14793.69 | 14790.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 15029.00 | 14888.21 | 14840.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 15060.00 | 15109.50 | 15018.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:15:00 | 15345.00 | 15109.50 | 15018.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 14937.00 | 15221.47 | 15160.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 14937.00 | 15221.47 | 15160.81 | SL hit (close<ema400) qty=1.00 sl=15160.81 alert=retest1 |

### Cycle 138 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 14722.00 | 15075.58 | 15102.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 14598.00 | 14766.56 | 14892.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 14150.00 | 14117.35 | 14353.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 14131.00 | 14117.35 | 14353.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 13908.00 | 14057.51 | 14155.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 13896.00 | 14030.83 | 14051.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:00:00 | 13855.00 | 13995.67 | 14033.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 13900.00 | 13783.71 | 13826.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 13885.00 | 13820.38 | 13835.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 14033.00 | 13885.08 | 13863.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 14:15:00 | 13853.00 | 13878.66 | 13862.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 14:15:00 | 13853.00 | 13878.66 | 13862.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 13853.00 | 13878.66 | 13862.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 13853.00 | 13878.66 | 13862.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 13798.00 | 13862.53 | 13857.08 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 13800.00 | 13850.03 | 13851.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 13726.00 | 13825.22 | 13840.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 13643.00 | 13641.35 | 13722.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 10:30:00 | 13650.00 | 13641.35 | 13722.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 13725.00 | 13658.08 | 13722.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:30:00 | 13780.00 | 13658.08 | 13722.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 13757.00 | 13677.86 | 13726.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:00:00 | 13757.00 | 13677.86 | 13726.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 13736.00 | 13689.49 | 13726.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:45:00 | 13706.00 | 13689.49 | 13726.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 14065.00 | 13764.59 | 13757.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 14289.00 | 14044.05 | 13923.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 14226.00 | 14226.60 | 14111.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:30:00 | 14239.00 | 14226.60 | 14111.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 15093.00 | 14399.88 | 14200.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 15200.00 | 14921.01 | 14626.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:00:00 | 15102.00 | 15043.31 | 14866.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 15399.00 | 14995.44 | 14874.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 15157.00 | 15273.23 | 15186.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 15172.00 | 15252.99 | 15185.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 15168.00 | 15252.99 | 15185.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 15075.00 | 15217.39 | 15175.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 15075.00 | 15217.39 | 15175.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 15189.00 | 15211.71 | 15176.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 14860.00 | 15059.67 | 15110.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 15001.00 | 14980.29 | 15056.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:45:00 | 14961.00 | 14980.29 | 15056.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 15016.00 | 14987.43 | 15053.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 15030.00 | 14987.43 | 15053.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 14962.00 | 14982.35 | 15044.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 15028.00 | 14982.35 | 15044.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 15027.00 | 14991.28 | 15043.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 15027.00 | 14991.28 | 15043.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 15111.00 | 15015.22 | 15049.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 15120.00 | 15015.22 | 15049.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 15111.00 | 15034.38 | 15054.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 15111.00 | 15034.38 | 15054.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 15164.00 | 15060.30 | 15064.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 15322.00 | 15060.30 | 15064.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 15360.00 | 15120.24 | 15091.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 15600.00 | 15269.29 | 15185.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 14:15:00 | 15801.00 | 15819.87 | 15654.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:15:00 | 15963.00 | 15816.50 | 15667.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 15922.00 | 15837.60 | 15691.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:30:00 | 16359.00 | 15983.52 | 15842.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 15960.00 | 16127.75 | 15985.53 | SL hit (close<ema400) qty=1.00 sl=15985.53 alert=retest1 |

### Cycle 144 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 15789.00 | 15916.59 | 15927.45 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 16045.00 | 15937.69 | 15934.87 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 15776.00 | 15905.56 | 15920.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 15728.00 | 15870.05 | 15903.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 15388.00 | 15361.72 | 15506.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:00:00 | 15388.00 | 15361.72 | 15506.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 15368.00 | 15362.98 | 15493.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 15499.00 | 15362.98 | 15493.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 15638.00 | 15393.60 | 15472.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 15552.00 | 15393.60 | 15472.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 15684.00 | 15451.68 | 15491.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 15553.00 | 15451.68 | 15491.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 15599.00 | 15532.44 | 15523.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 15843.00 | 15612.98 | 15564.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 15732.00 | 15745.80 | 15671.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 15732.00 | 15745.80 | 15671.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 15651.00 | 15726.84 | 15669.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 15651.00 | 15726.84 | 15669.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 15615.00 | 15704.47 | 15664.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 15615.00 | 15704.47 | 15664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 15580.00 | 15679.58 | 15656.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 15580.00 | 15679.58 | 15656.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 15456.00 | 15634.86 | 15638.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 09:15:00 | 15412.00 | 15565.72 | 15603.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 12:15:00 | 15558.00 | 15542.88 | 15582.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:00:00 | 15558.00 | 15542.88 | 15582.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 15500.00 | 15534.31 | 15574.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:45:00 | 15567.00 | 15534.31 | 15574.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 15608.00 | 15539.24 | 15569.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 15340.00 | 15539.24 | 15569.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 14573.00 | 14717.28 | 14989.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 14349.00 | 14328.00 | 14544.58 | SL hit (close>ema200) qty=0.50 sl=14328.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 13825.00 | 13746.96 | 13744.64 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 13545.00 | 13708.85 | 13727.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 13440.00 | 13655.08 | 13701.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 13572.00 | 13550.08 | 13625.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:15:00 | 13636.00 | 13550.08 | 13625.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 13474.00 | 13534.87 | 13611.68 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 13812.00 | 13663.27 | 13653.22 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 13421.00 | 13616.54 | 13633.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 13086.00 | 13510.43 | 13584.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 13440.00 | 13300.82 | 13441.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 13440.00 | 13300.82 | 13441.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 13440.00 | 13300.82 | 13441.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 13440.00 | 13300.82 | 13441.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 13061.00 | 13252.85 | 13407.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 13398.00 | 13252.85 | 13407.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 13387.00 | 13279.68 | 13405.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 13190.00 | 13268.15 | 13388.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:45:00 | 13219.00 | 13271.21 | 13369.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 13564.00 | 13390.54 | 13381.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 13564.00 | 13390.54 | 13381.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 15:15:00 | 13802.00 | 13596.22 | 13524.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 13591.00 | 13595.18 | 13530.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 13591.00 | 13595.18 | 13530.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 13591.00 | 13595.18 | 13530.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:00:00 | 13914.00 | 13613.52 | 13557.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 13726.00 | 14056.58 | 14022.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 13800.00 | 14056.58 | 14022.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 13722.00 | 13989.66 | 13995.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 13722.00 | 13989.66 | 13995.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 13571.00 | 13790.92 | 13883.57 | Break + close below crossover candle low |

### Cycle 155 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 14542.00 | 13847.43 | 13843.43 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 14127.00 | 14235.34 | 14246.48 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 13:15:00 | 14365.00 | 14178.61 | 14174.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 14506.00 | 14293.13 | 14232.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 14443.00 | 14465.97 | 14361.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 15:15:00 | 14443.00 | 14465.97 | 14361.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 14443.00 | 14465.97 | 14361.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 14699.00 | 14531.38 | 14400.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 14977.00 | 15086.40 | 15093.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 14977.00 | 15086.40 | 15093.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 14858.00 | 15040.72 | 15072.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 12:15:00 | 15102.00 | 15024.38 | 15057.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 12:15:00 | 15102.00 | 15024.38 | 15057.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 15102.00 | 15024.38 | 15057.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 15034.00 | 15024.38 | 15057.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 14872.00 | 14993.90 | 15040.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:45:00 | 15158.00 | 14993.90 | 15040.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 14952.00 | 14985.52 | 15032.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 14952.00 | 14985.52 | 15032.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 14870.00 | 14962.42 | 15017.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 14985.00 | 14962.42 | 15017.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 15101.00 | 14990.14 | 15025.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 15101.00 | 14990.14 | 15025.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 15071.00 | 15006.31 | 15029.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 14985.00 | 15006.31 | 15029.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 15019.00 | 15008.85 | 15028.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 14977.00 | 14999.66 | 15021.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 15000.00 | 14938.16 | 14976.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 15000.00 | 14950.53 | 14978.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 15000.00 | 14950.53 | 14978.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 14961.00 | 14919.62 | 14952.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 14902.00 | 14919.62 | 14952.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 14919.00 | 14919.49 | 14949.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 14840.00 | 14895.02 | 14931.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 14855.00 | 14709.66 | 14769.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:15:00 | 16678.50 | 2024-06-04 15:15:00 | 16120.10 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2024-06-13 14:15:00 | 17355.15 | 2024-06-14 14:15:00 | 17300.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-06-13 15:00:00 | 17381.45 | 2024-06-18 10:15:00 | 17339.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-06-14 09:30:00 | 17358.15 | 2024-06-18 10:15:00 | 17339.95 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-06-14 10:15:00 | 17474.95 | 2024-06-18 10:15:00 | 17339.95 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-06-14 13:00:00 | 17759.05 | 2024-06-18 10:15:00 | 17339.95 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-06-25 09:15:00 | 15909.20 | 2024-06-28 11:15:00 | 15845.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2024-06-25 10:00:00 | 15901.55 | 2024-06-28 11:15:00 | 15845.00 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-07-02 14:45:00 | 15991.35 | 2024-07-05 12:15:00 | 15740.05 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-07-03 14:00:00 | 15976.00 | 2024-07-05 12:15:00 | 15740.05 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-07-03 14:45:00 | 15994.30 | 2024-07-05 12:15:00 | 15740.05 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-07-03 15:15:00 | 15979.00 | 2024-07-05 12:15:00 | 15740.05 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-04 15:15:00 | 15995.30 | 2024-07-05 12:15:00 | 15740.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-05 09:45:00 | 16016.05 | 2024-07-05 12:15:00 | 15740.05 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-07-15 14:30:00 | 15760.00 | 2024-07-15 15:15:00 | 15880.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-07-16 14:30:00 | 15720.90 | 2024-07-22 09:15:00 | 14934.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 15:15:00 | 15741.85 | 2024-07-22 09:15:00 | 14954.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 10:00:00 | 15722.15 | 2024-07-22 09:15:00 | 14936.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 14:30:00 | 15720.90 | 2024-07-22 10:15:00 | 15280.10 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2024-07-16 15:15:00 | 15741.85 | 2024-07-22 10:15:00 | 15280.10 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2024-07-18 10:00:00 | 15722.15 | 2024-07-22 10:15:00 | 15280.10 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-07-19 09:15:00 | 15434.75 | 2024-07-24 10:15:00 | 15581.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-08-01 09:30:00 | 15877.35 | 2024-08-01 13:15:00 | 15766.75 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-08-01 10:00:00 | 15880.20 | 2024-08-01 13:15:00 | 15766.75 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-01 15:00:00 | 15854.70 | 2024-08-05 10:15:00 | 15721.95 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-08-02 10:30:00 | 15887.75 | 2024-08-05 10:15:00 | 15721.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-08-13 09:15:00 | 15341.55 | 2024-08-14 15:15:00 | 15999.90 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-08-27 09:45:00 | 15487.55 | 2024-08-30 13:15:00 | 15475.00 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-08-27 11:00:00 | 15457.50 | 2024-08-30 13:15:00 | 15475.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-08-29 12:15:00 | 15464.60 | 2024-08-30 13:15:00 | 15475.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-09-09 12:45:00 | 16161.75 | 2024-09-17 10:15:00 | 16622.55 | STOP_HIT | 1.00 | 2.85% |
| SELL | retest2 | 2024-09-20 14:45:00 | 16349.70 | 2024-09-24 14:15:00 | 16479.30 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-23 15:15:00 | 16365.00 | 2024-09-24 14:15:00 | 16479.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-04 09:15:00 | 15667.80 | 2024-10-08 09:15:00 | 14884.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 09:15:00 | 15667.80 | 2024-10-08 12:15:00 | 15132.75 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-10-25 09:30:00 | 13745.45 | 2024-10-28 14:15:00 | 13998.90 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-10-25 15:15:00 | 13751.00 | 2024-10-28 14:15:00 | 13998.90 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-11-06 12:00:00 | 14588.20 | 2024-11-06 12:15:00 | 14419.10 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-11-07 09:15:00 | 15007.40 | 2024-11-13 12:15:00 | 14800.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-12-06 15:15:00 | 12325.00 | 2024-12-11 14:15:00 | 11708.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 09:30:00 | 12328.40 | 2024-12-11 14:15:00 | 11711.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 15:15:00 | 12325.00 | 2024-12-16 12:15:00 | 11524.40 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2024-12-09 09:30:00 | 12328.40 | 2024-12-16 12:15:00 | 11524.40 | STOP_HIT | 0.50 | 6.52% |
| BUY | retest2 | 2024-12-23 09:15:00 | 11723.55 | 2024-12-24 11:15:00 | 11550.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-12-24 09:15:00 | 11764.35 | 2024-12-24 11:15:00 | 11550.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-01-02 09:15:00 | 11988.30 | 2025-01-02 13:15:00 | 11897.80 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-01-10 09:15:00 | 11344.90 | 2025-01-15 13:15:00 | 11384.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-01-10 10:45:00 | 11400.25 | 2025-01-15 13:15:00 | 11384.85 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-01-10 12:15:00 | 11423.65 | 2025-01-15 13:15:00 | 11384.85 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-01-31 14:30:00 | 11001.00 | 2025-02-05 12:15:00 | 10833.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-01 13:30:00 | 10987.10 | 2025-02-05 12:15:00 | 10833.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-01 14:45:00 | 11001.55 | 2025-02-05 12:15:00 | 10833.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-03 09:15:00 | 11074.90 | 2025-02-05 12:15:00 | 10833.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-02-12 09:15:00 | 11341.25 | 2025-02-12 09:15:00 | 10903.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-02-17 10:45:00 | 10477.20 | 2025-02-17 14:15:00 | 10986.00 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2025-02-17 13:45:00 | 10487.00 | 2025-02-17 14:15:00 | 10986.00 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2025-02-18 12:30:00 | 10345.80 | 2025-02-19 13:15:00 | 10761.30 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-02-19 09:45:00 | 10446.60 | 2025-02-19 13:15:00 | 10761.30 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-02-24 11:30:00 | 10872.00 | 2025-02-27 12:15:00 | 10776.55 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-24 14:45:00 | 10825.85 | 2025-02-27 12:15:00 | 10776.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-02-24 15:15:00 | 10845.00 | 2025-02-27 12:15:00 | 10776.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-02-27 11:15:00 | 10845.60 | 2025-02-27 12:15:00 | 10776.55 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-03-10 09:30:00 | 11456.20 | 2025-03-10 14:15:00 | 11250.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-03-20 09:15:00 | 12230.00 | 2025-03-20 09:15:00 | 11994.65 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-04-01 09:30:00 | 13164.10 | 2025-04-04 10:15:00 | 12904.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-04-02 13:30:00 | 13077.00 | 2025-04-04 10:15:00 | 12904.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-04-02 14:15:00 | 13160.00 | 2025-04-04 10:15:00 | 12904.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-04-03 09:15:00 | 13139.05 | 2025-04-04 10:15:00 | 12904.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-05-02 09:15:00 | 12408.00 | 2025-05-07 14:15:00 | 12406.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-05-02 10:00:00 | 12413.00 | 2025-05-07 14:15:00 | 12406.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-05-19 09:15:00 | 13400.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-19 10:15:00 | 13315.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-19 14:15:00 | 13410.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-20 14:00:00 | 13291.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-05-28 12:00:00 | 12875.00 | 2025-05-30 12:15:00 | 13252.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-05-28 13:30:00 | 12843.00 | 2025-05-30 12:15:00 | 13252.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest1 | 2025-06-04 09:15:00 | 13594.00 | 2025-06-04 12:15:00 | 13375.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest1 | 2025-06-04 10:45:00 | 13620.00 | 2025-06-04 12:15:00 | 13375.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-06-05 09:45:00 | 13523.00 | 2025-06-10 14:15:00 | 13536.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-06-05 14:15:00 | 13515.00 | 2025-06-10 14:15:00 | 13536.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-06-06 09:15:00 | 13854.00 | 2025-06-10 14:15:00 | 13536.00 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-06-20 15:15:00 | 13110.00 | 2025-06-25 09:15:00 | 13268.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-06-24 13:30:00 | 13126.00 | 2025-06-25 09:15:00 | 13268.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-30 14:30:00 | 13369.00 | 2025-07-01 09:15:00 | 13280.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-09 09:45:00 | 13100.00 | 2025-07-15 10:15:00 | 13143.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-10 09:45:00 | 13112.00 | 2025-07-15 10:15:00 | 13143.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-10 10:45:00 | 13112.00 | 2025-07-15 10:15:00 | 13143.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-07-17 10:30:00 | 13242.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-18 15:15:00 | 13189.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-21 11:30:00 | 13192.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-21 12:00:00 | 13250.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-07-22 10:30:00 | 13261.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-30 10:15:00 | 13192.00 | 2025-07-30 13:15:00 | 13420.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-08-13 15:15:00 | 13650.00 | 2025-08-25 11:15:00 | 15015.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 10:15:00 | 13662.00 | 2025-08-25 11:15:00 | 15028.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 11:00:00 | 14084.00 | 2025-08-26 11:15:00 | 14322.00 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-09-23 10:30:00 | 13040.00 | 2025-09-29 11:15:00 | 12388.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13010.00 | 2025-09-29 11:15:00 | 12387.05 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-09-24 11:15:00 | 13039.00 | 2025-09-29 15:15:00 | 12359.50 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-09-23 10:30:00 | 13040.00 | 2025-09-30 09:15:00 | 12554.00 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13010.00 | 2025-09-30 09:15:00 | 12554.00 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-09-24 11:15:00 | 13039.00 | 2025-09-30 09:15:00 | 12554.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2025-10-13 10:45:00 | 13096.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-15 12:45:00 | 13049.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-16 09:30:00 | 13091.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-16 14:00:00 | 13114.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-10-16 15:15:00 | 12976.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-17 09:30:00 | 13011.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-10-17 12:45:00 | 13042.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-10-20 15:00:00 | 13025.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-11 09:15:00 | 12629.00 | 2025-11-12 14:15:00 | 12867.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-11-12 10:15:00 | 12616.00 | 2025-11-12 14:15:00 | 12867.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-11-27 11:00:00 | 13050.00 | 2025-12-01 13:15:00 | 14355.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-27 12:45:00 | 13063.00 | 2025-12-01 13:15:00 | 14369.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-28 12:00:00 | 13052.00 | 2025-12-01 13:15:00 | 14357.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 10:15:00 | 14991.00 | 2025-12-23 13:15:00 | 14982.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-18 12:15:00 | 14850.00 | 2025-12-23 13:15:00 | 14982.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-12-30 10:15:00 | 14670.00 | 2025-12-31 09:15:00 | 14953.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest1 | 2026-01-06 09:15:00 | 15345.00 | 2026-01-07 09:15:00 | 14937.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-01-20 09:15:00 | 13896.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-20 10:00:00 | 13855.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-22 10:15:00 | 13900.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-01-22 11:45:00 | 13885.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-02-01 12:30:00 | 15200.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-02 14:00:00 | 15102.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-02-03 09:15:00 | 15399.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-02-05 09:15:00 | 15157.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2026-02-13 09:15:00 | 15963.00 | 2026-02-16 14:15:00 | 15960.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2026-02-16 09:30:00 | 16359.00 | 2026-02-17 13:15:00 | 15789.00 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-02-27 09:15:00 | 15340.00 | 2026-03-04 09:15:00 | 14573.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 15340.00 | 2026-03-05 12:15:00 | 14349.00 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2026-03-24 10:30:00 | 13190.00 | 2026-03-25 11:15:00 | 13564.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-24 12:45:00 | 13219.00 | 2026-03-25 11:15:00 | 13564.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-03-30 15:00:00 | 13914.00 | 2026-04-06 10:15:00 | 13722.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-06 09:30:00 | 13726.00 | 2026-04-06 10:15:00 | 13722.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-04-06 10:00:00 | 13800.00 | 2026-04-06 10:15:00 | 13722.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-04-20 09:45:00 | 14699.00 | 2026-04-24 09:15:00 | 14977.00 | STOP_HIT | 1.00 | 1.89% |

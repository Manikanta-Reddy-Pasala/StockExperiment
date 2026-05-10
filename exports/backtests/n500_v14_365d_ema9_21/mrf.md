# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 130490.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 44 |
| ALERT2 | 43 |
| ALERT2_SKIP | 19 |
| ALERT3 | 105 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 62 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 18 / 48
- **Target hits / Stop hits / Partials:** 0 / 63 / 3
- **Avg / median % per leg:** -0.04% / -0.49%
- **Sum % (uncompounded):** -2.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 6 | 24.0% | 0 | 25 | 0 | -0.52% | -12.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.68% | -5.0% |
| BUY @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 0 | 22 | 0 | -0.36% | -7.9% |
| SELL (all) | 41 | 12 | 29.3% | 0 | 38 | 3 | 0.26% | 10.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| SELL @ 3rd Alert (retest2) | 40 | 12 | 30.0% | 0 | 37 | 3 | 0.29% | 11.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.53% | -6.1% |
| retest2 (combined) | 62 | 18 | 29.0% | 0 | 59 | 3 | 0.06% | 3.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 140130.00 | 140673.96 | 140726.99 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 142045.00 | 140712.62 | 140552.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 142125.00 | 140995.09 | 140695.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 141655.00 | 141704.21 | 141235.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:45:00 | 141800.00 | 141704.21 | 141235.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 144615.00 | 145336.31 | 144653.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 144615.00 | 145336.31 | 144653.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 144315.00 | 145132.05 | 144622.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 144315.00 | 145132.05 | 144622.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 143840.00 | 144873.64 | 144551.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 143840.00 | 144873.64 | 144551.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 142695.00 | 144080.70 | 144244.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 141990.00 | 142974.10 | 143469.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 139590.00 | 139308.22 | 140367.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:30:00 | 140085.00 | 139308.22 | 140367.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 140330.00 | 139512.58 | 140363.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 140330.00 | 139512.58 | 140363.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 140695.00 | 139749.06 | 140393.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 139835.00 | 139646.80 | 140239.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 140050.00 | 138433.86 | 138444.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 140080.00 | 138433.86 | 138444.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 139995.00 | 138746.09 | 138585.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 139995.00 | 138746.09 | 138585.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 139995.00 | 138746.09 | 138585.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 139995.00 | 138746.09 | 138585.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 140300.00 | 139056.87 | 138741.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 138995.00 | 139181.80 | 138861.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 138995.00 | 139181.80 | 138861.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 138995.00 | 139181.80 | 138861.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 141010.00 | 139382.55 | 139109.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 138530.00 | 139069.07 | 139101.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 138530.00 | 139069.07 | 139101.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 137445.00 | 138744.26 | 138951.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 138235.00 | 137876.78 | 138279.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 138235.00 | 137876.78 | 138279.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 138235.00 | 137876.78 | 138279.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 137835.00 | 137975.51 | 138232.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 137295.00 | 136686.78 | 136664.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 137295.00 | 136686.78 | 136664.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 137830.00 | 137120.83 | 136885.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 137620.00 | 137893.31 | 137520.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 138315.00 | 137893.31 | 137520.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 136415.00 | 137597.65 | 137419.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 136415.00 | 137597.65 | 137419.88 | SL hit (close<ema400) qty=1.00 sl=137419.88 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 136415.00 | 137597.65 | 137419.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 137085.00 | 137495.12 | 137389.44 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 12:15:00 | 136865.00 | 137273.88 | 137301.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 135000.00 | 136719.28 | 137038.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 136025.00 | 135723.68 | 136140.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 136025.00 | 135723.68 | 136140.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 136025.00 | 135723.68 | 136140.60 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 136755.00 | 136322.64 | 136315.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 137610.00 | 136703.69 | 136497.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 142830.00 | 142932.77 | 141150.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 142830.00 | 142932.77 | 141150.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 142725.00 | 143457.70 | 142388.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 142725.00 | 143457.70 | 142388.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 142005.00 | 143167.16 | 142353.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 143230.00 | 143167.16 | 142353.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:00:00 | 142915.00 | 142698.35 | 142357.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 143215.00 | 142619.36 | 142411.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 143690.00 | 144325.02 | 144336.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 143690.00 | 144325.02 | 144336.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 143690.00 | 144325.02 | 144336.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 143690.00 | 144325.02 | 144336.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 142940.00 | 143929.13 | 144143.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 143240.00 | 143130.54 | 143598.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 15:00:00 | 143240.00 | 143130.54 | 143598.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 143965.00 | 143196.55 | 143467.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 143965.00 | 143196.55 | 143467.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 144275.00 | 143412.24 | 143540.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 144300.00 | 143412.24 | 143540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 144905.00 | 143876.83 | 143739.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 146500.00 | 144421.17 | 144012.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 147590.00 | 148461.07 | 146885.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 147590.00 | 148461.07 | 146885.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 148220.00 | 148412.86 | 147007.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 149860.00 | 147910.86 | 147511.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 149030.00 | 148164.37 | 147815.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 148995.00 | 148173.49 | 147851.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 148480.00 | 150271.01 | 150330.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 148480.00 | 150271.01 | 150330.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 148480.00 | 150271.01 | 150330.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 148480.00 | 150271.01 | 150330.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 148000.00 | 149234.87 | 149768.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 14:15:00 | 148585.00 | 148258.74 | 148910.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 15:00:00 | 148585.00 | 148258.74 | 148910.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 149750.00 | 148595.60 | 148952.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 150070.00 | 148595.60 | 148952.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 149285.00 | 148733.48 | 148983.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 148855.00 | 148773.78 | 148978.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 149095.00 | 148827.03 | 148984.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:00:00 | 148950.00 | 148736.90 | 148858.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 148990.00 | 148859.62 | 148896.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 149035.00 | 148894.69 | 148908.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 149170.00 | 148894.69 | 148908.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 151710.00 | 149749.04 | 149321.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 150165.00 | 150493.05 | 150023.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 150165.00 | 150493.05 | 150023.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 150165.00 | 150493.05 | 150023.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 150130.00 | 150493.05 | 150023.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 150245.00 | 150443.44 | 150043.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 150010.00 | 150443.44 | 150043.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 150105.00 | 150375.75 | 150048.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 150165.00 | 150375.75 | 150048.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 149795.00 | 150259.60 | 150025.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 149870.00 | 150259.60 | 150025.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 150040.00 | 150215.68 | 150027.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 149835.00 | 150215.68 | 150027.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 149875.00 | 150147.55 | 150013.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 149875.00 | 150147.55 | 150013.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 148755.00 | 149869.04 | 149898.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 148310.00 | 149291.31 | 149599.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 149410.00 | 148865.71 | 149290.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 149410.00 | 148865.71 | 149290.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 149410.00 | 148865.71 | 149290.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 149410.00 | 148865.71 | 149290.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 149240.00 | 148940.57 | 149285.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 149415.00 | 148940.57 | 149285.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 149485.00 | 149049.45 | 149303.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 149485.00 | 149049.45 | 149303.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 148700.00 | 148979.56 | 149248.79 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 149530.00 | 149342.72 | 149328.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 149825.00 | 149487.54 | 149400.11 | Break + close above crossover candle high |

### Cycle 15 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 148000.00 | 149352.66 | 149376.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 147695.00 | 149021.13 | 149223.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 11:15:00 | 145360.00 | 145056.13 | 145856.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:00:00 | 145360.00 | 145056.13 | 145856.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 146070.00 | 145286.74 | 145766.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 146070.00 | 145286.74 | 145766.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 145600.00 | 145349.39 | 145751.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 146860.00 | 145349.39 | 145751.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 146000.00 | 145479.51 | 145774.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:00:00 | 145725.00 | 145528.61 | 145769.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 146895.00 | 145990.11 | 145949.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 146895.00 | 145990.11 | 145949.31 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 144825.00 | 145919.29 | 145949.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 144605.00 | 145466.94 | 145726.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 140860.00 | 139130.88 | 139826.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 140860.00 | 139130.88 | 139826.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 140860.00 | 139130.88 | 139826.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 140860.00 | 139130.88 | 139826.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 143715.00 | 140047.70 | 140180.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 144175.00 | 140047.70 | 140180.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 143400.00 | 140718.16 | 140472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 144945.00 | 141563.53 | 140879.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 148000.00 | 148009.21 | 146868.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:00:00 | 148000.00 | 148009.21 | 146868.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 147410.00 | 147896.65 | 147180.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 147410.00 | 147896.65 | 147180.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 147210.00 | 147759.32 | 147183.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 147210.00 | 147759.32 | 147183.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 147220.00 | 147651.46 | 147186.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:30:00 | 147125.00 | 147651.46 | 147186.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 147190.00 | 147559.17 | 147187.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 147375.00 | 147643.33 | 147259.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 146605.00 | 147370.07 | 147225.76 | SL hit (close<static) qty=1.00 sl=147045.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 147475.00 | 147278.24 | 147202.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 146955.00 | 147213.59 | 147179.61 | SL hit (close<static) qty=1.00 sl=147045.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 146670.00 | 147066.70 | 147116.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 145325.00 | 146706.09 | 146943.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 141920.00 | 141369.94 | 142412.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 141920.00 | 141369.94 | 142412.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 141920.00 | 141369.94 | 142412.37 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 145000.00 | 142845.84 | 142769.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 148505.00 | 144312.74 | 143470.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 150585.00 | 150707.94 | 148247.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:15:00 | 150325.00 | 150707.94 | 148247.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 149700.00 | 150230.47 | 149581.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 149140.00 | 150230.47 | 149581.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 149100.00 | 150004.38 | 149537.49 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 148295.00 | 149253.34 | 149310.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 147100.00 | 148640.54 | 149011.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 11:15:00 | 147770.00 | 147714.66 | 148199.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:30:00 | 147820.00 | 147714.66 | 148199.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 148600.00 | 147945.38 | 148223.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 148595.00 | 147945.38 | 148223.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 149395.00 | 148235.31 | 148330.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 149395.00 | 148235.31 | 148330.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 149150.00 | 148418.25 | 148404.62 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 10:15:00 | 147850.00 | 148332.08 | 148369.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 11:15:00 | 147035.00 | 148072.66 | 148248.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 146930.00 | 146568.99 | 147260.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 146930.00 | 146568.99 | 147260.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 147650.00 | 146700.56 | 147007.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 147650.00 | 146700.56 | 147007.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 148040.00 | 146968.45 | 147101.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 148200.00 | 146968.45 | 147101.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 146760.00 | 146961.41 | 147077.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 146675.00 | 146919.13 | 147047.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:00:00 | 146280.00 | 146791.30 | 146977.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 148350.00 | 147104.43 | 147087.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 148350.00 | 147104.43 | 147087.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 148350.00 | 147104.43 | 147087.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 151400.00 | 148321.91 | 147750.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 150405.00 | 150495.50 | 149458.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:30:00 | 150390.00 | 150495.50 | 149458.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 149850.00 | 150236.34 | 149832.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 149850.00 | 150236.34 | 149832.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 149490.00 | 150087.07 | 149801.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 149490.00 | 150087.07 | 149801.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 149450.00 | 149959.66 | 149769.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 149845.00 | 149655.78 | 149654.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 148770.00 | 149478.63 | 149574.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 148770.00 | 149478.63 | 149574.34 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 150100.00 | 149623.24 | 149601.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 151345.00 | 150213.21 | 149895.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 154630.00 | 154655.59 | 153229.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 153385.00 | 154201.14 | 153363.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 153385.00 | 154201.14 | 153363.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 153420.00 | 154201.14 | 153363.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 153440.00 | 154048.91 | 153370.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 153440.00 | 154048.91 | 153370.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 153800.00 | 153999.13 | 153409.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:15:00 | 153300.00 | 153999.13 | 153409.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 153300.00 | 153859.31 | 153399.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 152340.00 | 153859.31 | 153399.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 151810.00 | 153449.44 | 153254.89 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 151640.00 | 153087.56 | 153108.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 151025.00 | 152675.04 | 152918.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 147230.00 | 146751.72 | 147721.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 147230.00 | 146751.72 | 147721.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 147230.00 | 146751.72 | 147721.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 146330.00 | 146707.37 | 147613.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:15:00 | 146365.00 | 146707.37 | 147613.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 146430.00 | 146657.90 | 147508.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 148310.00 | 146862.84 | 147377.86 | SL hit (close>static) qty=1.00 sl=147855.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 148310.00 | 146862.84 | 147377.86 | SL hit (close>static) qty=1.00 sl=147855.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 148310.00 | 146862.84 | 147377.86 | SL hit (close>static) qty=1.00 sl=147855.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 149585.00 | 147677.22 | 147675.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 150555.00 | 148252.78 | 147937.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 151495.00 | 151845.72 | 150993.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 151495.00 | 151845.72 | 150993.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 154890.00 | 155666.34 | 155052.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:45:00 | 154460.00 | 155666.34 | 155052.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 154445.00 | 155422.07 | 154996.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 154445.00 | 155422.07 | 154996.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 155200.00 | 155355.72 | 155039.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 155200.00 | 155355.72 | 155039.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 155850.00 | 155479.26 | 155151.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 156830.00 | 155479.26 | 155151.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 156395.00 | 155652.41 | 155260.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 156390.00 | 155789.93 | 155358.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 156400.00 | 155888.94 | 155442.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 156450.00 | 157187.32 | 156681.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 156450.00 | 157187.32 | 156681.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 155600.00 | 156869.86 | 156583.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 155600.00 | 156869.86 | 156583.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 156300.00 | 156755.89 | 156557.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 157900.00 | 156155.23 | 156034.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 159145.00 | 157270.88 | 156618.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 161000.00 | 161640.98 | 160286.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 162850.00 | 161872.78 | 160515.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 160590.00 | 161541.38 | 160595.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 160590.00 | 161541.38 | 160595.57 | SL hit (close<ema400) qty=1.00 sl=160595.57 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 160590.00 | 161541.38 | 160595.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 160700.00 | 161373.10 | 160605.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 160620.00 | 161373.10 | 160605.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 160000.00 | 161098.48 | 160550.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 160000.00 | 161098.48 | 160550.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 160105.00 | 160899.79 | 160509.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 160105.00 | 160899.79 | 160509.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 160000.00 | 160719.83 | 160463.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 160620.00 | 160719.83 | 160463.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 160695.00 | 160655.69 | 160474.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 160530.00 | 160625.55 | 160477.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 159315.00 | 160235.60 | 160326.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 159315.00 | 160235.60 | 160326.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 159315.00 | 160235.60 | 160326.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 159315.00 | 160235.60 | 160326.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 158455.00 | 159670.86 | 160014.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 159145.00 | 158963.62 | 159482.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 159145.00 | 158963.62 | 159482.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 159145.00 | 158963.62 | 159482.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:45:00 | 158600.00 | 158986.73 | 159365.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:45:00 | 158500.00 | 159188.39 | 159355.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 158450.00 | 159034.71 | 159270.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 158255.00 | 158597.87 | 158917.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 158330.00 | 158544.30 | 158863.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 157885.00 | 158544.30 | 158863.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 158030.00 | 158513.44 | 158820.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 158260.00 | 158189.97 | 158435.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 158210.00 | 158232.98 | 158433.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 158205.00 | 158227.38 | 158412.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 159060.00 | 158485.86 | 158490.68 | SL hit (close>static) qty=1.00 sl=158890.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 159060.00 | 158485.86 | 158490.68 | SL hit (close>static) qty=1.00 sl=158890.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 159060.00 | 158485.86 | 158490.68 | SL hit (close>static) qty=1.00 sl=158890.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 159060.00 | 158485.86 | 158490.68 | SL hit (close>static) qty=1.00 sl=158890.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 157950.00 | 158327.75 | 158415.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 157605.00 | 158191.21 | 158333.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 157735.00 | 158226.37 | 158324.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 158070.00 | 158248.28 | 158316.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 158460.00 | 158290.62 | 158329.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 158460.00 | 158290.62 | 158329.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 158445.00 | 158321.50 | 158340.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 156930.00 | 158043.20 | 158211.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 12:45:00 | 157825.00 | 157671.33 | 157913.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 158580.00 | 157853.06 | 157974.34 | SL hit (close>static) qty=1.00 sl=158500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 158580.00 | 157853.06 | 157974.34 | SL hit (close>static) qty=1.00 sl=158500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 158580.00 | 157853.06 | 157974.34 | SL hit (close>static) qty=1.00 sl=158500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 158580.00 | 157853.06 | 157974.34 | SL hit (close>static) qty=1.00 sl=158500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 158760.00 | 158034.45 | 158045.76 | SL hit (close>static) qty=1.00 sl=158600.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 158760.00 | 158034.45 | 158045.76 | SL hit (close>static) qty=1.00 sl=158600.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 158605.00 | 158148.56 | 158096.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 158605.00 | 158148.56 | 158096.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 158605.00 | 158148.56 | 158096.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 158605.00 | 158148.56 | 158096.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 158605.00 | 158148.56 | 158096.60 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 157725.00 | 158176.02 | 158183.17 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 158910.00 | 158260.02 | 158173.19 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 157650.00 | 158121.91 | 158143.50 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 158800.00 | 158257.53 | 158203.18 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 157920.00 | 158263.80 | 158298.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 155830.00 | 156989.46 | 157459.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 153150.00 | 152821.68 | 153820.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 153150.00 | 152821.68 | 153820.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 153150.00 | 152821.68 | 153820.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 153870.00 | 152821.68 | 153820.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 152655.00 | 152602.73 | 153202.77 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 157395.00 | 153852.66 | 153583.50 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 153600.00 | 154076.05 | 154117.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 152650.00 | 153719.94 | 153932.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 152610.00 | 152304.01 | 152780.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 152610.00 | 152304.01 | 152780.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 152100.00 | 152263.21 | 152718.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 154220.00 | 152263.21 | 152718.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 154800.00 | 152770.57 | 152907.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 154800.00 | 152770.57 | 152907.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 155135.00 | 153243.45 | 153110.22 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 152610.00 | 153372.25 | 153382.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 152250.00 | 153017.84 | 153211.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 152815.00 | 152693.09 | 152989.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 152815.00 | 152693.09 | 152989.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 152815.00 | 152693.09 | 152989.57 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 153540.00 | 153120.04 | 153114.11 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 152810.00 | 153152.45 | 153170.98 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 153200.00 | 153151.40 | 153149.28 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 152500.00 | 153021.12 | 153090.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 151365.00 | 152040.10 | 152407.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 152235.00 | 151831.53 | 152195.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 152235.00 | 151831.53 | 152195.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 152235.00 | 151831.53 | 152195.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 152235.00 | 151831.53 | 152195.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 152700.00 | 152005.23 | 152241.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 152465.00 | 152005.23 | 152241.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 152660.00 | 152136.18 | 152279.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:15:00 | 152720.00 | 152136.18 | 152279.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 153700.00 | 152448.94 | 152408.51 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 151765.00 | 152425.16 | 152477.65 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 152790.00 | 152555.04 | 152529.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 153200.00 | 152712.83 | 152607.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 152725.00 | 152729.21 | 152634.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 152725.00 | 152729.21 | 152634.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 152725.00 | 152729.21 | 152634.10 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 152400.00 | 152582.92 | 152601.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 152000.00 | 152466.34 | 152547.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 13:15:00 | 152300.00 | 152013.39 | 152203.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 152300.00 | 152013.39 | 152203.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 152300.00 | 152013.39 | 152203.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 152300.00 | 152013.39 | 152203.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 152215.00 | 152053.71 | 152204.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 152010.00 | 152053.71 | 152204.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 152375.00 | 152117.97 | 152220.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 153850.00 | 152117.97 | 152220.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 153995.00 | 152493.37 | 152381.57 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 152900.00 | 153310.58 | 153352.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 152125.00 | 153073.47 | 153240.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 149185.00 | 149166.82 | 150257.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:00:00 | 149185.00 | 149166.82 | 150257.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 150000.00 | 149163.45 | 149829.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 150000.00 | 149163.45 | 149829.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 152500.00 | 149830.76 | 150072.42 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 152895.00 | 150443.61 | 150329.02 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 151160.00 | 151611.95 | 151617.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 151005.00 | 151490.56 | 151561.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 150695.00 | 150518.52 | 150918.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 150695.00 | 150518.52 | 150918.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 149950.00 | 150291.46 | 150738.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 149500.00 | 150272.65 | 150389.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:45:00 | 149370.00 | 150093.12 | 150297.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 142025.00 | 143135.63 | 144437.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 141901.50 | 143135.63 | 144437.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 143195.00 | 143041.56 | 144061.00 | SL hit (close>ema200) qty=0.50 sl=143041.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 143195.00 | 143041.56 | 144061.00 | SL hit (close>ema200) qty=0.50 sl=143041.56 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 133800.00 | 132957.37 | 132911.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 135180.00 | 133401.89 | 133117.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 135300.00 | 135630.01 | 134584.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:15:00 | 139100.00 | 135630.01 | 134584.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 135930.00 | 137182.44 | 136253.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 135930.00 | 137182.44 | 136253.20 | SL hit (close<ema400) qty=1.00 sl=136253.20 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 135960.00 | 137182.44 | 136253.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 135385.00 | 136822.95 | 136174.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 135385.00 | 136822.95 | 136174.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 135585.00 | 136575.36 | 136120.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 135675.00 | 136421.29 | 136092.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 134750.00 | 135742.50 | 135837.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 134750.00 | 135742.50 | 135837.94 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 137175.00 | 136029.00 | 135959.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 143855.00 | 137915.61 | 136863.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 12:15:00 | 151045.00 | 151076.58 | 149122.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:00:00 | 151045.00 | 151076.58 | 149122.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 149170.00 | 150722.33 | 149589.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 149170.00 | 150722.33 | 149589.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 150000.00 | 150577.86 | 149626.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 151030.00 | 150668.29 | 149754.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 147920.00 | 149616.58 | 149687.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 147920.00 | 149616.58 | 149687.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 147105.00 | 148679.57 | 149205.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 147940.00 | 147806.66 | 148398.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 147940.00 | 147806.66 | 148398.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 147860.00 | 147887.46 | 148336.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 147155.00 | 148066.06 | 148271.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 146580.00 | 145419.99 | 145370.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 146580.00 | 145419.99 | 145370.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 147570.00 | 146054.79 | 145680.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 145880.00 | 146203.07 | 145824.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 145880.00 | 146203.07 | 145824.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 145880.00 | 146203.07 | 145824.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 145880.00 | 146203.07 | 145824.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 144695.00 | 145901.45 | 145721.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 144695.00 | 145901.45 | 145721.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 144245.00 | 145570.16 | 145587.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 14:15:00 | 143550.00 | 144693.12 | 145140.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 138990.00 | 136576.90 | 138277.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 138990.00 | 136576.90 | 138277.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 138990.00 | 136576.90 | 138277.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 138990.00 | 136576.90 | 138277.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 138045.00 | 136870.52 | 138256.36 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 138815.00 | 138683.49 | 138679.07 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 134970.00 | 138003.79 | 138375.99 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 137795.00 | 136999.14 | 136968.84 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 135640.00 | 137003.72 | 137004.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 135435.00 | 136371.30 | 136684.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 134200.00 | 133070.20 | 134175.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 134200.00 | 133070.20 | 134175.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 134200.00 | 133070.20 | 134175.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 134200.00 | 133070.20 | 134175.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 133060.00 | 133068.16 | 134073.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 132490.00 | 133871.14 | 133905.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 125865.50 | 128151.61 | 130155.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 125930.00 | 125759.66 | 127647.38 | SL hit (close>ema200) qty=0.50 sl=125759.66 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 130375.00 | 128251.76 | 128057.74 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 125890.00 | 128645.09 | 128669.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 124890.00 | 127894.07 | 128326.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 127400.00 | 127351.68 | 127929.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:00:00 | 127400.00 | 127351.68 | 127929.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 128545.00 | 127590.35 | 127985.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:30:00 | 129725.00 | 127590.35 | 127985.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 128800.00 | 127832.28 | 128059.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 131510.00 | 127832.28 | 128059.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 131105.00 | 128486.82 | 128336.00 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 126320.00 | 128333.96 | 128590.63 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 133010.00 | 128240.09 | 127718.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 134980.00 | 133052.40 | 131152.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 135155.00 | 136002.02 | 134515.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 135155.00 | 136002.02 | 134515.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 135155.00 | 136002.02 | 134515.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 135300.00 | 136002.02 | 134515.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 135270.00 | 135844.62 | 134578.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 136420.00 | 135025.09 | 134619.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 137670.00 | 139106.31 | 139179.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 137670.00 | 139106.31 | 139179.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 137670.00 | 139106.31 | 139179.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 137670.00 | 139106.31 | 139179.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 14:15:00 | 137265.00 | 138191.97 | 138672.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 15:15:00 | 132990.00 | 132525.00 | 133513.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 131700.00 | 132275.00 | 133310.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 133100.00 | 131311.09 | 132147.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 133100.00 | 131311.09 | 132147.76 | SL hit (close>ema400) qty=1.00 sl=132147.76 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 133410.00 | 131311.09 | 132147.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 131495.00 | 131347.87 | 132088.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 130405.00 | 131058.47 | 131764.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 129195.00 | 130907.42 | 131566.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:00:00 | 130645.00 | 130434.53 | 131018.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 130160.00 | 130267.62 | 130889.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 130335.00 | 130238.28 | 130765.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 129750.00 | 130218.50 | 130665.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 131780.00 | 130050.37 | 129806.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 13:15:00 | 130555.00 | 130740.72 | 130272.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 14:00:00 | 130555.00 | 130740.72 | 130272.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 129835.00 | 130559.58 | 130232.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 131135.00 | 130477.66 | 130224.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 130800.00 | 130468.13 | 130243.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 130665.00 | 130503.44 | 130319.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-03 09:30:00 | 139835.00 | 2025-06-05 13:15:00 | 139995.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-06-05 12:30:00 | 140050.00 | 2025-06-05 13:15:00 | 139995.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-06-05 13:00:00 | 140080.00 | 2025-06-05 13:15:00 | 139995.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-06-09 09:15:00 | 141010.00 | 2025-06-09 15:15:00 | 138530.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-11 12:30:00 | 137835.00 | 2025-06-18 11:15:00 | 137295.00 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest1 | 2025-06-20 09:15:00 | 138315.00 | 2025-06-20 09:15:00 | 136415.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-01 09:15:00 | 143230.00 | 2025-07-04 13:15:00 | 143690.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-07-01 13:00:00 | 142915.00 | 2025-07-04 13:15:00 | 143690.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-07-02 09:15:00 | 143215.00 | 2025-07-04 13:15:00 | 143690.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-14 09:15:00 | 149860.00 | 2025-07-18 10:15:00 | 148480.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-14 14:30:00 | 149030.00 | 2025-07-18 10:15:00 | 148480.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-15 09:15:00 | 148995.00 | 2025-07-18 10:15:00 | 148480.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-22 11:30:00 | 148855.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-22 12:45:00 | 149095.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-23 11:00:00 | 148950.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-23 12:45:00 | 148990.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-08-06 11:00:00 | 145725.00 | 2025-08-06 12:15:00 | 146895.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-22 13:30:00 | 147375.00 | 2025-08-25 09:15:00 | 146605.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-25 12:00:00 | 147475.00 | 2025-08-25 12:15:00 | 146955.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-12 13:45:00 | 146675.00 | 2025-09-15 09:15:00 | 148350.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-09-12 15:00:00 | 146280.00 | 2025-09-15 09:15:00 | 148350.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-19 09:15:00 | 149845.00 | 2025-09-19 09:15:00 | 148770.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-01 10:30:00 | 146330.00 | 2025-10-01 14:15:00 | 148310.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-10-01 11:15:00 | 146365.00 | 2025-10-01 14:15:00 | 148310.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-01 11:45:00 | 146430.00 | 2025-10-01 14:15:00 | 148310.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-13 10:15:00 | 156830.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-13 11:15:00 | 156395.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-13 12:15:00 | 156390.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-13 12:45:00 | 156400.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-10-24 09:30:00 | 162850.00 | 2025-10-24 11:15:00 | 160590.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-10-27 09:15:00 | 160620.00 | 2025-10-27 14:15:00 | 159315.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-27 11:00:00 | 160695.00 | 2025-10-27 14:15:00 | 159315.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-27 11:45:00 | 160530.00 | 2025-10-27 14:15:00 | 159315.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-29 12:45:00 | 158600.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-10-30 10:45:00 | 158500.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-10-30 11:30:00 | 158450.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-31 09:45:00 | 158255.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-31 11:15:00 | 157885.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-31 12:15:00 | 158030.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-03 11:30:00 | 158260.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-03 13:15:00 | 158210.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-11-04 11:30:00 | 157950.00 | 2025-11-07 14:15:00 | 158760.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-04 15:00:00 | 157605.00 | 2025-11-07 14:15:00 | 158760.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-06 10:15:00 | 157735.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-06 12:15:00 | 158070.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-11-06 15:00:00 | 156930.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-07 12:45:00 | 157825.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-08 09:15:00 | 149500.00 | 2026-01-19 11:15:00 | 142025.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:45:00 | 149370.00 | 2026-01-19 11:15:00 | 141901.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 149500.00 | 2026-01-19 14:15:00 | 143195.00 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2026-01-08 09:45:00 | 149370.00 | 2026-01-19 14:15:00 | 143195.00 | STOP_HIT | 0.50 | 4.13% |
| BUY | retest1 | 2026-02-04 09:15:00 | 139100.00 | 2026-02-05 09:15:00 | 135930.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-02-05 12:30:00 | 135675.00 | 2026-02-05 15:15:00 | 134750.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-02-13 12:00:00 | 151030.00 | 2026-02-16 11:15:00 | 147920.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-02-19 09:15:00 | 147155.00 | 2026-02-25 12:15:00 | 146580.00 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2026-03-19 09:15:00 | 132490.00 | 2026-03-23 09:15:00 | 125865.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 132490.00 | 2026-03-24 09:15:00 | 125930.00 | STOP_HIT | 0.50 | 4.95% |
| BUY | retest2 | 2026-04-13 10:15:00 | 135300.00 | 2026-04-22 10:15:00 | 137670.00 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2026-04-13 10:45:00 | 135270.00 | 2026-04-22 10:15:00 | 137670.00 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-04-15 09:15:00 | 136420.00 | 2026-04-22 10:15:00 | 137670.00 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest1 | 2026-04-28 09:30:00 | 131700.00 | 2026-04-29 09:15:00 | 133100.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-04-29 13:30:00 | 130405.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2026-04-30 09:15:00 | 129195.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-30 14:00:00 | 130645.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-04-30 14:45:00 | 130160.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-05-04 12:15:00 | 129750.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | -0.42% |

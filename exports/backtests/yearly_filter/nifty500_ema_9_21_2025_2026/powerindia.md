# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 33960.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 47 |
| ALERT2 | 47 |
| ALERT2_SKIP | 27 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 69 |
| PARTIAL | 4 |
| TARGET_HIT | 7 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 47
- **Target hits / Stop hits / Partials:** 6 / 63 / 4
- **Avg / median % per leg:** 0.59% / -0.80%
- **Sum % (uncompounded):** 42.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 13 | 44.8% | 6 | 23 | 0 | 1.72% | 49.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 13 | 44.8% | 6 | 23 | 0 | 1.72% | 49.9% |
| SELL (all) | 44 | 13 | 29.5% | 0 | 40 | 4 | -0.16% | -7.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 13 | 29.5% | 0 | 40 | 4 | -0.16% | -7.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 73 | 26 | 35.6% | 6 | 63 | 4 | 0.59% | 42.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 15580.00 | 15941.39 | 15958.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 15990.00 | 15897.64 | 15890.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 16185.00 | 15999.78 | 15946.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 15980.00 | 16000.34 | 15956.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 15980.00 | 16000.34 | 15956.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 15980.00 | 16000.34 | 15956.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 15980.00 | 16000.34 | 15956.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 15888.00 | 15977.87 | 15950.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 15888.00 | 15977.87 | 15950.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 15814.00 | 15945.10 | 15937.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 15814.00 | 15945.10 | 15937.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 15807.00 | 15917.48 | 15925.99 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 15:15:00 | 15996.00 | 15933.18 | 15932.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 16196.00 | 15985.74 | 15956.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 13:15:00 | 17124.00 | 17235.70 | 16943.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:15:00 | 17042.00 | 17235.70 | 16943.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 16727.00 | 17099.23 | 16951.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 16727.00 | 17099.23 | 16951.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 16637.00 | 17006.78 | 16922.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 16637.00 | 17006.78 | 16922.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 16588.00 | 16826.45 | 16853.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 14:15:00 | 16294.00 | 16719.96 | 16802.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 16831.00 | 16674.97 | 16763.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 16831.00 | 16674.97 | 16763.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 16831.00 | 16674.97 | 16763.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 17077.00 | 16674.97 | 16763.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 16838.00 | 16707.58 | 16770.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:15:00 | 16949.00 | 16707.58 | 16770.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 16973.00 | 16760.66 | 16789.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 17055.00 | 16760.66 | 16789.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 17097.00 | 16850.22 | 16826.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 17311.00 | 16942.38 | 16870.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 19780.00 | 19847.44 | 19656.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 19780.00 | 19847.44 | 19656.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 19780.00 | 19847.44 | 19656.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 19664.00 | 19847.44 | 19656.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 19421.00 | 19762.15 | 19634.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 19421.00 | 19762.15 | 19634.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 19451.00 | 19699.92 | 19618.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 19138.00 | 19699.92 | 19618.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 18551.00 | 19470.14 | 19521.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 18430.00 | 19262.11 | 19421.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 18011.00 | 17935.53 | 18187.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 18011.00 | 17935.53 | 18187.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 18011.00 | 17935.53 | 18187.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 18159.00 | 17935.53 | 18187.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 17446.00 | 17208.99 | 17457.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 17440.00 | 17208.99 | 17457.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 17534.00 | 17273.99 | 17464.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:00:00 | 17534.00 | 17273.99 | 17464.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 17545.00 | 17328.19 | 17471.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:15:00 | 17590.00 | 17328.19 | 17471.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 17732.00 | 17572.75 | 17557.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 18073.00 | 17694.50 | 17628.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 18471.00 | 18593.70 | 18379.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 18471.00 | 18593.70 | 18379.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 18465.00 | 18567.96 | 18387.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 18408.00 | 18567.96 | 18387.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 18390.00 | 18532.37 | 18387.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 18390.00 | 18532.37 | 18387.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 18388.00 | 18503.50 | 18387.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 18201.00 | 18503.50 | 18387.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 18702.00 | 18543.20 | 18416.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 18754.00 | 18590.36 | 18449.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 18812.00 | 18519.89 | 18430.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 18890.00 | 18531.70 | 18455.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 18819.00 | 18664.49 | 18532.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 18662.00 | 18671.15 | 18569.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 18813.00 | 18697.92 | 18590.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 20629.40 | 20052.72 | 19794.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 19480.00 | 19822.36 | 19856.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 19100.00 | 19677.88 | 19787.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 20080.00 | 19659.24 | 19742.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 20080.00 | 19659.24 | 19742.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 20080.00 | 19659.24 | 19742.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 20080.00 | 19659.24 | 19742.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 19995.00 | 19726.39 | 19765.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 19890.00 | 19759.11 | 19776.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 19975.00 | 19813.75 | 19797.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 19975.00 | 19813.75 | 19797.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 15:15:00 | 19980.00 | 19847.00 | 19814.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 19775.00 | 19832.60 | 19810.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 19775.00 | 19832.60 | 19810.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 19775.00 | 19832.60 | 19810.66 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 19700.00 | 19779.41 | 19789.90 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 19970.00 | 19805.50 | 19795.60 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 19600.00 | 19801.33 | 19817.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 19560.00 | 19720.85 | 19775.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 19490.00 | 19333.32 | 19487.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 19490.00 | 19333.32 | 19487.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 19490.00 | 19333.32 | 19487.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 19480.00 | 19333.32 | 19487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 19475.00 | 19361.66 | 19486.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 19520.00 | 19361.66 | 19486.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 19620.00 | 19413.32 | 19498.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 19620.00 | 19413.32 | 19498.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 19655.00 | 19461.66 | 19512.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 19675.00 | 19461.66 | 19512.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 19855.00 | 19575.26 | 19557.45 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 19460.00 | 19555.85 | 19568.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 19050.00 | 19454.68 | 19521.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 09:15:00 | 18490.00 | 18483.68 | 18646.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 18545.00 | 18499.68 | 18613.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 18545.00 | 18499.68 | 18613.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 18595.00 | 18499.68 | 18613.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 18575.00 | 18515.00 | 18592.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 18710.00 | 18515.00 | 18592.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 19040.00 | 18620.00 | 18632.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 19040.00 | 18620.00 | 18632.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 19170.00 | 18730.00 | 18681.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 11:15:00 | 19270.00 | 18838.00 | 18735.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 12:15:00 | 19100.00 | 19198.87 | 19033.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 13:00:00 | 19100.00 | 19198.87 | 19033.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 19050.00 | 19169.10 | 19035.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 19050.00 | 19169.10 | 19035.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 19070.00 | 19149.28 | 19038.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:30:00 | 19135.00 | 19148.54 | 19056.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 19290.00 | 19705.30 | 19743.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 19290.00 | 19705.30 | 19743.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 19130.00 | 19542.99 | 19659.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 19245.00 | 19112.52 | 19301.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 19245.00 | 19112.52 | 19301.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 19245.00 | 19112.52 | 19301.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 19405.00 | 19112.52 | 19301.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 19385.00 | 19167.02 | 19309.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 19385.00 | 19167.02 | 19309.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 19330.00 | 19199.62 | 19310.92 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 19895.00 | 19448.56 | 19406.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 20585.00 | 19732.08 | 19546.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 19825.00 | 20412.62 | 20087.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 19825.00 | 20412.62 | 20087.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 19825.00 | 20412.62 | 20087.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 19660.00 | 20412.62 | 20087.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 19850.00 | 20300.09 | 20065.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:15:00 | 19840.00 | 20300.09 | 20065.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 19990.00 | 20238.07 | 20059.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 20030.00 | 20196.46 | 20056.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 20010.00 | 20167.17 | 20055.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 20185.00 | 20147.73 | 20057.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 20705.00 | 20883.55 | 20885.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 20705.00 | 20883.55 | 20885.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 20420.00 | 20742.27 | 20817.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 20875.00 | 20718.85 | 20790.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 20875.00 | 20718.85 | 20790.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 20875.00 | 20718.85 | 20790.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 20960.00 | 20718.85 | 20790.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 20940.00 | 20763.08 | 20804.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 20940.00 | 20763.08 | 20804.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 20715.00 | 20753.47 | 20796.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 20575.00 | 20747.77 | 20789.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 20305.00 | 20724.62 | 20768.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 20465.00 | 20606.28 | 20682.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:00:00 | 20585.00 | 20601.94 | 20653.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 20770.00 | 20635.55 | 20663.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 20770.00 | 20635.55 | 20663.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 20715.00 | 20651.44 | 20668.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 20880.00 | 20697.15 | 20687.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 20880.00 | 20697.15 | 20687.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 21585.00 | 20895.18 | 20780.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 20855.00 | 21040.87 | 20906.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 20855.00 | 21040.87 | 20906.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 20855.00 | 21040.87 | 20906.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 20855.00 | 21040.87 | 20906.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 20740.00 | 20980.69 | 20891.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:45:00 | 20750.00 | 20980.69 | 20891.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 20660.00 | 20916.55 | 20870.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 20980.00 | 20916.55 | 20870.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 20485.00 | 20889.67 | 20920.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 20485.00 | 20889.67 | 20920.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 20400.00 | 20728.59 | 20838.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 20300.00 | 19919.95 | 20114.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 20300.00 | 19919.95 | 20114.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 20300.00 | 19919.95 | 20114.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 20300.00 | 19919.95 | 20114.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 20170.00 | 19969.96 | 20119.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:30:00 | 20065.00 | 20033.17 | 20124.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 13:15:00 | 20095.00 | 20033.17 | 20124.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 13:45:00 | 20045.00 | 20019.54 | 20110.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 09:45:00 | 20045.00 | 20029.00 | 20091.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 20245.00 | 20072.20 | 20105.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 20245.00 | 20072.20 | 20105.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 20100.00 | 20080.61 | 20103.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 20100.00 | 20080.61 | 20103.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 20065.00 | 20081.63 | 20098.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 20210.00 | 20081.63 | 20098.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 20090.00 | 20083.31 | 20097.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 19770.00 | 20083.31 | 20097.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:15:00 | 19505.00 | 19474.73 | 19584.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:45:00 | 19840.00 | 19499.63 | 19577.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 19950.00 | 19668.22 | 19639.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 13:15:00 | 19950.00 | 19668.22 | 19639.98 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 19500.00 | 19622.06 | 19623.12 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 19815.00 | 19660.65 | 19640.57 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 19415.00 | 19630.85 | 19634.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 19085.00 | 19481.54 | 19563.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 19051.00 | 18980.48 | 19106.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 19051.00 | 18980.48 | 19106.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 19051.00 | 18980.48 | 19106.86 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 19322.00 | 19177.53 | 19163.59 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 18839.00 | 19105.67 | 19137.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 18814.00 | 18955.15 | 19038.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 18901.00 | 18886.16 | 18967.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:45:00 | 18952.00 | 18886.16 | 18967.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 18849.00 | 18865.85 | 18931.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 18709.00 | 18816.75 | 18896.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 18729.00 | 18816.75 | 18896.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 18695.00 | 18802.00 | 18882.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 19394.00 | 18944.85 | 18922.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 19394.00 | 18944.85 | 18922.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 19449.00 | 19115.30 | 19008.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 19648.00 | 19700.25 | 19455.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 19648.00 | 19700.25 | 19455.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 19455.00 | 19651.20 | 19455.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 19455.00 | 19651.20 | 19455.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 19500.00 | 19620.96 | 19459.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 19722.00 | 19620.96 | 19459.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 19641.00 | 19948.52 | 19971.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 19641.00 | 19948.52 | 19971.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 19449.00 | 19792.69 | 19892.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 19356.00 | 19254.94 | 19462.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:45:00 | 19420.00 | 19254.94 | 19462.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 19434.00 | 19290.75 | 19459.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 19463.00 | 19290.75 | 19459.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 19566.00 | 19345.80 | 19469.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 19541.00 | 19345.80 | 19469.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 19479.00 | 19372.44 | 19470.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 19628.00 | 19372.44 | 19470.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 19190.00 | 19335.95 | 19444.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:30:00 | 19550.00 | 19335.95 | 19444.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 19344.00 | 19276.33 | 19384.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 19400.00 | 19276.33 | 19384.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 18948.00 | 19210.66 | 19344.78 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 15:15:00 | 19247.00 | 19134.42 | 19128.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-26 09:15:00 | 19530.00 | 19213.53 | 19165.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 14:15:00 | 19069.00 | 19323.99 | 19255.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 19069.00 | 19323.99 | 19255.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 19069.00 | 19323.99 | 19255.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 19069.00 | 19323.99 | 19255.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 19210.00 | 19301.20 | 19251.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 19637.00 | 19301.20 | 19251.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 18978.00 | 19233.57 | 19239.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 18978.00 | 19233.57 | 19239.13 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 19333.00 | 19253.45 | 19247.66 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 18682.00 | 19134.33 | 19194.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 12:15:00 | 18144.00 | 18820.33 | 19026.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 18225.00 | 18148.41 | 18425.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 18429.00 | 18148.41 | 18425.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 18286.00 | 18175.93 | 18412.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 18113.00 | 18172.64 | 18370.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 18150.00 | 18236.87 | 18282.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:15:00 | 18159.00 | 18263.30 | 18290.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 17207.35 | 17468.55 | 17573.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 17242.50 | 17468.55 | 17573.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 17251.05 | 17468.55 | 17573.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 17460.00 | 17428.92 | 17525.59 | SL hit (close>ema200) qty=0.50 sl=17428.92 alert=retest2 |

### Cycle 34 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 17731.00 | 17523.95 | 17523.93 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 17450.00 | 17541.25 | 17547.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 17410.00 | 17515.00 | 17535.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 17570.00 | 17519.60 | 17533.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 17570.00 | 17519.60 | 17533.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 17570.00 | 17519.60 | 17533.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 17571.00 | 17519.60 | 17533.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 17402.00 | 17496.08 | 17521.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 17317.00 | 17456.62 | 17485.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 17175.00 | 17003.51 | 16984.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 17175.00 | 17003.51 | 16984.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 17321.00 | 17102.93 | 17035.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 17872.00 | 17875.84 | 17617.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 17872.00 | 17875.84 | 17617.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 17762.00 | 17809.00 | 17698.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:15:00 | 17731.00 | 17809.00 | 17698.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 17365.00 | 17720.20 | 17668.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 17365.00 | 17720.20 | 17668.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 17478.00 | 17671.76 | 17651.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:00:00 | 17600.00 | 17657.41 | 17646.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-04 09:15:00 | 19360.00 | 18234.46 | 17917.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 21352.00 | 21488.51 | 21503.59 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 21674.00 | 21520.93 | 21509.08 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 21466.00 | 21509.42 | 21509.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 21149.00 | 21437.34 | 21476.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 21420.00 | 21395.92 | 21444.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 12:15:00 | 21420.00 | 21395.92 | 21444.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 21420.00 | 21395.92 | 21444.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:00:00 | 21420.00 | 21395.92 | 21444.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 21567.00 | 21430.13 | 21455.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 21567.00 | 21430.13 | 21455.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 21393.00 | 21422.71 | 21449.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:30:00 | 21493.00 | 21422.71 | 21449.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 21378.00 | 21413.76 | 21443.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 21301.00 | 21413.76 | 21443.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 21330.00 | 21400.57 | 21432.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 21575.00 | 21435.46 | 21445.40 | SL hit (close>static) qty=1.00 sl=21444.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 21543.00 | 21456.96 | 21454.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 22172.00 | 21657.23 | 21553.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 21956.00 | 22142.48 | 21914.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 21956.00 | 22142.48 | 21914.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 21813.00 | 22076.58 | 21905.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 21813.00 | 22076.58 | 21905.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 21768.00 | 22014.86 | 21892.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 21768.00 | 22014.86 | 21892.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 21565.00 | 21825.89 | 21828.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 21424.00 | 21644.02 | 21733.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 21665.00 | 21590.19 | 21680.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 21665.00 | 21590.19 | 21680.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 21529.00 | 21577.96 | 21667.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 21462.00 | 21577.96 | 21667.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 21778.00 | 21632.02 | 21653.41 | SL hit (close>static) qty=1.00 sl=21724.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 21955.00 | 21720.29 | 21691.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 22169.00 | 21843.75 | 21754.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 21982.00 | 22005.85 | 21878.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:00:00 | 21982.00 | 22005.85 | 21878.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 21918.00 | 21988.28 | 21882.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 22020.00 | 21923.00 | 21889.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 22040.00 | 21978.47 | 21927.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 22005.00 | 21990.54 | 21941.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 21840.00 | 22244.96 | 22246.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 21840.00 | 22244.96 | 22246.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 21695.00 | 22134.97 | 22195.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 19125.00 | 19052.67 | 19716.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 19125.00 | 19052.67 | 19716.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 19645.00 | 19258.64 | 19607.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 19645.00 | 19258.64 | 19607.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 19425.00 | 19291.91 | 19590.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 13:15:00 | 19325.00 | 19291.91 | 19590.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:30:00 | 19400.00 | 19367.62 | 19575.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 19405.00 | 19392.10 | 19567.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 19555.00 | 19300.48 | 19278.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 19555.00 | 19300.48 | 19278.22 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 19245.00 | 19353.89 | 19367.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 19225.00 | 19321.09 | 19350.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 19410.00 | 19338.87 | 19355.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 19410.00 | 19338.87 | 19355.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 19410.00 | 19338.87 | 19355.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 19530.00 | 19338.87 | 19355.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 19395.00 | 19350.10 | 19359.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 19290.00 | 19350.10 | 19359.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 12:15:00 | 18325.50 | 18748.57 | 19016.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 18325.00 | 18272.75 | 18563.04 | SL hit (close>ema200) qty=0.50 sl=18272.75 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 13:15:00 | 18630.00 | 18611.05 | 18609.31 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 18415.00 | 18571.84 | 18591.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 18335.00 | 18495.38 | 18551.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 18350.00 | 18320.93 | 18398.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 18350.00 | 18320.93 | 18398.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 18380.00 | 18332.74 | 18397.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 18380.00 | 18332.74 | 18397.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 18295.00 | 18325.19 | 18387.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 18215.00 | 18304.15 | 18372.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:30:00 | 18210.00 | 18133.93 | 18245.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 18145.00 | 18160.52 | 18238.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:30:00 | 18220.00 | 18207.33 | 18248.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 18340.00 | 18233.86 | 18256.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 18420.00 | 18271.09 | 18271.30 | SL hit (close>static) qty=1.00 sl=18390.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 18326.00 | 18280.96 | 18275.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 18501.00 | 18343.67 | 18308.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 18842.00 | 18903.30 | 18773.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 18842.00 | 18903.30 | 18773.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 18888.00 | 18900.24 | 18783.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 19021.00 | 18848.05 | 18792.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 18460.00 | 19312.19 | 19239.74 | SL hit (close<static) qty=1.00 sl=18772.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 18388.00 | 19127.35 | 19162.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 17810.00 | 18863.88 | 19039.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 17446.00 | 17444.05 | 17887.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 17539.00 | 17444.05 | 17887.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 16903.00 | 16533.68 | 16799.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 16861.00 | 16533.68 | 16799.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 16795.00 | 16585.95 | 16798.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 16724.00 | 16796.87 | 16838.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 16675.00 | 16790.29 | 16831.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 16655.00 | 16772.90 | 16815.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:45:00 | 16715.00 | 16706.70 | 16775.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 16665.00 | 16687.69 | 16754.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 16538.00 | 16688.55 | 16748.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:00:00 | 16555.00 | 16698.48 | 16741.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 16858.00 | 16759.94 | 16755.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 16858.00 | 16759.94 | 16755.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 16926.00 | 16793.15 | 16770.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 16637.00 | 16796.82 | 16784.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 16637.00 | 16796.82 | 16784.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 16637.00 | 16796.82 | 16784.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 16633.00 | 16796.82 | 16784.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 16609.00 | 16759.26 | 16768.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 16484.00 | 16704.20 | 16743.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 16829.00 | 16616.19 | 16671.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 16829.00 | 16616.19 | 16671.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 16829.00 | 16616.19 | 16671.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 16829.00 | 16616.19 | 16671.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 16596.00 | 16612.15 | 16664.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 11:15:00 | 16585.00 | 16612.15 | 16664.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 16587.00 | 16593.37 | 16640.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 16835.00 | 16657.96 | 16663.06 | SL hit (close>static) qty=1.00 sl=16831.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 17116.00 | 16749.57 | 16704.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 17457.00 | 16968.72 | 16816.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 18549.00 | 18649.06 | 18288.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 18549.00 | 18649.06 | 18288.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 18456.00 | 18610.45 | 18303.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 18388.00 | 18610.45 | 18303.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 18012.00 | 18462.37 | 18287.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 18012.00 | 18462.37 | 18287.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 18128.00 | 18395.49 | 18273.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 18466.00 | 18395.49 | 18273.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 20312.60 | 18903.03 | 18609.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 19051.00 | 19085.02 | 19085.40 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 19189.00 | 19105.82 | 19094.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 21560.00 | 19603.72 | 19323.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 22391.00 | 22391.54 | 21883.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 22488.00 | 22391.54 | 21883.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 22252.00 | 22673.99 | 22527.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 22366.00 | 22673.99 | 22527.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 22333.00 | 22605.79 | 22509.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 22480.00 | 22600.43 | 22515.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 22773.00 | 23013.55 | 23014.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 22773.00 | 23013.55 | 23014.36 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 23314.00 | 23073.64 | 23041.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 23620.00 | 23182.91 | 23094.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 25508.00 | 25508.59 | 25259.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:15:00 | 25660.00 | 25508.59 | 25259.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 25780.00 | 25562.87 | 25307.22 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 24510.00 | 25182.06 | 25245.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 24335.00 | 25012.65 | 25163.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 25135.00 | 24764.90 | 24951.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 25135.00 | 24764.90 | 24951.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 25135.00 | 24764.90 | 24951.20 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 25240.00 | 25080.28 | 25060.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 25400.00 | 25144.22 | 25091.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 24945.00 | 25544.33 | 25402.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 24945.00 | 25544.33 | 25402.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 24945.00 | 25544.33 | 25402.22 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 24825.00 | 25246.97 | 25282.21 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 25890.00 | 25283.95 | 25266.65 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 25045.00 | 25291.18 | 25303.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 24940.00 | 25220.94 | 25270.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 25130.00 | 24965.03 | 25095.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 25130.00 | 24965.03 | 25095.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 25130.00 | 24965.03 | 25095.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 25130.00 | 24965.03 | 25095.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 24915.00 | 24955.02 | 25078.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 24915.00 | 24955.02 | 25078.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 24645.00 | 24833.90 | 24966.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 24575.00 | 24833.90 | 24966.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 24770.00 | 24454.32 | 24427.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 24770.00 | 24454.32 | 24427.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 25010.00 | 24598.36 | 24499.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 24565.00 | 24826.65 | 24704.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 24565.00 | 24826.65 | 24704.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 24565.00 | 24826.65 | 24704.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 24810.00 | 24803.32 | 24705.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:15:00 | 24765.00 | 24785.65 | 24706.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:30:00 | 24795.00 | 24772.02 | 24713.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:30:00 | 24765.00 | 24739.61 | 24703.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 24750.00 | 24741.69 | 24708.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 25630.00 | 24741.69 | 24708.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 24245.00 | 24958.83 | 24920.84 | SL hit (close<static) qty=1.00 sl=24335.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 24055.00 | 24778.06 | 24842.13 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 25140.00 | 24732.35 | 24716.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 25680.00 | 25050.79 | 24882.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 25350.00 | 25479.57 | 25236.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 25350.00 | 25479.57 | 25236.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 25350.00 | 25479.57 | 25236.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 25195.00 | 25479.57 | 25236.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 25150.00 | 25413.66 | 25228.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 25150.00 | 25413.66 | 25228.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 24960.00 | 25322.93 | 25204.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 24960.00 | 25322.93 | 25204.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 24920.00 | 25242.34 | 25178.22 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 24710.00 | 25079.50 | 25111.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 24350.00 | 24880.88 | 25012.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 24955.00 | 24550.45 | 24720.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 24955.00 | 24550.45 | 24720.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 24955.00 | 24550.45 | 24720.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 25125.00 | 24550.45 | 24720.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 25520.00 | 24943.27 | 24873.77 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 24250.00 | 24867.53 | 24869.46 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 25365.00 | 24922.03 | 24874.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 25780.00 | 25163.70 | 24996.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 25125.00 | 25155.96 | 25008.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 25125.00 | 25155.96 | 25008.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 25050.00 | 25179.29 | 25060.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:45:00 | 24975.00 | 25179.29 | 25060.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 24980.00 | 25139.43 | 25052.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 24980.00 | 25139.43 | 25052.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 25080.00 | 25127.55 | 25055.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 24850.00 | 25127.55 | 25055.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 24985.00 | 25099.04 | 25049.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:30:00 | 25070.00 | 25054.23 | 25033.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 25155.00 | 25051.38 | 25033.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:15:00 | 25065.00 | 25051.38 | 25033.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:00:00 | 25090.00 | 25059.11 | 25038.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 25055.00 | 25058.29 | 25040.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 25055.00 | 25058.29 | 25040.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 25045.00 | 25055.63 | 25040.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 25045.00 | 25055.63 | 25040.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 24930.00 | 25030.50 | 25030.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 15:15:00 | 24930.00 | 25030.50 | 25030.74 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 25440.00 | 25112.40 | 25067.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 25995.00 | 25462.63 | 25254.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 28305.00 | 28398.31 | 27777.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 28305.00 | 28398.31 | 27777.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 27935.00 | 28375.95 | 28149.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 27925.00 | 28375.95 | 28149.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 27995.00 | 28299.76 | 28135.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:45:00 | 28215.00 | 28129.50 | 28089.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 31036.50 | 30422.90 | 30036.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 15:15:00 | 33425.00 | 33583.77 | 33587.28 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 34415.00 | 33750.01 | 33662.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 34555.00 | 33911.01 | 33743.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 34095.00 | 34400.52 | 34116.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 34095.00 | 34400.52 | 34116.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 34095.00 | 34400.52 | 34116.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 34095.00 | 34400.52 | 34116.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 33595.00 | 34239.42 | 34068.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 33595.00 | 34239.42 | 34068.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 33570.00 | 34105.54 | 34023.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 33570.00 | 34105.54 | 34023.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 33960.00 | 33990.81 | 33982.19 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 10:45:00 | 18754.00 | 2025-06-30 09:15:00 | 20629.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 11:30:00 | 18812.00 | 2025-06-30 09:15:00 | 20693.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 15:15:00 | 18890.00 | 2025-06-30 09:15:00 | 20694.30 | TARGET_HIT | 1.00 | 9.55% |
| BUY | retest2 | 2025-06-23 09:30:00 | 18819.00 | 2025-07-01 12:15:00 | 19480.00 | STOP_HIT | 1.00 | 3.51% |
| BUY | retest2 | 2025-06-23 13:30:00 | 18813.00 | 2025-07-01 12:15:00 | 19480.00 | STOP_HIT | 1.00 | 3.55% |
| SELL | retest2 | 2025-07-02 12:00:00 | 19890.00 | 2025-07-02 14:15:00 | 19975.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-21 09:30:00 | 19135.00 | 2025-07-25 10:15:00 | 19290.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-07-31 13:00:00 | 20030.00 | 2025-08-06 12:15:00 | 20705.00 | STOP_HIT | 1.00 | 3.37% |
| BUY | retest2 | 2025-07-31 13:30:00 | 20010.00 | 2025-08-06 12:15:00 | 20705.00 | STOP_HIT | 1.00 | 3.47% |
| BUY | retest2 | 2025-07-31 14:30:00 | 20185.00 | 2025-08-06 12:15:00 | 20705.00 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2025-08-07 13:15:00 | 20575.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-08-08 09:15:00 | 20305.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-08-08 14:30:00 | 20465.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-11 12:00:00 | 20585.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-08-13 09:15:00 | 20980.00 | 2025-08-14 12:15:00 | 20485.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-08-20 12:30:00 | 20065.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-08-20 13:15:00 | 20095.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2025-08-20 13:45:00 | 20045.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-08-21 09:45:00 | 20045.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-08-22 10:15:00 | 19770.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-26 15:15:00 | 19505.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-08-28 09:45:00 | 19840.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-09-09 11:30:00 | 18709.00 | 2025-09-10 09:15:00 | 19394.00 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2025-09-09 12:00:00 | 18729.00 | 2025-09-10 09:15:00 | 19394.00 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-09-09 13:15:00 | 18695.00 | 2025-09-10 09:15:00 | 19394.00 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-09-12 09:15:00 | 19722.00 | 2025-09-18 10:15:00 | 19641.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-29 09:15:00 | 19637.00 | 2025-09-29 12:15:00 | 18978.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-10-03 12:00:00 | 18113.00 | 2025-10-15 09:15:00 | 17207.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 10:15:00 | 18150.00 | 2025-10-15 09:15:00 | 17242.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 11:15:00 | 18159.00 | 2025-10-15 09:15:00 | 17251.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 12:00:00 | 18113.00 | 2025-10-15 12:15:00 | 17460.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-10-07 10:15:00 | 18150.00 | 2025-10-15 12:15:00 | 17460.00 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-10-07 11:15:00 | 18159.00 | 2025-10-15 12:15:00 | 17460.00 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2025-10-23 12:00:00 | 17317.00 | 2025-10-29 10:15:00 | 17175.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-11-03 14:00:00 | 17600.00 | 2025-11-04 09:15:00 | 19360.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-19 09:15:00 | 21301.00 | 2025-11-19 11:15:00 | 21575.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-19 11:00:00 | 21330.00 | 2025-11-19 11:15:00 | 21575.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-25 09:15:00 | 21462.00 | 2025-11-25 14:15:00 | 21778.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-11-28 09:45:00 | 22020.00 | 2025-12-03 12:15:00 | 21840.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-28 13:15:00 | 22040.00 | 2025-12-03 12:15:00 | 21840.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-28 14:30:00 | 22005.00 | 2025-12-03 12:15:00 | 21840.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-09 13:15:00 | 19325.00 | 2025-12-12 11:15:00 | 19555.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-09 14:30:00 | 19400.00 | 2025-12-12 11:15:00 | 19555.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-10 09:15:00 | 19405.00 | 2025-12-12 11:15:00 | 19555.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-17 11:15:00 | 19290.00 | 2025-12-18 12:15:00 | 18325.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-17 11:15:00 | 19290.00 | 2025-12-19 13:15:00 | 18325.00 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 09:45:00 | 18215.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-12-30 14:30:00 | 18210.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-31 09:45:00 | 18145.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-31 11:30:00 | 18220.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-31 15:15:00 | 18265.00 | 2026-01-01 09:15:00 | 18326.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-01-07 09:15:00 | 19021.00 | 2026-01-08 14:15:00 | 18460.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2026-01-20 09:15:00 | 16724.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-20 10:15:00 | 16675.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-20 12:30:00 | 16655.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-20 14:45:00 | 16715.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-21 10:30:00 | 16538.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-21 14:00:00 | 16555.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-27 11:15:00 | 16585.00 | 2026-01-27 15:15:00 | 16835.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-27 14:15:00 | 16587.00 | 2026-01-27 15:15:00 | 16835.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-02 09:15:00 | 18466.00 | 2026-02-03 09:15:00 | 20312.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 11:45:00 | 22480.00 | 2026-02-19 15:15:00 | 22773.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2026-03-13 10:15:00 | 24575.00 | 2026-03-17 14:15:00 | 24770.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-03-19 10:30:00 | 24810.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-19 12:15:00 | 24765.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-19 13:30:00 | 24795.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-19 14:30:00 | 24765.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-20 09:15:00 | 25630.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest2 | 2026-04-07 10:30:00 | 25070.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-04-07 11:45:00 | 25155.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-04-07 12:15:00 | 25065.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-04-07 13:00:00 | 25090.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-16 14:45:00 | 28215.00 | 2026-04-23 09:15:00 | 31036.50 | TARGET_HIT | 1.00 | 10.00% |

# Maruti Suzuki India Ltd. (MARUTI)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 13733.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 50 |
| ALERT2 | 51 |
| ALERT2_SKIP | 27 |
| ALERT3 | 140 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 62 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 20 / 45
- **Target hits / Stop hits / Partials:** 3 / 58 / 4
- **Avg / median % per leg:** 0.46% / -0.64%
- **Sum % (uncompounded):** 29.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 11 | 34.4% | 2 | 30 | 0 | 0.68% | 21.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.18% | -1.2% |
| BUY @ 3rd Alert (retest2) | 31 | 11 | 35.5% | 2 | 29 | 0 | 0.74% | 22.9% |
| SELL (all) | 33 | 9 | 27.3% | 1 | 28 | 4 | 0.24% | 8.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 9 | 27.3% | 1 | 28 | 4 | 0.24% | 8.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.18% | -1.2% |
| retest2 (combined) | 64 | 20 | 31.2% | 3 | 57 | 4 | 0.48% | 30.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 12534.00 | 12407.49 | 12402.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 12545.00 | 12434.99 | 12415.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 12529.00 | 12533.54 | 12479.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 12543.00 | 12533.54 | 12479.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 12453.00 | 12518.46 | 12482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 12441.00 | 12518.46 | 12482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 12443.00 | 12503.37 | 12478.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:45:00 | 12506.00 | 12499.10 | 12479.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 12500.00 | 12492.28 | 12477.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 12700.00 | 12845.83 | 12861.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 12700.00 | 12845.83 | 12861.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 12640.00 | 12804.66 | 12841.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 14:15:00 | 12416.00 | 12387.47 | 12432.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 12416.00 | 12387.47 | 12432.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 12391.00 | 12357.08 | 12386.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 12391.00 | 12357.08 | 12386.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 12361.00 | 12357.86 | 12383.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 12338.00 | 12357.86 | 12383.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 12339.00 | 12349.31 | 12374.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 12427.00 | 12370.17 | 12378.14 | SL hit (close>static) qty=1.00 sl=12415.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 12438.00 | 12383.73 | 12383.58 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 12337.00 | 12374.39 | 12379.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 12280.00 | 12355.51 | 12370.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 12279.00 | 12272.41 | 12304.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 12279.00 | 12272.41 | 12304.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 12279.00 | 12272.41 | 12304.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 12279.00 | 12272.41 | 12304.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 12273.00 | 12272.53 | 12301.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 12217.00 | 12272.53 | 12301.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 12217.00 | 12261.42 | 12294.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 12154.00 | 12235.34 | 12279.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 12164.00 | 12176.74 | 12233.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 12157.00 | 12172.20 | 12221.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 12169.00 | 12177.57 | 12215.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 12212.00 | 12186.92 | 12213.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 12212.00 | 12186.92 | 12213.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 12167.00 | 12182.94 | 12209.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:30:00 | 12189.00 | 12182.94 | 12209.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 12062.00 | 12158.60 | 12193.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 12193.00 | 12158.60 | 12193.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 12230.00 | 12148.54 | 12165.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 12230.00 | 12148.54 | 12165.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 12471.00 | 12213.03 | 12193.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 12471.00 | 12213.03 | 12193.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 12515.00 | 12273.43 | 12222.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 12570.00 | 12572.10 | 12469.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 12570.00 | 12572.10 | 12469.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 12519.00 | 12570.35 | 12502.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 12519.00 | 12570.35 | 12502.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 12472.00 | 12550.68 | 12499.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 12541.00 | 12550.68 | 12499.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 12487.00 | 12537.94 | 12498.30 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 12424.00 | 12474.90 | 12480.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 12383.00 | 12439.91 | 12461.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 12:15:00 | 12386.00 | 12371.18 | 12407.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 12:15:00 | 12386.00 | 12371.18 | 12407.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 12386.00 | 12371.18 | 12407.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:45:00 | 12398.00 | 12371.18 | 12407.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 12390.00 | 12374.94 | 12406.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 12386.00 | 12374.94 | 12406.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 12416.00 | 12383.15 | 12406.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 12416.00 | 12383.15 | 12406.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 12405.00 | 12387.52 | 12406.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 12391.00 | 12387.52 | 12406.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 12435.00 | 12397.02 | 12409.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 12435.00 | 12397.02 | 12409.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 12518.00 | 12421.21 | 12419.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 11:15:00 | 12545.00 | 12445.97 | 12430.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 12782.00 | 12783.12 | 12716.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 12849.00 | 12783.12 | 12716.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 12803.00 | 12779.48 | 12735.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 12749.00 | 12779.48 | 12735.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 12698.00 | 12766.30 | 12743.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 12698.00 | 12766.30 | 12743.91 | SL hit (close<ema400) qty=1.00 sl=12743.91 alert=retest1 |

### Cycle 8 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 12706.00 | 12727.63 | 12729.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 12802.00 | 12736.96 | 12733.39 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 12636.00 | 12718.82 | 12727.21 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 12787.00 | 12726.28 | 12720.76 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 09:15:00 | 12635.00 | 12716.13 | 12724.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 12579.00 | 12661.42 | 12690.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 12435.00 | 12433.57 | 12518.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:30:00 | 12435.00 | 12433.57 | 12518.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 12484.00 | 12443.76 | 12496.34 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 12631.00 | 12535.43 | 12525.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 12743.00 | 12589.67 | 12552.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 12638.00 | 12699.17 | 12642.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 12638.00 | 12699.17 | 12642.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 12638.00 | 12699.17 | 12642.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 12638.00 | 12699.17 | 12642.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 12631.00 | 12685.54 | 12641.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 12638.00 | 12685.54 | 12641.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 12637.00 | 12675.83 | 12640.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:00:00 | 12656.00 | 12671.86 | 12642.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 12650.00 | 12667.49 | 12642.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 12508.00 | 12628.57 | 12630.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 12508.00 | 12628.57 | 12630.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 12497.00 | 12571.05 | 12601.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 12485.00 | 12450.64 | 12496.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 12485.00 | 12450.64 | 12496.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 12485.00 | 12450.64 | 12496.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 12485.00 | 12450.64 | 12496.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 12515.00 | 12468.21 | 12497.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 12519.00 | 12468.21 | 12497.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 12471.00 | 12468.77 | 12494.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 12459.00 | 12470.02 | 12492.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:15:00 | 12453.00 | 12470.01 | 12490.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 12589.00 | 12491.09 | 12496.67 | SL hit (close>static) qty=1.00 sl=12518.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 12570.00 | 12506.87 | 12503.33 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 12462.00 | 12541.73 | 12548.11 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 12569.00 | 12533.42 | 12530.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 12580.00 | 12542.74 | 12535.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 12540.00 | 12548.47 | 12540.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 12540.00 | 12548.47 | 12540.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 12540.00 | 12548.47 | 12540.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 12540.00 | 12548.47 | 12540.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 12501.00 | 12538.98 | 12536.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 12501.00 | 12538.98 | 12536.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 12476.00 | 12526.38 | 12531.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 12466.00 | 12508.49 | 12521.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 12423.00 | 12402.31 | 12427.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:00:00 | 12423.00 | 12402.31 | 12427.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 12476.00 | 12417.05 | 12431.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 12510.00 | 12417.05 | 12431.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 12498.00 | 12433.24 | 12437.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 12496.00 | 12433.24 | 12437.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 12488.00 | 12444.19 | 12442.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 12661.00 | 12497.12 | 12467.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 12594.00 | 12603.12 | 12552.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:00:00 | 12594.00 | 12603.12 | 12552.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 12545.00 | 12591.50 | 12551.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 12545.00 | 12591.50 | 12551.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 12559.00 | 12585.00 | 12552.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 12559.00 | 12585.00 | 12552.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 12553.00 | 12578.60 | 12552.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 12546.00 | 12578.60 | 12552.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 12558.00 | 12574.48 | 12552.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 12545.00 | 12574.48 | 12552.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 12565.00 | 12572.58 | 12554.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 12485.00 | 12572.58 | 12554.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 12450.00 | 12548.07 | 12544.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 12452.00 | 12548.07 | 12544.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 12405.00 | 12519.45 | 12531.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 12333.00 | 12418.83 | 12460.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 12368.00 | 12344.81 | 12393.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 12368.00 | 12344.81 | 12393.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 12443.00 | 12364.45 | 12398.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 12443.00 | 12364.45 | 12398.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 12455.00 | 12382.56 | 12403.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 12446.00 | 12382.56 | 12403.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 12527.00 | 12428.32 | 12421.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 12593.00 | 12461.25 | 12437.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 12479.00 | 12539.17 | 12496.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 12479.00 | 12539.17 | 12496.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 12479.00 | 12539.17 | 12496.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 12479.00 | 12539.17 | 12496.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 12525.00 | 12536.34 | 12499.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 12574.00 | 12536.34 | 12499.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:45:00 | 12560.00 | 12544.94 | 12509.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 12573.00 | 12550.55 | 12515.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 12706.00 | 12543.27 | 12518.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 12504.00 | 12535.42 | 12516.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 12504.00 | 12535.42 | 12516.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 12442.00 | 12516.73 | 12510.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 12442.00 | 12516.73 | 12510.10 | SL hit (close<static) qty=1.00 sl=12464.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 12431.00 | 12499.59 | 12502.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 12370.00 | 12458.06 | 12482.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 12360.00 | 12348.81 | 12400.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 12360.00 | 12348.81 | 12400.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 12413.00 | 12364.39 | 12394.79 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 12556.00 | 12440.20 | 12424.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 12581.00 | 12507.11 | 12464.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 12:15:00 | 12506.00 | 12518.02 | 12481.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:00:00 | 12506.00 | 12518.02 | 12481.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 12488.00 | 12509.91 | 12488.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 12488.00 | 12509.91 | 12488.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 12501.00 | 12508.13 | 12489.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 12466.00 | 12508.13 | 12489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 12479.00 | 12502.30 | 12488.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:45:00 | 12475.00 | 12502.30 | 12488.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 12473.00 | 12496.44 | 12487.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:45:00 | 12473.00 | 12496.44 | 12487.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 12488.00 | 12494.75 | 12487.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 12456.00 | 12494.75 | 12487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 12632.00 | 12522.20 | 12500.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 12672.00 | 12582.21 | 12539.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 12695.00 | 12636.86 | 12591.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 13939.20 | 13075.12 | 12924.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 14772.00 | 14821.59 | 14826.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 14645.00 | 14786.28 | 14809.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 14840.00 | 14780.02 | 14801.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 14840.00 | 14780.02 | 14801.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 14840.00 | 14780.02 | 14801.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 14860.00 | 14780.02 | 14801.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 14863.00 | 14796.61 | 14807.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 14863.00 | 14796.61 | 14807.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 14777.00 | 14788.43 | 14801.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 14786.00 | 14788.43 | 14801.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 14808.00 | 14792.35 | 14802.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 14808.00 | 14792.35 | 14802.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 14903.00 | 14814.48 | 14811.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 15056.00 | 14876.95 | 14841.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 15282.00 | 15287.55 | 15171.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 15190.00 | 15264.91 | 15181.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 15190.00 | 15264.91 | 15181.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 15190.00 | 15264.91 | 15181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 15183.00 | 15248.53 | 15181.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:45:00 | 15185.00 | 15248.53 | 15181.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 15179.00 | 15234.63 | 15181.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:15:00 | 15104.00 | 15234.63 | 15181.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 15099.00 | 15207.50 | 15173.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 15087.00 | 15207.50 | 15173.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 15134.00 | 15192.80 | 15169.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 15060.00 | 15166.24 | 15159.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 15064.00 | 15145.79 | 15151.25 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 15311.00 | 15162.38 | 15154.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 15325.00 | 15194.91 | 15169.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 11:15:00 | 15280.00 | 15289.57 | 15246.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 12:00:00 | 15280.00 | 15289.57 | 15246.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 15269.00 | 15286.38 | 15255.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 15269.00 | 15286.38 | 15255.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 15251.00 | 15279.31 | 15255.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 15310.00 | 15279.31 | 15255.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 16065.00 | 16198.27 | 16208.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 16065.00 | 16198.27 | 16208.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 15973.00 | 16138.29 | 16177.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 16035.00 | 16029.74 | 16083.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 16059.00 | 16029.74 | 16083.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 15930.00 | 16009.79 | 16069.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 15877.00 | 15969.62 | 16039.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 15849.00 | 15952.67 | 16007.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:45:00 | 15817.00 | 15923.34 | 15988.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 16042.00 | 15946.71 | 15942.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 16042.00 | 15946.71 | 15942.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 16128.00 | 15998.93 | 15968.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 16079.00 | 16115.74 | 16072.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 16079.00 | 16115.74 | 16072.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 16079.00 | 16115.74 | 16072.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 16079.00 | 16115.74 | 16072.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 16005.00 | 16093.59 | 16066.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 16027.00 | 16093.59 | 16066.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 16014.00 | 16077.67 | 16061.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 15976.00 | 16077.67 | 16061.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 15906.00 | 16043.34 | 16047.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 15890.00 | 16002.22 | 16027.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 15974.00 | 15962.30 | 15999.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 15974.00 | 15962.30 | 15999.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 15948.00 | 15959.44 | 15995.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 16040.00 | 15972.35 | 15997.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 16116.00 | 16001.08 | 16008.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 16099.00 | 16001.08 | 16008.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 16179.00 | 16036.67 | 16023.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 16240.00 | 16077.33 | 16043.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 16170.00 | 16262.01 | 16202.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 16170.00 | 16262.01 | 16202.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 16170.00 | 16262.01 | 16202.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 16170.00 | 16262.01 | 16202.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 16163.00 | 16242.21 | 16199.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:15:00 | 16247.00 | 16215.44 | 16195.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 16330.00 | 16221.32 | 16202.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 16322.00 | 16248.33 | 16232.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 16310.00 | 16368.78 | 16373.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 16310.00 | 16368.78 | 16373.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 16282.00 | 16351.43 | 16364.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 16363.00 | 16308.21 | 16332.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 16363.00 | 16308.21 | 16332.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 16363.00 | 16308.21 | 16332.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 16363.00 | 16308.21 | 16332.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 16330.00 | 16312.57 | 16332.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:00:00 | 16309.00 | 16320.08 | 16332.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 16320.00 | 16320.07 | 16331.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 16400.00 | 16336.05 | 16337.80 | SL hit (close>static) qty=1.00 sl=16388.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 16382.00 | 16345.24 | 16341.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 16423.00 | 16360.79 | 16349.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 16355.00 | 16365.27 | 16353.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 16355.00 | 16365.27 | 16353.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 16355.00 | 16365.27 | 16353.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 16355.00 | 16365.27 | 16353.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 16344.00 | 16361.01 | 16352.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:15:00 | 16332.00 | 16361.01 | 16352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 16296.00 | 16348.01 | 16347.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 16296.00 | 16348.01 | 16347.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 16316.00 | 16341.61 | 16344.70 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 16416.00 | 16356.86 | 16350.59 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 16104.00 | 16305.56 | 16329.09 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 16365.00 | 16290.59 | 16283.14 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 16155.00 | 16256.60 | 16270.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 15664.00 | 16138.08 | 16215.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 15448.00 | 15440.18 | 15592.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:30:00 | 15459.00 | 15440.18 | 15592.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 15490.00 | 15439.99 | 15498.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 15441.00 | 15439.99 | 15498.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 15477.00 | 15447.39 | 15496.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:15:00 | 15511.00 | 15447.39 | 15496.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 15488.00 | 15455.52 | 15495.56 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 15608.00 | 15522.98 | 15519.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 15644.00 | 15585.20 | 15556.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 15580.00 | 15601.96 | 15573.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 15580.00 | 15601.96 | 15573.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 15620.00 | 15605.57 | 15577.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 15581.00 | 15605.57 | 15577.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 15617.00 | 15706.51 | 15674.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 15593.00 | 15706.51 | 15674.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 15629.00 | 15691.01 | 15670.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 15606.00 | 15691.01 | 15670.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 15691.00 | 15669.64 | 15664.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 15616.00 | 15669.64 | 15664.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 15708.00 | 15677.31 | 15668.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 15733.00 | 15677.31 | 15668.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 15751.00 | 15692.05 | 15675.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 15801.00 | 15731.19 | 15696.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 15773.00 | 15833.59 | 15820.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 15776.00 | 15811.90 | 15812.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 15776.00 | 15811.90 | 15812.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 15740.00 | 15797.52 | 15805.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 15762.00 | 15740.31 | 15769.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 13:15:00 | 15762.00 | 15740.31 | 15769.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 15762.00 | 15740.31 | 15769.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 15762.00 | 15740.31 | 15769.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 15809.00 | 15754.05 | 15772.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 15809.00 | 15754.05 | 15772.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 15822.00 | 15767.64 | 15777.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 15813.00 | 15767.64 | 15777.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 09:15:00 | 15889.00 | 15791.91 | 15787.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 16000.00 | 15851.30 | 15816.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 15945.00 | 15985.04 | 15936.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 15945.00 | 15985.04 | 15936.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 15950.00 | 15978.04 | 15937.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 15999.00 | 15978.04 | 15937.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:00:00 | 15978.00 | 15985.38 | 15948.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 15897.00 | 15970.31 | 15953.81 | SL hit (close<static) qty=1.00 sl=15911.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 15904.00 | 16008.77 | 16016.92 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 16103.00 | 15978.68 | 15974.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 16199.00 | 16070.80 | 16021.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 16099.00 | 16156.03 | 16094.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 16099.00 | 16156.03 | 16094.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 16099.00 | 16156.03 | 16094.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 16113.00 | 16156.03 | 16094.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 16038.00 | 16132.42 | 16089.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 16038.00 | 16132.42 | 16089.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 16041.00 | 16114.14 | 16085.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 16090.00 | 16102.25 | 16084.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 16087.00 | 16099.20 | 16084.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 16087.00 | 16094.36 | 16083.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 15987.00 | 16072.89 | 16074.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 15987.00 | 16072.89 | 16074.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 15980.00 | 16054.31 | 16066.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 16119.00 | 16022.32 | 16036.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 16119.00 | 16022.32 | 16036.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 16119.00 | 16022.32 | 16036.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 16075.00 | 16022.32 | 16036.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 16196.00 | 16057.05 | 16051.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 16279.00 | 16156.26 | 16105.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 16183.00 | 16184.60 | 16128.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 16183.00 | 16184.60 | 16128.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 16181.00 | 16178.24 | 16142.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 16117.00 | 16178.24 | 16142.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 16058.00 | 16155.27 | 16141.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 16005.00 | 16155.27 | 16141.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 16086.00 | 16141.42 | 16136.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 16069.00 | 16141.42 | 16136.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 16136.00 | 16142.87 | 16138.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 16124.00 | 16142.87 | 16138.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 16026.00 | 16119.50 | 16128.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 15:15:00 | 15995.00 | 16094.60 | 16116.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 12:15:00 | 16098.00 | 16081.46 | 16101.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 12:15:00 | 16098.00 | 16081.46 | 16101.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 16098.00 | 16081.46 | 16101.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 16033.00 | 16067.57 | 16093.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 16176.00 | 16072.39 | 16087.77 | SL hit (close>static) qty=1.00 sl=16110.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 16206.00 | 16099.11 | 16098.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 16260.00 | 16131.29 | 16113.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 16438.00 | 16442.20 | 16343.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:00:00 | 16438.00 | 16442.20 | 16343.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 16400.00 | 16424.62 | 16371.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 16398.00 | 16424.62 | 16371.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 16357.00 | 16411.10 | 16370.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 16357.00 | 16411.10 | 16370.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 16392.00 | 16407.28 | 16372.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 16454.00 | 16378.30 | 16371.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 16251.00 | 16365.88 | 16373.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 16251.00 | 16365.88 | 16373.91 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 16448.00 | 16378.75 | 16371.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 16518.00 | 16427.47 | 16400.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 16586.00 | 16586.28 | 16530.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 16610.00 | 16586.28 | 16530.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 16592.00 | 16647.66 | 16608.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 16592.00 | 16647.66 | 16608.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 16578.00 | 16633.73 | 16605.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 16616.00 | 16630.18 | 16606.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 16613.00 | 16620.32 | 16606.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 16664.00 | 16614.25 | 16604.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 16530.00 | 16594.00 | 16596.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 16530.00 | 16594.00 | 16596.81 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 16686.00 | 16604.48 | 16597.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 16976.00 | 16766.79 | 16714.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 17152.00 | 17208.62 | 17121.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 17152.00 | 17208.62 | 17121.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 17152.00 | 17208.62 | 17121.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:15:00 | 17069.00 | 17208.62 | 17121.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 17030.00 | 17172.90 | 17113.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 17030.00 | 17172.90 | 17113.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 17024.00 | 17143.12 | 17105.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 17024.00 | 17143.12 | 17105.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 16787.00 | 17071.90 | 17076.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 13:15:00 | 16668.00 | 16991.12 | 17039.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 16542.00 | 16464.52 | 16585.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 16542.00 | 16464.52 | 16585.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 16557.00 | 16492.37 | 16577.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 16614.00 | 16492.37 | 16577.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 16605.00 | 16514.90 | 16580.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 16611.00 | 16514.90 | 16580.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 16560.00 | 16523.92 | 16578.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 16499.00 | 16515.74 | 16569.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:15:00 | 15674.05 | 15736.37 | 15816.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-28 09:15:00 | 14849.10 | 15200.74 | 15396.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 14870.00 | 14508.38 | 14479.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 14978.00 | 14731.91 | 14612.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 14948.00 | 14971.08 | 14822.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 14920.00 | 14971.08 | 14822.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 14989.00 | 15024.46 | 14928.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 15089.00 | 15034.37 | 14950.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 15136.00 | 15025.55 | 14990.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 15218.00 | 15278.09 | 15282.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 15218.00 | 15278.09 | 15282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 15070.00 | 15236.47 | 15262.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 15157.00 | 15113.47 | 15168.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 15157.00 | 15113.47 | 15168.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 15157.00 | 15113.47 | 15168.04 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 15180.00 | 15162.33 | 15161.90 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 15101.00 | 15150.06 | 15156.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 14905.00 | 15092.23 | 15127.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 15046.00 | 15020.19 | 15065.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 15046.00 | 15020.19 | 15065.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 15046.00 | 15020.19 | 15065.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 15060.00 | 15020.19 | 15065.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 15051.00 | 15020.44 | 15053.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 15053.00 | 15020.44 | 15053.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 15053.00 | 15026.95 | 15053.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 15020.00 | 15032.29 | 15051.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 15025.00 | 15028.23 | 15048.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 15013.00 | 15039.91 | 15050.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:00:00 | 14989.00 | 14962.55 | 14994.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 15044.00 | 14978.84 | 14998.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 15044.00 | 14978.84 | 14998.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 15127.00 | 15008.47 | 15010.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 15127.00 | 15008.47 | 15010.32 | SL hit (close>static) qty=1.00 sl=15080.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 15028.00 | 15012.38 | 15011.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 15158.00 | 15066.29 | 15038.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 15070.00 | 15076.27 | 15048.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 15070.00 | 15076.27 | 15048.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 15070.00 | 15076.27 | 15048.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 15061.00 | 15076.27 | 15048.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 14984.00 | 15112.58 | 15083.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:15:00 | 14912.00 | 15112.58 | 15083.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 14970.00 | 15084.06 | 15073.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 14918.00 | 15084.06 | 15073.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 14929.00 | 15053.05 | 15060.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 14865.00 | 14998.08 | 15031.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 14165.00 | 14153.62 | 14408.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:45:00 | 14168.00 | 14153.62 | 14408.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 14347.00 | 14222.99 | 14320.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 14469.00 | 14222.99 | 14320.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 14471.00 | 14272.59 | 14334.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 14318.00 | 14272.59 | 14334.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 14320.00 | 14316.49 | 14340.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 14326.00 | 14318.39 | 14339.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 13602.10 | 14104.18 | 14231.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 13604.00 | 14104.18 | 14231.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 13609.70 | 14104.18 | 14231.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 13694.00 | 13644.52 | 13857.43 | SL hit (close>ema200) qty=0.50 sl=13644.52 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 13105.00 | 12934.07 | 12911.97 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 12643.00 | 12910.99 | 12934.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 12612.00 | 12851.19 | 12905.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 12420.00 | 12406.13 | 12510.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 12444.00 | 12406.13 | 12510.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 12570.00 | 12438.90 | 12515.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 12594.00 | 12438.90 | 12515.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 12543.00 | 12459.72 | 12518.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:15:00 | 12510.00 | 12459.72 | 12518.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 12720.00 | 12514.99 | 12528.73 | SL hit (close>static) qty=1.00 sl=12614.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 12779.00 | 12567.80 | 12551.48 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 12433.00 | 12581.08 | 12581.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 12385.00 | 12541.87 | 12563.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 12655.00 | 12410.15 | 12442.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 12655.00 | 12410.15 | 12442.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 12655.00 | 12410.15 | 12442.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 12593.00 | 12410.15 | 12442.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 12526.00 | 12462.11 | 12460.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 12526.00 | 12462.11 | 12460.57 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 13:15:00 | 12434.00 | 12456.49 | 12458.16 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 12511.00 | 12467.39 | 12462.96 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 12397.00 | 12458.53 | 12460.03 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 12632.00 | 12454.37 | 12451.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 12659.00 | 12535.47 | 12499.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 12582.00 | 12609.94 | 12552.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 12582.00 | 12609.94 | 12552.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 12582.00 | 12609.94 | 12552.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 12681.00 | 12617.32 | 12566.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:45:00 | 12655.00 | 12632.26 | 12577.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 13174.00 | 13427.57 | 13436.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 13174.00 | 13427.57 | 13436.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 13076.00 | 13357.25 | 13403.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 13342.00 | 13241.74 | 13322.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 13342.00 | 13241.74 | 13322.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 13342.00 | 13241.74 | 13322.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:15:00 | 13427.00 | 13241.74 | 13322.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 13378.00 | 13268.99 | 13327.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:45:00 | 13461.00 | 13268.99 | 13327.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 13311.00 | 13289.23 | 13327.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 13278.00 | 13289.23 | 13327.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:00:00 | 13279.00 | 13287.19 | 13322.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 13287.00 | 13287.15 | 13319.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 13248.00 | 13291.78 | 13315.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 13340.00 | 13301.42 | 13318.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 13340.00 | 13301.42 | 13318.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 13325.00 | 13306.14 | 13318.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 13349.00 | 13306.14 | 13318.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 13310.00 | 13306.91 | 13317.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 13369.00 | 13306.91 | 13317.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 13312.00 | 13307.93 | 13317.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 13332.00 | 13307.93 | 13317.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 13330.00 | 13312.34 | 13318.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:45:00 | 13337.00 | 13312.34 | 13318.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 13336.00 | 13317.07 | 13320.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 13478.00 | 13317.07 | 13320.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 13611.00 | 13375.86 | 13346.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 13611.00 | 13375.86 | 13346.58 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 13343.00 | 13434.87 | 13442.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 13276.00 | 13374.40 | 13407.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 13109.00 | 13085.67 | 13170.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 13109.00 | 13085.67 | 13170.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 13109.00 | 13085.67 | 13170.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:30:00 | 13055.00 | 13090.94 | 13164.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 13235.00 | 13131.60 | 13171.15 | SL hit (close>static) qty=1.00 sl=13189.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 13256.00 | 13195.81 | 13192.91 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 13136.00 | 13189.64 | 13190.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 13122.00 | 13176.11 | 13184.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 13489.00 | 13157.81 | 13166.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 13489.00 | 13157.81 | 13166.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 13489.00 | 13157.81 | 13166.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 13497.00 | 13157.81 | 13166.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 13478.00 | 13221.85 | 13194.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 13510.00 | 13279.48 | 13223.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 13280.00 | 13305.94 | 13252.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:45:00 | 13261.00 | 13305.94 | 13252.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 13265.00 | 13297.75 | 13253.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 13044.00 | 13297.75 | 13253.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 13015.00 | 13241.20 | 13231.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 13014.00 | 13241.20 | 13231.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 13040.00 | 13200.96 | 13214.25 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 13320.00 | 13230.32 | 13218.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 13748.00 | 13333.86 | 13267.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 13448.00 | 13494.88 | 13403.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 13448.00 | 13494.88 | 13403.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 13448.00 | 13494.88 | 13403.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 13584.00 | 13487.95 | 13440.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 13600.00 | 13480.94 | 13451.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 14:45:00 | 12506.00 | 2025-05-20 13:15:00 | 12700.00 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-05-14 09:15:00 | 12500.00 | 2025-05-20 13:15:00 | 12700.00 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2025-05-29 11:15:00 | 12338.00 | 2025-05-29 15:15:00 | 12427.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-05-29 13:15:00 | 12339.00 | 2025-05-29 15:15:00 | 12427.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-03 10:45:00 | 12154.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-06-03 14:45:00 | 12164.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-04 09:30:00 | 12157.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-06-04 11:45:00 | 12169.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2025-06-20 09:15:00 | 12849.00 | 2025-06-23 09:15:00 | 12698.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-04 13:00:00 | 12656.00 | 2025-07-07 09:15:00 | 12508.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-04 14:00:00 | 12650.00 | 2025-07-07 09:15:00 | 12508.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-09 13:30:00 | 12459.00 | 2025-07-10 09:15:00 | 12589.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-09 15:15:00 | 12453.00 | 2025-07-10 09:15:00 | 12589.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-31 11:15:00 | 12574.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-31 12:45:00 | 12560.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-31 14:00:00 | 12573.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-08-01 09:15:00 | 12706.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-08 11:45:00 | 12672.00 | 2025-08-18 09:15:00 | 13939.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:30:00 | 12695.00 | 2025-08-18 09:15:00 | 13964.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-16 09:15:00 | 15310.00 | 2025-09-29 11:15:00 | 16065.00 | STOP_HIT | 1.00 | 4.93% |
| SELL | retest2 | 2025-10-01 11:45:00 | 15877.00 | 2025-10-06 13:15:00 | 16042.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-03 09:15:00 | 15849.00 | 2025-10-06 13:15:00 | 16042.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-03 09:45:00 | 15817.00 | 2025-10-06 13:15:00 | 16042.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-14 14:15:00 | 16247.00 | 2025-10-24 10:15:00 | 16310.00 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2025-10-15 09:15:00 | 16330.00 | 2025-10-24 10:15:00 | 16310.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-16 09:15:00 | 16322.00 | 2025-10-24 10:15:00 | 16310.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-27 13:00:00 | 16309.00 | 2025-10-27 14:15:00 | 16400.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-27 14:00:00 | 16320.00 | 2025-10-27 14:15:00 | 16400.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-11-17 11:30:00 | 15801.00 | 2025-11-19 14:15:00 | 15776.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-11-19 12:30:00 | 15773.00 | 2025-11-19 14:15:00 | 15776.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-11-25 09:15:00 | 15999.00 | 2025-11-25 14:15:00 | 15897.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-11-25 11:00:00 | 15978.00 | 2025-11-25 14:15:00 | 15897.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-26 09:15:00 | 16020.00 | 2025-11-27 12:15:00 | 15901.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-27 13:45:00 | 15970.00 | 2025-11-27 14:15:00 | 15904.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-03 14:00:00 | 16090.00 | 2025-12-04 09:15:00 | 15987.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-03 15:00:00 | 16087.00 | 2025-12-04 09:15:00 | 15987.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-04 09:15:00 | 16087.00 | 2025-12-04 09:15:00 | 15987.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-10 13:30:00 | 16033.00 | 2025-12-11 09:15:00 | 16176.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-17 09:15:00 | 16454.00 | 2025-12-18 09:15:00 | 16251.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 13:00:00 | 16616.00 | 2025-12-29 10:15:00 | 16530.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-26 15:00:00 | 16613.00 | 2025-12-29 10:15:00 | 16530.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-29 09:15:00 | 16664.00 | 2025-12-29 10:15:00 | 16530.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-13 10:45:00 | 16499.00 | 2026-01-23 10:15:00 | 15674.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:45:00 | 16499.00 | 2026-01-28 09:15:00 | 14849.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-06 11:30:00 | 15089.00 | 2026-02-16 09:15:00 | 15218.00 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2026-02-10 09:30:00 | 15136.00 | 2026-02-16 09:15:00 | 15218.00 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2026-02-23 13:00:00 | 15020.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-02-23 13:30:00 | 15025.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-24 09:15:00 | 15013.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-02-25 10:00:00 | 14989.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-03-06 09:15:00 | 14318.00 | 2026-03-09 09:15:00 | 13602.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 14320.00 | 2026-03-09 09:15:00 | 13604.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:00:00 | 14326.00 | 2026-03-09 09:15:00 | 13609.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 14318.00 | 2026-03-10 10:15:00 | 13694.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-03-06 12:15:00 | 14320.00 | 2026-03-10 10:15:00 | 13694.00 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2026-03-06 13:00:00 | 14326.00 | 2026-03-10 10:15:00 | 13694.00 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2026-03-24 14:15:00 | 12510.00 | 2026-03-25 09:15:00 | 12720.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-01 10:15:00 | 12593.00 | 2026-04-01 12:15:00 | 12526.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2026-04-07 11:45:00 | 12681.00 | 2026-04-13 11:15:00 | 13174.00 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2026-04-07 12:45:00 | 12655.00 | 2026-04-13 11:15:00 | 13174.00 | STOP_HIT | 1.00 | 4.10% |
| SELL | retest2 | 2026-04-15 13:15:00 | 13278.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-15 14:00:00 | 13279.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-15 15:00:00 | 13287.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-04-16 09:45:00 | 13248.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-04-27 10:30:00 | 13055.00 | 2026-04-27 12:15:00 | 13235.00 | STOP_HIT | 1.00 | -1.38% |

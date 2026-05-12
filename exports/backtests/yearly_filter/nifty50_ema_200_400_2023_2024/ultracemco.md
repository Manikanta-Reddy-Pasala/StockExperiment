# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 11930.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT2_SKIP | 4 |
| ALERT3 | 69 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 46 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 40
- **Target hits / Stop hits / Partials:** 2 / 48 / 8
- **Avg / median % per leg:** 0.13% / -0.99%
- **Sum % (uncompounded):** 7.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 10 | 41.7% | 2 | 18 | 4 | 0.71% | 17.1% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.42% | 27.3% |
| BUY @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 2 | 14 | 0 | -0.64% | -10.2% |
| SELL (all) | 34 | 8 | 23.5% | 0 | 30 | 4 | -0.28% | -9.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 8 | 23.5% | 0 | 30 | 4 | -0.28% | -9.5% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.42% | 27.3% |
| retest2 (combined) | 50 | 10 | 20.0% | 2 | 44 | 4 | -0.39% | -19.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 12:15:00 | 9482.90 | 9751.75 | 9751.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 9397.85 | 9748.23 | 9750.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 9692.75 | 9687.35 | 9715.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:45:00 | 9704.25 | 9687.35 | 9715.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 9687.45 | 9687.51 | 9715.48 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 15:15:00 | 9981.05 | 9740.80 | 9739.92 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 12:15:00 | 9522.90 | 9739.40 | 9739.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 14:15:00 | 9514.15 | 9735.07 | 9737.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 13:15:00 | 9685.15 | 9684.53 | 9710.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-14 14:00:00 | 9685.15 | 9684.53 | 9710.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 9714.00 | 9674.61 | 9703.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 9703.90 | 9674.61 | 9703.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 9700.00 | 9674.86 | 9703.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 9706.10 | 9674.86 | 9703.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 9773.40 | 9675.84 | 9703.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 9777.90 | 9675.84 | 9703.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 9858.60 | 9677.66 | 9704.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:00:00 | 9858.60 | 9677.66 | 9704.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 10165.75 | 9729.92 | 9728.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 10225.60 | 9738.99 | 9732.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 9842.95 | 9902.82 | 9826.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 12:00:00 | 9842.95 | 9902.82 | 9826.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 10020.10 | 9903.98 | 9827.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 10:30:00 | 10134.70 | 9913.74 | 9837.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 10134.30 | 9919.72 | 9842.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-13 09:15:00 | 11148.17 | 10145.13 | 9972.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 10776.45 | 11382.71 | 11384.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 10697.85 | 11116.87 | 11219.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 11151.90 | 11058.55 | 11178.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 11151.90 | 11058.55 | 11178.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 11187.00 | 11059.82 | 11178.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 11187.00 | 11059.82 | 11178.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 11234.90 | 11061.57 | 11178.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 11234.90 | 11061.57 | 11178.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 11246.05 | 11063.40 | 11178.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 11246.05 | 11063.40 | 11178.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 11174.70 | 11105.36 | 11193.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:45:00 | 11177.65 | 11105.36 | 11193.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 11138.50 | 11102.96 | 11188.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 11165.65 | 11102.96 | 11188.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 11111.50 | 11103.92 | 11187.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 11084.25 | 11103.92 | 11187.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:45:00 | 11081.00 | 11103.49 | 11186.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 11034.05 | 11102.67 | 11185.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 11203.65 | 11102.64 | 11181.77 | SL hit (close>static) qty=1.00 sl=11190.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 12:15:00 | 11830.00 | 11250.84 | 11250.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 13:15:00 | 11890.60 | 11257.21 | 11253.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 11495.85 | 11550.53 | 11429.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 10:00:00 | 11495.85 | 11550.53 | 11429.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 11444.40 | 11548.61 | 11430.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 11444.40 | 11548.61 | 11430.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 11401.05 | 11546.31 | 11430.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 11401.05 | 11546.31 | 11430.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 11460.95 | 11545.47 | 11430.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 11400.00 | 11544.44 | 11430.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 11431.75 | 11541.15 | 11431.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:00:00 | 11431.75 | 11541.15 | 11431.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 11477.10 | 11540.51 | 11431.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 15:15:00 | 11465.00 | 11540.51 | 11431.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 11465.00 | 11539.76 | 11431.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:15:00 | 11428.05 | 11539.76 | 11431.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 11454.15 | 11538.91 | 11431.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 11609.90 | 11498.30 | 11430.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:45:00 | 11588.00 | 11531.58 | 11452.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:30:00 | 11604.85 | 11531.33 | 11453.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:00:00 | 11623.25 | 11531.33 | 11453.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 11461.95 | 11534.69 | 11458.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 11461.95 | 11534.69 | 11458.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 11428.70 | 11533.64 | 11458.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:00:00 | 11428.70 | 11533.64 | 11458.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 11432.35 | 11532.63 | 11458.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:30:00 | 11435.10 | 11532.63 | 11458.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 11408.80 | 11531.40 | 11458.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:30:00 | 11422.60 | 11531.40 | 11458.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-08 15:15:00 | 11370.00 | 11529.79 | 11457.99 | SL hit (close<static) qty=1.00 sl=11400.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 10525.35 | 11388.07 | 11392.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 13:15:00 | 10504.80 | 11379.28 | 11387.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11234.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11234.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11234.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 11249.85 | 11111.81 | 11234.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 11427.40 | 11114.95 | 11235.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:45:00 | 11417.00 | 11114.95 | 11235.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 11300.00 | 11132.40 | 11239.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 11246.65 | 11132.40 | 11239.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 11219.50 | 11133.89 | 11239.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 11:15:00 | 11318.15 | 11136.36 | 11239.89 | SL hit (close>static) qty=1.00 sl=11300.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 13:15:00 | 11537.95 | 11304.75 | 11304.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 12:15:00 | 11559.00 | 11312.89 | 11308.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11320.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11320.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11320.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 11287.50 | 11335.36 | 11320.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 11263.45 | 11334.65 | 11319.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 11263.45 | 11334.65 | 11319.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 11222.95 | 11333.53 | 11319.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 11222.95 | 11333.53 | 11319.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 11256.00 | 11330.64 | 11318.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 11253.95 | 11330.64 | 11318.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 11330.00 | 11328.96 | 11317.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 13:30:00 | 11427.40 | 11329.61 | 11317.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 09:45:00 | 11381.70 | 11333.30 | 11320.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 10:45:00 | 11380.00 | 11333.32 | 11320.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 11:30:00 | 11364.95 | 11333.09 | 11320.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 11307.25 | 11332.83 | 11319.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 12:45:00 | 11288.30 | 11332.83 | 11319.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 11290.30 | 11332.41 | 11319.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 11290.30 | 11332.41 | 11319.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 11340.10 | 11332.48 | 11319.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:15:00 | 11283.05 | 11332.48 | 11319.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 11283.05 | 11331.99 | 11319.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 11313.45 | 11331.99 | 11319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 11407.45 | 11332.74 | 11320.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 10:15:00 | 11440.90 | 11332.74 | 11320.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 11:00:00 | 11450.00 | 11333.91 | 11320.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 11076.10 | 11328.42 | 11318.87 | SL hit (close<static) qty=1.00 sl=11244.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 11127.10 | 11309.08 | 11309.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11306.44 | 11308.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.50 | 10987.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:00:00 | 10832.10 | 10797.50 | 10987.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 10884.00 | 10802.38 | 10984.99 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 11395.10 | 11099.39 | 11098.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.20 | 11131.71 | 11114.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11572.00 | 11385.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:45:00 | 11577.00 | 11572.00 | 11385.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11588.84 | 11423.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 11369.00 | 11588.84 | 11423.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 11346.00 | 11586.42 | 11422.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:00:00 | 11346.00 | 11586.42 | 11422.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 11366.00 | 11582.59 | 11422.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:30:00 | 11350.00 | 11582.59 | 11422.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 11363.00 | 11580.41 | 11422.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:15:00 | 11321.00 | 11580.41 | 11422.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 11458.00 | 11670.17 | 11528.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 11458.00 | 11670.17 | 11528.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 11427.00 | 11667.75 | 11527.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 11427.00 | 11667.75 | 11527.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 11421.00 | 11663.38 | 11526.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 11421.00 | 11663.38 | 11526.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 11259.00 | 11430.13 | 11430.20 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 11453.00 | 11430.33 | 11430.26 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11429.41 | 11429.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.27 | 11427.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 11437.00 | 11410.21 | 11419.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 11438.00 | 11410.49 | 11419.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 11448.00 | 11410.49 | 11419.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 11493.00 | 11411.31 | 11420.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 11493.00 | 11411.31 | 11420.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 11421.00 | 11412.94 | 11420.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 11397.00 | 11412.67 | 11420.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:00:00 | 11397.00 | 11412.23 | 11420.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 11396.00 | 11411.66 | 11420.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 11395.00 | 11411.65 | 11420.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 11417.00 | 11411.70 | 11419.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 11405.00 | 11411.70 | 11419.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 11391.00 | 11411.50 | 11419.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 11461.00 | 11411.79 | 11419.87 | SL hit (close>static) qty=1.00 sl=11435.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11767.00 | 11429.13 | 11427.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11458.38 | 11442.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12206.86 | 11965.91 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:30:00 | 12294.00 | 12208.15 | 11974.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 15:00:00 | 12271.00 | 12213.37 | 11983.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 09:30:00 | 12267.00 | 12213.61 | 11985.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:30:00 | 12268.00 | 12214.18 | 11988.21 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12884.55 | 12265.91 | 12087.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12880.35 | 12265.91 | 12087.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12881.40 | 12265.91 | 12087.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:15:00 | 12908.70 | 12339.54 | 12138.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 12500.00 | 12549.20 | 12348.11 | SL hit (close<ema200) qty=0.50 sl=12549.20 alert=retest1 |

### Cycle 15 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 12163.00 | 12306.12 | 12306.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 12136.00 | 12294.56 | 12300.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 12282.00 | 12278.51 | 12292.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 12314.00 | 12278.86 | 12292.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 12314.00 | 12278.86 | 12292.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 12287.00 | 12278.94 | 12292.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 12363.00 | 12278.94 | 12292.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 12334.00 | 12279.49 | 12292.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 12190.00 | 12288.18 | 12296.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 12258.00 | 12288.06 | 12295.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 12251.00 | 12291.27 | 12297.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:45:00 | 12252.00 | 12291.04 | 12297.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11645.10 | 11988.66 | 12103.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11638.45 | 11988.66 | 12103.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11639.40 | 11988.66 | 12103.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 11580.50 | 11931.92 | 12061.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.07 | SL hit (close>ema200) qty=0.50 sl=11843.48 alert=retest2 |

### Cycle 16 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11875.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12461.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 12650.00 | 12709.64 | 12461.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12468.32 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10762.00 | 12281.28 | 12291.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.66 | 11710.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 11620.00 | 11358.66 | 11710.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11710.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 11736.00 | 11362.41 | 11710.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 11678.00 | 11365.55 | 11710.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 11616.00 | 11374.95 | 11709.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 11759.00 | 11409.18 | 11688.40 | SL hit (close>static) qty=1.00 sl=11746.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-06 10:30:00 | 10134.70 | 2024-06-13 09:15:00 | 11148.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 09:15:00 | 10134.30 | 2024-06-13 09:15:00 | 11147.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-28 10:15:00 | 11084.25 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-11-28 11:45:00 | 11081.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-11-28 12:30:00 | 11034.05 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-01-02 11:45:00 | 11609.90 | 2025-01-08 15:15:00 | 11370.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-01-06 12:45:00 | 11588.00 | 2025-01-08 15:15:00 | 11370.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-01-07 09:30:00 | 11604.85 | 2025-01-08 15:15:00 | 11370.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-01-07 10:00:00 | 11623.25 | 2025-01-08 15:15:00 | 11370.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-01-27 09:15:00 | 11246.65 | 2025-01-27 11:15:00 | 11318.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-01-27 10:15:00 | 11219.50 | 2025-01-27 11:15:00 | 11318.15 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-01-27 13:00:00 | 11249.55 | 2025-01-28 13:15:00 | 11341.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-27 14:00:00 | 11256.60 | 2025-01-28 13:15:00 | 11341.70 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-28 09:15:00 | 11115.85 | 2025-01-28 13:15:00 | 11341.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-02-01 12:30:00 | 11160.00 | 2025-02-01 13:15:00 | 11274.65 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-02-03 09:30:00 | 11211.80 | 2025-02-04 10:15:00 | 11387.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-02-17 13:30:00 | 11427.40 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-02-18 09:45:00 | 11381.70 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-02-18 10:45:00 | 11380.00 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-18 11:30:00 | 11364.95 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-02-19 10:15:00 | 11440.90 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-02-19 11:00:00 | 11450.00 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-06-17 11:30:00 | 11397.00 | 2025-06-19 09:15:00 | 11461.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-06-17 14:00:00 | 11397.00 | 2025-06-19 09:15:00 | 11461.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-06-18 10:30:00 | 11396.00 | 2025-06-19 09:15:00 | 11461.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-06-18 12:15:00 | 11395.00 | 2025-06-19 09:15:00 | 11461.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-19 15:15:00 | 11359.00 | 2025-06-20 09:15:00 | 11472.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-20 12:15:00 | 11373.00 | 2025-06-20 13:15:00 | 11436.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-23 09:15:00 | 11347.00 | 2025-06-23 13:15:00 | 11474.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-23 11:00:00 | 11359.00 | 2025-06-23 13:15:00 | 11474.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest1 | 2025-07-30 09:30:00 | 12294.00 | 2025-08-18 09:15:00 | 12884.55 | PARTIAL | 0.50 | 4.80% |
| BUY | retest1 | 2025-07-30 15:00:00 | 12271.00 | 2025-08-18 09:15:00 | 12880.35 | PARTIAL | 0.50 | 4.97% |
| BUY | retest1 | 2025-07-31 09:30:00 | 12267.00 | 2025-08-18 09:15:00 | 12881.40 | PARTIAL | 0.50 | 5.01% |
| BUY | retest1 | 2025-07-31 11:30:00 | 12268.00 | 2025-08-20 10:15:00 | 12908.70 | PARTIAL | 0.50 | 5.22% |
| BUY | retest1 | 2025-07-30 09:30:00 | 12294.00 | 2025-09-10 10:15:00 | 12500.00 | STOP_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-07-30 15:00:00 | 12271.00 | 2025-09-10 10:15:00 | 12500.00 | STOP_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2025-07-31 09:30:00 | 12267.00 | 2025-09-10 10:15:00 | 12500.00 | STOP_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2025-07-31 11:30:00 | 12268.00 | 2025-09-10 10:15:00 | 12500.00 | STOP_HIT | 0.50 | 1.89% |
| BUY | retest2 | 2025-09-12 11:45:00 | 12450.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-15 12:30:00 | 12446.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-16 09:15:00 | 12508.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-09-23 13:00:00 | 12433.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-10-20 09:30:00 | 12190.00 | 2025-11-19 13:15:00 | 11645.10 | PARTIAL | 0.50 | 4.47% |
| SELL | retest2 | 2025-10-20 11:30:00 | 12258.00 | 2025-11-19 13:15:00 | 11638.45 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-10-23 12:15:00 | 12251.00 | 2025-11-19 13:15:00 | 11639.40 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-10-23 12:45:00 | 12252.00 | 2025-11-24 14:15:00 | 11580.50 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2025-10-20 09:30:00 | 12190.00 | 2025-12-01 15:15:00 | 12013.00 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-10-20 11:30:00 | 12258.00 | 2025-12-01 15:15:00 | 12013.00 | STOP_HIT | 0.50 | 2.00% |
| SELL | retest2 | 2025-10-23 12:15:00 | 12251.00 | 2025-12-01 15:15:00 | 12013.00 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2025-10-23 12:45:00 | 12252.00 | 2025-12-01 15:15:00 | 12013.00 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2025-12-02 09:30:00 | 11632.00 | 2026-01-01 09:15:00 | 11843.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-12-02 11:30:00 | 11640.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-12-02 12:00:00 | 11635.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-12-02 12:45:00 | 11636.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-12-30 09:15:00 | 11711.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-04-08 14:45:00 | 11616.00 | 2026-04-15 11:15:00 | 11759.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-04-30 09:15:00 | 11536.00 | 2026-05-04 10:15:00 | 11749.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-05-04 13:15:00 | 11604.00 | 2026-05-04 14:15:00 | 11761.00 | STOP_HIT | 1.00 | -1.35% |

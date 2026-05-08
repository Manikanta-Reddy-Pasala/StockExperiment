# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 11950.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 20 |
| ALERT1 | 16 |
| ALERT2 | 16 |
| ALERT2_SKIP | 13 |
| ALERT3 | 16 |
| PENDING | 32 |
| PENDING_CANCEL | 10 |
| ENTRY1 | 8 |
| ENTRY2 | 14 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 0 / 22 / 2
- **Avg / median % per leg:** 0.92% / -1.33%
- **Sum % (uncompounded):** 21.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 0 | 15 | 2 | 2.15% | 36.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 7 | 0 | -0.92% | -6.5% |
| BUY @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 0 | 8 | 2 | 4.31% | 43.1% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.09% | -14.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.03% | -1.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.27% | -13.6% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 8 | 0 | -0.94% | -7.5% |
| retest2 (combined) | 16 | 4 | 25.0% | 0 | 14 | 2 | 1.84% | 29.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 11:15:00 | 8515.50 | 8222.52 | 8222.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 14:15:00 | 8584.75 | 8231.89 | 8227.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 8393.55 | 8409.52 | 8333.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 13:15:00 | 8339.85 | 8407.77 | 8333.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 8339.85 | 8407.77 | 8333.71 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 13:15:00 | 8160.00 | 8287.14 | 8287.56 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 15:15:00 | 8350.90 | 8287.94 | 8287.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 15:15:00 | 8351.75 | 8291.61 | 8289.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 14:15:00 | 8286.00 | 8298.32 | 8293.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 14:15:00 | 8286.00 | 8298.32 | 8293.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 8286.00 | 8298.32 | 8293.38 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-10-19 14:15:00 | 8513.80 | 8299.41 | 8294.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 15:15:00 | 8511.70 | 8301.52 | 8295.50 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-25 11:15:00 | 8282.45 | 8316.06 | 8303.68 | SL hit (close<static) qty=1.00 sl=8285.55 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-30 13:15:00 | 8381.85 | 8300.67 | 8296.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 14:15:00 | 8414.00 | 8301.80 | 8297.35 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 15:15:00 | 9676.10 | 8880.66 | 8671.42 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-10 11:15:00 | 9763.50 | 9791.96 | 9387.50 | SL hit (close<ema200) qty=0.50 sl=9791.96 alert=retest2 |

### Cycle 4 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 9373.40 | 9743.61 | 9744.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 9307.00 | 9735.71 | 9740.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 9692.30 | 9686.18 | 9712.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 09:15:00 | 9687.45 | 9686.17 | 9712.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 9687.45 | 9686.17 | 9712.03 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 9993.70 | 9734.84 | 9734.42 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 14:15:00 | 9514.15 | 9734.36 | 9734.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 09:15:00 | 9468.30 | 9729.52 | 9732.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 13:15:00 | 9685.15 | 9684.02 | 9707.78 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-15 09:15:00 | 9611.75 | 9682.93 | 9706.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:15:00 | 9614.80 | 9682.26 | 9706.42 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 9714.15 | 9674.29 | 9701.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 9714.15 | 9674.29 | 9701.04 | SL hit (close>ema400) qty=1.00 sl=9701.04 alert=retest1 |

### Cycle 7 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 10165.75 | 9725.28 | 9724.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 10226.25 | 9734.65 | 9729.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 9842.95 | 9900.58 | 9824.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 10020.10 | 9901.77 | 9825.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 10020.10 | 9901.77 | 9825.16 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-06 10:15:00 | 10145.75 | 9911.36 | 9834.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-06 11:15:00 | 10034.30 | 9912.58 | 9835.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-07 09:15:00 | 10126.45 | 9919.49 | 9840.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:15:00 | 10220.90 | 9922.49 | 9842.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:15:00 | 11754.04 | 10542.69 | 10248.57 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 11238.55 | 11279.43 | 10844.32 | SL hit (close<ema200) qty=0.50 sl=11279.43 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 10776.45 | 11382.30 | 11384.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 10698.05 | 11118.98 | 11221.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 11151.90 | 11059.52 | 11179.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 10:15:00 | 11187.00 | 11060.79 | 11179.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 11187.00 | 11060.79 | 11179.25 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-27 09:15:00 | 11043.90 | 11106.66 | 11192.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:15:00 | 11000.00 | 11105.60 | 11191.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 11060.00 | 11104.53 | 11187.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:15:00 | 11105.00 | 11104.53 | 11187.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 11203.65 | 11102.84 | 11182.52 | SL hit (close>static) qty=1.00 sl=11198.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 11203.65 | 11102.84 | 11182.52 | SL hit (close>static) qty=1.00 sl=11198.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 11890.60 | 11257.79 | 11254.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 11977.85 | 11264.96 | 11258.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 11495.85 | 11551.06 | 11430.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 11444.40 | 11549.17 | 11431.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 11444.40 | 11549.17 | 11431.08 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-23 10:15:00 | 11519.25 | 11544.34 | 11431.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-23 11:15:00 | 11453.70 | 11543.44 | 11431.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-02 09:15:00 | 11517.60 | 11496.08 | 11428.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:15:00 | 11563.05 | 11496.75 | 11429.37 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 09:15:00 | 11623.25 | 11531.56 | 11454.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 11583.25 | 11532.07 | 11454.93 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 11408.75 | 11531.62 | 11458.88 | SL hit (close<static) qty=1.00 sl=11427.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 11408.75 | 11531.62 | 11458.88 | SL hit (close<static) qty=1.00 sl=11427.55 alert=retest2 |

### Cycle 10 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 10561.45 | 11396.93 | 11397.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 10525.35 | 11388.26 | 11392.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.45 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 11559.00 | 11304.75 | 11304.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 11590.00 | 11313.09 | 11308.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-02-17 14:15:00 | 11497.00 | 11325.46 | 11315.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 15:15:00 | 11470.30 | 11326.90 | 11316.45 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-19 10:15:00 | 11450.00 | 11328.90 | 11317.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-19 11:15:00 | 11395.60 | 11329.57 | 11318.31 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 11076.10 | 11324.39 | 11316.34 | SL hit (close<static) qty=1.00 sl=11269.45 alert=retest2 |

### Cycle 12 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 11145.40 | 11307.19 | 11307.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11302.79 | 11305.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.16 | 10986.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 10921.85 | 10808.59 | 10981.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 10921.85 | 10808.59 | 10981.21 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 11395.10 | 11098.50 | 11097.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.15 | 11131.19 | 11114.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11572.00 | 11385.11 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-05 09:15:00 | 11679.00 | 11574.67 | 11391.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 11693.00 | 11575.85 | 11392.57 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 14:15:00 | 11697.00 | 11583.79 | 11406.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 15:15:00 | 11660.00 | 11584.55 | 11407.74 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 11:15:00 | 11663.00 | 11585.47 | 11410.83 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 12:15:00 | 11650.00 | 11586.11 | 11412.03 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-07 14:15:00 | 11664.00 | 11587.31 | 11414.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 15:15:00 | 11658.00 | 11588.02 | 11415.58 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-08 12:15:00 | 11671.00 | 11590.78 | 11420.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-08 13:15:00 | 11621.00 | 11591.08 | 11421.39 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11589.82 | 11423.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11369.00 | 11589.82 | 11423.28 | SL hit (close<ema400) qty=1.00 sl=11423.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11369.00 | 11589.82 | 11423.28 | SL hit (close<ema400) qty=1.00 sl=11423.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11369.00 | 11589.82 | 11423.28 | SL hit (close<ema400) qty=1.00 sl=11423.28 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 11603.00 | 11577.08 | 11422.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 11625.00 | 11577.55 | 11423.52 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 11295.00 | 11647.05 | 11522.40 | SL hit (close<static) qty=1.00 sl=11330.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 11266.00 | 11429.40 | 11429.88 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 11493.00 | 11430.46 | 11430.39 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11430.14 | 11430.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.99 | 11428.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11765.00 | 11430.17 | 11428.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11459.16 | 11443.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12207.48 | 11966.53 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 12291.00 | 12208.63 | 11975.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 12295.00 | 12209.49 | 11976.99 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 12:15:00 | 12295.00 | 12215.54 | 11990.33 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 13:15:00 | 12327.00 | 12216.65 | 11992.01 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-04 13:15:00 | 12289.00 | 12212.63 | 12005.13 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-04 14:15:00 | 12245.00 | 12212.96 | 12006.33 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-05 09:15:00 | 12327.00 | 12214.52 | 12009.17 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:15:00 | 12322.00 | 12215.59 | 12010.73 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 12271.00 | 12221.82 | 12031.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 15:15:00 | 12309.00 | 12222.69 | 12033.10 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 12391.00 | 12535.50 | 12354.00 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-12 11:15:00 | 12466.00 | 12533.67 | 12354.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 12:15:00 | 12420.00 | 12532.54 | 12355.21 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.75 | 12355.20 | SL hit (close<ema400) qty=1.00 sl=12355.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.75 | 12355.20 | SL hit (close<ema400) qty=1.00 sl=12355.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.75 | 12355.20 | SL hit (close<ema400) qty=1.00 sl=12355.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.75 | 12355.20 | SL hit (close<ema400) qty=1.00 sl=12355.20 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-15 12:15:00 | 12455.00 | 12522.12 | 12356.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 12454.00 | 12521.44 | 12356.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-23 12:15:00 | 12433.00 | 12544.71 | 12400.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 13:15:00 | 12419.00 | 12543.46 | 12400.17 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 12315.00 | 12536.44 | 12399.46 | SL hit (close<static) qty=1.00 sl=12346.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 14:15:00 | 12177.00 | 12307.18 | 12307.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 12036.00 | 12290.03 | 12298.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12289.00 | 12278.55 | 12292.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 12289.00 | 12278.55 | 12292.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 12289.00 | 12278.55 | 12292.18 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-23 13:15:00 | 12189.00 | 12290.76 | 12296.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:15:00 | 12126.00 | 12289.12 | 12296.10 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-14 15:15:00 | 12255.00 | 11868.86 | 11871.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-16 09:15:00 | 12284.00 | 11872.99 | 11873.46 | ENTRY2 sustain failed after 2520m |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 12304.00 | 11877.28 | 11875.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.28 | 11875.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.91 | 11877.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12662.00 | 12704.29 | 12453.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 12469.00 | 12701.44 | 12460.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12701.44 | 12460.78 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11089.00 | 12293.72 | 12293.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10765.00 | 12278.51 | 12286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11357.16 | 11706.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 11736.00 | 11360.93 | 11706.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11360.93 | 11706.66 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 14:15:00 | 11576.00 | 11373.48 | 11706.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-08 15:15:00 | 11603.00 | 11375.77 | 11705.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 11452.00 | 11376.52 | 11704.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 11474.00 | 11377.49 | 11703.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 11562.00 | 11393.82 | 11692.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 11325.00 | 11393.13 | 11690.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11412.26 | 11685.98 | SL hit (close>static) qty=1.00 sl=11764.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11412.26 | 11685.98 | SL hit (close>static) qty=1.00 sl=11764.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 11528.00 | 11692.94 | 11769.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 11523.00 | 11691.25 | 11768.26 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 11779.00 | 11688.34 | 11761.83 | SL hit (close>static) qty=1.00 sl=11764.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-19 15:15:00 | 8511.70 | 2023-10-25 11:15:00 | 8282.45 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2023-10-30 14:15:00 | 8414.00 | 2023-12-11 15:15:00 | 9676.10 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-10-30 14:15:00 | 8414.00 | 2024-01-10 11:15:00 | 9763.50 | STOP_HIT | 0.50 | 16.04% |
| SELL | retest1 | 2024-05-15 10:15:00 | 9614.80 | 2024-05-16 14:15:00 | 9714.15 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-06-07 10:15:00 | 10220.90 | 2024-06-27 09:15:00 | 11754.04 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-07 10:15:00 | 10220.90 | 2024-07-19 14:15:00 | 11238.55 | STOP_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2024-11-27 10:15:00 | 11000.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-11-28 11:15:00 | 11105.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-01-02 10:15:00 | 11563.05 | 2025-01-08 14:15:00 | 11408.75 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-07 10:15:00 | 11583.25 | 2025-01-08 14:15:00 | 11408.75 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-02-17 15:15:00 | 11470.30 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest1 | 2025-05-05 10:15:00 | 11693.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2025-05-06 15:15:00 | 11660.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2025-05-07 15:15:00 | 11658.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-05-12 10:15:00 | 11625.00 | 2025-05-28 11:15:00 | 11295.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2025-07-30 10:15:00 | 12295.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest1 | 2025-07-31 13:15:00 | 12327.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest1 | 2025-08-05 10:15:00 | 12322.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest1 | 2025-08-07 15:15:00 | 12309.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-09-15 13:15:00 | 12454.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-23 14:15:00 | 12126.00 | 2026-01-16 10:15:00 | 12304.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-04-09 10:15:00 | 11474.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2026-04-30 10:15:00 | 11523.00 | 2026-05-05 09:15:00 | 11779.00 | STOP_HIT | 1.00 | -2.22% |

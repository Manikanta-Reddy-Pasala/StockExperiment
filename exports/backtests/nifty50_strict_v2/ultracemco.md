# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 11950.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 22 |
| ALERT1 | 17 |
| ALERT2 | 17 |
| ALERT2_SKIP | 13 |
| ALERT3 | 18 |
| PENDING | 40 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 9 |
| ENTRY2 | 18 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 21
- **Target hits / Stop hits / Partials:** 2 / 25 / 5
- **Avg / median % per leg:** 0.29% / -1.20%
- **Sum % (uncompounded):** 9.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 11 | 45.8% | 2 | 17 | 5 | 1.13% | 27.0% |
| BUY @ 2nd Alert (retest1) | 13 | 10 | 76.9% | 1 | 7 | 5 | 2.65% | 34.4% |
| BUY @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.67% | -7.3% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.21% | -17.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.03% | -1.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.37% | -16.6% |
| retest1 (combined) | 14 | 10 | 71.4% | 1 | 8 | 5 | 2.38% | 33.4% |
| retest2 (combined) | 18 | 1 | 5.6% | 1 | 17 | 0 | -1.33% | -24.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 15:15:00 | 8550.00 | 8269.55 | 8268.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 11:15:00 | 8556.95 | 8312.46 | 8291.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 8393.55 | 8410.08 | 8348.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 8349.95 | 8409.01 | 8349.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 8349.95 | 8409.01 | 8349.30 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 12:15:00 | 8171.20 | 8308.34 | 8308.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 8146.05 | 8301.18 | 8305.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 10:15:00 | 8331.80 | 8285.22 | 8296.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 10:15:00 | 8331.80 | 8285.22 | 8296.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 8331.80 | 8285.22 | 8296.47 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-10-18 14:15:00 | 8267.65 | 8298.46 | 8301.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-18 15:15:00 | 8287.00 | 8298.35 | 8301.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-19 09:15:00 | 8263.00 | 8298.00 | 8301.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-19 10:15:00 | 8300.00 | 8298.02 | 8301.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-19 12:15:00 | 8268.60 | 8297.72 | 8301.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 13:15:00 | 8265.50 | 8297.40 | 8301.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-19 14:15:00 | 8513.80 | 8299.55 | 8302.37 | SL hit (close>static) qty=1.00 sl=8334.50 alert=retest2 |

### Cycle 3 — BUY (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 11:15:00 | 8431.95 | 8305.91 | 8305.53 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 15:15:00 | 8205.30 | 8305.48 | 8305.74 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 8415.90 | 8306.51 | 8306.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 13:15:00 | 8429.40 | 8307.73 | 8306.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 8546.30 | 8555.00 | 8463.06 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-11-28 09:15:00 | 8665.60 | 8556.22 | 8464.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 10:15:00 | 8699.10 | 8557.65 | 8465.76 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 09:15:00 | 9134.05 | 8604.75 | 8498.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-12-11 10:15:00 | 9569.01 | 8842.40 | 8649.70 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 9744.95 | 9957.78 | 9721.24 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-02-15 14:15:00 | 9820.05 | 9936.19 | 9723.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 15:15:00 | 9814.30 | 9934.98 | 9724.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-29 10:15:00 | 9833.45 | 9930.74 | 9780.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 11:15:00 | 9843.40 | 9929.88 | 9781.14 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-06 10:15:00 | 9696.05 | 9934.31 | 9802.58 | SL hit (close<static) qty=1.00 sl=9704.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-06 10:15:00 | 9696.05 | 9934.31 | 9802.58 | SL hit (close<static) qty=1.00 sl=9704.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-28 12:15:00 | 9820.00 | 9735.89 | 9730.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 13:15:00 | 9822.00 | 9736.75 | 9731.01 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-01 09:15:00 | 9870.85 | 9738.07 | 9731.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 10:15:00 | 9890.00 | 9739.58 | 9732.55 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 9765.15 | 9808.75 | 9773.14 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-04-10 10:15:00 | 9823.00 | 9807.75 | 9773.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-10 11:15:00 | 9803.75 | 9807.71 | 9773.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-10 12:15:00 | 9853.45 | 9808.17 | 9773.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 13:15:00 | 9844.10 | 9808.52 | 9774.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-12 09:15:00 | 9721.15 | 9807.67 | 9774.15 | SL hit (close<static) qty=1.00 sl=9759.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-12 13:15:00 | 9700.10 | 9805.75 | 9773.85 | SL hit (close<static) qty=1.00 sl=9704.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-12 13:15:00 | 9700.10 | 9805.75 | 9773.85 | SL hit (close<static) qty=1.00 sl=9704.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 9373.40 | 9743.61 | 9744.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 9307.00 | 9735.71 | 9740.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 9692.30 | 9686.18 | 9712.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 09:15:00 | 9687.45 | 9686.17 | 9712.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 9687.45 | 9686.17 | 9712.13 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 9993.70 | 9734.84 | 9734.51 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 14:15:00 | 9514.15 | 9734.36 | 9735.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 09:15:00 | 9468.30 | 9729.52 | 9732.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 13:15:00 | 9685.15 | 9684.02 | 9707.85 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-15 09:15:00 | 9611.75 | 9682.93 | 9706.95 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:15:00 | 9614.80 | 9682.26 | 9706.49 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 9714.15 | 9674.29 | 9701.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 9714.15 | 9674.29 | 9701.10 | SL hit (close>ema400) qty=1.00 sl=9701.10 alert=retest1 |

### Cycle 9 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 10165.75 | 9725.28 | 9724.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 10226.25 | 9734.65 | 9729.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 9842.95 | 9900.58 | 9824.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 10020.10 | 9901.77 | 9825.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 10020.10 | 9901.77 | 9825.20 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-06 10:15:00 | 10145.75 | 9911.36 | 9834.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-06 11:15:00 | 10034.30 | 9912.58 | 9835.55 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-07 09:15:00 | 10126.45 | 9919.49 | 9840.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:15:00 | 10220.90 | 9922.49 | 9842.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-06-14 09:15:00 | 11242.99 | 10210.51 | 10009.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-10-23 10:15:00)

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

### Cycle 11 — BUY (started 2024-12-05 13:15:00)

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

### Cycle 12 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 10561.45 | 11396.93 | 11397.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 10525.35 | 11388.26 | 11392.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.46 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-02-12 12:15:00)

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

### Cycle 14 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 11145.40 | 11307.19 | 11307.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11302.79 | 11305.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.16 | 10986.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 10921.85 | 10808.59 | 10981.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 10921.85 | 10808.59 | 10981.21 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-04-08 09:15:00)

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

### Cycle 16 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 11266.00 | 11429.40 | 11429.88 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 11493.00 | 11430.46 | 11430.39 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11430.14 | 11430.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.99 | 11428.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-06-24 12:15:00)

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
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:15:00 | 12909.75 | 12340.02 | 12138.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 11:15:00 | 12924.45 | 12345.68 | 12142.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 09:15:00 | 12943.35 | 12522.22 | 12304.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 09:15:00 | 12938.10 | 12522.22 | 12304.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 12533.00 | 12533.70 | 12319.82 | SL hit (close<ema200) qty=0.50 sl=12533.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 12533.00 | 12533.70 | 12319.82 | SL hit (close<ema200) qty=0.50 sl=12533.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 12533.00 | 12533.70 | 12319.82 | SL hit (close<ema200) qty=0.50 sl=12533.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 12533.00 | 12533.70 | 12319.82 | SL hit (close<ema200) qty=0.50 sl=12533.70 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 12391.00 | 12535.50 | 12354.00 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-12 11:15:00 | 12466.00 | 12533.67 | 12354.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 12:15:00 | 12420.00 | 12532.54 | 12355.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-15 12:15:00 | 12455.00 | 12522.12 | 12356.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 12454.00 | 12521.44 | 12356.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-23 12:15:00 | 12433.00 | 12544.71 | 12400.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 13:15:00 | 12419.00 | 12543.46 | 12400.17 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 12315.00 | 12536.44 | 12399.46 | SL hit (close<static) qty=1.00 sl=12346.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-10-09 14:15:00)

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

### Cycle 21 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.28 | 11875.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.91 | 11877.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12662.00 | 12704.29 | 12453.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 12469.00 | 12701.44 | 12460.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12701.44 | 12460.78 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2026-03-12 15:15:00)

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
| SELL | retest2 | 2023-10-19 13:15:00 | 8265.50 | 2023-10-19 14:15:00 | 8513.80 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest1 | 2023-11-28 10:15:00 | 8699.10 | 2023-12-01 09:15:00 | 9134.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-28 10:15:00 | 8699.10 | 2023-12-11 10:15:00 | 9569.01 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-02-15 15:15:00 | 9814.30 | 2024-03-06 10:15:00 | 9696.05 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-02-29 11:15:00 | 9843.40 | 2024-03-06 10:15:00 | 9696.05 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-03-28 13:15:00 | 9822.00 | 2024-04-12 09:15:00 | 9721.15 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-04-01 10:15:00 | 9890.00 | 2024-04-12 13:15:00 | 9700.10 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-04-10 13:15:00 | 9844.10 | 2024-04-12 13:15:00 | 9700.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest1 | 2024-05-15 10:15:00 | 9614.80 | 2024-05-16 14:15:00 | 9714.15 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-06-07 10:15:00 | 10220.90 | 2024-06-14 09:15:00 | 11242.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-27 10:15:00 | 11000.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-11-28 11:15:00 | 11105.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-01-02 10:15:00 | 11563.05 | 2025-01-08 14:15:00 | 11408.75 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-07 10:15:00 | 11583.25 | 2025-01-08 14:15:00 | 11408.75 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-02-17 15:15:00 | 11470.30 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest1 | 2025-05-05 10:15:00 | 11693.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2025-05-06 15:15:00 | 11660.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2025-05-07 15:15:00 | 11658.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-05-12 10:15:00 | 11625.00 | 2025-05-28 11:15:00 | 11295.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2025-07-30 10:15:00 | 12295.00 | 2025-08-20 10:15:00 | 12909.75 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-31 13:15:00 | 12327.00 | 2025-08-20 11:15:00 | 12924.45 | PARTIAL | 0.50 | 4.85% |
| BUY | retest1 | 2025-08-05 10:15:00 | 12322.00 | 2025-09-04 09:15:00 | 12943.35 | PARTIAL | 0.50 | 5.04% |
| BUY | retest1 | 2025-08-07 15:15:00 | 12309.00 | 2025-09-04 09:15:00 | 12938.10 | PARTIAL | 0.50 | 5.11% |
| BUY | retest1 | 2025-07-30 10:15:00 | 12295.00 | 2025-09-05 11:15:00 | 12533.00 | STOP_HIT | 0.50 | 1.94% |
| BUY | retest1 | 2025-07-31 13:15:00 | 12327.00 | 2025-09-05 11:15:00 | 12533.00 | STOP_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-08-05 10:15:00 | 12322.00 | 2025-09-05 11:15:00 | 12533.00 | STOP_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2025-08-07 15:15:00 | 12309.00 | 2025-09-05 11:15:00 | 12533.00 | STOP_HIT | 0.50 | 1.82% |
| BUY | retest2 | 2025-09-15 13:15:00 | 12454.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-23 14:15:00 | 12126.00 | 2026-01-16 10:15:00 | 12304.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-04-09 10:15:00 | 11474.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2026-04-30 10:15:00 | 11523.00 | 2026-05-05 09:15:00 | 11779.00 | STOP_HIT | 1.00 | -2.22% |

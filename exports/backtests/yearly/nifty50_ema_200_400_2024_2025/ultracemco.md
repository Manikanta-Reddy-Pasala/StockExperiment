# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2024-10-14 09:15:00 → 2026-05-08 15:15:00 (2706 bars)
- **Last close:** 11930.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 30 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 26
- **Target hits / Stop hits / Partials:** 0 / 34 / 8
- **Avg / median % per leg:** -0.06% / -1.20%
- **Sum % (uncompounded):** -2.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 8 | 30.8% | 0 | 22 | 4 | -0.40% | -10.3% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.42% | 27.3% |
| BUY @ 3rd Alert (retest2) | 18 | 0 | 0.0% | 0 | 18 | 0 | -2.09% | -37.7% |
| SELL (all) | 16 | 8 | 50.0% | 0 | 12 | 4 | 0.50% | 7.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 0 | 12 | 4 | 0.50% | 7.9% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.42% | 27.3% |
| retest2 (combined) | 34 | 8 | 23.5% | 0 | 30 | 4 | -0.87% | -29.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 10550.15 | 11200.61 | 11201.18 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 14:15:00 | 11476.95 | 11187.21 | 11187.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 11527.90 | 11190.60 | 11188.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 11113.60 | 11223.99 | 11205.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 11113.60 | 11223.99 | 11205.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 11113.60 | 11223.99 | 11205.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 11064.05 | 11223.99 | 11205.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 11274.65 | 11224.50 | 11206.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 11420.80 | 11224.50 | 11206.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 11:15:00 | 11048.65 | 11221.86 | 11205.45 | SL hit (close<static) qty=1.00 sl=11059.30 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 10426.85 | 11234.08 | 11236.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 10:15:00 | 10311.80 | 11201.74 | 11220.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.27 | 10965.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:00:00 | 10832.10 | 10797.27 | 10965.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 10884.00 | 10802.16 | 10963.51 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 09:15:00 | 11483.00 | 11070.21 | 11068.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.20 | 11131.62 | 11101.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11571.96 | 11376.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:45:00 | 11577.00 | 11571.96 | 11376.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11588.81 | 11415.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 11369.00 | 11588.81 | 11415.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 11346.00 | 11586.40 | 11415.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:00:00 | 11346.00 | 11586.40 | 11415.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 11366.00 | 11582.57 | 11415.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:30:00 | 11350.00 | 11582.57 | 11415.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 11363.00 | 11580.38 | 11414.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:15:00 | 11321.00 | 11580.38 | 11414.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 11458.00 | 11670.16 | 11523.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 11458.00 | 11670.16 | 11523.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 11427.00 | 11667.74 | 11522.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 11427.00 | 11667.74 | 11522.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 11421.00 | 11663.36 | 11522.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 11421.00 | 11663.36 | 11522.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 11433.00 | 11428.90 | 11426.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 11424.00 | 11428.90 | 11426.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 11429.00 | 11428.91 | 11426.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:15:00 | 11412.00 | 11428.91 | 11426.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 11459.00 | 11429.20 | 11426.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:15:00 | 11465.00 | 11428.55 | 11426.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:45:00 | 11461.00 | 11430.68 | 11427.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 11471.00 | 11431.02 | 11427.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:45:00 | 11463.00 | 11431.27 | 11427.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 11435.00 | 11431.31 | 11427.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 11435.00 | 11431.31 | 11427.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 11325.00 | 11430.49 | 11427.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 11325.00 | 11430.49 | 11427.29 | SL hit (close<static) qty=1.00 sl=11396.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 11130.00 | 11422.33 | 11423.24 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 11802.00 | 11421.88 | 11421.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11458.38 | 11440.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12206.86 | 11964.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:30:00 | 12294.00 | 12208.15 | 11973.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 15:00:00 | 12271.00 | 12213.36 | 11982.31 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 09:30:00 | 12267.00 | 12213.61 | 11984.74 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:30:00 | 12268.00 | 12214.18 | 11987.30 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12884.55 | 12265.91 | 12086.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12880.35 | 12265.91 | 12086.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12881.40 | 12265.91 | 12086.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:15:00 | 12908.70 | 12339.54 | 12137.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 12500.00 | 12549.20 | 12347.75 | SL hit (close<ema200) qty=0.50 sl=12549.20 alert=retest1 |

### Cycle 7 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 12163.00 | 12306.12 | 12306.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 12136.00 | 12294.56 | 12300.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12291.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12291.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12291.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 12282.00 | 12278.51 | 12291.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 12314.00 | 12278.86 | 12292.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 12314.00 | 12278.86 | 12292.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 12287.00 | 12278.94 | 12292.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 12363.00 | 12278.94 | 12292.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 12334.00 | 12279.49 | 12292.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 12190.00 | 12288.18 | 12295.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 12258.00 | 12288.06 | 12295.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 12251.00 | 12291.27 | 12297.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:45:00 | 12252.00 | 12291.04 | 12296.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11645.10 | 11988.66 | 12103.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11638.45 | 11988.66 | 12103.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11639.40 | 11988.66 | 12103.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 11580.50 | 11931.92 | 12061.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.02 | SL hit (close>ema200) qty=0.50 sl=11843.48 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11875.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12461.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 12650.00 | 12709.64 | 12461.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12468.32 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10762.00 | 12281.28 | 12291.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.66 | 11710.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 11620.00 | 11358.66 | 11710.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11710.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 11736.00 | 11362.41 | 11710.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 11678.00 | 11365.55 | 11710.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 11616.00 | 11374.95 | 11709.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 11759.00 | 11409.18 | 11688.40 | SL hit (close>static) qty=1.00 sl=11746.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-02-01 14:15:00 | 11420.80 | 2025-02-03 11:15:00 | 11048.65 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-02-04 11:00:00 | 11387.40 | 2025-02-14 12:15:00 | 11222.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-04 12:00:00 | 11376.10 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-02-04 12:30:00 | 11374.00 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-02-12 11:45:00 | 11371.10 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-02-17 13:30:00 | 11427.40 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-02-18 10:30:00 | 11362.00 | 2025-02-21 09:15:00 | 11076.10 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-02-18 11:30:00 | 11364.95 | 2025-02-21 10:15:00 | 11037.70 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-02-19 10:15:00 | 11440.90 | 2025-02-21 10:15:00 | 11037.70 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2025-02-19 11:00:00 | 11450.00 | 2025-02-21 10:15:00 | 11037.70 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-06-11 10:15:00 | 11465.00 | 2025-06-12 13:15:00 | 11325.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-06-11 14:45:00 | 11461.00 | 2025-06-12 13:15:00 | 11325.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-12 10:15:00 | 11471.00 | 2025-06-12 13:15:00 | 11325.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-06-12 10:45:00 | 11463.00 | 2025-06-12 13:15:00 | 11325.00 | STOP_HIT | 1.00 | -1.20% |
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

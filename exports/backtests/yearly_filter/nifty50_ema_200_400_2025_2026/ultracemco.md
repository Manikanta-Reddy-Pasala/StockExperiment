# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 11930.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 24 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 20
- **Target hits / Stop hits / Partials:** 0 / 28 / 8
- **Avg / median % per leg:** 0.69% / -0.56%
- **Sum % (uncompounded):** 24.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.89% | 22.7% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.42% | 27.3% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.16% | -4.6% |
| SELL (all) | 24 | 8 | 33.3% | 0 | 20 | 4 | 0.08% | 2.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 8 | 33.3% | 0 | 20 | 4 | 0.08% | 2.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.42% | 27.3% |
| retest2 (combined) | 28 | 8 | 28.6% | 0 | 24 | 4 | -0.09% | -2.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 11266.00 | 11428.50 | 11429.31 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 12:15:00 | 11475.00 | 11430.10 | 11430.08 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11429.41 | 11429.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.27 | 11427.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 11437.00 | 11410.21 | 11419.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 11438.00 | 11410.49 | 11419.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 11448.00 | 11410.49 | 11419.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 11493.00 | 11411.31 | 11420.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 11493.00 | 11411.31 | 11420.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 11421.00 | 11412.94 | 11420.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 11397.00 | 11412.67 | 11420.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:00:00 | 11397.00 | 11412.23 | 11420.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 11396.00 | 11411.66 | 11419.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 11395.00 | 11411.65 | 11419.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 11417.00 | 11411.70 | 11419.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 11405.00 | 11411.70 | 11419.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 11391.00 | 11411.50 | 11419.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 11461.00 | 11411.79 | 11419.81 | SL hit (close>static) qty=1.00 sl=11435.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11767.00 | 11429.13 | 11427.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11458.38 | 11442.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12206.86 | 11965.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:30:00 | 12294.00 | 12208.15 | 11974.82 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 15:00:00 | 12271.00 | 12213.37 | 11983.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 09:30:00 | 12267.00 | 12213.61 | 11985.64 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:30:00 | 12268.00 | 12214.18 | 11988.19 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12884.55 | 12265.91 | 12087.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12880.35 | 12265.91 | 12087.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 12881.40 | 12265.91 | 12087.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:15:00 | 12908.70 | 12339.54 | 12138.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 12500.00 | 12549.20 | 12348.10 | SL hit (close<ema200) qty=0.50 sl=12549.20 alert=retest1 |

### Cycle 5 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 12163.00 | 12306.12 | 12306.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 12136.00 | 12294.56 | 12300.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 12282.00 | 12278.51 | 12292.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 12314.00 | 12278.86 | 12292.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 12314.00 | 12278.86 | 12292.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 12287.00 | 12278.94 | 12292.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 12363.00 | 12278.94 | 12292.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 12334.00 | 12279.49 | 12292.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 12190.00 | 12288.18 | 12296.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 12258.00 | 12288.06 | 12295.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 12251.00 | 12291.27 | 12297.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:45:00 | 12252.00 | 12291.04 | 12297.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11645.10 | 11988.66 | 12103.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11638.45 | 11988.66 | 12103.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 11639.40 | 11988.66 | 12103.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 11580.50 | 11931.92 | 12061.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.07 | SL hit (close>ema200) qty=0.50 sl=11843.48 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11875.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12461.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 12650.00 | 12709.64 | 12461.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12468.32 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-12 15:15:00)

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

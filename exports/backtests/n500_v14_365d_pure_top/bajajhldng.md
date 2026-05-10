# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 10678.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 5
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 2.33% / 1.61%
- **Sum % (uncompounded):** 30.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 8 | 61.5% | 1 | 8 | 4 | 2.33% | 30.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 1 | 8 | 4 | 2.33% | 30.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 8 | 61.5% | 1 | 8 | 4 | 2.33% | 30.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 12620.00 | 13595.83 | 13597.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 12599.00 | 13558.31 | 13578.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 13358.00 | 13309.78 | 13424.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 13358.00 | 13309.78 | 13424.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 13545.00 | 13312.12 | 13424.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 13545.00 | 13312.12 | 13424.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 13550.00 | 13314.49 | 13425.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 13702.00 | 13314.49 | 13425.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 13100.00 | 12600.57 | 12898.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 13100.00 | 12600.57 | 12898.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 13114.00 | 12605.68 | 12899.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 13013.00 | 12631.23 | 12905.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 13029.00 | 12635.45 | 12905.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:45:00 | 13055.00 | 12639.56 | 12906.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 13044.00 | 12643.59 | 12907.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 13155.00 | 12656.25 | 12909.65 | SL hit (close>static) qty=1.00 sl=13150.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 13155.00 | 12656.25 | 12909.65 | SL hit (close>static) qty=1.00 sl=13150.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 13155.00 | 12656.25 | 12909.65 | SL hit (close>static) qty=1.00 sl=13150.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 13155.00 | 12656.25 | 12909.65 | SL hit (close>static) qty=1.00 sl=13150.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 12965.00 | 12687.07 | 12914.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 12882.00 | 12695.44 | 12915.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 12237.90 | 12650.61 | 12866.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 12675.00 | 12582.77 | 12815.29 | SL hit (close>ema200) qty=0.50 sl=12582.77 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 12882.00 | 12585.74 | 12815.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 13105.00 | 12590.91 | 12817.07 | SL hit (close>static) qty=1.00 sl=13037.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 12716.00 | 12606.88 | 12821.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 12891.00 | 12611.48 | 12815.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 09:15:00 | 12246.45 | 12611.15 | 12808.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 14:15:00 | 12080.20 | 12590.10 | 12792.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 12539.00 | 12507.56 | 12731.13 | SL hit (close>ema200) qty=0.50 sl=12507.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 12539.00 | 12507.56 | 12731.13 | SL hit (close>ema200) qty=0.50 sl=12507.56 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 12312.00 | 12506.98 | 12726.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 12270.00 | 12505.14 | 12724.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:15:00 | 11656.50 | 12341.20 | 12608.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-02 09:15:00 | 11043.00 | 12015.68 | 12371.09 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-24 09:15:00 | 13013.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-24 09:45:00 | 13029.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-24 10:45:00 | 13055.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-24 12:00:00 | 13044.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-28 13:15:00 | 12882.00 | 2025-11-03 09:15:00 | 12237.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 13:15:00 | 12882.00 | 2025-11-06 10:15:00 | 12675.00 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-11-06 12:00:00 | 12882.00 | 2025-11-06 12:15:00 | 13105.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-07 09:15:00 | 12716.00 | 2025-11-11 09:15:00 | 12246.45 | PARTIAL | 0.50 | 3.69% |
| SELL | retest2 | 2025-11-10 09:45:00 | 12891.00 | 2025-11-11 14:15:00 | 12080.20 | PARTIAL | 0.50 | 6.29% |
| SELL | retest2 | 2025-11-07 09:15:00 | 12716.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-11-10 09:45:00 | 12891.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-11-17 11:15:00 | 12270.00 | 2025-11-21 10:15:00 | 11656.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:15:00 | 12270.00 | 2025-12-02 09:15:00 | 11043.00 | TARGET_HIT | 0.50 | 10.00% |

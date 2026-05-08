# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 11930.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 8 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -3.46% / -3.63%
- **Sum % (uncompounded):** -17.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.46% | -17.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.46% | -17.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.46% | -17.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 11928.00 | 12267.54 | 12268.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 11847.00 | 12155.65 | 12205.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 15:15:00 | 12013.00 | 11843.49 | 11985.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 15:15:00 | 12013.00 | 11843.49 | 11985.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 12013.00 | 11843.49 | 11985.35 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-02 09:15:00 | 11608.00 | 11841.15 | 11983.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 10:15:00 | 11645.00 | 11839.20 | 11981.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-02 11:15:00 | 11635.00 | 11837.17 | 11980.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 12:15:00 | 11641.00 | 11835.22 | 11978.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 09:15:00 | 11559.00 | 11827.65 | 11971.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 11547.00 | 11824.86 | 11969.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-16 09:15:00 | 11619.00 | 11700.85 | 11858.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 11617.00 | 11700.02 | 11857.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 12039.00 | 11712.24 | 11801.98 | SL hit (close>static) qty=1.00 sl=12021.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 12039.00 | 11712.24 | 11801.98 | SL hit (close>static) qty=1.00 sl=12021.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 12284.00 | 11873.02 | 11871.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11873.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12460.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12467.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12467.56 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12298.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10762.00 | 12281.28 | 12291.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.66 | 11709.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11709.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11709.98 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 14:15:00 | 11576.00 | 11374.95 | 11709.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-08 15:15:00 | 11590.00 | 11377.09 | 11708.83 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 11452.00 | 11377.84 | 11707.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 11474.00 | 11378.79 | 11706.39 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 11562.00 | 11394.71 | 11695.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 11325.00 | 11394.02 | 11693.61 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11413.15 | 11688.71 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11413.15 | 11688.71 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 11528.00 | 11703.07 | 11774.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 11523.00 | 11701.27 | 11773.13 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 11779.00 | 11696.99 | 11766.31 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-03 10:15:00 | 11547.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-12-16 10:15:00 | 11617.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-04-09 10:15:00 | 11474.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2026-04-30 10:15:00 | 11523.00 | 2026-05-05 09:15:00 | 11779.00 | STOP_HIT | 1.00 | -2.22% |

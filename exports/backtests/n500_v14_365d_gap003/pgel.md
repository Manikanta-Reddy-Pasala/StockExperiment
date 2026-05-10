# PG Electroplast Ltd. (PGEL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 530.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -8.22% / -7.91%
- **Sum % (uncompounded):** -32.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -8.22% | -32.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -8.22% | -32.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -8.22% | -32.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 628.30 | 578.84 | 578.63 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 533.65 | 580.06 | 580.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 528.85 | 579.55 | 580.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 562.60 | 561.82 | 569.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 562.60 | 561.82 | 569.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 582.45 | 562.04 | 569.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:00:00 | 571.35 | 562.14 | 569.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:45:00 | 578.20 | 562.13 | 569.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 578.60 | 562.13 | 569.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:45:00 | 579.70 | 562.29 | 569.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 569.20 | 565.09 | 570.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| CROSSOVER_SKIP | 2026-02-12 13:15:00 | 624.25 | 575.15 | 575.09 | min_gap filter: gap=0.010% < 0.030% |
| TREND_RESET | 2026-02-12 13:15:00 | 624.25 | 575.15 | 575.09 | EMA inversion without crossover edge (EMA200=575.15 EMA400=575.09) — end cycle |
| CROSSOVER_SKIP | 2026-03-13 11:15:00 | 513.05 | 583.49 | 583.53 | min_gap filter: gap=0.008% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-03 11:00:00 | 571.35 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -9.28% |
| SELL | retest2 | 2026-02-04 09:45:00 | 578.20 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -7.98% |
| SELL | retest2 | 2026-02-04 10:15:00 | 578.60 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -7.91% |
| SELL | retest2 | 2026-02-04 10:45:00 | 579.70 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -7.70% |

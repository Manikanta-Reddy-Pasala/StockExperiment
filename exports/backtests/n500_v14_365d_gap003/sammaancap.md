# Sammaan Capital Ltd. (SAMMAANCAP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 148.78
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 1.32% / -0.39%
- **Sum % (uncompounded):** 6.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.32% | 6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.32% | 6.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.32% | 6.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 116.72 | 126.94 | 126.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 115.56 | 126.83 | 126.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 125.38 | 124.53 | 125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 125.66 | 124.54 | 125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 125.66 | 124.54 | 125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 124.45 | 124.54 | 125.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 123.80 | 124.52 | 125.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 122.89 | 124.51 | 125.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 117.61 | 123.76 | 125.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:15:00 | 116.75 | 123.63 | 124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | SL hit (close>ema200) qty=0.50 sl=123.41 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | SL hit (close>ema200) qty=0.50 sl=123.41 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:30:00 | 123.61 | 123.50 | 124.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 125.91 | 123.55 | 124.83 | SL hit (close>static) qty=1.00 sl=125.87 alert=retest2 |
| CROSSOVER_SKIP | 2025-09-04 12:15:00 | 138.44 | 126.01 | 125.99 | min_gap filter: gap=0.011% < 0.030% |
| TREND_RESET | 2025-09-04 12:15:00 | 138.44 | 126.01 | 125.99 | EMA inversion without crossover edge (EMA200=126.01 EMA400=125.99) — end cycle |
| CROSSOVER_SKIP | 2025-12-12 12:15:00 | 149.13 | 157.24 | 157.26 | min_gap filter: gap=0.015% < 0.030% |
| CROSSOVER_SKIP | 2026-02-26 13:15:00 | 154.32 | 148.80 | 148.78 | min_gap filter: gap=0.013% < 0.030% |
| CROSSOVER_SKIP | 2026-03-04 10:15:00 | 141.00 | 148.72 | 148.75 | min_gap filter: gap=0.018% < 0.030% |
| CROSSOVER_SKIP | 2026-04-13 14:15:00 | 154.10 | 146.78 | 146.76 | min_gap filter: gap=0.015% < 0.030% |
| CROSSOVER_SKIP | 2026-04-29 15:15:00 | 140.90 | 147.13 | 147.15 | min_gap filter: gap=0.016% < 0.030% |
| CROSSOVER_SKIP | 2026-05-07 14:15:00 | 151.74 | 147.18 | 147.16 | min_gap filter: gap=0.014% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-21 13:30:00 | 123.80 | 2025-08-28 09:15:00 | 117.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:00:00 | 122.89 | 2025-08-28 11:15:00 | 116.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 123.80 | 2025-08-29 10:15:00 | 124.28 | STOP_HIT | 0.50 | -0.39% |
| SELL | retest2 | 2025-08-21 15:00:00 | 122.89 | 2025-08-29 10:15:00 | 124.28 | STOP_HIT | 0.50 | -1.13% |
| SELL | retest2 | 2025-09-01 10:30:00 | 123.61 | 2025-09-01 12:15:00 | 125.91 | STOP_HIT | 1.00 | -1.86% |

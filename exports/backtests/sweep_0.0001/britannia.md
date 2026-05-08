# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 5516.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 3 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -2.55% / -2.53%
- **Sum % (uncompounded):** -5.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.55% | -5.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.55% | -5.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.55% | -5.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 5371.50 | 5605.30 | 5605.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 5336.00 | 5598.10 | 5602.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 5592.00 | 5569.88 | 5586.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 10:15:00 | 5592.00 | 5569.88 | 5586.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 5592.00 | 5569.88 | 5586.77 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-22 11:15:00 | 5550.00 | 5578.46 | 5590.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 5547.50 | 5578.15 | 5589.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 5544.50 | 5577.57 | 5589.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-22 15:15:00 | 5568.00 | 5577.48 | 5589.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-25 14:15:00 | 5549.50 | 5577.70 | 5589.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 5549.50 | 5577.42 | 5588.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 5690.00 | 5578.54 | 5589.40 | SL hit (close>static) qty=1.00 sl=5609.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 5690.00 | 5578.54 | 5589.40 | SL hit (close>static) qty=1.00 sl=5609.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-29 09:15:00 | 5786.00 | 5600.45 | 5600.00 | min_gap filter: gap=0.008% < 0.010% |
| TREND_RESET | 2025-08-29 09:15:00 | 5786.00 | 5600.45 | 5600.00 | EMA inversion without crossover edge (EMA200=5600.45 EMA400=5600.00) — end cycle |
| CROSSOVER_SKIP | 2025-12-02 12:15:00 | 5831.00 | 5880.06 | 5880.30 | min_gap filter: gap=0.004% < 0.010% |
| CROSSOVER_SKIP | 2025-12-15 12:15:00 | 6032.00 | 5880.08 | 5879.86 | min_gap filter: gap=0.004% < 0.010% |

### Cycle 2 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.44 | EMA200 below EMA400 |
| CROSSOVER_SKIP | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.53 | min_gap filter: gap=0.006% < 0.010% |
| TREND_RESET | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.53 | EMA inversion without crossover edge (EMA200=5925.92 EMA400=5925.53) — end cycle |
| CROSSOVER_SKIP | 2026-03-16 13:15:00 | 5806.50 | 5955.73 | 5955.85 | min_gap filter: gap=0.002% < 0.010% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-22 12:15:00 | 5547.50 | 2025-08-26 09:15:00 | 5690.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-08-25 15:15:00 | 5549.50 | 2025-08-26 09:15:00 | 5690.00 | STOP_HIT | 1.00 | -2.53% |

# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 12121.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.04% / -2.88%
- **Sum % (uncompounded):** -9.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.04% | -9.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.04% | -9.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.04% | -9.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 10726.00 | 12265.09 | 12265.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 10698.00 | 12234.77 | 12250.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.41 | 11699.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 11736.00 | 11362.16 | 11699.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11362.16 | 11699.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 14:15:00 | 11576.00 | 11374.71 | 11699.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-08 15:15:00 | 11590.00 | 11376.85 | 11698.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 11452.00 | 11377.60 | 11697.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 11478.00 | 11379.55 | 11695.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 11562.00 | 11394.50 | 11685.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 11325.00 | 11393.81 | 11684.10 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11412.96 | 11679.67 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11412.96 | 11679.67 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 11528.00 | 11702.98 | 11768.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 11549.00 | 11699.67 | 11765.83 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 11779.00 | 11696.91 | 11760.49 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-09 11:15:00 | 11478.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2026-04-30 11:15:00 | 11549.00 | 2026-05-05 09:15:00 | 11779.00 | STOP_HIT | 1.00 | -1.99% |

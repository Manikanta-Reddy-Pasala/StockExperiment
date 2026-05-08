# MARUTI (MARUTI)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 13726.00
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
| ALERT3 | 1 |
| PENDING | 5 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -4.44% / -4.10%
- **Sum % (uncompounded):** -17.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.44% | -17.8% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.44% | -17.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.44% | -17.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 14494.00 | 16078.00 | 16080.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 10:15:00 | 14363.00 | 15968.96 | 16025.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13627.00 | 13309.02 | 14079.67 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 13180.00 | 13370.51 | 14035.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 13132.00 | 13368.14 | 14030.76 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-23 10:15:00 | 13183.00 | 13370.61 | 13897.87 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 13219.00 | 13369.10 | 13894.48 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-27 14:15:00 | 13231.00 | 13330.09 | 13831.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 15:15:00 | 13220.00 | 13328.99 | 13828.23 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 10:15:00 | 13232.00 | 13327.31 | 13822.42 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 13132.00 | 13325.36 | 13818.97 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 13746.00 | 13312.44 | 13767.33 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-05-04 11:15:00 | 13525.00 | 13318.57 | 13765.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 13502.00 | 13320.39 | 13764.57 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13744.48 | SL hit (close>ema400) qty=1.00 sl=13744.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13744.48 | SL hit (close>ema400) qty=1.00 sl=13744.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13744.48 | SL hit (close>ema400) qty=1.00 sl=13744.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13744.48 | SL hit (close>ema400) qty=1.00 sl=13744.48 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-13 10:15:00 | 13132.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest1 | 2026-04-23 11:15:00 | 13219.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest1 | 2026-04-27 15:15:00 | 13220.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest1 | 2026-04-28 11:15:00 | 13132.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.79% |

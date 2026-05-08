# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 948.45
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
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -2.86% / -2.79%
- **Sum % (uncompounded):** -8.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.86% | -8.6% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.86% | -8.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.86% | -8.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 823.80 | 887.82 | 888.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 817.30 | 886.50 | 887.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.13 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 12:15:00 | 831.40 | 835.23 | 856.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 13:15:00 | 830.40 | 835.18 | 856.77 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 826.10 | 835.10 | 856.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 828.00 | 835.03 | 856.26 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 14:15:00 | 830.85 | 833.92 | 854.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 15:15:00 | 830.00 | 833.88 | 854.43 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | SL hit (close>ema400) qty=1.00 sl=852.62 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | SL hit (close>ema400) qty=1.00 sl=852.62 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.62 | SL hit (close>ema400) qty=1.00 sl=852.62 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 840.50 | 841.62 | 853.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-24 14:15:00 | 848.85 | 841.69 | 853.22 | ENTRY2 sustain failed after 60m |
| CROSSOVER_SKIP | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.48 | min_gap filter: gap=0.004% < 0.020% |
| TREND_RESET | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.48 | EMA inversion without crossover edge (EMA200=862.52 EMA400=862.48) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-08 13:15:00 | 830.40 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest1 | 2026-04-09 10:15:00 | 828.00 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest1 | 2026-04-10 15:15:00 | 830.00 | 2026-04-16 09:15:00 | 853.15 | STOP_HIT | 1.00 | -2.79% |

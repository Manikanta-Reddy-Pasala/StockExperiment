# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 381.00
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
| ALERT3 | 1 |
| PENDING | 2 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 1 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 1 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 1
- **Target hits / Stop hits / Partials:** 0 / 1 / 0
- **Avg / median % per leg:** -2.47% / -2.47%
- **Sum % (uncompounded):** -2.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.47% | -2.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.47% | -2.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.47% | -2.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 398.62 | 423.59 | 423.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 390.22 | 414.58 | 418.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 399.62 | 397.77 | 404.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 11:15:00 | 404.56 | 397.88 | 404.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 404.56 | 397.88 | 404.80 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 402.20 | 401.56 | 405.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-25 15:15:00 | 402.96 | 401.57 | 405.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-26 09:15:00 | 400.00 | 401.55 | 405.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:15:00 | 397.88 | 401.52 | 405.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 407.70 | 401.12 | 404.65 | SL hit (close>static) qty=1.00 sl=406.10 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-09 10:15:00 | 426.64 | 407.62 | 407.59 | min_gap filter: gap=0.007% < 0.030% |
| TREND_RESET | 2025-10-09 10:15:00 | 426.64 | 407.62 | 407.59 | EMA inversion without crossover edge (EMA200=407.62 EMA400=407.59) — end cycle |
| CROSSOVER_SKIP | 2026-01-29 09:15:00 | 410.40 | 425.22 | 425.27 | min_gap filter: gap=0.011% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-26 10:15:00 | 397.88 | 2025-10-01 10:15:00 | 407.70 | STOP_HIT | 1.00 | -2.47% |

# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 742.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 3 / 3 / 0
- **Avg / median % per leg:** -6.67% / 10.00%
- **Sum % (uncompounded):** -40.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -23.35% | -70.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -23.35% | -70.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -23.35% | -70.0% |
| retest2 (combined) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 10:15:00 | 621.85 | 656.02 | 656.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 610.30 | 653.87 | 654.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 625.35 | 623.60 | 635.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 533.05 | 588.96 | 609.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 621.00 | 588.96 | 609.74 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:15:00 | 617.65 | 588.96 | 609.74 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 724.80 | 590.31 | 610.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 724.80 | 590.31 | 610.31 | SL hit (close>ema400) qty=1.00 sl=610.31 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 09:15:00 | 582.05 | 2025-09-02 15:15:00 | 640.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 15:00:00 | 580.30 | 2025-09-02 15:15:00 | 638.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 10:00:00 | 578.75 | 2025-09-02 15:15:00 | 636.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-02-05 09:15:00 | 533.05 | 2026-05-06 09:15:00 | 724.80 | STOP_HIT | 1.00 | -35.97% |
| SELL | retest1 | 2026-02-18 10:00:00 | 621.00 | 2026-05-06 09:15:00 | 724.80 | STOP_HIT | 1.00 | -16.71% |
| SELL | retest1 | 2026-03-23 09:15:00 | 617.65 | 2026-05-06 09:15:00 | 724.80 | STOP_HIT | 1.00 | -17.35% |

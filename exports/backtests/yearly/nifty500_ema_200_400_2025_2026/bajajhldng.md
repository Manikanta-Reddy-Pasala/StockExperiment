# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1857 bars)
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
| ENTRY1 | 1 |
| ENTRY2 | 14 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 7
- **Target hits / Stop hits / Partials:** 2 / 13 / 8
- **Avg / median % per leg:** 2.48% / 1.61%
- **Sum % (uncompounded):** 57.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 16 | 69.6% | 2 | 13 | 8 | 2.48% | 57.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.71% | -4.7% |
| SELL @ 3rd Alert (retest2) | 22 | 16 | 72.7% | 2 | 12 | 8 | 2.81% | 61.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.71% | -4.7% |
| retest2 (combined) | 22 | 16 | 72.7% | 2 | 12 | 8 | 2.81% | 61.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 12746.00 | 13543.07 | 13546.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 12665.00 | 13284.79 | 13373.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 11:15:00 | 12650.00 | 12589.93 | 12906.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 12511.00 | 12591.71 | 12899.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 13100.00 | 12600.57 | 12890.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 13100.00 | 12600.57 | 12890.54 | SL hit (close>ema400) qty=1.00 sl=12890.54 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-10-20 09:30:00 | 12511.00 | 2025-10-23 09:15:00 | 13100.00 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2025-10-24 09:15:00 | 13013.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-24 09:45:00 | 13029.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-24 10:45:00 | 13055.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-24 12:00:00 | 13044.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-28 13:15:00 | 12882.00 | 2025-11-03 09:15:00 | 12237.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 13:15:00 | 12882.00 | 2025-11-06 10:15:00 | 12675.00 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-11-06 12:00:00 | 12882.00 | 2025-11-06 12:15:00 | 13105.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-07 09:15:00 | 12716.00 | 2025-11-11 09:15:00 | 12246.45 | PARTIAL | 0.50 | 3.69% |
| SELL | retest2 | 2025-11-10 09:45:00 | 12891.00 | 2025-11-11 14:15:00 | 12080.20 | PARTIAL | 0.50 | 6.29% |
| SELL | retest2 | 2025-11-10 11:30:00 | 12650.00 | 2025-11-11 14:15:00 | 12056.45 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2025-11-10 13:00:00 | 12691.00 | 2025-11-12 09:15:00 | 12017.50 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2025-11-10 15:00:00 | 12589.00 | 2025-11-12 09:15:00 | 11959.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 09:15:00 | 12716.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-11-10 09:45:00 | 12891.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-11-10 11:30:00 | 12650.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest2 | 2025-11-10 13:00:00 | 12691.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 1.20% |
| SELL | retest2 | 2025-11-10 15:00:00 | 12589.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2025-11-17 09:30:00 | 12552.00 | 2025-11-18 11:15:00 | 11924.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 09:30:00 | 12552.00 | 2025-11-27 14:15:00 | 11296.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 15:15:00 | 11388.00 | 2026-02-23 12:15:00 | 11502.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-02-24 12:45:00 | 11373.00 | 2026-02-27 09:15:00 | 10804.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 12:45:00 | 11373.00 | 2026-03-09 10:15:00 | 10235.70 | TARGET_HIT | 0.50 | 10.00% |

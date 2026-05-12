# PI Industries Ltd. (PIIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3103.60
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
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 1.38% / -1.32%
- **Sum % (uncompounded):** 6.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.38% | 6.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.38% | -6.8% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.56% | 13.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.38% | -6.8% |
| retest2 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.56% | 13.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 3865.30 | 3963.08 | 3963.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 3826.90 | 3956.79 | 3960.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 3612.40 | 3610.69 | 3696.05 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 14:30:00 | 3559.30 | 3610.08 | 3690.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 12:45:00 | 3560.70 | 3606.88 | 3681.82 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3680.40 | 3606.59 | 3680.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3680.40 | 3606.59 | 3680.19 | SL hit (close>ema400) qty=1.00 sl=3680.19 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-10-28 14:30:00 | 3559.30 | 2025-11-03 09:15:00 | 3680.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest1 | 2025-10-31 12:45:00 | 3560.70 | 2025-11-03 09:15:00 | 3680.40 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-11-04 12:30:00 | 3636.90 | 2025-11-04 14:15:00 | 3684.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-11-12 11:45:00 | 3631.90 | 2025-11-19 09:15:00 | 3450.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:45:00 | 3631.90 | 2025-12-09 09:15:00 | 3268.71 | TARGET_HIT | 0.50 | 10.00% |

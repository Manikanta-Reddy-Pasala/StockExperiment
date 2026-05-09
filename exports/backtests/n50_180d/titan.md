# TITAN (TITAN)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 4517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 0
- **Avg / median % per leg:** 1.60% / -1.90%
- **Sum % (uncompounded):** 9.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.60% | 9.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.60% | 9.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.60% | 9.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

_No CROSSOVER signals fired in window._

## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-28 09:15:00 | 4011.70 | 2026-01-29 09:15:00 | 3891.70 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-02-01 13:00:00 | 4044.90 | 2026-02-01 15:15:00 | 3944.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-03 09:15:00 | 4070.00 | 2026-03-23 09:15:00 | 3946.40 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-25 10:00:00 | 4013.00 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-02 15:00:00 | 4100.40 | 2026-04-10 12:15:00 | 4510.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:15:00 | 4152.60 | 2026-05-08 14:15:00 | 4567.86 | TARGET_HIT | 1.00 | 10.00% |

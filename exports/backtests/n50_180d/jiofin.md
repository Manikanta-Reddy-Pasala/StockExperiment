# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 249.01
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 3
- **Avg / median % per leg:** 4.05% / 5.00%
- **Sum % (uncompounded):** 40.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 6 | 60.0% | 3 | 4 | 3 | 4.05% | 40.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 3 | 4 | 3 | 4.05% | 40.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 6 | 60.0% | 3 | 4 | 3 | 4.05% | 40.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

_No CROSSOVER signals fired in window._

## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-02 14:30:00 | 301.30 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 300.50 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-05 13:00:00 | 301.00 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-06 10:00:00 | 300.80 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-07 13:15:00 | 302.90 | 2026-01-09 14:15:00 | 287.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:15:00 | 303.20 | 2026-01-09 14:15:00 | 288.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 301.70 | 2026-01-09 14:15:00 | 286.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 302.90 | 2026-01-20 09:15:00 | 272.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 14:15:00 | 303.20 | 2026-01-20 09:15:00 | 272.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 301.70 | 2026-01-20 09:15:00 | 271.53 | TARGET_HIT | 0.50 | 10.00% |

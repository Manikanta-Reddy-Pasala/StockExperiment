# Swan Corp Ltd. (SWANCORP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 353.15
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
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 0 / 9 / 4
- **Avg / median % per leg:** 0.79% / -0.61%
- **Sum % (uncompounded):** 10.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 6 | 46.2% | 0 | 9 | 4 | 0.79% | 10.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 0 | 9 | 4 | 0.79% | 10.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 6 | 46.2% | 0 | 9 | 4 | 0.79% | 10.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

_No CROSSOVER signals fired in window._

## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-19 09:30:00 | 450.65 | 2025-05-20 14:15:00 | 430.82 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-05-19 09:30:00 | 450.65 | 2025-05-20 14:15:00 | 430.75 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-05-19 13:00:00 | 453.50 | 2025-05-21 09:15:00 | 428.12 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-05-19 13:00:00 | 453.50 | 2025-05-21 09:15:00 | 434.95 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-06-04 13:30:00 | 453.60 | 2025-06-05 10:15:00 | 463.05 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-06 09:15:00 | 451.75 | 2025-06-09 13:15:00 | 468.00 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-06-13 09:15:00 | 435.00 | 2025-06-19 12:15:00 | 413.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 10:30:00 | 439.95 | 2025-06-19 12:15:00 | 417.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 09:15:00 | 435.00 | 2025-06-24 09:15:00 | 452.45 | STOP_HIT | 0.50 | -4.01% |
| SELL | retest2 | 2025-06-13 10:30:00 | 439.95 | 2025-06-24 09:15:00 | 452.45 | STOP_HIT | 0.50 | -2.84% |
| SELL | retest2 | 2025-07-01 13:00:00 | 439.70 | 2025-07-03 09:15:00 | 442.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-02 09:30:00 | 439.00 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-07-03 09:15:00 | 437.10 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -2.78% |

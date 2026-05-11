# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 902.50
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
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 3.13% / 5.00%
- **Sum % (uncompounded):** 25.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.13% | 25.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.13% | 25.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.13% | 25.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

_No CROSSOVER signals fired in window._

## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-26 11:15:00 | 1274.00 | 2025-06-26 14:15:00 | 1297.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-07-10 12:00:00 | 1279.80 | 2025-07-10 14:15:00 | 1292.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-11 11:00:00 | 1276.70 | 2025-07-14 13:15:00 | 1292.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-11 11:45:00 | 1280.20 | 2025-07-14 13:15:00 | 1292.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1299.80 | 2025-07-24 14:15:00 | 1234.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 1298.60 | 2025-07-24 15:15:00 | 1233.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1299.80 | 2025-08-07 11:15:00 | 1169.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 1298.60 | 2025-08-07 11:15:00 | 1168.74 | TARGET_HIT | 0.50 | 10.00% |

# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2024-10-14 09:15:00 → 2026-05-08 15:15:00 (2706 bars)
- **Last close:** 1649.50
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
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -2.27% / -2.40%
- **Sum % (uncompounded):** -15.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1624.00 | 1447.03 | 1446.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1638.80 | 1448.94 | 1447.26 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 13:45:00 | 2439.90 | 2025-05-27 11:15:00 | 2379.80 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-05-19 14:15:00 | 2438.40 | 2025-05-27 11:15:00 | 2379.80 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-05-20 14:30:00 | 2439.00 | 2025-05-27 11:15:00 | 2379.80 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-05-23 11:00:00 | 2452.90 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-05-26 11:45:00 | 2413.20 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-26 13:45:00 | 2410.00 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-05-26 14:15:00 | 2422.60 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -2.07% |

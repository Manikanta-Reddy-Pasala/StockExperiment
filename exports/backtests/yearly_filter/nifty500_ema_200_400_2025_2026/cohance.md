# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 487.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 0.41% / -3.05%
- **Sum % (uncompounded):** 2.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.41% | 2.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.41% | 2.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.41% | 2.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 474.40 | 374.01 | 373.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 483.75 | 383.33 | 378.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-07 09:15:00 | 1023.85 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-07-07 10:00:00 | 1024.45 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-07-11 10:30:00 | 1023.55 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-07-14 15:15:00 | 1022.50 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-07-24 13:45:00 | 1039.05 | 2025-07-29 10:15:00 | 987.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:45:00 | 1039.05 | 2025-08-06 10:15:00 | 935.14 | TARGET_HIT | 0.50 | 10.00% |

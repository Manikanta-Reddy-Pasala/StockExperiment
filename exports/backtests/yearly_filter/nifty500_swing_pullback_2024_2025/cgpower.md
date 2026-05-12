# CG Power and Industrial Solutions Ltd. (CGPOWER)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 873.55
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
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 3.59% / 6.72%
- **Sum % (uncompounded):** 14.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.59% | 14.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.59% | 14.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.59% | 14.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 00:00:00 | 729.15 | 599.45 | 699.02 | Stage2 pullback-breakout RSI=58 vol=4.4x ATR=24.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 00:00:00 | 778.18 | 610.49 | 725.84 | T1 booked 50% @ 778.18 |
| Stop hit — per-position SL triggered | 2024-10-03 00:00:00 | 729.15 | 619.17 | 740.19 | SL hit (bars_held=14) |

### Cycle 2 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 736.70 | 656.94 | 724.98 | Stage2 pullback-breakout RSI=52 vol=2.5x ATR=28.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 00:00:00 | 792.96 | 666.77 | 749.67 | T1 booked 50% @ 792.96 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 736.70 | 675.68 | 759.75 | SL hit (bars_held=19) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-12 00:00:00 | 729.15 | 2024-09-24 00:00:00 | 778.18 | PARTIAL | 0.50 | 6.72% |
| BUY | retest1 | 2024-09-12 00:00:00 | 729.15 | 2024-10-03 00:00:00 | 729.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 736.70 | 2024-12-09 00:00:00 | 792.96 | PARTIAL | 0.50 | 7.64% |
| BUY | retest1 | 2024-11-25 00:00:00 | 736.70 | 2024-12-20 00:00:00 | 736.70 | STOP_HIT | 0.50 | 0.00% |

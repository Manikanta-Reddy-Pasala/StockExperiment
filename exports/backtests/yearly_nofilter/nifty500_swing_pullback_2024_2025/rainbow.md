# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1304.50
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 1.38% / 1.77%
- **Sum % (uncompounded):** 4.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 1.38% | 4.1% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 1.38% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 1.38% | 4.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 00:00:00 | 1497.65 | 1271.84 | 1405.72 | Stage2 pullback-breakout RSI=65 vol=4.1x ATR=52.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 00:00:00 | 1602.12 | 1279.21 | 1436.77 | T1 booked 50% @ 1602.12 |
| Target hit | 2024-11-28 00:00:00 | 1524.15 | 1330.72 | 1566.44 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-01-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 00:00:00 | 1635.90 | 1391.08 | 1568.59 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=50.37 |
| Stop hit — per-position SL triggered | 2025-01-10 00:00:00 | 1560.35 | 1396.84 | 1572.62 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-28 00:00:00 | 1497.65 | 2024-10-31 00:00:00 | 1602.12 | PARTIAL | 0.50 | 6.98% |
| BUY | retest1 | 2024-10-28 00:00:00 | 1497.65 | 2024-11-28 00:00:00 | 1524.15 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2025-01-07 00:00:00 | 1635.90 | 2025-01-10 00:00:00 | 1560.35 | STOP_HIT | 1.00 | -4.62% |

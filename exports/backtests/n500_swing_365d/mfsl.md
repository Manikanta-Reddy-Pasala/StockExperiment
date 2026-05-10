# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1699.70
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 1.99% / 4.26%
- **Sum % (uncompounded):** 9.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.99% | 10.0% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.99% | 10.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.99% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 05:30:00 | 1551.50 | 1325.09 | 1523.59 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=34.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 05:30:00 | 1620.30 | 1338.24 | 1552.82 | T1 booked 50% @ 1620.30 |
| Stop hit — per-position SL triggered | 2025-09-04 05:30:00 | 1551.50 | 1370.60 | 1598.90 | SL hit (bars_held=17) |

### Cycle 2 — BUY (started 2025-11-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 05:30:00 | 1592.40 | 1437.10 | 1557.40 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=33.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 05:30:00 | 1660.23 | 1446.79 | 1590.30 | T1 booked 50% @ 1660.23 |
| Target hit | 2025-12-03 05:30:00 | 1664.80 | 1481.12 | 1671.29 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 1703.70 | 1521.01 | 1678.09 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=37.28 |
| Stop hit — per-position SL triggered | 2026-01-06 05:30:00 | 1647.79 | 1523.11 | 1683.16 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-08 05:30:00 | 1551.50 | 2025-08-18 05:30:00 | 1620.30 | PARTIAL | 0.50 | 4.43% |
| BUY | retest1 | 2025-08-08 05:30:00 | 1551.50 | 2025-09-04 05:30:00 | 1551.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 05:30:00 | 1592.40 | 2025-11-12 05:30:00 | 1660.23 | PARTIAL | 0.50 | 4.26% |
| BUY | retest1 | 2025-11-04 05:30:00 | 1592.40 | 2025-12-03 05:30:00 | 1664.80 | TARGET_HIT | 0.50 | 4.55% |
| BUY | retest1 | 2026-01-05 05:30:00 | 1703.70 | 2026-01-06 05:30:00 | 1647.79 | STOP_HIT | 1.00 | -3.28% |

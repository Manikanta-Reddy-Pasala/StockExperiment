# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2025-09-03 05:30:00 (745 bars)
- **Last close:** 1674.80
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
- **Avg / median % per leg:** 4.84% / 6.13%
- **Sum % (uncompounded):** 24.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.84% | 24.2% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.84% | 24.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.84% | 24.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 05:30:00 | 1397.05 | 1190.64 | 1326.04 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=42.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 05:30:00 | 1482.63 | 1197.51 | 1351.39 | T1 booked 50% @ 1482.63 |
| Target hit | 2023-09-12 05:30:00 | 1624.00 | 1320.25 | 1662.41 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 05:30:00 | 1669.35 | 1370.32 | 1620.48 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=54.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 05:30:00 | 1777.82 | 1384.08 | 1654.80 | T1 booked 50% @ 1777.82 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 1669.35 | 1396.57 | 1669.27 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-02-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 05:30:00 | 1547.55 | 1475.40 | 1453.46 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=48.13 |
| Stop hit — per-position SL triggered | 2024-03-13 05:30:00 | 1475.35 | 1482.28 | 1514.20 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-19 05:30:00 | 1397.05 | 2023-07-24 05:30:00 | 1482.63 | PARTIAL | 0.50 | 6.13% |
| BUY | retest1 | 2023-07-19 05:30:00 | 1397.05 | 2023-09-12 05:30:00 | 1624.00 | TARGET_HIT | 0.50 | 16.24% |
| BUY | retest1 | 2023-10-11 05:30:00 | 1669.35 | 2023-10-17 05:30:00 | 1777.82 | PARTIAL | 0.50 | 6.50% |
| BUY | retest1 | 2023-10-11 05:30:00 | 1669.35 | 2023-10-23 05:30:00 | 1669.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-29 05:30:00 | 1547.55 | 2024-03-13 05:30:00 | 1475.35 | STOP_HIT | 1.00 | -4.67% |

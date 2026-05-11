# Caplin Point Laboratories Ltd. (CAPLIPOINT)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1856.30
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 4.09% / 7.75%
- **Sum % (uncompounded):** 24.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 4.09% | 24.5% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 4.09% | 24.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 4.09% | 24.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 1478.20 | 1296.25 | 1389.60 | Stage2 pullback-breakout RSI=65 vol=3.6x ATR=49.00 |
| Stop hit — per-position SL triggered | 2024-07-11 00:00:00 | 1517.10 | 1312.61 | 1443.35 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-10-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 00:00:00 | 2078.20 | 1515.47 | 1931.61 | Stage2 pullback-breakout RSI=64 vol=4.5x ATR=83.43 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 1953.05 | 1524.78 | 1941.22 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 00:00:00 | 1979.50 | 1584.58 | 1890.99 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=78.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 00:00:00 | 2135.96 | 1611.42 | 1963.20 | T1 booked 50% @ 2135.96 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1979.50 | 1626.54 | 1973.68 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2024-11-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 00:00:00 | 2113.95 | 1650.76 | 1982.67 | Stage2 pullback-breakout RSI=63 vol=3.6x ATR=81.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 00:00:00 | 2277.78 | 1670.58 | 2041.98 | T1 booked 50% @ 2277.78 |
| Target hit | 2025-01-10 00:00:00 | 2373.05 | 1860.63 | 2451.80 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 1478.20 | 2024-07-11 00:00:00 | 1517.10 | STOP_HIT | 1.00 | 2.63% |
| BUY | retest1 | 2024-10-01 00:00:00 | 2078.20 | 2024-10-04 00:00:00 | 1953.05 | STOP_HIT | 1.00 | -6.02% |
| BUY | retest1 | 2024-10-30 00:00:00 | 1979.50 | 2024-11-07 00:00:00 | 2135.96 | PARTIAL | 0.50 | 7.90% |
| BUY | retest1 | 2024-10-30 00:00:00 | 1979.50 | 2024-11-13 00:00:00 | 1979.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 00:00:00 | 2113.95 | 2024-12-02 00:00:00 | 2277.78 | PARTIAL | 0.50 | 7.75% |
| BUY | retest1 | 2024-11-26 00:00:00 | 2113.95 | 2025-01-10 00:00:00 | 2373.05 | TARGET_HIT | 0.50 | 12.26% |

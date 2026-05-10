# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1818.30
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 2.51% / 3.36%
- **Sum % (uncompounded):** 10.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 1 | 1 | 2 | 2.51% | 10.0% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 2.51% | 10.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 2.51% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 05:30:00 | 2002.20 | 1920.88 | 1955.96 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=35.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 05:30:00 | 2073.25 | 1927.50 | 1992.99 | T1 booked 50% @ 2073.25 |
| Target hit | 2025-09-25 05:30:00 | 2035.60 | 1939.58 | 2036.47 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 05:30:00 | 2084.10 | 1949.48 | 2027.90 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=35.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-21 05:30:00 | 2154.14 | 1956.20 | 2059.74 | T1 booked 50% @ 2154.14 |
| Stop hit — per-position SL triggered | 2025-10-30 05:30:00 | 2114.60 | 1967.45 | 2098.80 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-04 05:30:00 | 2002.20 | 2025-09-12 05:30:00 | 2073.25 | PARTIAL | 0.50 | 3.55% |
| BUY | retest1 | 2025-09-04 05:30:00 | 2002.20 | 2025-09-25 05:30:00 | 2035.60 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-10-15 05:30:00 | 2084.10 | 2025-10-21 05:30:00 | 2154.14 | PARTIAL | 0.50 | 3.36% |
| BUY | retest1 | 2025-10-15 05:30:00 | 2084.10 | 2025-10-30 05:30:00 | 2114.60 | STOP_HIT | 0.50 | 1.46% |

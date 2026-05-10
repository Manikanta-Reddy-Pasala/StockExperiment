# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1954.90
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
- **Avg / median % per leg:** 5.32% / 7.31%
- **Sum % (uncompounded):** 26.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 5.32% | 26.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 5.32% | 26.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 5.32% | 26.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 05:30:00 | 2066.90 | 1520.50 | 1870.25 | Stage2 pullback-breakout RSI=69 vol=8.1x ATR=91.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 05:30:00 | 2250.33 | 1551.54 | 1985.92 | T1 booked 50% @ 2250.33 |
| Stop hit — per-position SL triggered | 2025-08-07 05:30:00 | 2066.90 | 1569.52 | 2029.83 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 2540.60 | 1785.67 | 2424.70 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=128.75 |
| Stop hit — per-position SL triggered | 2025-10-06 05:30:00 | 2493.90 | 1849.54 | 2448.79 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 2665.40 | 1934.43 | 2497.53 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=97.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 05:30:00 | 2860.25 | 1946.38 | 2558.26 | T1 booked 50% @ 2860.25 |
| Target hit | 2025-12-02 05:30:00 | 2991.50 | 2183.51 | 3025.42 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-28 05:30:00 | 2066.90 | 2025-08-04 05:30:00 | 2250.33 | PARTIAL | 0.50 | 8.87% |
| BUY | retest1 | 2025-07-28 05:30:00 | 2066.90 | 2025-08-07 05:30:00 | 2066.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 05:30:00 | 2540.60 | 2025-10-06 05:30:00 | 2493.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2025-10-27 05:30:00 | 2665.40 | 2025-10-28 05:30:00 | 2860.25 | PARTIAL | 0.50 | 7.31% |
| BUY | retest1 | 2025-10-27 05:30:00 | 2665.40 | 2025-12-02 05:30:00 | 2991.50 | TARGET_HIT | 0.50 | 12.23% |

# Dalmia Bharat Ltd. (DALBHARAT)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1823.60
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
- **Avg / median % per leg:** 2.52% / 4.24%
- **Sum % (uncompounded):** 12.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.52% | 12.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.52% | 12.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.52% | 12.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 05:30:00 | 2082.80 | 1937.17 | 1995.92 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=44.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 05:30:00 | 2171.16 | 1943.67 | 2039.11 | T1 booked 50% @ 2171.16 |
| Target hit | 2023-09-22 05:30:00 | 2245.60 | 1990.25 | 2245.85 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 05:30:00 | 2207.65 | 2050.96 | 2134.67 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=55.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 05:30:00 | 2318.45 | 2067.00 | 2193.35 | T1 booked 50% @ 2318.45 |
| Stop hit — per-position SL triggered | 2023-12-20 05:30:00 | 2207.65 | 2098.06 | 2290.50 | SL hit (bars_held=22) |

### Cycle 3 — BUY (started 2024-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 05:30:00 | 2390.70 | 2119.26 | 2306.88 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=71.15 |
| Stop hit — per-position SL triggered | 2024-01-10 05:30:00 | 2283.98 | 2124.85 | 2306.96 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-30 05:30:00 | 2082.80 | 2023-09-04 05:30:00 | 2171.16 | PARTIAL | 0.50 | 4.24% |
| BUY | retest1 | 2023-08-30 05:30:00 | 2082.80 | 2023-09-22 05:30:00 | 2245.60 | TARGET_HIT | 0.50 | 7.82% |
| BUY | retest1 | 2023-11-17 05:30:00 | 2207.65 | 2023-12-04 05:30:00 | 2318.45 | PARTIAL | 0.50 | 5.02% |
| BUY | retest1 | 2023-11-17 05:30:00 | 2207.65 | 2023-12-20 05:30:00 | 2207.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-05 05:30:00 | 2390.70 | 2024-01-10 05:30:00 | 2283.98 | STOP_HIT | 1.00 | -4.46% |

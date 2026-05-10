# Dalmia Bharat Ltd. (DALBHARAT)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.08% / -0.92%
- **Sum % (uncompounded):** 0.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.08% | 0.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.08% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.08% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 2191.80 | 1908.95 | 2083.53 | Stage2 pullback-breakout RSI=68 vol=2.7x ATR=50.32 |
| Stop hit — per-position SL triggered | 2025-07-10 05:30:00 | 2171.70 | 1935.36 | 2146.03 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 05:30:00 | 2251.80 | 1950.05 | 2166.42 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=49.22 |
| Stop hit — per-position SL triggered | 2025-08-01 05:30:00 | 2204.40 | 1977.91 | 2210.66 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 2330.80 | 2004.62 | 2244.57 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=55.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 05:30:00 | 2441.06 | 2053.35 | 2352.16 | T1 booked 50% @ 2441.06 |
| Target hit | 2025-09-22 05:30:00 | 2378.50 | 2088.15 | 2394.53 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-01-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 05:30:00 | 2177.30 | 2091.96 | 2097.17 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=48.86 |
| Stop hit — per-position SL triggered | 2026-01-23 05:30:00 | 2104.01 | 2096.02 | 2124.79 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 2191.80 | 2025-07-10 05:30:00 | 2171.70 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest1 | 2025-07-18 05:30:00 | 2251.80 | 2025-08-01 05:30:00 | 2204.40 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest1 | 2025-08-18 05:30:00 | 2330.80 | 2025-09-08 05:30:00 | 2441.06 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2025-08-18 05:30:00 | 2330.80 | 2025-09-22 05:30:00 | 2378.50 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2026-01-14 05:30:00 | 2177.30 | 2026-01-23 05:30:00 | 2104.01 | STOP_HIT | 1.00 | -3.37% |

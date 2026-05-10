# Hyundai Motor India Ltd. (HYUNDAI)

## Backtest Summary

- **Window:** 2024-10-22 05:30:00 → 2026-05-08 05:30:00 (382 bars)
- **Last close:** 1852.80
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
- **Avg / median % per leg:** 1.92% / -2.39%
- **Sum % (uncompounded):** 9.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.92% | 9.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.92% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.92% | 9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 05:30:00 | 2246.70 | 1901.48 | 2147.33 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=58.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 05:30:00 | 2363.80 | 1910.02 | 2181.67 | T1 booked 50% @ 2363.80 |
| Target hit | 2025-09-30 05:30:00 | 2584.40 | 2082.10 | 2608.94 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 05:30:00 | 2413.70 | 2136.88 | 2392.41 | Stage2 pullback-breakout RSI=50 vol=2.0x ATR=73.81 |
| Stop hit — per-position SL triggered | 2025-11-14 05:30:00 | 2356.10 | 2160.16 | 2383.79 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 05:30:00 | 2395.70 | 2179.52 | 2356.25 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=64.93 |
| Stop hit — per-position SL triggered | 2025-12-08 05:30:00 | 2298.30 | 2187.56 | 2349.49 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2026-02-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 05:30:00 | 2293.40 | 2214.51 | 2212.01 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=63.84 |
| Stop hit — per-position SL triggered | 2026-02-24 05:30:00 | 2197.64 | 2214.66 | 2213.32 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-13 05:30:00 | 2246.70 | 2025-08-18 05:30:00 | 2363.80 | PARTIAL | 0.50 | 5.21% |
| BUY | retest1 | 2025-08-13 05:30:00 | 2246.70 | 2025-09-30 05:30:00 | 2584.40 | TARGET_HIT | 0.50 | 15.03% |
| BUY | retest1 | 2025-10-30 05:30:00 | 2413.70 | 2025-11-14 05:30:00 | 2356.10 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2025-12-01 05:30:00 | 2395.70 | 2025-12-08 05:30:00 | 2298.30 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2026-02-20 05:30:00 | 2293.40 | 2026-02-24 05:30:00 | 2197.64 | STOP_HIT | 1.00 | -4.18% |

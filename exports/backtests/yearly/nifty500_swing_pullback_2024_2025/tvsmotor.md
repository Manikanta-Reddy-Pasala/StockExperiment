# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 3695.20
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** 0.49% / 0.82%
- **Sum % (uncompounded):** 2.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.49% | 2.4% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.49% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.49% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 05:30:00 | 2508.00 | 2075.06 | 2428.90 | Stage2 pullback-breakout RSI=63 vol=1.5x ATR=64.92 |
| Stop hit — per-position SL triggered | 2024-08-09 05:30:00 | 2581.45 | 2118.62 | 2497.15 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-12-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 05:30:00 | 2556.20 | 2391.88 | 2471.03 | Stage2 pullback-breakout RSI=57 vol=2.3x ATR=66.72 |
| Stop hit — per-position SL triggered | 2024-12-17 05:30:00 | 2456.11 | 2403.29 | 2495.52 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2025-01-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 05:30:00 | 2501.45 | 2405.15 | 2445.39 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=63.52 |
| Stop hit — per-position SL triggered | 2025-01-06 05:30:00 | 2406.17 | 2406.00 | 2445.62 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 05:30:00 | 2464.75 | 2389.10 | 2330.63 | Stage2 pullback-breakout RSI=60 vol=3.6x ATR=78.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 05:30:00 | 2622.48 | 2392.08 | 2372.69 | T1 booked 50% @ 2622.48 |
| Stop hit — per-position SL triggered | 2025-02-11 05:30:00 | 2485.00 | 2405.64 | 2478.94 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 05:30:00 | 2508.00 | 2024-08-09 05:30:00 | 2581.45 | STOP_HIT | 1.00 | 2.93% |
| BUY | retest1 | 2024-12-03 05:30:00 | 2556.20 | 2024-12-17 05:30:00 | 2456.11 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2025-01-02 05:30:00 | 2501.45 | 2025-01-06 05:30:00 | 2406.17 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest1 | 2025-01-29 05:30:00 | 2464.75 | 2025-02-01 05:30:00 | 2622.48 | PARTIAL | 0.50 | 6.40% |
| BUY | retest1 | 2025-01-29 05:30:00 | 2464.75 | 2025-02-11 05:30:00 | 2485.00 | STOP_HIT | 0.50 | 0.82% |

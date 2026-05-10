# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 2424.80
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 3
- **Avg / median % per leg:** 2.25% / 0.00%
- **Sum % (uncompounded):** 15.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 4 | 3 | 2.25% | 15.8% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 4 | 3 | 2.25% | 15.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 4 | 3 | 2.25% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 2867.00 | 2307.60 | 2749.90 | Stage2 pullback-breakout RSI=61 vol=2.8x ATR=98.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 05:30:00 | 3064.77 | 2314.63 | 2775.08 | T1 booked 50% @ 3064.77 |
| Stop hit — per-position SL triggered | 2025-07-07 05:30:00 | 2867.00 | 2350.58 | 2842.73 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2025-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 05:30:00 | 3017.00 | 2374.94 | 2849.25 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=124.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 05:30:00 | 3266.60 | 2382.35 | 2875.04 | T1 booked 50% @ 3266.60 |
| Stop hit — per-position SL triggered | 2025-07-24 05:30:00 | 3017.00 | 2429.46 | 2975.15 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2025-08-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 05:30:00 | 3291.50 | 2475.21 | 3011.25 | Stage2 pullback-breakout RSI=63 vol=3.7x ATR=144.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 05:30:00 | 3579.62 | 2486.58 | 3069.05 | T1 booked 50% @ 3579.62 |
| Stop hit — per-position SL triggered | 2025-08-07 05:30:00 | 3291.50 | 2495.00 | 3094.18 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-08-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 05:30:00 | 3688.00 | 2560.13 | 3239.66 | Stage2 pullback-breakout RSI=65 vol=4.7x ATR=200.28 |
| Stop hit — per-position SL triggered | 2025-09-02 05:30:00 | 3387.58 | 2636.57 | 3405.56 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 2867.00 | 2025-06-27 05:30:00 | 3064.77 | PARTIAL | 0.50 | 6.90% |
| BUY | retest1 | 2025-06-26 05:30:00 | 2867.00 | 2025-07-07 05:30:00 | 2867.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 05:30:00 | 3017.00 | 2025-07-15 05:30:00 | 3266.60 | PARTIAL | 0.50 | 8.27% |
| BUY | retest1 | 2025-07-14 05:30:00 | 3017.00 | 2025-07-24 05:30:00 | 3017.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-05 05:30:00 | 3291.50 | 2025-08-06 05:30:00 | 3579.62 | PARTIAL | 0.50 | 8.75% |
| BUY | retest1 | 2025-08-05 05:30:00 | 3291.50 | 2025-08-07 05:30:00 | 3291.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 05:30:00 | 3688.00 | 2025-09-02 05:30:00 | 3387.58 | STOP_HIT | 1.00 | -8.15% |

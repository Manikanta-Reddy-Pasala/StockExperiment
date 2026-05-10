# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 2.18% / 0.00%
- **Sum % (uncompounded):** 17.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 2.18% | 17.4% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 2.18% | 17.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 5 | 2 | 2.18% | 17.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 2837.10 | 2616.51 | 2774.41 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=65.08 |
| Stop hit — per-position SL triggered | 2025-07-08 05:30:00 | 2830.70 | 2643.25 | 2847.85 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 05:30:00 | 2858.20 | 2672.62 | 2822.43 | Stage2 pullback-breakout RSI=56 vol=3.6x ATR=62.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 05:30:00 | 2982.76 | 2678.37 | 2848.09 | T1 booked 50% @ 2982.76 |
| Target hit | 2025-09-25 05:30:00 | 3408.40 | 2875.02 | 3433.57 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 05:30:00 | 3574.90 | 2949.25 | 3478.41 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=66.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 05:30:00 | 3707.66 | 2982.34 | 3537.40 | T1 booked 50% @ 3707.66 |
| Stop hit — per-position SL triggered | 2025-10-28 05:30:00 | 3574.90 | 3000.61 | 3553.44 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2025-12-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 05:30:00 | 3661.80 | 3100.25 | 3505.60 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=72.38 |
| Stop hit — per-position SL triggered | 2025-12-09 05:30:00 | 3553.23 | 3131.68 | 3565.18 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2026-01-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 05:30:00 | 3847.80 | 3215.52 | 3665.59 | Stage2 pullback-breakout RSI=69 vol=1.8x ATR=73.24 |
| Stop hit — per-position SL triggered | 2026-01-12 05:30:00 | 3737.93 | 3250.35 | 3729.35 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 3728.40 | 3287.81 | 3670.43 | Stage2 pullback-breakout RSI=55 vol=2.7x ATR=95.34 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 3585.39 | 3298.27 | 3662.53 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 2837.10 | 2025-07-08 05:30:00 | 2830.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-01 05:30:00 | 2858.20 | 2025-08-05 05:30:00 | 2982.76 | PARTIAL | 0.50 | 4.36% |
| BUY | retest1 | 2025-08-01 05:30:00 | 2858.20 | 2025-09-25 05:30:00 | 3408.40 | TARGET_HIT | 0.50 | 19.25% |
| BUY | retest1 | 2025-10-15 05:30:00 | 3574.90 | 2025-10-23 05:30:00 | 3707.66 | PARTIAL | 0.50 | 3.71% |
| BUY | retest1 | 2025-10-15 05:30:00 | 3574.90 | 2025-10-28 05:30:00 | 3574.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 05:30:00 | 3661.80 | 2025-12-09 05:30:00 | 3553.23 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest1 | 2026-01-02 05:30:00 | 3847.80 | 2026-01-12 05:30:00 | 3737.93 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest1 | 2026-01-28 05:30:00 | 3728.40 | 2026-02-01 05:30:00 | 3585.39 | STOP_HIT | 1.00 | -3.84% |

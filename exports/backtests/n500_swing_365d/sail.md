# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 184.88
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 2.66% / 4.56%
- **Sum % (uncompounded):** 18.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 2 | 3 | 2 | 2.66% | 18.6% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 2 | 3 | 2 | 2.66% | 18.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 2 | 3 | 2 | 2.66% | 18.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 133.62 | 120.51 | 128.69 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-07-10 05:30:00 | 134.78 | 121.83 | 132.32 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 129.68 | 123.10 | 123.53 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=3.23 |
| Stop hit — per-position SL triggered | 2025-09-17 05:30:00 | 132.75 | 123.86 | 128.53 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 132.16 | 125.89 | 131.09 | Stage2 pullback-breakout RSI=53 vol=3.1x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 138.85 | 126.04 | 131.99 | T1 booked 50% @ 138.85 |
| Target hit | 2025-11-20 05:30:00 | 138.19 | 128.04 | 138.72 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 141.02 | 129.00 | 132.66 | Stage2 pullback-breakout RSI=66 vol=2.8x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 05:30:00 | 148.83 | 129.18 | 134.03 | T1 booked 50% @ 148.83 |
| Target hit | 2026-02-01 05:30:00 | 148.63 | 133.09 | 148.68 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 164.93 | 137.29 | 157.45 | Stage2 pullback-breakout RSI=63 vol=2.6x ATR=5.66 |
| Stop hit — per-position SL triggered | 2026-03-04 05:30:00 | 156.44 | 138.30 | 159.19 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 133.62 | 2025-07-10 05:30:00 | 134.78 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest1 | 2025-09-03 05:30:00 | 129.68 | 2025-09-17 05:30:00 | 132.75 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest1 | 2025-10-28 05:30:00 | 132.16 | 2025-10-29 05:30:00 | 138.85 | PARTIAL | 0.50 | 5.06% |
| BUY | retest1 | 2025-10-28 05:30:00 | 132.16 | 2025-11-20 05:30:00 | 138.19 | TARGET_HIT | 0.50 | 4.56% |
| BUY | retest1 | 2025-12-30 05:30:00 | 141.02 | 2025-12-31 05:30:00 | 148.83 | PARTIAL | 0.50 | 5.54% |
| BUY | retest1 | 2025-12-30 05:30:00 | 141.02 | 2026-02-01 05:30:00 | 148.63 | TARGET_HIT | 0.50 | 5.40% |
| BUY | retest1 | 2026-02-25 05:30:00 | 164.93 | 2026-03-04 05:30:00 | 156.44 | STOP_HIT | 1.00 | -5.15% |

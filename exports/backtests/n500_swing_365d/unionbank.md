# Union Bank of India (UNIONBANK)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 166.24
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 3.10% / 4.32%
- **Sum % (uncompounded):** 24.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.10% | 24.8% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.10% | 24.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.10% | 24.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 153.60 | 127.07 | 146.42 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=4.50 |
| Stop hit — per-position SL triggered | 2025-07-09 05:30:00 | 146.85 | 128.71 | 148.66 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2025-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 05:30:00 | 137.03 | 130.87 | 132.33 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-10-01 05:30:00 | 138.75 | 131.55 | 135.87 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 05:30:00 | 141.33 | 132.31 | 137.71 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 05:30:00 | 148.40 | 133.09 | 140.88 | T1 booked 50% @ 148.40 |
| Target hit | 2025-12-03 05:30:00 | 151.38 | 137.17 | 152.03 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 162.36 | 140.23 | 153.45 | Stage2 pullback-breakout RSI=70 vol=2.9x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 05:30:00 | 169.24 | 142.01 | 160.35 | T1 booked 50% @ 169.24 |
| Target hit | 2026-02-01 05:30:00 | 169.37 | 145.55 | 170.84 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-02-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 05:30:00 | 188.89 | 149.36 | 177.51 | Stage2 pullback-breakout RSI=69 vol=1.8x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 05:30:00 | 200.35 | 151.59 | 184.59 | T1 booked 50% @ 200.35 |
| Stop hit — per-position SL triggered | 2026-03-04 05:30:00 | 191.13 | 153.87 | 189.89 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 153.60 | 2025-07-09 05:30:00 | 146.85 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest1 | 2025-09-17 05:30:00 | 137.03 | 2025-10-01 05:30:00 | 138.75 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest1 | 2025-10-20 05:30:00 | 141.33 | 2025-10-30 05:30:00 | 148.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-20 05:30:00 | 141.33 | 2025-12-03 05:30:00 | 151.38 | TARGET_HIT | 0.50 | 7.11% |
| BUY | retest1 | 2026-01-05 05:30:00 | 162.36 | 2026-01-14 05:30:00 | 169.24 | PARTIAL | 0.50 | 4.24% |
| BUY | retest1 | 2026-01-05 05:30:00 | 162.36 | 2026-02-01 05:30:00 | 169.37 | TARGET_HIT | 0.50 | 4.32% |
| BUY | retest1 | 2026-02-17 05:30:00 | 188.89 | 2026-02-24 05:30:00 | 200.35 | PARTIAL | 0.50 | 6.07% |
| BUY | retest1 | 2026-02-17 05:30:00 | 188.89 | 2026-03-04 05:30:00 | 191.13 | STOP_HIT | 0.50 | 1.19% |

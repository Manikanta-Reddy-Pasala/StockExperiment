# KEI Industries Ltd. (KEI)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 5099.60
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
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 1.65% / 3.86%
- **Sum % (uncompounded):** 11.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 3 | 2 | 1.65% | 11.6% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 3 | 2 | 1.65% | 11.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 3 | 2 | 1.65% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 05:30:00 | 3946.60 | 3755.42 | 3849.06 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=97.35 |
| Stop hit — per-position SL triggered | 2025-08-29 05:30:00 | 3800.58 | 3765.95 | 3875.20 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2025-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 05:30:00 | 4125.80 | 3775.46 | 3921.71 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=100.09 |
| Stop hit — per-position SL triggered | 2025-09-18 05:30:00 | 4149.40 | 3806.27 | 4040.77 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 05:30:00 | 4167.90 | 3944.52 | 4103.14 | Stage2 pullback-breakout RSI=55 vol=2.6x ATR=99.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 05:30:00 | 4366.05 | 3957.79 | 4153.36 | T1 booked 50% @ 4366.05 |
| Target hit | 2026-01-09 05:30:00 | 4328.80 | 4017.06 | 4367.78 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 4367.00 | 4028.21 | 4128.92 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=164.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 05:30:00 | 4696.55 | 4093.17 | 4457.37 | T1 booked 50% @ 4696.55 |
| Target hit | 2026-03-10 05:30:00 | 4537.00 | 4177.39 | 4741.65 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-05-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 05:30:00 | 5058.30 | 4269.24 | 4731.77 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=180.89 |
| Stop hit — per-position SL triggered | 2026-05-05 05:30:00 | 4786.96 | 4276.69 | 4759.05 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-19 05:30:00 | 3946.60 | 2025-08-29 05:30:00 | 3800.58 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest1 | 2025-09-04 05:30:00 | 4125.80 | 2025-09-18 05:30:00 | 4149.40 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest1 | 2025-12-15 05:30:00 | 4167.90 | 2025-12-22 05:30:00 | 4366.05 | PARTIAL | 0.50 | 4.75% |
| BUY | retest1 | 2025-12-15 05:30:00 | 4167.90 | 2026-01-09 05:30:00 | 4328.80 | TARGET_HIT | 0.50 | 3.86% |
| BUY | retest1 | 2026-02-03 05:30:00 | 4367.00 | 2026-02-20 05:30:00 | 4696.55 | PARTIAL | 0.50 | 7.55% |
| BUY | retest1 | 2026-02-03 05:30:00 | 4367.00 | 2026-03-10 05:30:00 | 4537.00 | TARGET_HIT | 0.50 | 3.89% |
| BUY | retest1 | 2026-05-04 05:30:00 | 5058.30 | 2026-05-05 05:30:00 | 4786.96 | STOP_HIT | 1.00 | -5.36% |

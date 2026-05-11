# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 4522.70
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.76% / 2.57%
- **Sum % (uncompounded):** 12.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.76% | 12.3% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.76% | 12.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.76% | 12.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 05:30:00 | 4385.85 | 3494.23 | 4287.86 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=97.43 |
| Stop hit — per-position SL triggered | 2024-07-23 05:30:00 | 4239.70 | 3536.24 | 4312.44 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-08-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 05:30:00 | 4483.15 | 3686.98 | 4314.73 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=104.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 05:30:00 | 4691.68 | 3697.16 | 4352.42 | T1 booked 50% @ 4691.68 |
| Target hit | 2024-09-25 05:30:00 | 4782.20 | 3934.59 | 4817.45 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 05:30:00 | 4352.65 | 4068.81 | 4165.51 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=119.34 |
| Stop hit — per-position SL triggered | 2024-12-12 05:30:00 | 4464.35 | 4103.26 | 4338.81 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-12-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 05:30:00 | 4612.25 | 4129.02 | 4398.77 | Stage2 pullback-breakout RSI=69 vol=2.3x ATR=117.75 |
| Stop hit — per-position SL triggered | 2025-01-06 05:30:00 | 4435.62 | 4161.42 | 4468.37 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2025-03-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 05:30:00 | 4698.10 | 4215.81 | 4443.56 | Stage2 pullback-breakout RSI=68 vol=1.6x ATR=131.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 05:30:00 | 4961.60 | 4261.43 | 4633.96 | T1 booked 50% @ 4961.60 |
| Stop hit — per-position SL triggered | 2025-04-07 05:30:00 | 4698.10 | 4350.26 | 4922.14 | SL hit (bars_held=21) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 05:30:00 | 4385.85 | 2024-07-23 05:30:00 | 4239.70 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2024-08-22 05:30:00 | 4483.15 | 2024-08-23 05:30:00 | 4691.68 | PARTIAL | 0.50 | 4.65% |
| BUY | retest1 | 2024-08-22 05:30:00 | 4483.15 | 2024-09-25 05:30:00 | 4782.20 | TARGET_HIT | 0.50 | 6.67% |
| BUY | retest1 | 2024-11-28 05:30:00 | 4352.65 | 2024-12-12 05:30:00 | 4464.35 | STOP_HIT | 1.00 | 2.57% |
| BUY | retest1 | 2024-12-24 05:30:00 | 4612.25 | 2025-01-06 05:30:00 | 4435.62 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest1 | 2025-03-05 05:30:00 | 4698.10 | 2025-03-19 05:30:00 | 4961.60 | PARTIAL | 0.50 | 5.61% |
| BUY | retest1 | 2025-03-05 05:30:00 | 4698.10 | 2025-04-07 05:30:00 | 4698.10 | STOP_HIT | 0.50 | 0.00% |

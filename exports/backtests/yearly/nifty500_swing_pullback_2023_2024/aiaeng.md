# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 3971.90
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.63% / 0.96%
- **Sum % (uncompounded):** 3.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.63% | 3.8% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.63% | 3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.63% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 05:30:00 | 3334.75 | 2841.49 | 3221.18 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=91.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 05:30:00 | 3517.85 | 2867.84 | 3286.81 | T1 booked 50% @ 3517.85 |
| Stop hit — per-position SL triggered | 2023-08-01 05:30:00 | 3334.75 | 2927.24 | 3410.76 | SL hit (bars_held=15) |

### Cycle 2 — BUY (started 2023-11-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 05:30:00 | 3679.35 | 3212.71 | 3500.71 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=113.06 |
| Stop hit — per-position SL triggered | 2023-11-08 05:30:00 | 3509.76 | 3234.34 | 3558.87 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-01-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 05:30:00 | 3755.85 | 3364.39 | 3628.08 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=89.14 |
| Stop hit — per-position SL triggered | 2024-01-18 05:30:00 | 3622.13 | 3384.44 | 3662.96 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 05:30:00 | 3886.30 | 3548.38 | 3735.45 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=106.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 05:30:00 | 4100.24 | 3578.74 | 3870.70 | T1 booked 50% @ 4100.24 |
| Stop hit — per-position SL triggered | 2024-04-12 05:30:00 | 3923.60 | 3591.52 | 3905.69 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 05:30:00 | 3334.75 | 2023-07-18 05:30:00 | 3517.85 | PARTIAL | 0.50 | 5.49% |
| BUY | retest1 | 2023-07-11 05:30:00 | 3334.75 | 2023-08-01 05:30:00 | 3334.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-01 05:30:00 | 3679.35 | 2023-11-08 05:30:00 | 3509.76 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest1 | 2024-01-10 05:30:00 | 3755.85 | 2024-01-18 05:30:00 | 3622.13 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest1 | 2024-03-27 05:30:00 | 3886.30 | 2024-04-08 05:30:00 | 4100.24 | PARTIAL | 0.50 | 5.51% |
| BUY | retest1 | 2024-03-27 05:30:00 | 3886.30 | 2024-04-12 05:30:00 | 3923.60 | STOP_HIT | 0.50 | 0.96% |

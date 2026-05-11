# Power Finance Corporation Ltd. (PFC)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 461.35
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.21% / 1.16%
- **Sum % (uncompounded):** 7.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.21% | 7.2% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.21% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.21% | 7.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 05:30:00 | 531.05 | 393.01 | 491.63 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=20.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 05:30:00 | 571.60 | 402.16 | 518.92 | T1 booked 50% @ 571.60 |
| Stop hit — per-position SL triggered | 2024-07-19 05:30:00 | 533.70 | 409.30 | 530.25 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 05:30:00 | 536.45 | 433.80 | 514.64 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=17.61 |
| Stop hit — per-position SL triggered | 2024-09-10 05:30:00 | 510.04 | 444.29 | 531.66 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2024-11-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 05:30:00 | 481.85 | 453.35 | 463.42 | Stage2 pullback-breakout RSI=55 vol=3.3x ATR=18.25 |
| Stop hit — per-position SL triggered | 2024-11-13 05:30:00 | 454.48 | 453.56 | 463.55 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 481.50 | 454.32 | 465.47 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=20.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 05:30:00 | 522.66 | 458.25 | 487.25 | T1 booked 50% @ 522.66 |
| Target hit | 2024-12-18 05:30:00 | 487.10 | 461.92 | 496.63 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 05:30:00 | 531.05 | 2024-07-11 05:30:00 | 571.60 | PARTIAL | 0.50 | 7.63% |
| BUY | retest1 | 2024-07-03 05:30:00 | 531.05 | 2024-07-19 05:30:00 | 533.70 | STOP_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-27 05:30:00 | 536.45 | 2024-09-10 05:30:00 | 510.04 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest1 | 2024-11-11 05:30:00 | 481.85 | 2024-11-13 05:30:00 | 454.48 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest1 | 2024-11-25 05:30:00 | 481.50 | 2024-12-06 05:30:00 | 522.66 | PARTIAL | 0.50 | 8.55% |
| BUY | retest1 | 2024-11-25 05:30:00 | 481.50 | 2024-12-18 05:30:00 | 487.10 | TARGET_HIT | 0.50 | 1.16% |

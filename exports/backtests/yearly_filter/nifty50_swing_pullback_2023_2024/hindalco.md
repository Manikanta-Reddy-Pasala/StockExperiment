# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1030.90
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
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.15% / 1.76%
- **Sum % (uncompounded):** 0.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.15% | 0.7% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.15% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.15% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 461.25 | 432.81 | 451.43 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=10.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 482.95 | 434.97 | 457.86 | T1 booked 50% @ 482.95 |
| Target hit | 2023-09-25 00:00:00 | 469.35 | 441.09 | 475.49 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 492.65 | 442.48 | 476.05 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=13.03 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 473.10 | 443.16 | 476.14 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 505.40 | 451.47 | 479.78 | Stage2 pullback-breakout RSI=70 vol=2.2x ATR=11.72 |
| Stop hit — per-position SL triggered | 2023-11-30 00:00:00 | 515.65 | 456.72 | 497.77 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 592.25 | 498.37 | 574.42 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=14.92 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 569.87 | 501.22 | 572.20 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-23 00:00:00 | 461.25 | 2023-09-04 00:00:00 | 482.95 | PARTIAL | 0.50 | 4.71% |
| BUY | retest1 | 2023-08-23 00:00:00 | 461.25 | 2023-09-25 00:00:00 | 469.35 | TARGET_HIT | 0.50 | 1.76% |
| BUY | retest1 | 2023-09-29 00:00:00 | 492.65 | 2023-10-04 00:00:00 | 473.10 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest1 | 2023-11-15 00:00:00 | 505.40 | 2023-11-30 00:00:00 | 515.65 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest1 | 2024-02-07 00:00:00 | 592.25 | 2024-02-13 00:00:00 | 569.87 | STOP_HIT | 1.00 | -3.78% |

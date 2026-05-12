# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 504.85
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 4.90% / 7.28%
- **Sum % (uncompounded):** 29.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.90% | 29.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.90% | 29.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.90% | 29.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 339.35 | 242.84 | 320.48 | Stage2 pullback-breakout RSI=65 vol=2.6x ATR=12.38 |
| Stop hit — per-position SL triggered | 2023-07-07 00:00:00 | 320.78 | 246.88 | 322.31 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 335.28 | 250.39 | 321.40 | Stage2 pullback-breakout RSI=60 vol=3.1x ATR=12.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 00:00:00 | 359.70 | 253.28 | 328.47 | T1 booked 50% @ 359.70 |
| Target hit | 2023-08-28 00:00:00 | 425.55 | 296.70 | 439.97 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 00:00:00 | 460.55 | 322.55 | 435.50 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=19.11 |
| Stop hit — per-position SL triggered | 2023-10-12 00:00:00 | 457.00 | 334.53 | 444.53 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 829.95 | 577.84 | 789.30 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=37.37 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 773.89 | 600.79 | 804.85 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 803.95 | 642.28 | 770.03 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=32.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 00:00:00 | 869.92 | 646.12 | 782.28 | T1 booked 50% @ 869.92 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 339.35 | 2023-07-07 00:00:00 | 320.78 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest1 | 2023-07-14 00:00:00 | 335.28 | 2023-07-19 00:00:00 | 359.70 | PARTIAL | 0.50 | 7.28% |
| BUY | retest1 | 2023-07-14 00:00:00 | 335.28 | 2023-08-28 00:00:00 | 425.55 | TARGET_HIT | 0.50 | 26.92% |
| BUY | retest1 | 2023-09-27 00:00:00 | 460.55 | 2023-10-12 00:00:00 | 457.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest1 | 2024-02-26 00:00:00 | 829.95 | 2024-03-11 00:00:00 | 773.89 | STOP_HIT | 1.00 | -6.75% |
| BUY | retest1 | 2024-04-25 00:00:00 | 803.95 | 2024-04-29 00:00:00 | 869.92 | PARTIAL | 0.50 | 8.21% |

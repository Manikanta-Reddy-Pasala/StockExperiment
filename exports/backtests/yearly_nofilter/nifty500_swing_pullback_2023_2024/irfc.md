# Indian Railway Finance Corporation Ltd. (IRFC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 106.03
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 18.00% / 0.69%
- **Sum % (uncompounded):** 108.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 4 | 1 | 18.00% | 108.0% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 4 | 1 | 18.00% | 108.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 18.00% | 108.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 00:00:00 | 33.45 | 29.37 | 32.83 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=0.73 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 32.35 | 29.63 | 32.81 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-07-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 00:00:00 | 33.25 | 29.73 | 32.84 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 00:00:00 | 34.49 | 29.78 | 33.04 | T1 booked 50% @ 34.49 |
| Target hit | 2023-10-09 00:00:00 | 71.15 | 42.81 | 72.94 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 79.15 | 44.77 | 74.59 | Stage2 pullback-breakout RSI=64 vol=2.9x ATR=3.40 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 74.05 | 46.00 | 75.01 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 76.90 | 50.49 | 74.01 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=2.68 |
| Stop hit — per-position SL triggered | 2023-12-04 00:00:00 | 76.40 | 52.92 | 75.18 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 145.70 | 97.41 | 141.21 | Stage2 pullback-breakout RSI=53 vol=2.1x ATR=8.49 |
| Stop hit — per-position SL triggered | 2024-04-10 00:00:00 | 146.70 | 101.99 | 144.14 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-05 00:00:00 | 33.45 | 2023-07-17 00:00:00 | 32.35 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest1 | 2023-07-20 00:00:00 | 33.25 | 2023-07-21 00:00:00 | 34.49 | PARTIAL | 0.50 | 3.73% |
| BUY | retest1 | 2023-07-20 00:00:00 | 33.25 | 2023-10-09 00:00:00 | 71.15 | TARGET_HIT | 0.50 | 113.98% |
| BUY | retest1 | 2023-10-17 00:00:00 | 79.15 | 2023-10-23 00:00:00 | 74.05 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest1 | 2023-11-17 00:00:00 | 76.90 | 2023-12-04 00:00:00 | 76.40 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2024-03-26 00:00:00 | 145.70 | 2024-04-10 00:00:00 | 146.70 | STOP_HIT | 1.00 | 0.69% |

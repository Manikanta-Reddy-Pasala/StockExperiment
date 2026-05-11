# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 751.75
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
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.75% / 5.74%
- **Sum % (uncompounded):** 10.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.75% | 10.5% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.75% | 10.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.75% | 10.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 05:30:00 | 432.75 | 362.74 | 411.88 | Stage2 pullback-breakout RSI=66 vol=2.9x ATR=12.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 05:30:00 | 457.57 | 366.35 | 422.28 | T1 booked 50% @ 457.57 |
| Stop hit — per-position SL triggered | 2023-08-08 05:30:00 | 432.75 | 367.03 | 423.47 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 05:30:00 | 482.75 | 421.05 | 469.67 | Stage2 pullback-breakout RSI=59 vol=3.2x ATR=15.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 05:30:00 | 513.38 | 426.57 | 480.58 | T1 booked 50% @ 513.38 |
| Target hit | 2023-12-20 05:30:00 | 512.05 | 438.40 | 513.58 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 05:30:00 | 586.20 | 466.84 | 546.27 | Stage2 pullback-breakout RSI=67 vol=5.4x ATR=22.04 |
| Stop hit — per-position SL triggered | 2024-02-12 05:30:00 | 553.15 | 472.76 | 561.72 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-02-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-22 05:30:00 | 607.85 | 479.62 | 563.46 | Stage2 pullback-breakout RSI=62 vol=5.7x ATR=29.24 |
| Stop hit — per-position SL triggered | 2024-03-06 05:30:00 | 595.80 | 491.56 | 590.68 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-31 05:30:00 | 432.75 | 2023-08-07 05:30:00 | 457.57 | PARTIAL | 0.50 | 5.74% |
| BUY | retest1 | 2023-07-31 05:30:00 | 432.75 | 2023-08-08 05:30:00 | 432.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 05:30:00 | 482.75 | 2023-12-01 05:30:00 | 513.38 | PARTIAL | 0.50 | 6.35% |
| BUY | retest1 | 2023-11-17 05:30:00 | 482.75 | 2023-12-20 05:30:00 | 512.05 | TARGET_HIT | 0.50 | 6.07% |
| BUY | retest1 | 2024-02-05 05:30:00 | 586.20 | 2024-02-12 05:30:00 | 553.15 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest1 | 2024-02-22 05:30:00 | 607.85 | 2024-03-06 05:30:00 | 595.80 | STOP_HIT | 1.00 | -1.98% |

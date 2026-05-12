# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 835.35
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 2.14% / 2.60%
- **Sum % (uncompounded):** 10.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 2.14% | 10.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 2.14% | 10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 2.14% | 10.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 00:00:00 | 641.30 | 562.10 | 616.39 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=16.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 00:00:00 | 674.30 | 568.39 | 637.18 | T1 booked 50% @ 674.30 |
| Stop hit — per-position SL triggered | 2024-07-24 00:00:00 | 657.95 | 572.25 | 646.83 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 00:00:00 | 665.25 | 596.39 | 658.68 | Stage2 pullback-breakout RSI=53 vol=2.8x ATR=16.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 00:00:00 | 698.89 | 602.35 | 672.27 | T1 booked 50% @ 698.89 |
| Target hit | 2024-10-07 00:00:00 | 678.80 | 613.69 | 687.89 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 00:00:00 | 649.90 | 621.05 | 622.80 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=17.94 |
| Stop hit — per-position SL triggered | 2024-12-09 00:00:00 | 622.99 | 622.15 | 628.52 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 00:00:00 | 641.30 | 2024-07-18 00:00:00 | 674.30 | PARTIAL | 0.50 | 5.15% |
| BUY | retest1 | 2024-07-08 00:00:00 | 641.30 | 2024-07-24 00:00:00 | 657.95 | STOP_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2024-09-06 00:00:00 | 665.25 | 2024-09-17 00:00:00 | 698.89 | PARTIAL | 0.50 | 5.06% |
| BUY | retest1 | 2024-09-06 00:00:00 | 665.25 | 2024-10-07 00:00:00 | 678.80 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2024-11-27 00:00:00 | 649.90 | 2024-12-09 00:00:00 | 622.99 | STOP_HIT | 1.00 | -4.14% |

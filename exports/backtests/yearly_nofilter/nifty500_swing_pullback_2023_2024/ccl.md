# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1123.10
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
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.80% / 0.00%
- **Sum % (uncompounded):** -4.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.80% | -4.8% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.80% | -4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.80% | -4.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 628.80 | 589.16 | 613.12 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=13.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 00:00:00 | 656.39 | 590.39 | 620.06 | T1 booked 50% @ 656.39 |
| Stop hit — per-position SL triggered | 2023-09-20 00:00:00 | 628.80 | 596.08 | 642.99 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 660.80 | 599.20 | 640.82 | Stage2 pullback-breakout RSI=59 vol=3.9x ATR=19.70 |
| Stop hit — per-position SL triggered | 2023-10-17 00:00:00 | 637.25 | 603.99 | 645.91 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 636.00 | 606.40 | 627.72 | Stage2 pullback-breakout RSI=52 vol=1.7x ATR=21.20 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 619.70 | 608.64 | 629.03 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 651.85 | 610.28 | 632.86 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=17.07 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 652.70 | 614.61 | 646.60 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 668.30 | 620.04 | 638.13 | Stage2 pullback-breakout RSI=68 vol=4.4x ATR=14.11 |
| Stop hit — per-position SL triggered | 2024-02-02 00:00:00 | 647.14 | 621.79 | 646.58 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 00:00:00 | 628.80 | 2023-09-07 00:00:00 | 656.39 | PARTIAL | 0.50 | 4.39% |
| BUY | retest1 | 2023-09-05 00:00:00 | 628.80 | 2023-09-20 00:00:00 | 628.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-03 00:00:00 | 660.80 | 2023-10-17 00:00:00 | 637.25 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest1 | 2023-11-06 00:00:00 | 636.00 | 2023-11-20 00:00:00 | 619.70 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest1 | 2023-11-29 00:00:00 | 651.85 | 2023-12-13 00:00:00 | 652.70 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2024-01-29 00:00:00 | 668.30 | 2024-02-02 00:00:00 | 647.14 | STOP_HIT | 1.00 | -3.17% |

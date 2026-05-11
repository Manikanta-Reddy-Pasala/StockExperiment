# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 821.20
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
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.93% / 2.45%
- **Sum % (uncompounded):** 6.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 0.93% | 6.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 5 | 2 | 0.93% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 0.93% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 512.58 | 465.22 | 495.68 | Stage2 pullback-breakout RSI=59 vol=6.9x ATR=14.37 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 491.02 | 467.83 | 499.28 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 536.62 | 468.97 | 486.44 | Stage2 pullback-breakout RSI=68 vol=4.8x ATR=16.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 570.56 | 469.88 | 493.53 | T1 booked 50% @ 570.56 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 551.13 | 481.96 | 545.34 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 579.40 | 502.99 | 554.50 | Stage2 pullback-breakout RSI=68 vol=2.4x ATR=14.79 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 557.22 | 509.39 | 567.65 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-02-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 00:00:00 | 597.91 | 513.46 | 573.36 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=18.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 00:00:00 | 634.18 | 516.46 | 584.25 | T1 booked 50% @ 634.18 |
| Stop hit — per-position SL triggered | 2024-03-05 00:00:00 | 612.55 | 523.31 | 600.87 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 617.82 | 530.14 | 589.61 | Stage2 pullback-breakout RSI=58 vol=2.6x ATR=25.91 |
| Stop hit — per-position SL triggered | 2024-04-09 00:00:00 | 599.48 | 537.84 | 604.25 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-12 00:00:00 | 512.58 | 2023-10-23 00:00:00 | 491.02 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest1 | 2023-11-13 00:00:00 | 536.62 | 2023-11-15 00:00:00 | 570.56 | PARTIAL | 0.50 | 6.32% |
| BUY | retest1 | 2023-11-13 00:00:00 | 536.62 | 2023-12-06 00:00:00 | 551.13 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest1 | 2024-01-31 00:00:00 | 579.40 | 2024-02-13 00:00:00 | 557.22 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest1 | 2024-02-21 00:00:00 | 597.91 | 2024-02-26 00:00:00 | 634.18 | PARTIAL | 0.50 | 6.07% |
| BUY | retest1 | 2024-02-21 00:00:00 | 597.91 | 2024-03-05 00:00:00 | 612.55 | STOP_HIT | 0.50 | 2.45% |
| BUY | retest1 | 2024-03-22 00:00:00 | 617.82 | 2024-04-09 00:00:00 | 599.48 | STOP_HIT | 1.00 | -2.97% |

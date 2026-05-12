# Indian Hotels Co. Ltd. (INDHOTEL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 673.05
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 2.21% / 2.40%
- **Sum % (uncompounded):** 8.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.21% | 8.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.21% | 8.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.21% | 8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 00:00:00 | 620.35 | 536.41 | 602.37 | Stage2 pullback-breakout RSI=57 vol=5.5x ATR=18.75 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 609.20 | 545.64 | 621.42 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 00:00:00 | 644.60 | 553.77 | 620.18 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=17.42 |
| Stop hit — per-position SL triggered | 2024-09-06 00:00:00 | 657.25 | 564.48 | 645.08 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 00:00:00 | 695.00 | 566.78 | 651.62 | Stage2 pullback-breakout RSI=70 vol=2.3x ATR=16.87 |
| Stop hit — per-position SL triggered | 2024-09-24 00:00:00 | 711.65 | 579.09 | 681.00 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 00:00:00 | 732.90 | 608.37 | 685.38 | Stage2 pullback-breakout RSI=65 vol=6.6x ATR=23.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 00:00:00 | 778.99 | 617.42 | 715.69 | T1 booked 50% @ 778.99 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-22 00:00:00 | 620.35 | 2024-08-05 00:00:00 | 609.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest1 | 2024-08-22 00:00:00 | 644.60 | 2024-09-06 00:00:00 | 657.25 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest1 | 2024-09-10 00:00:00 | 695.00 | 2024-09-24 00:00:00 | 711.65 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest1 | 2024-11-08 00:00:00 | 732.90 | 2024-11-21 00:00:00 | 778.99 | PARTIAL | 0.50 | 6.29% |

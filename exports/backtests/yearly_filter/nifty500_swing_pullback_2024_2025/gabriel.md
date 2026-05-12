# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1127.90
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
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -2.01% / -6.15%
- **Sum % (uncompounded):** -10.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -2.01% | -10.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -2.01% | -10.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -2.01% | -10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 499.00 | 389.78 | 479.25 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=20.44 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 468.34 | 395.88 | 485.41 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-08-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 00:00:00 | 511.65 | 402.60 | 491.17 | Stage2 pullback-breakout RSI=61 vol=4.3x ATR=22.06 |
| Stop hit — per-position SL triggered | 2024-08-19 00:00:00 | 478.56 | 404.47 | 492.35 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 548.35 | 415.55 | 516.33 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=23.58 |
| Stop hit — per-position SL triggered | 2024-09-09 00:00:00 | 512.98 | 422.26 | 522.13 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 00:00:00 | 480.80 | 440.27 | 445.78 | Stage2 pullback-breakout RSI=63 vol=10.9x ATR=17.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 00:00:00 | 515.90 | 441.56 | 456.70 | T1 booked 50% @ 515.90 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 489.10 | 446.73 | 484.61 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 00:00:00 | 499.00 | 2024-08-05 00:00:00 | 468.34 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest1 | 2024-08-14 00:00:00 | 511.65 | 2024-08-19 00:00:00 | 478.56 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest1 | 2024-08-30 00:00:00 | 548.35 | 2024-09-09 00:00:00 | 512.98 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest1 | 2024-12-06 00:00:00 | 480.80 | 2024-12-10 00:00:00 | 515.90 | PARTIAL | 0.50 | 7.30% |
| BUY | retest1 | 2024-12-06 00:00:00 | 480.80 | 2024-12-20 00:00:00 | 489.10 | STOP_HIT | 0.50 | 1.73% |

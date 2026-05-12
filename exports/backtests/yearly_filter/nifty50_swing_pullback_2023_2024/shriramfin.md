# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1007.75
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -2.96% / -3.63%
- **Sum % (uncompounded):** -14.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.96% | -14.8% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.96% | -14.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.96% | -14.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 00:00:00 | 389.11 | 314.71 | 380.60 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=9.43 |
| Stop hit — per-position SL triggered | 2023-09-26 00:00:00 | 374.97 | 315.33 | 380.25 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-10-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 00:00:00 | 387.04 | 326.52 | 374.65 | Stage2 pullback-breakout RSI=58 vol=3.4x ATR=12.37 |
| Stop hit — per-position SL triggered | 2023-11-10 00:00:00 | 390.44 | 332.57 | 385.09 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 418.89 | 341.84 | 396.73 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=9.65 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 404.41 | 344.50 | 400.86 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 423.66 | 346.98 | 403.34 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=10.96 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 407.22 | 349.42 | 405.09 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-03-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 00:00:00 | 496.19 | 399.00 | 482.84 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=15.64 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 472.74 | 400.49 | 481.20 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-25 00:00:00 | 389.11 | 2023-09-26 00:00:00 | 374.97 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest1 | 2023-10-27 00:00:00 | 387.04 | 2023-11-10 00:00:00 | 390.44 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest1 | 2023-12-04 00:00:00 | 418.89 | 2023-12-08 00:00:00 | 404.41 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest1 | 2023-12-14 00:00:00 | 423.66 | 2023-12-20 00:00:00 | 407.22 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest1 | 2024-03-11 00:00:00 | 496.19 | 2024-03-13 00:00:00 | 472.74 | STOP_HIT | 1.00 | -4.73% |

# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 598.05
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 1.41% / 7.13%
- **Sum % (uncompounded):** 5.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.41% | 5.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.41% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.41% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 00:00:00 | 925.35 | 748.53 | 883.57 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=32.67 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 876.35 | 752.17 | 879.95 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-08-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 00:00:00 | 854.05 | 763.01 | 846.88 | Stage2 pullback-breakout RSI=51 vol=5.9x ATR=30.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 00:00:00 | 914.91 | 766.40 | 855.18 | T1 booked 50% @ 914.91 |
| Target hit | 2024-09-20 00:00:00 | 941.25 | 796.60 | 950.99 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 1050.70 | 849.64 | 974.40 | Stage2 pullback-breakout RSI=62 vol=8.9x ATR=44.95 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 983.28 | 855.39 | 981.01 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-12-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 00:00:00 | 1236.70 | 914.68 | 1150.27 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=54.23 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-31 00:00:00 | 925.35 | 2024-08-05 00:00:00 | 876.35 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest1 | 2024-08-26 00:00:00 | 854.05 | 2024-08-29 00:00:00 | 914.91 | PARTIAL | 0.50 | 7.13% |
| BUY | retest1 | 2024-08-26 00:00:00 | 854.05 | 2024-09-20 00:00:00 | 941.25 | TARGET_HIT | 0.50 | 10.21% |
| BUY | retest1 | 2024-11-07 00:00:00 | 1050.70 | 2024-11-13 00:00:00 | 983.28 | STOP_HIT | 1.00 | -6.42% |

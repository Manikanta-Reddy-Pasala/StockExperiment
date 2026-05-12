# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 658.60
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
- **Avg / median % per leg:** 1.90% / 6.17%
- **Sum % (uncompounded):** 11.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.90% | 11.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.90% | 11.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.90% | 11.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 00:00:00 | 739.25 | 589.60 | 711.50 | Stage2 pullback-breakout RSI=65 vol=7.8x ATR=25.71 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 700.69 | 590.80 | 711.29 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2024-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 00:00:00 | 786.35 | 594.17 | 720.55 | Stage2 pullback-breakout RSI=70 vol=2.6x ATR=30.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 00:00:00 | 846.66 | 598.39 | 736.38 | T1 booked 50% @ 846.66 |
| Target hit | 2024-09-09 00:00:00 | 834.90 | 644.43 | 837.15 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-09-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 00:00:00 | 919.15 | 659.86 | 856.16 | Stage2 pullback-breakout RSI=66 vol=4.2x ATR=37.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 00:00:00 | 994.89 | 680.06 | 908.39 | T1 booked 50% @ 994.89 |
| Stop hit — per-position SL triggered | 2024-09-30 00:00:00 | 919.15 | 682.46 | 909.66 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-12-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 00:00:00 | 853.45 | 741.30 | 814.57 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=31.03 |
| Stop hit — per-position SL triggered | 2024-12-13 00:00:00 | 806.90 | 743.69 | 816.11 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-02 00:00:00 | 739.25 | 2024-08-05 00:00:00 | 700.69 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest1 | 2024-08-07 00:00:00 | 786.35 | 2024-08-09 00:00:00 | 846.66 | PARTIAL | 0.50 | 7.67% |
| BUY | retest1 | 2024-08-07 00:00:00 | 786.35 | 2024-09-09 00:00:00 | 834.90 | TARGET_HIT | 0.50 | 6.17% |
| BUY | retest1 | 2024-09-18 00:00:00 | 919.15 | 2024-09-27 00:00:00 | 994.89 | PARTIAL | 0.50 | 8.24% |
| BUY | retest1 | 2024-09-18 00:00:00 | 919.15 | 2024-09-30 00:00:00 | 919.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 00:00:00 | 853.45 | 2024-12-13 00:00:00 | 806.90 | STOP_HIT | 1.00 | -5.45% |

# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2023-12-20 00:00:00 → 2026-05-11 00:00:00 (591 bars)
- **Last close:** 2319.70
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
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -2.07% / -5.67%
- **Sum % (uncompounded):** -10.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -2.07% | -10.4% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -2.07% | -10.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -2.07% | -10.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 00:00:00 | 2924.40 | 2108.70 | 2723.95 | Stage2 pullback-breakout RSI=65 vol=6.6x ATR=121.31 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 2742.44 | 2122.24 | 2735.94 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 2785.10 | 2156.24 | 2682.48 | Stage2 pullback-breakout RSI=56 vol=3.3x ATR=127.32 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 2594.12 | 2208.68 | 2723.25 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 00:00:00 | 2840.60 | 2229.59 | 2731.22 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=119.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 00:00:00 | 3079.25 | 2244.51 | 2777.61 | T1 booked 50% @ 3079.25 |
| Stop hit — per-position SL triggered | 2024-12-05 00:00:00 | 2840.60 | 2297.48 | 2859.69 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2024-12-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 00:00:00 | 3052.05 | 2348.98 | 2921.57 | Stage2 pullback-breakout RSI=62 vol=3.6x ATR=115.46 |
| Stop hit — per-position SL triggered | 2024-12-19 00:00:00 | 2878.86 | 2361.72 | 2933.94 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-18 00:00:00 | 2924.40 | 2024-10-22 00:00:00 | 2742.44 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest1 | 2024-10-31 00:00:00 | 2785.10 | 2024-11-13 00:00:00 | 2594.12 | STOP_HIT | 1.00 | -6.86% |
| BUY | retest1 | 2024-11-21 00:00:00 | 2840.60 | 2024-11-25 00:00:00 | 3079.25 | PARTIAL | 0.50 | 8.40% |
| BUY | retest1 | 2024-11-21 00:00:00 | 2840.60 | 2024-12-05 00:00:00 | 2840.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 00:00:00 | 3052.05 | 2024-12-19 00:00:00 | 2878.86 | STOP_HIT | 1.00 | -5.67% |

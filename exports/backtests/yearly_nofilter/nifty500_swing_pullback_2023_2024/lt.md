# Larsen & Toubro Ltd. (LT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3917.40
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.24% / 2.84%
- **Sum % (uncompounded):** 8.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.24% | 9.0% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.24% | 9.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.24% | 9.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 3109.20 | 2628.95 | 3046.12 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=44.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 00:00:00 | 3197.39 | 2634.54 | 3059.89 | T1 booked 50% @ 3197.39 |
| Target hit | 2024-01-31 00:00:00 | 3479.75 | 2927.72 | 3557.75 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-03-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 00:00:00 | 3633.50 | 3019.00 | 3437.83 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=82.61 |
| Stop hit — per-position SL triggered | 2024-03-15 00:00:00 | 3537.15 | 3075.90 | 3545.72 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 3763.90 | 3117.21 | 3592.39 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=79.07 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 3645.30 | 3178.18 | 3687.68 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-30 00:00:00 | 3109.20 | 2023-12-01 00:00:00 | 3197.39 | PARTIAL | 0.50 | 2.84% |
| BUY | retest1 | 2023-11-30 00:00:00 | 3109.20 | 2024-01-31 00:00:00 | 3479.75 | TARGET_HIT | 0.50 | 11.92% |
| BUY | retest1 | 2024-03-01 00:00:00 | 3633.50 | 2024-03-15 00:00:00 | 3537.15 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2024-03-28 00:00:00 | 3763.90 | 2024-04-15 00:00:00 | 3645.30 | STOP_HIT | 1.00 | -3.15% |

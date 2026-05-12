# JBM Auto Ltd. (JBMA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 682.10
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
- **Avg / median % per leg:** 13.81% / 6.43%
- **Sum % (uncompounded):** 69.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 13.81% | 69.0% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 13.81% | 69.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 13.81% | 69.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 00:00:00 | 721.93 | 427.68 | 670.34 | Stage2 pullback-breakout RSI=61 vol=5.4x ATR=38.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 00:00:00 | 799.13 | 455.42 | 719.72 | T1 booked 50% @ 799.13 |
| Stop hit — per-position SL triggered | 2023-09-07 00:00:00 | 737.93 | 475.38 | 734.56 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 00:00:00 | 672.18 | 551.97 | 631.64 | Stage2 pullback-breakout RSI=67 vol=2.8x ATR=21.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 00:00:00 | 715.43 | 554.95 | 644.53 | T1 booked 50% @ 715.43 |
| Target hit | 2024-03-04 00:00:00 | 1044.50 | 728.78 | 1058.92 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 953.75 | 780.83 | 903.12 | Stage2 pullback-breakout RSI=60 vol=3.3x ATR=36.28 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 899.33 | 792.12 | 923.64 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-16 00:00:00 | 721.93 | 2023-08-29 00:00:00 | 799.13 | PARTIAL | 0.50 | 10.69% |
| BUY | retest1 | 2023-08-16 00:00:00 | 721.93 | 2023-09-07 00:00:00 | 737.93 | STOP_HIT | 0.50 | 2.22% |
| BUY | retest1 | 2023-12-11 00:00:00 | 672.18 | 2023-12-13 00:00:00 | 715.43 | PARTIAL | 0.50 | 6.43% |
| BUY | retest1 | 2023-12-11 00:00:00 | 672.18 | 2024-03-04 00:00:00 | 1044.50 | TARGET_HIT | 0.50 | 55.39% |
| BUY | retest1 | 2024-04-29 00:00:00 | 953.75 | 2024-05-09 00:00:00 | 899.33 | STOP_HIT | 1.00 | -5.71% |

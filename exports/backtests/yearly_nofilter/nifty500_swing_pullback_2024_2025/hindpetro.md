# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 378.55
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
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.39% / 0.08%
- **Sum % (uncompounded):** -1.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 4 | 1 | -0.39% | -2.0% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 4 | 1 | -0.39% | -2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | -0.39% | -2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 00:00:00 | 350.05 | 295.67 | 337.64 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=11.34 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 333.04 | 299.26 | 343.14 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 00:00:00 | 373.75 | 300.54 | 346.98 | Stage2 pullback-breakout RSI=66 vol=2.3x ATR=13.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 00:00:00 | 401.52 | 303.02 | 356.86 | T1 booked 50% @ 401.52 |
| Stop hit — per-position SL triggered | 2024-08-09 00:00:00 | 376.65 | 309.61 | 374.09 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 00:00:00 | 422.90 | 346.10 | 409.36 | Stage2 pullback-breakout RSI=57 vol=2.2x ATR=15.16 |
| Stop hit — per-position SL triggered | 2024-10-23 00:00:00 | 400.16 | 350.34 | 412.65 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 00:00:00 | 399.20 | 358.01 | 383.33 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=11.55 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 399.50 | 362.58 | 397.42 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-11 00:00:00 | 350.05 | 2024-07-23 00:00:00 | 333.04 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2024-07-25 00:00:00 | 373.75 | 2024-07-30 00:00:00 | 401.52 | PARTIAL | 0.50 | 7.43% |
| BUY | retest1 | 2024-07-25 00:00:00 | 373.75 | 2024-08-09 00:00:00 | 376.65 | STOP_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2024-10-15 00:00:00 | 422.90 | 2024-10-23 00:00:00 | 400.16 | STOP_HIT | 1.00 | -5.38% |
| BUY | retest1 | 2024-12-06 00:00:00 | 399.20 | 2024-12-20 00:00:00 | 399.50 | STOP_HIT | 1.00 | 0.08% |

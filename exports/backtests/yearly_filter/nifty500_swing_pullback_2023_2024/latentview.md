# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 313.00
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 3
- **Avg / median % per leg:** 2.58% / 4.83%
- **Sum % (uncompounded):** 20.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 2.58% | 20.6% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 2.58% | 20.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 3 | 2 | 3 | 2.58% | 20.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 401.25 | 360.97 | 377.94 | Stage2 pullback-breakout RSI=65 vol=4.5x ATR=13.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 00:00:00 | 428.93 | 363.02 | 388.46 | T1 booked 50% @ 428.93 |
| Target hit | 2023-09-12 00:00:00 | 420.65 | 375.58 | 433.59 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 431.95 | 383.39 | 414.82 | Stage2 pullback-breakout RSI=59 vol=4.4x ATR=13.54 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 411.64 | 385.11 | 418.68 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 450.50 | 389.96 | 416.09 | Stage2 pullback-breakout RSI=69 vol=7.4x ATR=13.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 00:00:00 | 478.42 | 390.69 | 420.64 | T1 booked 50% @ 478.42 |
| Target hit | 2023-12-12 00:00:00 | 453.65 | 401.55 | 459.97 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 00:00:00 | 466.15 | 411.39 | 456.96 | Stage2 pullback-breakout RSI=57 vol=2.8x ATR=14.65 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 450.80 | 416.54 | 461.98 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 507.75 | 443.04 | 474.90 | Stage2 pullback-breakout RSI=60 vol=7.5x ATR=21.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 550.48 | 445.68 | 489.97 | T1 booked 50% @ 550.48 |
| Target hit | 2024-04-18 00:00:00 | 515.85 | 453.73 | 517.65 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 401.25 | 2023-08-17 00:00:00 | 428.93 | PARTIAL | 0.50 | 6.90% |
| BUY | retest1 | 2023-08-09 00:00:00 | 401.25 | 2023-09-12 00:00:00 | 420.65 | TARGET_HIT | 0.50 | 4.83% |
| BUY | retest1 | 2023-10-17 00:00:00 | 431.95 | 2023-10-23 00:00:00 | 411.64 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest1 | 2023-11-20 00:00:00 | 450.50 | 2023-11-21 00:00:00 | 478.42 | PARTIAL | 0.50 | 6.20% |
| BUY | retest1 | 2023-11-20 00:00:00 | 450.50 | 2023-12-12 00:00:00 | 453.65 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2024-01-09 00:00:00 | 466.15 | 2024-01-23 00:00:00 | 450.80 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2024-03-28 00:00:00 | 507.75 | 2024-04-03 00:00:00 | 550.48 | PARTIAL | 0.50 | 8.42% |
| BUY | retest1 | 2024-03-28 00:00:00 | 507.75 | 2024-04-18 00:00:00 | 515.85 | TARGET_HIT | 0.50 | 1.60% |

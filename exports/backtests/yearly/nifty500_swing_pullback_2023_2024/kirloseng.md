# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1728.90
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 3 / 3 / 4
- **Avg / median % per leg:** 8.90% / 8.02%
- **Sum % (uncompounded):** 88.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 3 | 3 | 4 | 8.90% | 89.0% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 3 | 3 | 4 | 8.90% | 89.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 8 | 80.0% | 3 | 3 | 4 | 8.90% | 89.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 05:30:00 | 411.90 | 350.27 | 401.89 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=14.86 |
| Stop hit — per-position SL triggered | 2023-07-14 05:30:00 | 389.61 | 351.78 | 401.77 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-07-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 05:30:00 | 419.40 | 353.00 | 404.00 | Stage2 pullback-breakout RSI=59 vol=2.4x ATR=16.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 05:30:00 | 453.04 | 359.11 | 417.16 | T1 booked 50% @ 453.04 |
| Target hit | 2023-08-28 05:30:00 | 462.85 | 377.90 | 463.30 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-08-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 05:30:00 | 483.90 | 380.67 | 465.57 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=22.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 05:30:00 | 528.36 | 385.73 | 480.35 | T1 booked 50% @ 528.36 |
| Target hit | 2023-09-13 05:30:00 | 487.90 | 391.66 | 490.57 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-09-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 05:30:00 | 527.10 | 401.56 | 496.11 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=23.53 |
| Stop hit — per-position SL triggered | 2023-10-16 05:30:00 | 539.20 | 415.67 | 523.66 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-11-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 05:30:00 | 561.20 | 447.91 | 543.03 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=19.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 05:30:00 | 600.80 | 450.53 | 549.90 | T1 booked 50% @ 600.80 |
| Target hit | 2024-03-06 05:30:00 | 838.15 | 599.57 | 868.59 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 05:30:00 | 901.80 | 635.36 | 856.14 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=39.83 |
| Stop hit — per-position SL triggered | 2024-04-18 05:30:00 | 893.05 | 659.68 | 877.59 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-04-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 05:30:00 | 935.50 | 664.65 | 883.76 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=38.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 05:30:00 | 1012.80 | 676.51 | 912.15 | T1 booked 50% @ 1012.80 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 05:30:00 | 411.90 | 2023-07-14 05:30:00 | 389.61 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest1 | 2023-07-18 05:30:00 | 419.40 | 2023-07-31 05:30:00 | 453.04 | PARTIAL | 0.50 | 8.02% |
| BUY | retest1 | 2023-07-18 05:30:00 | 419.40 | 2023-08-28 05:30:00 | 462.85 | TARGET_HIT | 0.50 | 10.36% |
| BUY | retest1 | 2023-08-31 05:30:00 | 483.90 | 2023-09-06 05:30:00 | 528.36 | PARTIAL | 0.50 | 9.19% |
| BUY | retest1 | 2023-08-31 05:30:00 | 483.90 | 2023-09-13 05:30:00 | 487.90 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2023-09-28 05:30:00 | 527.10 | 2023-10-16 05:30:00 | 539.20 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest1 | 2023-11-28 05:30:00 | 561.20 | 2023-11-30 05:30:00 | 600.80 | PARTIAL | 0.50 | 7.06% |
| BUY | retest1 | 2023-11-28 05:30:00 | 561.20 | 2024-03-06 05:30:00 | 838.15 | TARGET_HIT | 0.50 | 49.35% |
| BUY | retest1 | 2024-04-02 05:30:00 | 901.80 | 2024-04-18 05:30:00 | 893.05 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2024-04-22 05:30:00 | 935.50 | 2024-04-26 05:30:00 | 1012.80 | PARTIAL | 0.50 | 8.26% |

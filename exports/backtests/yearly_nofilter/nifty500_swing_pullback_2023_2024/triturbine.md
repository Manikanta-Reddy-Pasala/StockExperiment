# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 585.00
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 2.94% / 4.99%
- **Sum % (uncompounded):** 26.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 2.94% | 26.4% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 2.94% | 26.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 2 | 4 | 3 | 2.94% | 26.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 00:00:00 | 406.25 | 349.85 | 390.62 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=10.95 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 389.83 | 351.91 | 394.39 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 433.35 | 353.18 | 398.48 | Stage2 pullback-breakout RSI=69 vol=3.4x ATR=15.24 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 410.50 | 355.97 | 406.71 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 391.70 | 362.26 | 368.88 | Stage2 pullback-breakout RSI=56 vol=3.5x ATR=19.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 00:00:00 | 429.83 | 370.01 | 405.23 | T1 booked 50% @ 429.83 |
| Target hit | 2023-12-11 00:00:00 | 411.25 | 373.62 | 413.71 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 00:00:00 | 417.40 | 381.42 | 409.89 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=12.58 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 398.53 | 382.10 | 408.35 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-02-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 00:00:00 | 424.80 | 382.35 | 390.75 | Stage2 pullback-breakout RSI=60 vol=12.3x ATR=17.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 00:00:00 | 460.50 | 383.22 | 398.26 | T1 booked 50% @ 460.50 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 424.80 | 385.32 | 412.52 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2024-03-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 00:00:00 | 488.35 | 404.70 | 469.45 | Stage2 pullback-breakout RSI=58 vol=5.0x ATR=26.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 00:00:00 | 541.27 | 408.66 | 482.07 | T1 booked 50% @ 541.27 |
| Target hit | 2024-04-18 00:00:00 | 519.15 | 426.82 | 529.33 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2024-05-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 00:00:00 | 559.60 | 436.37 | 535.18 | Stage2 pullback-breakout RSI=63 vol=3.7x ATR=22.97 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-07 00:00:00 | 406.25 | 2023-09-13 00:00:00 | 389.83 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest1 | 2023-09-15 00:00:00 | 433.35 | 2023-09-22 00:00:00 | 410.50 | STOP_HIT | 1.00 | -5.27% |
| BUY | retest1 | 2023-11-02 00:00:00 | 391.70 | 2023-11-30 00:00:00 | 429.83 | PARTIAL | 0.50 | 9.73% |
| BUY | retest1 | 2023-11-02 00:00:00 | 391.70 | 2023-12-11 00:00:00 | 411.25 | TARGET_HIT | 0.50 | 4.99% |
| BUY | retest1 | 2024-01-15 00:00:00 | 417.40 | 2024-01-18 00:00:00 | 398.53 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest1 | 2024-02-06 00:00:00 | 424.80 | 2024-02-07 00:00:00 | 460.50 | PARTIAL | 0.50 | 8.40% |
| BUY | retest1 | 2024-02-06 00:00:00 | 424.80 | 2024-02-12 00:00:00 | 424.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-19 00:00:00 | 488.35 | 2024-03-26 00:00:00 | 541.27 | PARTIAL | 0.50 | 10.84% |
| BUY | retest1 | 2024-03-19 00:00:00 | 488.35 | 2024-04-18 00:00:00 | 519.15 | TARGET_HIT | 0.50 | 6.31% |

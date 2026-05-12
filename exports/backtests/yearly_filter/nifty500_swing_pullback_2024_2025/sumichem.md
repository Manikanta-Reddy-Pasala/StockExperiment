# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 473.85
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 1
- **Avg / median % per leg:** -2.23% / -4.44%
- **Sum % (uncompounded):** -17.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -2.23% | -17.8% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -2.23% | -17.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 1 | 6 | 1 | -2.23% | -17.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 00:00:00 | 512.60 | 419.38 | 479.57 | Stage2 pullback-breakout RSI=67 vol=4.0x ATR=19.64 |
| Stop hit — per-position SL triggered | 2024-07-10 00:00:00 | 483.14 | 423.73 | 486.22 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-07-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 00:00:00 | 517.60 | 429.10 | 495.13 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=18.07 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 490.50 | 429.98 | 497.28 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 533.75 | 433.08 | 501.61 | Stage2 pullback-breakout RSI=66 vol=2.8x ATR=20.07 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 503.65 | 436.87 | 504.82 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 00:00:00 | 538.25 | 454.88 | 522.25 | Stage2 pullback-breakout RSI=59 vol=3.3x ATR=19.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 00:00:00 | 576.92 | 457.17 | 531.32 | T1 booked 50% @ 576.92 |
| Target hit | 2024-09-20 00:00:00 | 558.90 | 467.34 | 562.82 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 571.35 | 480.49 | 552.76 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=21.46 |
| Stop hit — per-position SL triggered | 2024-10-21 00:00:00 | 539.16 | 482.46 | 550.97 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-10-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 00:00:00 | 553.10 | 484.44 | 536.21 | Stage2 pullback-breakout RSI=54 vol=5.3x ATR=26.39 |
| Stop hit — per-position SL triggered | 2024-11-12 00:00:00 | 528.55 | 491.38 | 547.47 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 556.90 | 493.40 | 535.10 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=22.17 |
| Stop hit — per-position SL triggered | 2024-12-09 00:00:00 | 545.05 | 498.35 | 542.07 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-02 00:00:00 | 512.60 | 2024-07-10 00:00:00 | 483.14 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest1 | 2024-07-22 00:00:00 | 517.60 | 2024-07-23 00:00:00 | 490.50 | STOP_HIT | 1.00 | -5.24% |
| BUY | retest1 | 2024-07-29 00:00:00 | 533.75 | 2024-08-05 00:00:00 | 503.65 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest1 | 2024-09-06 00:00:00 | 538.25 | 2024-09-10 00:00:00 | 576.92 | PARTIAL | 0.50 | 7.18% |
| BUY | retest1 | 2024-09-06 00:00:00 | 538.25 | 2024-09-20 00:00:00 | 558.90 | TARGET_HIT | 0.50 | 3.84% |
| BUY | retest1 | 2024-10-16 00:00:00 | 571.35 | 2024-10-21 00:00:00 | 539.16 | STOP_HIT | 1.00 | -5.63% |
| BUY | retest1 | 2024-10-29 00:00:00 | 553.10 | 2024-11-12 00:00:00 | 528.55 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest1 | 2024-11-25 00:00:00 | 556.90 | 2024-12-09 00:00:00 | 545.05 | STOP_HIT | 1.00 | -2.13% |

# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1228.30
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.26% / 0.56%
- **Sum % (uncompounded):** 1.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.26% | 1.8% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.26% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.26% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 05:30:00 | 437.75 | 414.39 | 431.53 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=11.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 05:30:00 | 460.31 | 415.01 | 434.18 | T1 booked 50% @ 460.31 |
| Target hit | 2024-07-19 05:30:00 | 440.40 | 419.91 | 452.99 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 05:30:00 | 474.15 | 437.56 | 463.57 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=14.59 |
| Stop hit — per-position SL triggered | 2024-10-22 05:30:00 | 452.27 | 439.80 | 466.31 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 05:30:00 | 492.20 | 440.75 | 466.06 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=18.73 |
| Stop hit — per-position SL triggered | 2024-11-11 05:30:00 | 494.95 | 445.70 | 483.44 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-12-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 05:30:00 | 586.70 | 474.36 | 563.52 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=15.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 05:30:00 | 616.80 | 478.13 | 573.72 | T1 booked 50% @ 616.80 |
| Stop hit — per-position SL triggered | 2025-01-10 05:30:00 | 586.70 | 486.79 | 589.61 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2025-01-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 05:30:00 | 602.65 | 495.02 | 581.19 | Stage2 pullback-breakout RSI=59 vol=3.6x ATR=20.07 |
| Stop hit — per-position SL triggered | 2025-01-27 05:30:00 | 572.54 | 495.42 | 576.76 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-02 05:30:00 | 437.75 | 2024-07-04 05:30:00 | 460.31 | PARTIAL | 0.50 | 5.15% |
| BUY | retest1 | 2024-07-02 05:30:00 | 437.75 | 2024-07-19 05:30:00 | 440.40 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-11 05:30:00 | 474.15 | 2024-10-22 05:30:00 | 452.27 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest1 | 2024-10-28 05:30:00 | 492.20 | 2024-11-11 05:30:00 | 494.95 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest1 | 2024-12-27 05:30:00 | 586.70 | 2025-01-01 05:30:00 | 616.80 | PARTIAL | 0.50 | 5.13% |
| BUY | retest1 | 2024-12-27 05:30:00 | 586.70 | 2025-01-10 05:30:00 | 586.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-24 05:30:00 | 602.65 | 2025-01-27 05:30:00 | 572.54 | STOP_HIT | 1.00 | -5.00% |

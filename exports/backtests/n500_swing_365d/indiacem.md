# India Cements Ltd. (INDIACEM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 408.80
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** -0.12% / -3.87%
- **Sum % (uncompounded):** -0.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.12% | -0.7% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.12% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.12% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 05:30:00 | 370.95 | 328.28 | 345.59 | Stage2 pullback-breakout RSI=69 vol=6.8x ATR=11.95 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 353.03 | 329.31 | 350.04 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-08-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 05:30:00 | 390.95 | 335.59 | 366.41 | Stage2 pullback-breakout RSI=69 vol=12.4x ATR=12.32 |
| Stop hit — per-position SL triggered | 2025-08-25 05:30:00 | 372.48 | 336.45 | 368.75 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-10-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 05:30:00 | 400.00 | 356.33 | 391.00 | Stage2 pullback-breakout RSI=60 vol=11.0x ATR=10.32 |
| Stop hit — per-position SL triggered | 2025-11-07 05:30:00 | 384.52 | 358.16 | 391.69 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2025-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 05:30:00 | 412.15 | 360.23 | 393.18 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=11.60 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 394.76 | 362.52 | 398.40 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2025-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 05:30:00 | 412.15 | 364.87 | 395.37 | Stage2 pullback-breakout RSI=59 vol=8.7x ATR=14.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 05:30:00 | 440.97 | 369.37 | 408.97 | T1 booked 50% @ 440.97 |
| Target hit | 2026-01-21 05:30:00 | 453.10 | 386.67 | 454.44 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-22 05:30:00 | 370.95 | 2025-07-25 05:30:00 | 353.03 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest1 | 2025-08-21 05:30:00 | 390.95 | 2025-08-25 05:30:00 | 372.48 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest1 | 2025-10-30 05:30:00 | 400.00 | 2025-11-07 05:30:00 | 384.52 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest1 | 2025-11-17 05:30:00 | 412.15 | 2025-11-24 05:30:00 | 394.76 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest1 | 2025-12-04 05:30:00 | 412.15 | 2025-12-17 05:30:00 | 440.97 | PARTIAL | 0.50 | 6.99% |
| BUY | retest1 | 2025-12-04 05:30:00 | 412.15 | 2026-01-21 05:30:00 | 453.10 | TARGET_HIT | 0.50 | 9.94% |

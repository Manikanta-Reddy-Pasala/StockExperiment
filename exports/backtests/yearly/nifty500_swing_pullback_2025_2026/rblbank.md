# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 343.45
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 1.06% / 0.71%
- **Sum % (uncompounded):** 9.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 6 | 2 | 1.06% | 9.6% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 6 | 2 | 1.06% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 6 | 2 | 1.06% | 9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 05:30:00 | 266.83 | 207.60 | 256.28 | Stage2 pullback-breakout RSI=63 vol=2.8x ATR=8.95 |
| Stop hit — per-position SL triggered | 2025-08-12 05:30:00 | 253.40 | 211.73 | 258.77 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2025-08-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 05:30:00 | 261.05 | 216.33 | 256.88 | Stage2 pullback-breakout RSI=54 vol=4.9x ATR=8.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 05:30:00 | 278.96 | 218.91 | 261.88 | T1 booked 50% @ 278.96 |
| Stop hit — per-position SL triggered | 2025-09-16 05:30:00 | 267.30 | 222.46 | 266.55 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 05:30:00 | 286.45 | 229.62 | 273.35 | Stage2 pullback-breakout RSI=67 vol=2.9x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 05:30:00 | 303.35 | 232.66 | 280.85 | T1 booked 50% @ 303.35 |
| Target hit | 2025-11-13 05:30:00 | 315.45 | 247.97 | 315.78 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 315.80 | 264.50 | 306.05 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=7.18 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 305.02 | 267.93 | 310.32 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2026-01-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 05:30:00 | 324.60 | 269.65 | 311.08 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=8.75 |
| Stop hit — per-position SL triggered | 2026-01-19 05:30:00 | 311.48 | 269.98 | 310.30 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2026-02-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 05:30:00 | 325.80 | 276.80 | 309.67 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=8.56 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 312.96 | 280.48 | 317.41 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2026-04-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 05:30:00 | 318.15 | 284.16 | 302.27 | Stage2 pullback-breakout RSI=59 vol=4.6x ATR=10.50 |
| Stop hit — per-position SL triggered | 2026-04-21 05:30:00 | 320.40 | 287.38 | 312.26 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2026-04-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 05:30:00 | 341.20 | 289.35 | 316.76 | Stage2 pullback-breakout RSI=65 vol=3.0x ATR=11.52 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-31 05:30:00 | 266.83 | 2025-08-12 05:30:00 | 253.40 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest1 | 2025-08-29 05:30:00 | 261.05 | 2025-09-05 05:30:00 | 278.96 | PARTIAL | 0.50 | 6.86% |
| BUY | retest1 | 2025-08-29 05:30:00 | 261.05 | 2025-09-16 05:30:00 | 267.30 | STOP_HIT | 0.50 | 2.39% |
| BUY | retest1 | 2025-10-08 05:30:00 | 286.45 | 2025-10-15 05:30:00 | 303.35 | PARTIAL | 0.50 | 5.90% |
| BUY | retest1 | 2025-10-08 05:30:00 | 286.45 | 2025-11-13 05:30:00 | 315.45 | TARGET_HIT | 0.50 | 10.12% |
| BUY | retest1 | 2025-12-31 05:30:00 | 315.80 | 2026-01-09 05:30:00 | 305.02 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest1 | 2026-01-16 05:30:00 | 324.60 | 2026-01-19 05:30:00 | 311.48 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest1 | 2026-02-18 05:30:00 | 325.80 | 2026-03-02 05:30:00 | 312.96 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest1 | 2026-04-06 05:30:00 | 318.15 | 2026-04-21 05:30:00 | 320.40 | STOP_HIT | 1.00 | 0.71% |

# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 597.70
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
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 3
- **Avg / median % per leg:** 1.21% / 0.00%
- **Sum % (uncompounded):** 10.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 0 | 6 | 3 | 1.21% | 10.9% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 6 | 3 | 1.21% | 10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 6 | 3 | 1.21% | 10.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 05:30:00 | 335.55 | 246.74 | 316.82 | Stage2 pullback-breakout RSI=63 vol=5.6x ATR=12.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 05:30:00 | 361.43 | 250.52 | 325.74 | T1 booked 50% @ 361.43 |
| Stop hit — per-position SL triggered | 2023-08-10 05:30:00 | 342.85 | 258.28 | 339.42 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 05:30:00 | 355.59 | 263.34 | 342.24 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=12.41 |
| Stop hit — per-position SL triggered | 2023-09-04 05:30:00 | 349.38 | 271.67 | 347.45 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-09-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 05:30:00 | 364.23 | 274.90 | 349.79 | Stage2 pullback-breakout RSI=64 vol=7.2x ATR=11.38 |
| Stop hit — per-position SL triggered | 2023-09-12 05:30:00 | 347.16 | 276.39 | 349.78 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2023-10-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 05:30:00 | 358.25 | 285.54 | 347.56 | Stage2 pullback-breakout RSI=60 vol=3.3x ATR=10.86 |
| Stop hit — per-position SL triggered | 2023-10-18 05:30:00 | 348.02 | 292.22 | 352.65 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 05:30:00 | 369.32 | 321.45 | 358.43 | Stage2 pullback-breakout RSI=59 vol=5.5x ATR=13.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 05:30:00 | 395.48 | 322.13 | 361.45 | T1 booked 50% @ 395.48 |
| Stop hit — per-position SL triggered | 2024-02-12 05:30:00 | 369.32 | 324.28 | 366.05 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2024-03-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 05:30:00 | 361.77 | 326.41 | 340.82 | Stage2 pullback-breakout RSI=59 vol=5.9x ATR=14.48 |
| Stop hit — per-position SL triggered | 2024-03-13 05:30:00 | 340.05 | 326.60 | 341.27 | SL hit (bars_held=1) |

### Cycle 7 — BUY (started 2024-03-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 05:30:00 | 376.43 | 327.45 | 346.47 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=17.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 05:30:00 | 411.07 | 332.32 | 368.61 | T1 booked 50% @ 411.07 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-25 05:30:00 | 335.55 | 2023-07-31 05:30:00 | 361.43 | PARTIAL | 0.50 | 7.71% |
| BUY | retest1 | 2023-07-25 05:30:00 | 335.55 | 2023-08-10 05:30:00 | 342.85 | STOP_HIT | 0.50 | 2.18% |
| BUY | retest1 | 2023-08-21 05:30:00 | 355.59 | 2023-09-04 05:30:00 | 349.38 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest1 | 2023-09-08 05:30:00 | 364.23 | 2023-09-12 05:30:00 | 347.16 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest1 | 2023-10-04 05:30:00 | 358.25 | 2023-10-18 05:30:00 | 348.02 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest1 | 2024-02-05 05:30:00 | 369.32 | 2024-02-06 05:30:00 | 395.48 | PARTIAL | 0.50 | 7.08% |
| BUY | retest1 | 2024-02-05 05:30:00 | 369.32 | 2024-02-12 05:30:00 | 369.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-12 05:30:00 | 361.77 | 2024-03-13 05:30:00 | 340.05 | STOP_HIT | 1.00 | -6.00% |
| BUY | retest1 | 2024-03-15 05:30:00 | 376.43 | 2024-04-02 05:30:00 | 411.07 | PARTIAL | 0.50 | 9.20% |

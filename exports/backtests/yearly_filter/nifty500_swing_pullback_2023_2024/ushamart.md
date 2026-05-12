# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 472.85
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 1.15% / 0.61%
- **Sum % (uncompounded):** 12.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 1 | 7 | 3 | 1.15% | 12.7% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 1 | 7 | 3 | 1.15% | 12.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 1 | 7 | 3 | 1.15% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 294.85 | 205.41 | 275.97 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=9.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 314.19 | 209.38 | 286.30 | T1 booked 50% @ 314.19 |
| Target hit | 2023-08-14 00:00:00 | 324.10 | 231.65 | 326.30 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 00:00:00 | 347.00 | 235.70 | 329.47 | Stage2 pullback-breakout RSI=69 vol=2.1x ATR=10.23 |
| Stop hit — per-position SL triggered | 2023-09-04 00:00:00 | 349.10 | 246.49 | 342.35 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 00:00:00 | 333.90 | 277.55 | 313.61 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=13.89 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 322.15 | 282.31 | 322.18 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 312.65 | 286.05 | 304.81 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=12.08 |
| Stop hit — per-position SL triggered | 2024-01-09 00:00:00 | 294.53 | 286.60 | 304.66 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 322.20 | 287.14 | 306.43 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=12.48 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 303.48 | 288.37 | 308.61 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 340.50 | 288.89 | 311.64 | Stage2 pullback-breakout RSI=64 vol=5.0x ATR=14.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 00:00:00 | 369.23 | 290.03 | 317.92 | T1 booked 50% @ 369.23 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 340.50 | 290.70 | 321.67 | SL hit (bars_held=3) |

### Cycle 7 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 333.45 | 299.13 | 326.19 | Stage2 pullback-breakout RSI=53 vol=4.6x ATR=16.41 |
| Stop hit — per-position SL triggered | 2024-02-29 00:00:00 | 308.83 | 300.02 | 324.82 | SL hit (bars_held=4) |

### Cycle 8 — BUY (started 2024-04-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 00:00:00 | 364.30 | 305.05 | 332.01 | Stage2 pullback-breakout RSI=69 vol=3.1x ATR=14.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 00:00:00 | 392.62 | 309.66 | 350.47 | T1 booked 50% @ 392.62 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 371.25 | 310.90 | 354.38 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 294.85 | 2023-07-17 00:00:00 | 314.19 | PARTIAL | 0.50 | 6.56% |
| BUY | retest1 | 2023-07-11 00:00:00 | 294.85 | 2023-08-14 00:00:00 | 324.10 | TARGET_HIT | 0.50 | 9.92% |
| BUY | retest1 | 2023-08-21 00:00:00 | 347.00 | 2023-09-04 00:00:00 | 349.10 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest1 | 2023-11-21 00:00:00 | 333.90 | 2023-12-06 00:00:00 | 322.15 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest1 | 2024-01-04 00:00:00 | 312.65 | 2024-01-09 00:00:00 | 294.53 | STOP_HIT | 1.00 | -5.80% |
| BUY | retest1 | 2024-01-11 00:00:00 | 322.20 | 2024-01-18 00:00:00 | 303.48 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest1 | 2024-01-19 00:00:00 | 340.50 | 2024-01-23 00:00:00 | 369.23 | PARTIAL | 0.50 | 8.44% |
| BUY | retest1 | 2024-01-19 00:00:00 | 340.50 | 2024-01-24 00:00:00 | 340.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-23 00:00:00 | 333.45 | 2024-02-29 00:00:00 | 308.83 | STOP_HIT | 1.00 | -7.38% |
| BUY | retest1 | 2024-04-18 00:00:00 | 364.30 | 2024-04-30 00:00:00 | 392.62 | PARTIAL | 0.50 | 7.77% |
| BUY | retest1 | 2024-04-18 00:00:00 | 364.30 | 2024-05-03 00:00:00 | 371.25 | STOP_HIT | 0.50 | 1.91% |

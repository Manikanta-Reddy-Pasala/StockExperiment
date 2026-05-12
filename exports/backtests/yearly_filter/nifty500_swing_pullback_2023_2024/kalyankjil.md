# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 424.55
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 4.91% / 9.15%
- **Sum % (uncompounded):** 29.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.91% | 29.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.91% | 29.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.91% | 29.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 258.15 | 156.87 | 228.18 | Stage2 pullback-breakout RSI=68 vol=3.9x ATR=11.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 00:00:00 | 281.77 | 163.44 | 247.61 | T1 booked 50% @ 281.77 |
| Target hit | 2023-12-07 00:00:00 | 321.90 | 210.35 | 324.37 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-02-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 00:00:00 | 391.65 | 264.75 | 358.68 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=19.00 |
| Stop hit — per-position SL triggered | 2024-02-20 00:00:00 | 363.15 | 267.99 | 362.59 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-02-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 00:00:00 | 396.85 | 275.67 | 372.37 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=17.73 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 370.25 | 286.42 | 388.20 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 400.85 | 293.51 | 383.63 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=19.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 00:00:00 | 439.26 | 297.33 | 393.92 | T1 booked 50% @ 439.26 |
| Stop hit — per-position SL triggered | 2024-04-18 00:00:00 | 400.85 | 310.45 | 412.01 | SL hit (bars_held=14) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-06 00:00:00 | 258.15 | 2023-10-16 00:00:00 | 281.77 | PARTIAL | 0.50 | 9.15% |
| BUY | retest1 | 2023-10-06 00:00:00 | 258.15 | 2023-12-07 00:00:00 | 321.90 | TARGET_HIT | 0.50 | 24.69% |
| BUY | retest1 | 2024-02-15 00:00:00 | 391.65 | 2024-02-20 00:00:00 | 363.15 | STOP_HIT | 1.00 | -7.28% |
| BUY | retest1 | 2024-02-29 00:00:00 | 396.85 | 2024-03-13 00:00:00 | 370.25 | STOP_HIT | 1.00 | -6.70% |
| BUY | retest1 | 2024-03-26 00:00:00 | 400.85 | 2024-04-01 00:00:00 | 439.26 | PARTIAL | 0.50 | 9.58% |
| BUY | retest1 | 2024-03-26 00:00:00 | 400.85 | 2024-04-18 00:00:00 | 400.85 | STOP_HIT | 0.50 | 0.00% |

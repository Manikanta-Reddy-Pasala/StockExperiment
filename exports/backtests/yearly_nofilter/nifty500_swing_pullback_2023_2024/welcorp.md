# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1291.90
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 6.11% / 6.40%
- **Sum % (uncompounded):** 48.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.11% | 48.9% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.11% | 48.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.11% | 48.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 00:00:00 | 285.55 | 235.96 | 265.46 | Stage2 pullback-breakout RSI=69 vol=4.4x ATR=9.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 00:00:00 | 303.75 | 237.71 | 273.30 | T1 booked 50% @ 303.75 |
| Target hit | 2023-08-16 00:00:00 | 319.25 | 256.47 | 320.45 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 335.85 | 262.58 | 321.17 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=10.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 357.35 | 268.14 | 334.44 | T1 booked 50% @ 357.35 |
| Target hit | 2023-10-23 00:00:00 | 411.90 | 302.35 | 413.33 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 00:00:00 | 570.80 | 381.85 | 537.32 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=21.08 |
| Stop hit — per-position SL triggered | 2024-01-09 00:00:00 | 539.18 | 392.07 | 545.77 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-01-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-17 00:00:00 | 570.35 | 400.96 | 545.66 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=20.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 00:00:00 | 612.04 | 408.52 | 561.40 | T1 booked 50% @ 612.04 |
| Stop hit — per-position SL triggered | 2024-01-25 00:00:00 | 570.35 | 412.00 | 565.45 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 574.70 | 473.05 | 552.45 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=22.88 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 573.90 | 483.40 | 571.53 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-05 00:00:00 | 285.55 | 2023-07-10 00:00:00 | 303.75 | PARTIAL | 0.50 | 6.38% |
| BUY | retest1 | 2023-07-05 00:00:00 | 285.55 | 2023-08-16 00:00:00 | 319.25 | TARGET_HIT | 0.50 | 11.80% |
| BUY | retest1 | 2023-08-30 00:00:00 | 335.85 | 2023-09-08 00:00:00 | 357.35 | PARTIAL | 0.50 | 6.40% |
| BUY | retest1 | 2023-08-30 00:00:00 | 335.85 | 2023-10-23 00:00:00 | 411.90 | TARGET_HIT | 0.50 | 22.64% |
| BUY | retest1 | 2024-01-01 00:00:00 | 570.80 | 2024-01-09 00:00:00 | 539.18 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest1 | 2024-01-17 00:00:00 | 570.35 | 2024-01-23 00:00:00 | 612.04 | PARTIAL | 0.50 | 7.31% |
| BUY | retest1 | 2024-01-17 00:00:00 | 570.35 | 2024-01-25 00:00:00 | 570.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 00:00:00 | 574.70 | 2024-05-09 00:00:00 | 573.90 | STOP_HIT | 1.00 | -0.14% |

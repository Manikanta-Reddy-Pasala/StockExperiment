# Sapphire Foods India Ltd. (SAPPHIRE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 178.36
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
- **Avg / median % per leg:** 2.09% / 3.47%
- **Sum % (uncompounded):** 12.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.09% | 12.5% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.09% | 12.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.09% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 00:00:00 | 275.09 | 267.31 | 272.10 | Stage2 pullback-breakout RSI=54 vol=2.7x ATR=7.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 289.71 | 268.11 | 276.33 | T1 booked 50% @ 289.71 |
| Target hit | 2023-09-13 00:00:00 | 284.64 | 269.86 | 285.14 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 289.43 | 272.88 | 276.78 | Stage2 pullback-breakout RSI=69 vol=1.5x ATR=7.57 |
| Stop hit — per-position SL triggered | 2023-12-05 00:00:00 | 278.07 | 273.00 | 277.50 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 301.40 | 277.11 | 282.16 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=9.78 |
| Stop hit — per-position SL triggered | 2024-03-07 00:00:00 | 300.18 | 279.30 | 293.62 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 00:00:00 | 303.77 | 279.95 | 291.59 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=12.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 00:00:00 | 328.25 | 280.99 | 297.72 | T1 booked 50% @ 328.25 |
| Stop hit — per-position SL triggered | 2024-04-03 00:00:00 | 303.77 | 282.92 | 304.78 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-25 00:00:00 | 275.09 | 2023-09-04 00:00:00 | 289.71 | PARTIAL | 0.50 | 5.32% |
| BUY | retest1 | 2023-08-25 00:00:00 | 275.09 | 2023-09-13 00:00:00 | 284.64 | TARGET_HIT | 0.50 | 3.47% |
| BUY | retest1 | 2023-12-04 00:00:00 | 289.43 | 2023-12-05 00:00:00 | 278.07 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2024-02-23 00:00:00 | 301.40 | 2024-03-07 00:00:00 | 300.18 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-03-19 00:00:00 | 303.77 | 2024-03-22 00:00:00 | 328.25 | PARTIAL | 0.50 | 8.06% |
| BUY | retest1 | 2024-03-19 00:00:00 | 303.77 | 2024-04-03 00:00:00 | 303.77 | STOP_HIT | 0.50 | 0.00% |

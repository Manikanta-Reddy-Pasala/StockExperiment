# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 354.15
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.98% / 4.83%
- **Sum % (uncompounded):** 11.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.98% | 11.9% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.98% | 11.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.98% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 272.80 | 214.27 | 263.32 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=6.44 |
| Stop hit — per-position SL triggered | 2023-09-18 00:00:00 | 265.95 | 219.63 | 267.52 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 268.80 | 226.64 | 262.19 | Stage2 pullback-breakout RSI=59 vol=3.2x ATR=5.84 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 260.04 | 227.72 | 262.45 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 267.75 | 230.27 | 258.14 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 00:00:00 | 280.68 | 233.78 | 266.29 | T1 booked 50% @ 280.68 |
| Target hit | 2023-12-20 00:00:00 | 287.40 | 244.95 | 289.90 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 334.95 | 263.17 | 319.44 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=10.15 |
| Stop hit — per-position SL triggered | 2024-02-14 00:00:00 | 336.65 | 270.46 | 331.69 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 319.45 | 283.97 | 312.61 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=7.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 335.42 | 285.27 | 319.56 | T1 booked 50% @ 335.42 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-04 00:00:00 | 272.80 | 2023-09-18 00:00:00 | 265.95 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest1 | 2023-10-17 00:00:00 | 268.80 | 2023-10-20 00:00:00 | 260.04 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest1 | 2023-11-06 00:00:00 | 267.75 | 2023-11-17 00:00:00 | 280.68 | PARTIAL | 0.50 | 4.83% |
| BUY | retest1 | 2023-11-06 00:00:00 | 267.75 | 2023-12-20 00:00:00 | 287.40 | TARGET_HIT | 0.50 | 7.34% |
| BUY | retest1 | 2024-01-31 00:00:00 | 334.95 | 2024-02-14 00:00:00 | 336.65 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest1 | 2024-04-04 00:00:00 | 319.45 | 2024-04-08 00:00:00 | 335.42 | PARTIAL | 0.50 | 5.00% |

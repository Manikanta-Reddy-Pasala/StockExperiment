# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 915.95
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
- **Avg / median % per leg:** 9.06% / 5.95%
- **Sum % (uncompounded):** 72.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 9.06% | 72.5% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 9.06% | 72.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 9.06% | 72.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 362.40 | 306.98 | 348.70 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=11.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 00:00:00 | 385.24 | 308.37 | 353.89 | T1 booked 50% @ 385.24 |
| Target hit | 2023-09-11 00:00:00 | 451.25 | 351.37 | 463.28 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 461.45 | 368.26 | 443.79 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=13.84 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 440.69 | 374.52 | 451.68 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-11-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 00:00:00 | 456.30 | 382.75 | 446.01 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=13.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 483.46 | 385.39 | 452.82 | T1 booked 50% @ 483.46 |
| Stop hit — per-position SL triggered | 2023-11-17 00:00:00 | 456.30 | 386.86 | 453.97 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 469.90 | 391.88 | 453.46 | Stage2 pullback-breakout RSI=61 vol=3.8x ATR=13.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 496.93 | 395.50 | 464.18 | T1 booked 50% @ 496.93 |
| Target hit | 2024-02-12 00:00:00 | 657.70 | 486.15 | 678.00 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 653.40 | 537.24 | 613.14 | Stage2 pullback-breakout RSI=63 vol=3.2x ATR=24.06 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 617.31 | 542.71 | 620.55 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 362.40 | 2023-07-18 00:00:00 | 385.24 | PARTIAL | 0.50 | 6.30% |
| BUY | retest1 | 2023-07-14 00:00:00 | 362.40 | 2023-09-11 00:00:00 | 451.25 | TARGET_HIT | 0.50 | 24.52% |
| BUY | retest1 | 2023-10-12 00:00:00 | 461.45 | 2023-10-23 00:00:00 | 440.69 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest1 | 2023-11-10 00:00:00 | 456.30 | 2023-11-15 00:00:00 | 483.46 | PARTIAL | 0.50 | 5.95% |
| BUY | retest1 | 2023-11-10 00:00:00 | 456.30 | 2023-11-17 00:00:00 | 456.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 00:00:00 | 469.90 | 2023-12-06 00:00:00 | 496.93 | PARTIAL | 0.50 | 5.75% |
| BUY | retest1 | 2023-11-30 00:00:00 | 469.90 | 2024-02-12 00:00:00 | 657.70 | TARGET_HIT | 0.50 | 39.97% |
| BUY | retest1 | 2024-04-24 00:00:00 | 653.40 | 2024-05-03 00:00:00 | 617.31 | STOP_HIT | 1.00 | -5.52% |

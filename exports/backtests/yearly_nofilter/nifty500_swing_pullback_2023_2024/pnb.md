# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 107.24
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 3 / 1 / 3
- **Avg / median % per leg:** 11.75% / 5.42%
- **Sum % (uncompounded):** 82.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 11.75% | 82.3% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 11.75% | 82.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 3 | 1 | 3 | 11.75% | 82.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 53.85 | 48.07 | 51.44 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 56.25 | 48.16 | 51.97 | T1 booked 50% @ 56.25 |
| Target hit | 2023-08-02 00:00:00 | 59.50 | 50.66 | 60.38 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 65.40 | 52.87 | 62.51 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 68.67 | 53.70 | 64.65 | T1 booked 50% @ 68.67 |
| Target hit | 2023-10-09 00:00:00 | 73.35 | 57.54 | 75.49 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 80.70 | 63.17 | 77.28 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 85.08 | 63.59 | 78.57 | T1 booked 50% @ 85.08 |
| Target hit | 2024-02-28 00:00:00 | 120.75 | 83.31 | 122.03 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 135.15 | 91.70 | 124.29 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 128.53 | 94.93 | 129.28 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 53.85 | 2023-07-04 00:00:00 | 56.25 | PARTIAL | 0.50 | 4.46% |
| BUY | retest1 | 2023-07-03 00:00:00 | 53.85 | 2023-08-02 00:00:00 | 59.50 | TARGET_HIT | 0.50 | 10.49% |
| BUY | retest1 | 2023-09-01 00:00:00 | 65.40 | 2023-09-11 00:00:00 | 68.67 | PARTIAL | 0.50 | 5.01% |
| BUY | retest1 | 2023-09-01 00:00:00 | 65.40 | 2023-10-09 00:00:00 | 73.35 | TARGET_HIT | 0.50 | 12.16% |
| BUY | retest1 | 2023-12-01 00:00:00 | 80.70 | 2023-12-05 00:00:00 | 85.08 | PARTIAL | 0.50 | 5.42% |
| BUY | retest1 | 2023-12-01 00:00:00 | 80.70 | 2024-02-28 00:00:00 | 120.75 | TARGET_HIT | 0.50 | 49.63% |
| BUY | retest1 | 2024-04-03 00:00:00 | 135.15 | 2024-04-16 00:00:00 | 128.53 | STOP_HIT | 1.00 | -4.90% |

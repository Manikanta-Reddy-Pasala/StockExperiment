# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 134.34
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
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.36% / 2.69%
- **Sum % (uncompounded):** 9.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 4 | 2 | 1.36% | 9.5% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 4 | 2 | 1.36% | 9.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 4 | 2 | 1.36% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 63.70 | 58.34 | 60.86 | Stage2 pullback-breakout RSI=63 vol=3.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 00:00:00 | 66.47 | 58.61 | 62.31 | T1 booked 50% @ 66.47 |
| Target hit | 2023-08-02 00:00:00 | 65.45 | 60.01 | 66.52 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 00:00:00 | 72.95 | 61.71 | 67.75 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 00:00:00 | 76.91 | 62.43 | 70.64 | T1 booked 50% @ 76.91 |
| Stop hit — per-position SL triggered | 2023-09-28 00:00:00 | 74.91 | 62.91 | 71.98 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 00:00:00 | 76.16 | 64.84 | 73.47 | Stage2 pullback-breakout RSI=59 vol=3.2x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-11-10 00:00:00 | 77.48 | 66.00 | 75.79 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 80.89 | 66.27 | 76.44 | Stage2 pullback-breakout RSI=70 vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-11-24 00:00:00 | 77.97 | 67.29 | 78.06 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 123.37 | 92.83 | 118.71 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 118.63 | 94.62 | 120.80 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 63.70 | 2023-07-07 00:00:00 | 66.47 | PARTIAL | 0.50 | 4.35% |
| BUY | retest1 | 2023-07-03 00:00:00 | 63.70 | 2023-08-02 00:00:00 | 65.45 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2023-09-13 00:00:00 | 72.95 | 2023-09-22 00:00:00 | 76.91 | PARTIAL | 0.50 | 5.43% |
| BUY | retest1 | 2023-09-13 00:00:00 | 72.95 | 2023-09-28 00:00:00 | 74.91 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2023-10-27 00:00:00 | 76.16 | 2023-11-10 00:00:00 | 77.48 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest1 | 2023-11-13 00:00:00 | 80.89 | 2023-11-24 00:00:00 | 77.97 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest1 | 2024-04-25 00:00:00 | 123.37 | 2024-05-06 00:00:00 | 118.63 | STOP_HIT | 1.00 | -3.84% |

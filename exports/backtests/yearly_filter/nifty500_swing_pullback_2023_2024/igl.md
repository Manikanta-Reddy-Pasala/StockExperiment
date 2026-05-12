# Indraprastha Gas Ltd. (IGL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 165.99
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 1.57% / 0.00%
- **Sum % (uncompounded):** 7.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.57% | 7.9% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.57% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.57% | 7.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 00:00:00 | 247.65 | 224.66 | 239.34 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=5.24 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 247.15 | 226.59 | 243.15 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 00:00:00 | 231.33 | 226.33 | 224.91 | Stage2 pullback-breakout RSI=57 vol=3.6x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 240.79 | 226.93 | 230.28 | T1 booked 50% @ 240.79 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 231.33 | 227.08 | 231.00 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 230.93 | 227.43 | 229.71 | Stage2 pullback-breakout RSI=52 vol=1.7x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 00:00:00 | 240.11 | 227.88 | 232.02 | T1 booked 50% @ 240.11 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 230.93 | 228.17 | 233.35 | SL hit (bars_held=12) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-07 00:00:00 | 247.65 | 2023-07-21 00:00:00 | 247.15 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-29 00:00:00 | 231.33 | 2023-09-08 00:00:00 | 240.79 | PARTIAL | 0.50 | 4.09% |
| BUY | retest1 | 2023-08-29 00:00:00 | 231.33 | 2023-09-12 00:00:00 | 231.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-03 00:00:00 | 230.93 | 2023-10-16 00:00:00 | 240.11 | PARTIAL | 0.50 | 3.97% |
| BUY | retest1 | 2023-10-03 00:00:00 | 230.93 | 2023-10-19 00:00:00 | 230.93 | STOP_HIT | 0.50 | 0.00% |

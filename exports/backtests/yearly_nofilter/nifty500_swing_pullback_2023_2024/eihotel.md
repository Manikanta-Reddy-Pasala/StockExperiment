# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 331.85
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 2.28% / 2.94%
- **Sum % (uncompounded):** 13.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 4 | 1 | 2.28% | 13.7% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 4 | 1 | 2.28% | 13.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 2.28% | 13.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 214.45 | 186.64 | 209.38 | Stage2 pullback-breakout RSI=61 vol=6.5x ATR=6.46 |
| Stop hit — per-position SL triggered | 2023-07-26 00:00:00 | 214.20 | 189.40 | 213.52 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 00:00:00 | 214.95 | 192.38 | 211.40 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=5.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 00:00:00 | 226.79 | 193.13 | 214.90 | T1 booked 50% @ 226.79 |
| Target hit | 2023-09-12 00:00:00 | 238.65 | 200.95 | 241.19 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 227.90 | 208.14 | 224.97 | Stage2 pullback-breakout RSI=51 vol=3.3x ATR=8.20 |
| Stop hit — per-position SL triggered | 2023-11-12 00:00:00 | 234.60 | 210.32 | 229.38 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 243.30 | 211.51 | 231.69 | Stage2 pullback-breakout RSI=61 vol=4.1x ATR=7.61 |
| Stop hit — per-position SL triggered | 2023-12-05 00:00:00 | 239.95 | 214.01 | 235.55 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 249.40 | 214.61 | 237.18 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=7.19 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 238.61 | 217.03 | 239.93 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 214.45 | 2023-07-26 00:00:00 | 214.20 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-08-17 00:00:00 | 214.95 | 2023-08-21 00:00:00 | 226.79 | PARTIAL | 0.50 | 5.51% |
| BUY | retest1 | 2023-08-17 00:00:00 | 214.95 | 2023-09-12 00:00:00 | 238.65 | TARGET_HIT | 0.50 | 11.03% |
| BUY | retest1 | 2023-10-30 00:00:00 | 227.90 | 2023-11-12 00:00:00 | 234.60 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest1 | 2023-11-20 00:00:00 | 243.30 | 2023-12-05 00:00:00 | 239.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest1 | 2023-12-07 00:00:00 | 249.40 | 2023-12-20 00:00:00 | 238.61 | STOP_HIT | 1.00 | -4.33% |

# POWERGRID (POWERGRID)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 313.95
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -1.64% / -3.82%
- **Sum % (uncompounded):** -8.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.64% | -8.2% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.64% | -8.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.64% | -8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 00:00:00 | 349.80 | 280.88 | 338.55 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=9.48 |
| Stop hit — per-position SL triggered | 2024-08-13 00:00:00 | 335.58 | 287.15 | 342.97 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-09-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 00:00:00 | 350.05 | 299.53 | 337.89 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=6.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 00:00:00 | 362.80 | 300.16 | 340.36 | T1 booked 50% @ 362.80 |
| Stop hit — per-position SL triggered | 2024-09-30 00:00:00 | 350.05 | 301.86 | 344.70 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-11-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 00:00:00 | 329.80 | 307.76 | 322.71 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=8.41 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 317.19 | 308.01 | 322.26 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-11-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 00:00:00 | 336.95 | 308.62 | 322.04 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=8.87 |
| Stop hit — per-position SL triggered | 2024-12-04 00:00:00 | 323.64 | 310.51 | 327.54 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-30 00:00:00 | 349.80 | 2024-08-13 00:00:00 | 335.58 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest1 | 2024-09-24 00:00:00 | 350.05 | 2024-09-25 00:00:00 | 362.80 | PARTIAL | 0.50 | 3.64% |
| BUY | retest1 | 2024-09-24 00:00:00 | 350.05 | 2024-09-30 00:00:00 | 350.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 00:00:00 | 329.80 | 2024-11-13 00:00:00 | 317.19 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest1 | 2024-11-22 00:00:00 | 336.95 | 2024-12-04 00:00:00 | 323.64 | STOP_HIT | 1.00 | -3.95% |

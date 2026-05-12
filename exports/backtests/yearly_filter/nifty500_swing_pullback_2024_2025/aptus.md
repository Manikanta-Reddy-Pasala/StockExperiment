# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 282.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.06% / 1.32%
- **Sum % (uncompounded):** 15.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.06% | 15.3% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.06% | 15.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.06% | 15.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 00:00:00 | 322.40 | 317.10 | 313.12 | Stage2 pullback-breakout RSI=59 vol=3.0x ATR=9.33 |
| Stop hit — per-position SL triggered | 2024-09-12 00:00:00 | 323.65 | 317.69 | 319.37 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 336.00 | 317.87 | 320.95 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=10.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 00:00:00 | 356.95 | 319.08 | 330.06 | T1 booked 50% @ 356.95 |
| Target hit | 2024-10-03 00:00:00 | 340.45 | 322.18 | 345.28 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-10-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 00:00:00 | 369.75 | 323.02 | 348.83 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=13.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 00:00:00 | 396.88 | 325.54 | 359.06 | T1 booked 50% @ 396.88 |
| Stop hit — per-position SL triggered | 2024-10-21 00:00:00 | 369.75 | 328.07 | 365.93 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-29 00:00:00 | 322.40 | 2024-09-12 00:00:00 | 323.65 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest1 | 2024-09-13 00:00:00 | 336.00 | 2024-09-20 00:00:00 | 356.95 | PARTIAL | 0.50 | 6.24% |
| BUY | retest1 | 2024-09-13 00:00:00 | 336.00 | 2024-10-03 00:00:00 | 340.45 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2024-10-07 00:00:00 | 369.75 | 2024-10-14 00:00:00 | 396.88 | PARTIAL | 0.50 | 7.34% |
| BUY | retest1 | 2024-10-07 00:00:00 | 369.75 | 2024-10-21 00:00:00 | 369.75 | STOP_HIT | 0.50 | 0.00% |

# REC Ltd. (RECLTD)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 359.40
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
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 23.75% / 8.11%
- **Sum % (uncompounded):** 118.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 23.75% | 118.8% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 23.75% | 118.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 23.75% | 118.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 05:30:00 | 172.95 | 130.22 | 162.64 | Stage2 pullback-breakout RSI=68 vol=3.4x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 05:30:00 | 181.90 | 131.63 | 166.76 | T1 booked 50% @ 181.90 |
| Target hit | 2023-10-23 05:30:00 | 278.30 | 187.24 | 282.16 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 05:30:00 | 302.40 | 193.55 | 282.32 | Stage2 pullback-breakout RSI=63 vol=4.1x ATR=12.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-12 05:30:00 | 326.93 | 201.74 | 299.45 | T1 booked 50% @ 326.93 |
| Target hit | 2024-02-12 05:30:00 | 453.55 | 306.37 | 473.75 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-03-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 05:30:00 | 484.15 | 334.82 | 465.28 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=17.53 |
| Stop hit — per-position SL triggered | 2024-03-13 05:30:00 | 457.86 | 337.41 | 465.25 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-24 05:30:00 | 172.95 | 2023-07-27 05:30:00 | 181.90 | PARTIAL | 0.50 | 5.18% |
| BUY | retest1 | 2023-07-24 05:30:00 | 172.95 | 2023-10-23 05:30:00 | 278.30 | TARGET_HIT | 0.50 | 60.91% |
| BUY | retest1 | 2023-11-02 05:30:00 | 302.40 | 2023-11-12 05:30:00 | 326.93 | PARTIAL | 0.50 | 8.11% |
| BUY | retest1 | 2023-11-02 05:30:00 | 302.40 | 2024-02-12 05:30:00 | 453.55 | TARGET_HIT | 0.50 | 49.98% |
| BUY | retest1 | 2024-03-11 05:30:00 | 484.15 | 2024-03-13 05:30:00 | 457.86 | STOP_HIT | 1.00 | -5.43% |

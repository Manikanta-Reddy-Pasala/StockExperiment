# TMPV (TMPV)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 355.45
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
- **Avg / median % per leg:** 9.36% / 4.73%
- **Sum % (uncompounded):** 46.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 9.36% | 46.8% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 9.36% | 46.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 9.36% | 46.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 00:00:00 | 380.15 | 320.13 | 371.62 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=7.19 |
| Stop hit — per-position SL triggered | 2023-09-25 00:00:00 | 374.97 | 325.92 | 377.29 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 392.42 | 340.18 | 387.00 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=9.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 410.99 | 345.07 | 393.84 | T1 booked 50% @ 410.99 |
| Target hit | 2023-12-20 00:00:00 | 427.42 | 362.23 | 429.29 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 00:00:00 | 449.03 | 365.23 | 432.55 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 00:00:00 | 466.46 | 367.20 | 438.47 | T1 booked 50% @ 466.46 |
| Target hit | 2024-03-14 00:00:00 | 586.52 | 441.65 | 588.60 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-08 00:00:00 | 380.15 | 2023-09-25 00:00:00 | 374.97 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2023-11-03 00:00:00 | 392.42 | 2023-11-16 00:00:00 | 410.99 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2023-11-03 00:00:00 | 392.42 | 2023-12-20 00:00:00 | 427.42 | TARGET_HIT | 0.50 | 8.92% |
| BUY | retest1 | 2023-12-27 00:00:00 | 449.03 | 2023-12-29 00:00:00 | 466.46 | PARTIAL | 0.50 | 3.88% |
| BUY | retest1 | 2023-12-27 00:00:00 | 449.03 | 2024-03-14 00:00:00 | 586.52 | TARGET_HIT | 0.50 | 30.62% |

# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 717.60
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 10.10% / 4.66%
- **Sum % (uncompounded):** 40.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 1 | 1 | 2 | 10.10% | 40.4% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 10.10% | 40.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 10.10% | 40.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 05:30:00 | 675.95 | 652.69 | 665.95 | Stage2 pullback-breakout RSI=56 vol=8.3x ATR=15.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 05:30:00 | 707.42 | 654.84 | 672.88 | T1 booked 50% @ 707.42 |
| Stop hit — per-position SL triggered | 2025-08-07 05:30:00 | 690.25 | 656.30 | 680.79 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 05:30:00 | 686.05 | 662.14 | 681.48 | Stage2 pullback-breakout RSI=54 vol=3.2x ATR=13.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 05:30:00 | 713.76 | 663.48 | 686.61 | T1 booked 50% @ 713.76 |
| Target hit | 2025-10-17 05:30:00 | 889.00 | 710.05 | 900.54 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-21 05:30:00 | 675.95 | 2025-08-04 05:30:00 | 707.42 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2025-07-21 05:30:00 | 675.95 | 2025-08-07 05:30:00 | 690.25 | STOP_HIT | 0.50 | 2.12% |
| BUY | retest1 | 2025-09-11 05:30:00 | 686.05 | 2025-09-17 05:30:00 | 713.76 | PARTIAL | 0.50 | 4.04% |
| BUY | retest1 | 2025-09-11 05:30:00 | 686.05 | 2025-10-17 05:30:00 | 889.00 | TARGET_HIT | 0.50 | 29.58% |

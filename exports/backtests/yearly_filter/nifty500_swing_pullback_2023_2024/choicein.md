# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 674.50
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
| TARGET_HIT | 0 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 0 / 1 / 2
- **Avg / median % per leg:** 4.26% / 4.61%
- **Sum % (uncompounded):** 12.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 0 | 1 | 2 | 4.26% | 12.8% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 1 | 2 | 4.26% | 12.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 1 | 2 | 4.26% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 215.53 | 176.27 | 207.54 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-12 00:00:00 | 225.46 | 177.59 | 211.06 | T1 booked 50% @ 225.46 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 218.88 | 180.48 | 215.63 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 273.40 | 225.34 | 262.43 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=9.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 291.48 | 226.51 | 266.54 | T1 booked 50% @ 291.48 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-08 00:00:00 | 215.53 | 2023-11-12 00:00:00 | 225.46 | PARTIAL | 0.50 | 4.61% |
| BUY | retest1 | 2023-11-08 00:00:00 | 215.53 | 2023-11-22 00:00:00 | 218.88 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2024-03-28 00:00:00 | 273.40 | 2024-04-02 00:00:00 | 291.48 | PARTIAL | 0.50 | 6.61% |

# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 3769.10
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 1.52% / 2.99%
- **Sum % (uncompounded):** 4.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.52% | 4.6% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.52% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.52% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 3698.25 | 3213.79 | 3554.18 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=133.39 |
| Stop hit — per-position SL triggered | 2024-03-04 00:00:00 | 3498.16 | 3239.45 | 3595.39 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 3476.70 | 3242.74 | 3269.80 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=121.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 00:00:00 | 3719.86 | 3263.52 | 3421.50 | T1 booked 50% @ 3719.86 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 3580.65 | 3293.36 | 3544.49 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-02-26 00:00:00 | 3698.25 | 2024-03-04 00:00:00 | 3498.16 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest1 | 2024-04-10 00:00:00 | 3476.70 | 2024-04-22 00:00:00 | 3719.86 | PARTIAL | 0.50 | 6.99% |
| BUY | retest1 | 2024-04-10 00:00:00 | 3476.70 | 2024-05-03 00:00:00 | 3580.65 | STOP_HIT | 0.50 | 2.99% |

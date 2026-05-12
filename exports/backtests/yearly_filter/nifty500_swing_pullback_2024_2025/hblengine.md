# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (661 bars)
- **Last close:** 848.15
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.20% / 7.53%
- **Sum % (uncompounded):** 15.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.20% | 16.0% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.20% | 16.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.20% | 16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 00:00:00 | 523.80 | 443.72 | 501.05 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=19.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 00:00:00 | 563.24 | 445.83 | 510.07 | T1 booked 50% @ 563.24 |
| Target hit | 2024-08-05 00:00:00 | 567.35 | 475.32 | 595.61 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-08-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 00:00:00 | 656.55 | 487.54 | 611.35 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=33.86 |
| Stop hit — per-position SL triggered | 2024-09-02 00:00:00 | 636.30 | 502.96 | 634.16 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 00:00:00 | 629.15 | 524.34 | 616.72 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=20.19 |
| Stop hit — per-position SL triggered | 2024-10-07 00:00:00 | 598.86 | 527.05 | 616.15 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-11-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 00:00:00 | 616.40 | 538.95 | 564.07 | Stage2 pullback-breakout RSI=63 vol=3.7x ATR=24.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 00:00:00 | 665.94 | 545.51 | 601.56 | T1 booked 50% @ 665.94 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 00:00:00 | 523.80 | 2024-07-05 00:00:00 | 563.24 | PARTIAL | 0.50 | 7.53% |
| BUY | retest1 | 2024-07-03 00:00:00 | 523.80 | 2024-08-05 00:00:00 | 567.35 | TARGET_HIT | 0.50 | 8.31% |
| BUY | retest1 | 2024-08-19 00:00:00 | 656.55 | 2024-09-02 00:00:00 | 636.30 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2024-10-01 00:00:00 | 629.15 | 2024-10-07 00:00:00 | 598.86 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest1 | 2024-11-27 00:00:00 | 616.40 | 2024-12-06 00:00:00 | 665.94 | PARTIAL | 0.50 | 8.04% |

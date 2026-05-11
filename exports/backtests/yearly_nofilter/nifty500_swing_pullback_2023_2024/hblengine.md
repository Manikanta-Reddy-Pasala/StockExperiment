# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (910 bars)
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 3
- **Avg / median % per leg:** 12.93% / 9.67%
- **Sum % (uncompounded):** 77.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 1 | 3 | 12.93% | 77.6% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 1 | 3 | 12.93% | 77.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 2 | 1 | 3 | 12.93% | 77.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 266.90 | 166.52 | 258.22 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=12.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 00:00:00 | 292.71 | 171.05 | 266.16 | T1 booked 50% @ 292.71 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 266.90 | 183.26 | 280.25 | SL hit (bars_held=15) |

### Cycle 2 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 308.20 | 192.24 | 285.86 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=15.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 00:00:00 | 339.21 | 204.39 | 309.06 | T1 booked 50% @ 339.21 |
| Target hit | 2024-01-23 00:00:00 | 454.70 | 285.21 | 456.00 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 492.70 | 379.54 | 470.40 | Stage2 pullback-breakout RSI=57 vol=2.2x ATR=22.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 00:00:00 | 537.75 | 385.08 | 487.23 | T1 booked 50% @ 537.75 |
| Target hit | 2024-05-09 00:00:00 | 498.50 | 391.81 | 500.46 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-29 00:00:00 | 266.90 | 2023-10-06 00:00:00 | 292.71 | PARTIAL | 0.50 | 9.67% |
| BUY | retest1 | 2023-09-29 00:00:00 | 266.90 | 2023-10-23 00:00:00 | 266.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 00:00:00 | 308.20 | 2023-11-20 00:00:00 | 339.21 | PARTIAL | 0.50 | 10.06% |
| BUY | retest1 | 2023-11-06 00:00:00 | 308.20 | 2024-01-23 00:00:00 | 454.70 | TARGET_HIT | 0.50 | 47.53% |
| BUY | retest1 | 2024-04-25 00:00:00 | 492.70 | 2024-05-02 00:00:00 | 537.75 | PARTIAL | 0.50 | 9.14% |
| BUY | retest1 | 2024-04-25 00:00:00 | 492.70 | 2024-05-09 00:00:00 | 498.50 | TARGET_HIT | 0.50 | 1.18% |

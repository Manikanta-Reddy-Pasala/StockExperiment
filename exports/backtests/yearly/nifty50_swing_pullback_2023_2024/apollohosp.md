# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 8097.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 1.73% / 4.20%
- **Sum % (uncompounded):** 6.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 1.73% | 6.9% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 1.73% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 1.73% | 6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 05:30:00 | 5283.05 | 4562.80 | 5054.38 | Stage2 pullback-breakout RSI=69 vol=2.6x ATR=108.58 |
| Stop hit — per-position SL triggered | 2023-07-10 05:30:00 | 5120.18 | 4574.48 | 5072.20 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-09-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 05:30:00 | 4984.50 | 4719.40 | 4920.49 | Stage2 pullback-breakout RSI=53 vol=1.7x ATR=96.67 |
| Stop hit — per-position SL triggered | 2023-09-20 05:30:00 | 5006.30 | 4749.85 | 4997.31 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 05:30:00 | 5153.20 | 4811.77 | 4946.08 | Stage2 pullback-breakout RSI=62 vol=2.9x ATR=108.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 05:30:00 | 5369.76 | 4853.96 | 5154.78 | T1 booked 50% @ 5369.76 |
| Target hit | 2023-12-12 05:30:00 | 5428.95 | 4949.76 | 5437.15 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 05:30:00 | 5283.05 | 2023-07-10 05:30:00 | 5120.18 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2023-09-05 05:30:00 | 4984.50 | 2023-09-20 05:30:00 | 5006.30 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest1 | 2023-11-03 05:30:00 | 5153.20 | 2023-11-17 05:30:00 | 5369.76 | PARTIAL | 0.50 | 4.20% |
| BUY | retest1 | 2023-11-03 05:30:00 | 5153.20 | 2023-12-12 05:30:00 | 5428.95 | TARGET_HIT | 0.50 | 5.35% |

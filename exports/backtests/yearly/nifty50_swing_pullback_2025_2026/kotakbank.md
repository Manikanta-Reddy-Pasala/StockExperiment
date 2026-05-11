# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 380.80
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.45% / 3.30%
- **Sum % (uncompounded):** 1.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.45% | 1.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.45% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.45% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 05:30:00 | 444.90 | 398.57 | 431.79 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=8.23 |
| Stop hit — per-position SL triggered | 2025-07-16 05:30:00 | 432.55 | 401.03 | 435.64 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2025-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 05:30:00 | 410.06 | 400.87 | 397.28 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=6.29 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 400.62 | 401.18 | 401.15 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-10-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 05:30:00 | 412.66 | 401.24 | 401.80 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 05:30:00 | 426.26 | 401.71 | 405.99 | T1 booked 50% @ 426.26 |
| Target hit | 2025-10-30 05:30:00 | 427.44 | 406.65 | 428.65 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-08 05:30:00 | 444.90 | 2025-07-16 05:30:00 | 432.55 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-09-17 05:30:00 | 410.06 | 2025-09-26 05:30:00 | 400.62 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest1 | 2025-10-01 05:30:00 | 412.66 | 2025-10-06 05:30:00 | 426.26 | PARTIAL | 0.50 | 3.30% |
| BUY | retest1 | 2025-10-01 05:30:00 | 412.66 | 2025-10-30 05:30:00 | 427.44 | TARGET_HIT | 0.50 | 3.58% |

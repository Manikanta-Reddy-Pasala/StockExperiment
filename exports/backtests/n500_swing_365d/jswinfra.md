# JSW Infrastructure Ltd. (JSWINFRA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 283.65
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
- **Avg / median % per leg:** 0.00% / 3.07%
- **Sum % (uncompounded):** 0.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 05:30:00 | 317.05 | 299.73 | 309.04 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=7.79 |
| Stop hit — per-position SL triggered | 2025-07-18 05:30:00 | 305.37 | 300.30 | 309.80 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2025-07-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 05:30:00 | 321.70 | 300.84 | 312.11 | Stage2 pullback-breakout RSI=63 vol=3.5x ATR=8.11 |
| Stop hit — per-position SL triggered | 2025-07-28 05:30:00 | 309.54 | 301.21 | 312.36 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-09-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 05:30:00 | 313.05 | 301.76 | 303.91 | Stage2 pullback-breakout RSI=59 vol=3.6x ATR=6.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 05:30:00 | 326.85 | 302.50 | 309.21 | T1 booked 50% @ 326.85 |
| Target hit | 2025-09-26 05:30:00 | 322.65 | 304.96 | 323.23 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-11 05:30:00 | 317.05 | 2025-07-18 05:30:00 | 305.37 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest1 | 2025-07-23 05:30:00 | 321.70 | 2025-07-28 05:30:00 | 309.54 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest1 | 2025-09-09 05:30:00 | 313.05 | 2025-09-16 05:30:00 | 326.85 | PARTIAL | 0.50 | 4.41% |
| BUY | retest1 | 2025-09-09 05:30:00 | 313.05 | 2025-09-26 05:30:00 | 322.65 | TARGET_HIT | 0.50 | 3.07% |

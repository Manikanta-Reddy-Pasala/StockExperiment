# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 404.60
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 3.04% / 4.92%
- **Sum % (uncompounded):** 12.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.04% | 12.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.04% | 12.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.04% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 259.95 | 238.58 | 252.89 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=7.40 |
| Stop hit — per-position SL triggered | 2025-07-08 05:30:00 | 258.10 | 240.74 | 257.83 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 287.45 | 251.77 | 280.14 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=7.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 05:30:00 | 301.59 | 253.11 | 284.59 | T1 booked 50% @ 301.59 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 287.45 | 254.22 | 285.96 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 265.70 | 256.29 | 255.76 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=10.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 05:30:00 | 286.87 | 257.09 | 262.97 | T1 booked 50% @ 286.87 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 259.95 | 2025-07-08 05:30:00 | 258.10 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest1 | 2025-12-31 05:30:00 | 287.45 | 2026-01-05 05:30:00 | 301.59 | PARTIAL | 0.50 | 4.92% |
| BUY | retest1 | 2025-12-31 05:30:00 | 287.45 | 2026-01-08 05:30:00 | 287.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 05:30:00 | 265.70 | 2026-04-13 05:30:00 | 286.87 | PARTIAL | 0.50 | 7.97% |

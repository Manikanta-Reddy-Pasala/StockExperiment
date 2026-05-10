# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 263.90
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 4.00% / 3.50%
- **Sum % (uncompounded):** 20.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.00% | 20.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.00% | 20.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 248.78 | 237.19 | 240.72 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=5.84 |
| Stop hit — per-position SL triggered | 2025-07-04 05:30:00 | 240.02 | 237.44 | 241.53 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 05:30:00 | 245.89 | 238.51 | 238.80 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 05:30:00 | 254.51 | 238.87 | 241.91 | T1 booked 50% @ 254.51 |
| Target hit | 2025-11-24 05:30:00 | 281.90 | 250.97 | 282.52 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-02-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 05:30:00 | 303.25 | 270.28 | 293.55 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=8.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 05:30:00 | 319.57 | 273.12 | 303.56 | T1 booked 50% @ 319.57 |
| Stop hit — per-position SL triggered | 2026-03-04 05:30:00 | 303.25 | 274.27 | 305.59 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 248.78 | 2025-07-04 05:30:00 | 240.02 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest1 | 2025-09-17 05:30:00 | 245.89 | 2025-09-22 05:30:00 | 254.51 | PARTIAL | 0.50 | 3.50% |
| BUY | retest1 | 2025-09-17 05:30:00 | 245.89 | 2025-11-24 05:30:00 | 281.90 | TARGET_HIT | 0.50 | 14.64% |
| BUY | retest1 | 2026-02-17 05:30:00 | 303.25 | 2026-02-26 05:30:00 | 319.57 | PARTIAL | 0.50 | 5.38% |
| BUY | retest1 | 2026-02-17 05:30:00 | 303.25 | 2026-03-04 05:30:00 | 303.25 | STOP_HIT | 0.50 | 0.00% |

# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 168.57
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 3.68% / 4.29%
- **Sum % (uncompounded):** 29.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.68% | 29.5% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.68% | 29.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.68% | 29.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 124.46 | 113.99 | 119.14 | Stage2 pullback-breakout RSI=68 vol=1.6x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-07-10 05:30:00 | 124.90 | 115.06 | 122.97 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 05:30:00 | 121.96 | 116.63 | 121.23 | Stage2 pullback-breakout RSI=51 vol=2.2x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 05:30:00 | 127.65 | 116.78 | 122.24 | T1 booked 50% @ 127.65 |
| Target hit | 2025-10-06 05:30:00 | 137.78 | 122.11 | 138.11 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 140.81 | 124.09 | 137.49 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 05:30:00 | 146.85 | 125.90 | 140.38 | T1 booked 50% @ 146.85 |
| Stop hit — per-position SL triggered | 2025-11-12 05:30:00 | 142.53 | 125.90 | 140.38 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 05:30:00 | 150.41 | 126.15 | 141.33 | Stage2 pullback-breakout RSI=69 vol=3.8x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-11-20 05:30:00 | 144.87 | 127.17 | 143.56 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2026-01-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 05:30:00 | 192.98 | 143.75 | 183.09 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 05:30:00 | 204.12 | 147.72 | 190.15 | T1 booked 50% @ 204.12 |
| Target hit | 2026-03-04 05:30:00 | 200.46 | 158.06 | 205.23 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 124.46 | 2025-07-10 05:30:00 | 124.90 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest1 | 2025-08-14 05:30:00 | 121.96 | 2025-08-18 05:30:00 | 127.65 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2025-08-14 05:30:00 | 121.96 | 2025-10-06 05:30:00 | 137.78 | TARGET_HIT | 0.50 | 12.97% |
| BUY | retest1 | 2025-10-27 05:30:00 | 140.81 | 2025-11-12 05:30:00 | 146.85 | PARTIAL | 0.50 | 4.29% |
| BUY | retest1 | 2025-10-27 05:30:00 | 140.81 | 2025-11-12 05:30:00 | 142.53 | STOP_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-11-13 05:30:00 | 150.41 | 2025-11-20 05:30:00 | 144.87 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest1 | 2026-01-23 05:30:00 | 192.98 | 2026-02-04 05:30:00 | 204.12 | PARTIAL | 0.50 | 5.77% |
| BUY | retest1 | 2026-01-23 05:30:00 | 192.98 | 2026-03-04 05:30:00 | 200.46 | TARGET_HIT | 0.50 | 3.88% |

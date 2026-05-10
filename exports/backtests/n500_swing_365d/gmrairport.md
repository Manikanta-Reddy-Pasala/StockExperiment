# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 101.32
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -1.16% / -3.13%
- **Sum % (uncompounded):** -5.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.16% | -5.8% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.16% | -5.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.16% | -5.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 05:30:00 | 88.40 | 83.27 | 84.52 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 05:30:00 | 92.34 | 83.63 | 86.95 | T1 booked 50% @ 92.34 |
| Target hit | 2025-07-25 05:30:00 | 89.93 | 84.61 | 90.61 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 92.84 | 85.86 | 88.42 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-09-23 05:30:00 | 89.93 | 86.14 | 89.63 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 104.34 | 90.65 | 101.56 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-12-18 05:30:00 | 99.77 | 91.11 | 101.80 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2026-02-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 05:30:00 | 100.52 | 93.57 | 96.93 | Stage2 pullback-breakout RSI=57 vol=3.2x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 96.04 | 94.22 | 99.03 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2026-05-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 05:30:00 | 98.90 | 94.03 | 95.71 | Stage2 pullback-breakout RSI=58 vol=1.5x ATR=3.05 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-02 05:30:00 | 88.40 | 2025-07-09 05:30:00 | 92.34 | PARTIAL | 0.50 | 4.45% |
| BUY | retest1 | 2025-07-02 05:30:00 | 88.40 | 2025-07-25 05:30:00 | 89.93 | TARGET_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2025-09-16 05:30:00 | 92.84 | 2025-09-23 05:30:00 | 89.93 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest1 | 2025-12-12 05:30:00 | 104.34 | 2025-12-18 05:30:00 | 99.77 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest1 | 2026-02-16 05:30:00 | 100.52 | 2026-03-02 05:30:00 | 96.04 | STOP_HIT | 1.00 | -4.46% |

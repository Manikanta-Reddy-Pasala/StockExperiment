# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 107.24
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 0 / 7 / 4
- **Avg / median % per leg:** 1.22% / 0.20%
- **Sum % (uncompounded):** 13.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 0 | 7 | 4 | 1.22% | 13.4% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 0 | 7 | 4 | 1.22% | 13.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 0 | 7 | 4 | 1.22% | 13.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 110.50 | 102.81 | 106.04 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-07-14 05:30:00 | 110.72 | 103.63 | 109.30 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 05:30:00 | 114.97 | 103.83 | 110.09 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-07-22 05:30:00 | 111.17 | 104.16 | 110.75 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2025-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 05:30:00 | 107.76 | 104.52 | 104.98 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 05:30:00 | 111.65 | 104.70 | 106.38 | T1 booked 50% @ 111.65 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 107.76 | 105.17 | 108.92 | SL hit (bars_held=11) |

### Cycle 4 — BUY (started 2025-10-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 05:30:00 | 117.24 | 105.90 | 111.95 | Stage2 pullback-breakout RSI=68 vol=2.2x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-10-17 05:30:00 | 113.67 | 106.38 | 113.37 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2025-10-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 05:30:00 | 118.10 | 106.49 | 113.82 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 05:30:00 | 123.31 | 107.51 | 117.27 | T1 booked 50% @ 123.31 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 120.46 | 107.95 | 118.58 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2026-01-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 05:30:00 | 128.68 | 113.34 | 123.33 | Stage2 pullback-breakout RSI=69 vol=2.3x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 05:30:00 | 133.97 | 113.68 | 124.56 | T1 booked 50% @ 133.97 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 128.68 | 113.80 | 124.67 | SL hit (bars_held=3) |

### Cycle 7 — BUY (started 2026-02-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 05:30:00 | 124.82 | 115.46 | 122.89 | Stage2 pullback-breakout RSI=54 vol=2.4x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 05:30:00 | 131.56 | 115.97 | 124.79 | T1 booked 50% @ 131.56 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 124.82 | 116.63 | 126.56 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 110.50 | 2025-07-14 05:30:00 | 110.72 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest1 | 2025-07-16 05:30:00 | 114.97 | 2025-07-22 05:30:00 | 111.17 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest1 | 2025-09-11 05:30:00 | 107.76 | 2025-09-17 05:30:00 | 111.65 | PARTIAL | 0.50 | 3.61% |
| BUY | retest1 | 2025-09-11 05:30:00 | 107.76 | 2025-09-26 05:30:00 | 107.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 05:30:00 | 117.24 | 2025-10-17 05:30:00 | 113.67 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest1 | 2025-10-20 05:30:00 | 118.10 | 2025-10-31 05:30:00 | 123.31 | PARTIAL | 0.50 | 4.42% |
| BUY | retest1 | 2025-10-20 05:30:00 | 118.10 | 2025-11-06 05:30:00 | 120.46 | STOP_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2026-01-14 05:30:00 | 128.68 | 2026-01-19 05:30:00 | 133.97 | PARTIAL | 0.50 | 4.11% |
| BUY | retest1 | 2026-01-14 05:30:00 | 128.68 | 2026-01-20 05:30:00 | 128.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 05:30:00 | 124.82 | 2026-02-23 05:30:00 | 131.56 | PARTIAL | 0.50 | 5.40% |
| BUY | retest1 | 2026-02-17 05:30:00 | 124.82 | 2026-03-02 05:30:00 | 124.82 | STOP_HIT | 0.50 | 0.00% |

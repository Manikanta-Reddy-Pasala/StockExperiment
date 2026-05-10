# IDFC First Bank Ltd. (IDFCFIRSTB)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 71.27
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 3.09% / 3.76%
- **Sum % (uncompounded):** 18.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.09% | 18.5% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.09% | 18.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.09% | 18.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 05:30:00 | 72.23 | 68.36 | 70.00 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-09-18 05:30:00 | 71.95 | 68.73 | 71.36 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 71.09 | 68.86 | 70.38 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 05:30:00 | 73.76 | 69.02 | 71.27 | T1 booked 50% @ 73.76 |
| Target hit | 2025-11-20 05:30:00 | 78.93 | 71.31 | 79.23 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-12-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 05:30:00 | 81.98 | 71.98 | 79.75 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-12-05 05:30:00 | 79.63 | 72.23 | 79.93 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 82.29 | 72.64 | 80.26 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 05:30:00 | 85.81 | 73.31 | 82.08 | T1 booked 50% @ 85.81 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 84.54 | 73.76 | 82.95 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-04 05:30:00 | 72.23 | 2025-09-18 05:30:00 | 71.95 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-06 05:30:00 | 71.09 | 2025-10-10 05:30:00 | 73.76 | PARTIAL | 0.50 | 3.76% |
| BUY | retest1 | 2025-10-06 05:30:00 | 71.09 | 2025-11-20 05:30:00 | 78.93 | TARGET_HIT | 0.50 | 11.03% |
| BUY | retest1 | 2025-12-02 05:30:00 | 81.98 | 2025-12-05 05:30:00 | 79.63 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest1 | 2025-12-12 05:30:00 | 82.29 | 2025-12-22 05:30:00 | 85.81 | PARTIAL | 0.50 | 4.28% |
| BUY | retest1 | 2025-12-12 05:30:00 | 82.29 | 2025-12-29 05:30:00 | 84.54 | STOP_HIT | 0.50 | 2.73% |

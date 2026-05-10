# SBFC Finance Ltd. (SBFC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 98.77
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.02% / 0.00%
- **Sum % (uncompounded):** -0.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.02% | -0.1% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.02% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.02% | -0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 112.13 | 94.57 | 107.87 | Stage2 pullback-breakout RSI=59 vol=5.4x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-07-21 05:30:00 | 114.59 | 97.29 | 113.47 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 05:30:00 | 111.66 | 100.93 | 107.87 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 107.47 | 101.17 | 108.17 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 05:30:00 | 112.83 | 102.14 | 108.58 | Stage2 pullback-breakout RSI=61 vol=3.6x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 05:30:00 | 118.96 | 103.31 | 112.49 | T1 booked 50% @ 118.96 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 112.83 | 103.54 | 112.91 | SL hit (bars_held=12) |

### Cycle 4 — BUY (started 2026-01-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 05:30:00 | 107.01 | 104.30 | 103.56 | Stage2 pullback-breakout RSI=59 vol=3.7x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-01-21 05:30:00 | 102.78 | 104.28 | 103.57 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 112.13 | 2025-07-21 05:30:00 | 114.59 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest1 | 2025-09-23 05:30:00 | 111.66 | 2025-09-26 05:30:00 | 107.47 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2025-10-17 05:30:00 | 112.83 | 2025-11-03 05:30:00 | 118.96 | PARTIAL | 0.50 | 5.43% |
| BUY | retest1 | 2025-10-17 05:30:00 | 112.83 | 2025-11-06 05:30:00 | 112.83 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 05:30:00 | 107.01 | 2026-01-21 05:30:00 | 102.78 | STOP_HIT | 1.00 | -3.96% |

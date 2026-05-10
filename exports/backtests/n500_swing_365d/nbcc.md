# NBCC (India) Ltd. (NBCC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 100.64
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.48% / 1.28%
- **Sum % (uncompounded):** -1.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.48% | -1.9% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.48% | -1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.48% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 05:30:00 | 107.81 | 105.87 | 104.25 | Stage2 pullback-breakout RSI=57 vol=2.7x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 05:30:00 | 113.51 | 106.14 | 107.33 | T1 booked 50% @ 113.51 |
| Stop hit — per-position SL triggered | 2025-09-25 05:30:00 | 109.19 | 106.28 | 108.17 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 05:30:00 | 115.22 | 107.29 | 111.52 | Stage2 pullback-breakout RSI=60 vol=2.6x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-11-07 05:30:00 | 110.72 | 107.75 | 113.08 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 05:30:00 | 115.39 | 109.03 | 112.38 | Stage2 pullback-breakout RSI=55 vol=2.4x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-12-17 05:30:00 | 110.11 | 109.06 | 112.05 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-11 05:30:00 | 107.81 | 2025-09-22 05:30:00 | 113.51 | PARTIAL | 0.50 | 5.29% |
| BUY | retest1 | 2025-09-11 05:30:00 | 107.81 | 2025-09-25 05:30:00 | 109.19 | STOP_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2025-10-29 05:30:00 | 115.22 | 2025-11-07 05:30:00 | 110.72 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest1 | 2025-12-15 05:30:00 | 115.39 | 2025-12-17 05:30:00 | 110.11 | STOP_HIT | 1.00 | -4.58% |

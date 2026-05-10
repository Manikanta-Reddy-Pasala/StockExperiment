# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 831.30
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** -0.17% / 1.44%
- **Sum % (uncompounded):** -0.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.17% | -0.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.17% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.17% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 726.25 | 671.01 | 701.91 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=12.74 |
| Stop hit — per-position SL triggered | 2025-07-02 05:30:00 | 707.14 | 672.96 | 707.92 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 05:30:00 | 720.95 | 700.34 | 718.28 | Stage2 pullback-breakout RSI=52 vol=1.8x ATR=11.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 05:30:00 | 743.77 | 701.53 | 724.37 | T1 booked 50% @ 743.77 |
| Target hit | 2025-11-25 05:30:00 | 731.35 | 703.84 | 731.44 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 773.05 | 711.87 | 744.48 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=13.62 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 752.63 | 714.00 | 751.29 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2026-05-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 05:30:00 | 807.20 | 739.70 | 772.57 | Stage2 pullback-breakout RSI=68 vol=2.7x ATR=18.41 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 726.25 | 2025-07-02 05:30:00 | 707.14 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest1 | 2025-11-12 05:30:00 | 720.95 | 2025-11-17 05:30:00 | 743.77 | PARTIAL | 0.50 | 3.16% |
| BUY | retest1 | 2025-11-12 05:30:00 | 720.95 | 2025-11-25 05:30:00 | 731.35 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2026-01-05 05:30:00 | 773.05 | 2026-01-09 05:30:00 | 752.63 | STOP_HIT | 1.00 | -2.64% |

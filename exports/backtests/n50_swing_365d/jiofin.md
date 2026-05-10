# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 249.34
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
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.12% / 0.00%
- **Sum % (uncompounded):** -0.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 05:30:00 | 321.10 | 291.02 | 315.70 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=7.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 05:30:00 | 336.21 | 292.89 | 321.23 | T1 booked 50% @ 336.21 |
| Stop hit — per-position SL triggered | 2025-08-07 05:30:00 | 321.10 | 293.54 | 322.09 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2025-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 05:30:00 | 310.70 | 301.49 | 306.34 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=5.42 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 302.57 | 301.86 | 306.26 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2026-01-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 05:30:00 | 303.50 | 301.06 | 298.58 | Stage2 pullback-breakout RSI=56 vol=2.7x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 295.69 | 300.99 | 298.10 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-29 05:30:00 | 321.10 | 2025-08-05 05:30:00 | 336.21 | PARTIAL | 0.50 | 4.71% |
| BUY | retest1 | 2025-07-29 05:30:00 | 321.10 | 2025-08-07 05:30:00 | 321.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 05:30:00 | 310.70 | 2025-11-24 05:30:00 | 302.57 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest1 | 2026-01-07 05:30:00 | 303.50 | 2026-01-08 05:30:00 | 295.69 | STOP_HIT | 1.00 | -2.57% |

# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 569.20
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 24.15% / 7.21%
- **Sum % (uncompounded):** 72.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 24.15% | 72.4% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 24.15% | 72.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 24.15% | 72.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 05:30:00 | 359.70 | 278.31 | 337.67 | Stage2 pullback-breakout RSI=62 vol=4.1x ATR=13.58 |
| Stop hit — per-position SL triggered | 2025-11-12 05:30:00 | 339.33 | 278.92 | 337.87 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2025-12-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 05:30:00 | 339.30 | 284.91 | 330.42 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=12.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 05:30:00 | 363.76 | 286.77 | 335.18 | T1 booked 50% @ 363.76 |
| Target hit | 2026-02-06 05:30:00 | 579.85 | 369.23 | 583.52 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-05-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 05:30:00 | 568.60 | 443.41 | 542.60 | Stage2 pullback-breakout RSI=59 vol=2.4x ATR=21.29 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-11 05:30:00 | 359.70 | 2025-11-12 05:30:00 | 339.33 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest1 | 2025-12-01 05:30:00 | 339.30 | 2025-12-04 05:30:00 | 363.76 | PARTIAL | 0.50 | 7.21% |
| BUY | retest1 | 2025-12-01 05:30:00 | 339.30 | 2026-02-06 05:30:00 | 579.85 | TARGET_HIT | 0.50 | 70.90% |

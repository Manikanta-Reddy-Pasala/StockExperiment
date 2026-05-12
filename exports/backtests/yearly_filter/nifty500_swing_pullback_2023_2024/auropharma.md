# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1487.30
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 5.78% / 5.10%
- **Sum % (uncompounded):** 17.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 5.78% | 17.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 5.78% | 17.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 5.78% | 17.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 914.35 | 685.29 | 872.78 | Stage2 pullback-breakout RSI=65 vol=3.7x ATR=24.38 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 877.78 | 689.45 | 876.78 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 936.55 | 730.95 | 880.31 | Stage2 pullback-breakout RSI=67 vol=2.4x ATR=23.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-12 00:00:00 | 984.33 | 738.02 | 903.80 | T1 booked 50% @ 984.33 |
| Target hit | 2024-01-17 00:00:00 | 1088.65 | 854.26 | 1093.07 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-29 00:00:00 | 914.35 | 2023-10-04 00:00:00 | 877.78 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest1 | 2023-11-08 00:00:00 | 936.55 | 2023-11-12 00:00:00 | 984.33 | PARTIAL | 0.50 | 5.10% |
| BUY | retest1 | 2023-11-08 00:00:00 | 936.55 | 2024-01-17 00:00:00 | 1088.65 | TARGET_HIT | 0.50 | 16.24% |

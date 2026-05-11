# NHPC Ltd. (NHPC)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 80.77
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
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.17% / 5.11%
- **Sum % (uncompounded):** 13.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.17% | 13.0% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.17% | 13.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.17% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 05:30:00 | 52.70 | 44.59 | 50.39 | Stage2 pullback-breakout RSI=68 vol=5.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-09-12 05:30:00 | 50.69 | 45.01 | 51.42 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-09-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 05:30:00 | 55.90 | 45.18 | 51.89 | Stage2 pullback-breakout RSI=65 vol=3.2x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-09-21 05:30:00 | 52.77 | 45.54 | 52.63 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 05:30:00 | 54.40 | 47.62 | 51.86 | Stage2 pullback-breakout RSI=68 vol=2.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 05:30:00 | 57.18 | 48.11 | 53.39 | T1 booked 50% @ 57.18 |
| Target hit | 2023-12-20 05:30:00 | 60.35 | 49.89 | 60.46 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 05:30:00 | 96.20 | 71.32 | 91.01 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 05:30:00 | 102.34 | 72.16 | 93.28 | T1 booked 50% @ 102.34 |
| Stop hit — per-position SL triggered | 2024-05-07 05:30:00 | 96.20 | 72.41 | 93.65 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 05:30:00 | 52.70 | 2023-09-12 05:30:00 | 50.69 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest1 | 2023-09-14 05:30:00 | 55.90 | 2023-09-21 05:30:00 | 52.77 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest1 | 2023-11-21 05:30:00 | 54.40 | 2023-12-01 05:30:00 | 57.18 | PARTIAL | 0.50 | 5.11% |
| BUY | retest1 | 2023-11-21 05:30:00 | 54.40 | 2023-12-20 05:30:00 | 60.35 | TARGET_HIT | 0.50 | 10.94% |
| BUY | retest1 | 2024-04-30 05:30:00 | 96.20 | 2024-05-06 05:30:00 | 102.34 | PARTIAL | 0.50 | 6.39% |
| BUY | retest1 | 2024-04-30 05:30:00 | 96.20 | 2024-05-07 05:30:00 | 96.20 | STOP_HIT | 0.50 | 0.00% |

# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 508.85
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.84% / 0.90%
- **Sum % (uncompounded):** -7.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 4 | 0 | -1.84% | -7.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -1.84% | -7.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -1.84% | -7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 00:00:00 | 660.84 | 545.16 | 637.26 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=18.60 |
| Stop hit — per-position SL triggered | 2024-07-30 00:00:00 | 632.93 | 548.50 | 642.44 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-09-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 00:00:00 | 627.66 | 564.00 | 611.06 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=22.81 |
| Stop hit — per-position SL triggered | 2024-09-25 00:00:00 | 633.30 | 571.71 | 632.71 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 00:00:00 | 628.75 | 575.84 | 601.71 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=24.60 |
| Stop hit — per-position SL triggered | 2024-10-29 00:00:00 | 591.85 | 577.11 | 603.89 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-11-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 00:00:00 | 609.85 | 579.02 | 594.59 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=23.42 |
| Stop hit — per-position SL triggered | 2024-12-03 00:00:00 | 621.10 | 582.99 | 611.42 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-25 00:00:00 | 660.84 | 2024-07-30 00:00:00 | 632.93 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest1 | 2024-09-11 00:00:00 | 627.66 | 2024-09-25 00:00:00 | 633.30 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest1 | 2024-10-23 00:00:00 | 628.75 | 2024-10-29 00:00:00 | 591.85 | STOP_HIT | 1.00 | -5.87% |
| BUY | retest1 | 2024-11-18 00:00:00 | 609.85 | 2024-12-03 00:00:00 | 621.10 | STOP_HIT | 1.00 | 1.84% |

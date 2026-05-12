# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 571.45
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
- **Avg / median % per leg:** -0.38% / 2.67%
- **Sum % (uncompounded):** -1.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.38% | -1.5% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.38% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.38% | -1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 00:00:00 | 669.60 | 621.29 | 644.28 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=19.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 00:00:00 | 709.31 | 625.74 | 665.14 | T1 booked 50% @ 709.31 |
| Stop hit — per-position SL triggered | 2024-07-24 00:00:00 | 687.45 | 632.03 | 687.18 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-10-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 00:00:00 | 729.45 | 661.25 | 676.74 | Stage2 pullback-breakout RSI=63 vol=8.6x ATR=26.49 |
| Stop hit — per-position SL triggered | 2024-10-28 00:00:00 | 689.72 | 661.92 | 680.03 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 00:00:00 | 689.15 | 666.54 | 679.23 | Stage2 pullback-breakout RSI=54 vol=1.6x ATR=21.52 |
| Stop hit — per-position SL triggered | 2024-12-10 00:00:00 | 656.87 | 666.37 | 673.24 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-01 00:00:00 | 669.60 | 2024-07-11 00:00:00 | 709.31 | PARTIAL | 0.50 | 5.93% |
| BUY | retest1 | 2024-07-01 00:00:00 | 669.60 | 2024-07-24 00:00:00 | 687.45 | STOP_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2024-10-24 00:00:00 | 729.45 | 2024-10-28 00:00:00 | 689.72 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest1 | 2024-12-04 00:00:00 | 689.15 | 2024-12-10 00:00:00 | 656.87 | STOP_HIT | 1.00 | -4.68% |

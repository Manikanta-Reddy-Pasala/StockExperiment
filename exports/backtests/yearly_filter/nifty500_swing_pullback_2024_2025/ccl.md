# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1123.10
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 4.18% / 6.76%
- **Sum % (uncompounded):** 16.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.18% | 16.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.18% | 16.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.18% | 16.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 678.15 | 647.41 | 659.23 | Stage2 pullback-breakout RSI=54 vol=2.2x ATR=22.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 00:00:00 | 724.00 | 648.88 | 668.19 | T1 booked 50% @ 724.00 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 678.15 | 651.32 | 680.22 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-11-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 00:00:00 | 740.55 | 654.12 | 692.19 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=26.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 00:00:00 | 793.82 | 657.39 | 711.43 | T1 booked 50% @ 793.82 |
| Target hit | 2024-12-17 00:00:00 | 761.10 | 674.38 | 765.94 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-31 00:00:00 | 678.15 | 2024-11-06 00:00:00 | 724.00 | PARTIAL | 0.50 | 6.76% |
| BUY | retest1 | 2024-10-31 00:00:00 | 678.15 | 2024-11-13 00:00:00 | 678.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 00:00:00 | 740.55 | 2024-11-27 00:00:00 | 793.82 | PARTIAL | 0.50 | 7.19% |
| BUY | retest1 | 2024-11-22 00:00:00 | 740.55 | 2024-12-17 00:00:00 | 761.10 | TARGET_HIT | 0.50 | 2.77% |

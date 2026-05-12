# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 3852.70
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
| TARGET_HIT | 0 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 1
- **Target hits / Stop hits / Partials:** 0 / 1 / 1
- **Avg / median % per leg:** 1.65% / 9.24%
- **Sum % (uncompounded):** 3.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.65% | 3.3% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.65% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.65% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 00:00:00 | 852.15 | 769.04 | 806.60 | Stage2 pullback-breakout RSI=59 vol=3.1x ATR=33.75 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 801.52 | 771.24 | 815.52 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-12-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 00:00:00 | 1731.63 | 1083.58 | 1537.99 | Stage2 pullback-breakout RSI=68 vol=3.5x ATR=79.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 00:00:00 | 1891.59 | 1120.14 | 1655.44 | T1 booked 50% @ 1891.59 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-31 00:00:00 | 852.15 | 2024-08-05 00:00:00 | 801.52 | STOP_HIT | 1.00 | -5.94% |
| BUY | retest1 | 2024-12-05 00:00:00 | 1731.63 | 2024-12-12 00:00:00 | 1891.59 | PARTIAL | 0.50 | 9.24% |

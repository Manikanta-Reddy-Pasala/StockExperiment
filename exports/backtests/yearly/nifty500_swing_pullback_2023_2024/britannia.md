# Britannia Industries Ltd. (BRITANNIA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 5520.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 2.82% / 2.87%
- **Sum % (uncompounded):** 11.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.82% | 11.3% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.82% | 11.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.82% | 11.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 05:30:00 | 4803.65 | 4547.17 | 4670.61 | Stage2 pullback-breakout RSI=68 vol=1.5x ATR=69.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 05:30:00 | 4941.72 | 4554.40 | 4714.88 | T1 booked 50% @ 4941.72 |
| Stop hit — per-position SL triggered | 2023-12-13 05:30:00 | 4919.30 | 4584.92 | 4843.07 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 05:30:00 | 5059.60 | 4605.88 | 4891.72 | Stage2 pullback-breakout RSI=70 vol=2.6x ATR=92.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 05:30:00 | 5244.74 | 4623.58 | 4973.34 | T1 booked 50% @ 5244.74 |
| Stop hit — per-position SL triggered | 2024-01-08 05:30:00 | 5177.35 | 4674.52 | 5142.37 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-05-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 05:30:00 | 5061.60 | 4803.00 | 4819.29 | Stage2 pullback-breakout RSI=68 vol=5.7x ATR=107.30 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-29 05:30:00 | 4803.65 | 2023-12-01 05:30:00 | 4941.72 | PARTIAL | 0.50 | 2.87% |
| BUY | retest1 | 2023-11-29 05:30:00 | 4803.65 | 2023-12-13 05:30:00 | 4919.30 | STOP_HIT | 0.50 | 2.41% |
| BUY | retest1 | 2023-12-21 05:30:00 | 5059.60 | 2023-12-27 05:30:00 | 5244.74 | PARTIAL | 0.50 | 3.66% |
| BUY | retest1 | 2023-12-21 05:30:00 | 5059.60 | 2024-01-08 05:30:00 | 5177.35 | STOP_HIT | 0.50 | 2.33% |

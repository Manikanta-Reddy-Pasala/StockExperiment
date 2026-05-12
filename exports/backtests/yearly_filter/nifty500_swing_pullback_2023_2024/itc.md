# ITC Ltd. (ITC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 307.45
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 1.41% / 2.58%
- **Sum % (uncompounded):** 5.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.41% | 5.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.41% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.41% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 449.80 | 425.17 | 438.58 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 461.41 | 426.09 | 443.27 | T1 booked 50% @ 461.41 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 449.80 | 426.64 | 445.11 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 464.10 | 430.17 | 453.15 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=7.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 00:00:00 | 478.27 | 431.72 | 458.72 | T1 booked 50% @ 478.27 |
| Stop hit — per-position SL triggered | 2024-01-09 00:00:00 | 464.10 | 433.22 | 462.29 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-01 00:00:00 | 449.80 | 2023-12-06 00:00:00 | 461.41 | PARTIAL | 0.50 | 2.58% |
| BUY | retest1 | 2023-12-01 00:00:00 | 449.80 | 2023-12-08 00:00:00 | 449.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 00:00:00 | 464.10 | 2024-01-03 00:00:00 | 478.27 | PARTIAL | 0.50 | 3.05% |
| BUY | retest1 | 2023-12-28 00:00:00 | 464.10 | 2024-01-09 00:00:00 | 464.10 | STOP_HIT | 0.50 | 0.00% |

# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 34005.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.72% / 7.47%
- **Sum % (uncompounded):** 10.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.72% | 10.3% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.72% | 10.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.72% | 10.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 05:30:00 | 20115.00 | 15032.70 | 19147.90 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=789.76 |
| Stop hit — per-position SL triggered | 2025-07-28 05:30:00 | 18930.36 | 15206.33 | 19238.29 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 05:30:00 | 20825.00 | 15307.39 | 19436.94 | Stage2 pullback-breakout RSI=62 vol=2.8x ATR=855.19 |
| Stop hit — per-position SL triggered | 2025-07-31 05:30:00 | 19542.21 | 15355.03 | 19499.62 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2026-01-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 05:30:00 | 19582.00 | 18063.06 | 19034.24 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=634.38 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 18630.43 | 18066.85 | 18978.03 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2026-02-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 05:30:00 | 21871.00 | 18044.62 | 18599.85 | Stage2 pullback-breakout RSI=70 vol=3.7x ATR=967.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 05:30:00 | 23806.81 | 18506.81 | 21380.22 | T1 booked 50% @ 23806.81 |
| Target hit | 2026-03-23 05:30:00 | 24255.00 | 19668.22 | 24353.60 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-04-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 05:30:00 | 27315.00 | 20208.11 | 25064.38 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=1020.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 05:30:00 | 29355.90 | 20701.18 | 26721.92 | T1 booked 50% @ 29355.90 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-22 05:30:00 | 20115.00 | 2025-07-28 05:30:00 | 18930.36 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest1 | 2025-07-30 05:30:00 | 20825.00 | 2025-07-31 05:30:00 | 19542.21 | STOP_HIT | 1.00 | -6.16% |
| BUY | retest1 | 2026-01-07 05:30:00 | 19582.00 | 2026-01-08 05:30:00 | 18630.43 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2026-02-06 05:30:00 | 21871.00 | 2026-02-20 05:30:00 | 23806.81 | PARTIAL | 0.50 | 8.85% |
| BUY | retest1 | 2026-02-06 05:30:00 | 21871.00 | 2026-03-23 05:30:00 | 24255.00 | TARGET_HIT | 0.50 | 10.90% |
| BUY | retest1 | 2026-04-09 05:30:00 | 27315.00 | 2026-04-20 05:30:00 | 29355.90 | PARTIAL | 0.50 | 7.47% |

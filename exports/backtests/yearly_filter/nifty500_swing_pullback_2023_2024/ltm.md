# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 4349.80
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
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.43% / 0.00%
- **Sum % (uncompounded):** 3.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.43% | 3.0% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.43% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.43% | 3.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 00:00:00 | 5163.05 | 4662.21 | 4962.33 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=112.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 00:00:00 | 5388.24 | 4688.38 | 5057.87 | T1 booked 50% @ 5388.24 |
| Stop hit — per-position SL triggered | 2023-07-07 00:00:00 | 5163.05 | 4697.44 | 5073.63 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 00:00:00 | 5134.85 | 4713.76 | 5035.01 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=144.04 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 4918.78 | 4723.99 | 5014.15 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 5457.80 | 4953.57 | 5261.69 | Stage2 pullback-breakout RSI=61 vol=5.1x ATR=120.67 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 5276.79 | 4960.60 | 5269.30 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2023-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 00:00:00 | 5639.85 | 5064.74 | 5465.71 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=109.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 00:00:00 | 5858.29 | 5104.51 | 5599.51 | T1 booked 50% @ 5858.29 |
| Target hit | 2024-01-03 00:00:00 | 5961.05 | 5234.52 | 6016.10 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 6237.05 | 5283.62 | 5996.77 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=141.59 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 6024.66 | 5315.82 | 6024.40 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-27 00:00:00 | 5163.05 | 2023-07-05 00:00:00 | 5388.24 | PARTIAL | 0.50 | 4.36% |
| BUY | retest1 | 2023-06-27 00:00:00 | 5163.05 | 2023-07-07 00:00:00 | 5163.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-17 00:00:00 | 5134.85 | 2023-07-21 00:00:00 | 4918.78 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest1 | 2023-10-19 00:00:00 | 5457.80 | 2023-10-23 00:00:00 | 5276.79 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest1 | 2023-12-06 00:00:00 | 5639.85 | 2023-12-14 00:00:00 | 5858.29 | PARTIAL | 0.50 | 3.87% |
| BUY | retest1 | 2023-12-06 00:00:00 | 5639.85 | 2024-01-03 00:00:00 | 5961.05 | TARGET_HIT | 0.50 | 5.70% |
| BUY | retest1 | 2024-01-12 00:00:00 | 6237.05 | 2024-01-18 00:00:00 | 6024.66 | STOP_HIT | 1.00 | -3.41% |

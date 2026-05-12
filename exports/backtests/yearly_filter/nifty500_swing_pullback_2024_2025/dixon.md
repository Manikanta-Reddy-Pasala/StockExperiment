# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 10803.00
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.73% / -1.34%
- **Sum % (uncompounded):** -3.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.73% | -3.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.73% | -3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.73% | -3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 00:00:00 | 11977.35 | 8345.66 | 11649.13 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=427.80 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 11335.65 | 8474.82 | 11635.19 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-09-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 00:00:00 | 13990.30 | 9516.58 | 12767.48 | Stage2 pullback-breakout RSI=69 vol=2.1x ATR=455.84 |
| Stop hit — per-position SL triggered | 2024-09-30 00:00:00 | 13802.95 | 9948.97 | 13581.84 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 00:00:00 | 14519.00 | 10141.51 | 13710.82 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=502.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 00:00:00 | 15523.60 | 10576.99 | 14616.13 | T1 booked 50% @ 15523.60 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 14908.00 | 10620.08 | 14643.93 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 15647.60 | 11031.84 | 14606.22 | Stage2 pullback-breakout RSI=62 vol=2.9x ATR=682.47 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 14623.89 | 11239.12 | 14853.91 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-30 00:00:00 | 11977.35 | 2024-08-05 00:00:00 | 11335.65 | STOP_HIT | 1.00 | -5.36% |
| BUY | retest1 | 2024-09-16 00:00:00 | 13990.30 | 2024-09-30 00:00:00 | 13802.95 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest1 | 2024-10-08 00:00:00 | 14519.00 | 2024-10-21 00:00:00 | 15523.60 | PARTIAL | 0.50 | 6.92% |
| BUY | retest1 | 2024-10-08 00:00:00 | 14519.00 | 2024-10-22 00:00:00 | 14908.00 | STOP_HIT | 0.50 | 2.68% |
| BUY | retest1 | 2024-11-06 00:00:00 | 15647.60 | 2024-11-13 00:00:00 | 14623.89 | STOP_HIT | 1.00 | -6.54% |

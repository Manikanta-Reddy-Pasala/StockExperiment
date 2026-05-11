# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1103.40
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 2.57% / 2.87%
- **Sum % (uncompounded):** 12.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.57% | 12.8% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.57% | 12.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.57% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 05:30:00 | 322.50 | 271.46 | 313.76 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=7.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 05:30:00 | 338.14 | 272.62 | 316.66 | T1 booked 50% @ 338.14 |
| Stop hit — per-position SL triggered | 2023-09-20 05:30:00 | 327.65 | 277.27 | 325.51 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 05:30:00 | 391.45 | 295.54 | 362.05 | Stage2 pullback-breakout RSI=69 vol=1.8x ATR=14.15 |
| Stop hit — per-position SL triggered | 2023-11-13 05:30:00 | 402.70 | 305.30 | 385.56 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-02-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 05:30:00 | 528.95 | 392.93 | 502.40 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=18.71 |
| Stop hit — per-position SL triggered | 2024-03-07 05:30:00 | 500.88 | 402.16 | 507.28 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-04-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 05:30:00 | 504.70 | 411.70 | 477.21 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=22.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 05:30:00 | 549.29 | 417.40 | 497.71 | T1 booked 50% @ 549.29 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 05:30:00 | 322.50 | 2023-09-07 05:30:00 | 338.14 | PARTIAL | 0.50 | 4.85% |
| BUY | retest1 | 2023-09-05 05:30:00 | 322.50 | 2023-09-20 05:30:00 | 327.65 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2023-10-31 05:30:00 | 391.45 | 2023-11-13 05:30:00 | 402.70 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest1 | 2024-02-27 05:30:00 | 528.95 | 2024-03-07 05:30:00 | 500.88 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest1 | 2024-04-03 05:30:00 | 504.70 | 2024-04-10 05:30:00 | 549.29 | PARTIAL | 0.50 | 8.83% |

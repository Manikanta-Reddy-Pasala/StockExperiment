# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 3621.40
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 2.73% / 4.10%
- **Sum % (uncompounded):** 16.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 3 | 3 | 2.73% | 16.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 3 | 3 | 2.73% | 16.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 3 | 3 | 2.73% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 00:00:00 | 3567.97 | 3017.54 | 3407.11 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=73.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 00:00:00 | 3714.25 | 3041.23 | 3480.23 | T1 booked 50% @ 3714.25 |
| Stop hit — per-position SL triggered | 2024-10-07 00:00:00 | 3567.97 | 3057.08 | 3504.08 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-10-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 00:00:00 | 3816.63 | 3070.63 | 3547.71 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=109.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 00:00:00 | 4034.97 | 3110.11 | 3681.21 | T1 booked 50% @ 4034.97 |
| Stop hit — per-position SL triggered | 2024-10-17 00:00:00 | 3816.63 | 3117.34 | 3696.02 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 3653.23 | 3192.91 | 3454.69 | Stage2 pullback-breakout RSI=61 vol=5.0x ATR=119.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 00:00:00 | 3892.07 | 3230.21 | 3607.87 | T1 booked 50% @ 3892.07 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 3653.23 | 3299.90 | 3754.62 | SL hit (bars_held=19) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-25 00:00:00 | 3567.97 | 2024-10-01 00:00:00 | 3714.25 | PARTIAL | 0.50 | 4.10% |
| BUY | retest1 | 2024-09-25 00:00:00 | 3567.97 | 2024-10-07 00:00:00 | 3567.97 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 00:00:00 | 3816.63 | 2024-10-16 00:00:00 | 4034.97 | PARTIAL | 0.50 | 5.72% |
| BUY | retest1 | 2024-10-09 00:00:00 | 3816.63 | 2024-10-17 00:00:00 | 3816.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 3653.23 | 2024-12-04 00:00:00 | 3892.07 | PARTIAL | 0.50 | 6.54% |
| BUY | retest1 | 2024-11-25 00:00:00 | 3653.23 | 2024-12-20 00:00:00 | 3653.23 | STOP_HIT | 0.50 | 0.00% |

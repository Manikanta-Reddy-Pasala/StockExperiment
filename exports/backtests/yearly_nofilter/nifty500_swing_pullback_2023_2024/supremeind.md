# Supreme Industries Ltd. (SUPREMEIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3610.00
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 3.63% / 6.61%
- **Sum % (uncompounded):** 21.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.63% | 21.8% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.63% | 21.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.63% | 21.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 4338.95 | 3320.59 | 4139.65 | Stage2 pullback-breakout RSI=61 vol=2.8x ATR=143.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 00:00:00 | 4625.71 | 3333.57 | 4185.93 | T1 booked 50% @ 4625.71 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 4338.95 | 3383.07 | 4317.01 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 4431.15 | 3583.62 | 4223.02 | Stage2 pullback-breakout RSI=59 vol=6.9x ATR=177.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 4786.86 | 3613.68 | 4322.25 | T1 booked 50% @ 4786.86 |
| Stop hit — per-position SL triggered | 2023-12-12 00:00:00 | 4431.15 | 3657.79 | 4397.34 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 4169.65 | 3892.93 | 3966.66 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=165.10 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 4150.85 | 3922.22 | 4113.68 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 4338.45 | 3945.25 | 4170.92 | Stage2 pullback-breakout RSI=59 vol=3.6x ATR=164.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 00:00:00 | 4667.77 | 3955.30 | 4245.61 | T1 booked 50% @ 4667.77 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-16 00:00:00 | 4338.95 | 2023-10-17 00:00:00 | 4625.71 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2023-10-16 00:00:00 | 4338.95 | 2023-10-23 00:00:00 | 4338.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 00:00:00 | 4431.15 | 2023-12-05 00:00:00 | 4786.86 | PARTIAL | 0.50 | 8.03% |
| BUY | retest1 | 2023-11-30 00:00:00 | 4431.15 | 2023-12-12 00:00:00 | 4431.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-27 00:00:00 | 4169.65 | 2024-04-12 00:00:00 | 4150.85 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-04-26 00:00:00 | 4338.45 | 2024-04-29 00:00:00 | 4667.77 | PARTIAL | 0.50 | 7.59% |

# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 343.45
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 2 / 2 / 3
- **Avg / median % per leg:** 6.04% / 6.06%
- **Sum % (uncompounded):** 42.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 2 | 2 | 3 | 6.04% | 42.3% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 2 | 2 | 3 | 6.04% | 42.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 2 | 2 | 3 | 6.04% | 42.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 05:30:00 | 181.95 | 151.76 | 170.29 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 05:30:00 | 192.97 | 152.76 | 174.37 | T1 booked 50% @ 192.97 |
| Target hit | 2023-08-16 05:30:00 | 218.05 | 168.87 | 218.20 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 05:30:00 | 237.75 | 183.46 | 230.50 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=8.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 05:30:00 | 254.77 | 185.85 | 235.22 | T1 booked 50% @ 254.77 |
| Stop hit — per-position SL triggered | 2023-10-16 05:30:00 | 243.70 | 191.66 | 242.35 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 05:30:00 | 246.60 | 204.34 | 237.94 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 05:30:00 | 262.81 | 205.33 | 240.97 | T1 booked 50% @ 262.81 |
| Target hit | 2023-12-26 05:30:00 | 260.55 | 213.40 | 265.31 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 05:30:00 | 254.65 | 235.86 | 246.15 | Stage2 pullback-breakout RSI=56 vol=3.3x ATR=9.35 |
| Stop hit — per-position SL triggered | 2024-04-19 05:30:00 | 240.62 | 237.38 | 250.08 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 05:30:00 | 181.95 | 2023-07-05 05:30:00 | 192.97 | PARTIAL | 0.50 | 6.06% |
| BUY | retest1 | 2023-06-30 05:30:00 | 181.95 | 2023-08-16 05:30:00 | 218.05 | TARGET_HIT | 0.50 | 19.84% |
| BUY | retest1 | 2023-09-25 05:30:00 | 237.75 | 2023-09-29 05:30:00 | 254.77 | PARTIAL | 0.50 | 7.16% |
| BUY | retest1 | 2023-09-25 05:30:00 | 237.75 | 2023-10-16 05:30:00 | 243.70 | STOP_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2023-12-05 05:30:00 | 246.60 | 2023-12-07 05:30:00 | 262.81 | PARTIAL | 0.50 | 6.57% |
| BUY | retest1 | 2023-12-05 05:30:00 | 246.60 | 2023-12-26 05:30:00 | 260.55 | TARGET_HIT | 0.50 | 5.66% |
| BUY | retest1 | 2024-04-04 05:30:00 | 254.65 | 2024-04-19 05:30:00 | 240.62 | STOP_HIT | 1.00 | -5.51% |

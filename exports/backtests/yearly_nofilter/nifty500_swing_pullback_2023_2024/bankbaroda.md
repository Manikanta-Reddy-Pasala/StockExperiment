# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 263.65
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 4
- **Avg / median % per leg:** 4.11% / 5.22%
- **Sum % (uncompounded):** 32.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 2 | 2 | 4 | 4.11% | 32.9% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 2 | 2 | 4 | 4.11% | 32.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 2 | 4 | 4.11% | 32.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 199.10 | 169.60 | 190.26 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 00:00:00 | 207.48 | 171.01 | 195.37 | T1 booked 50% @ 207.48 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 199.10 | 172.24 | 197.63 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 209.05 | 189.38 | 198.37 | Stage2 pullback-breakout RSI=66 vol=2.1x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 00:00:00 | 218.25 | 191.02 | 206.44 | T1 booked 50% @ 218.25 |
| Target hit | 2024-01-08 00:00:00 | 223.45 | 196.85 | 225.42 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 247.60 | 201.77 | 230.57 | Stage2 pullback-breakout RSI=68 vol=3.0x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 00:00:00 | 263.65 | 205.32 | 242.60 | T1 booked 50% @ 263.65 |
| Target hit | 2024-02-28 00:00:00 | 261.75 | 213.11 | 262.14 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 268.65 | 229.06 | 263.11 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=7.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 00:00:00 | 282.67 | 230.39 | 266.06 | T1 booked 50% @ 282.67 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 268.65 | 231.67 | 267.90 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 199.10 | 2023-07-07 00:00:00 | 207.48 | PARTIAL | 0.50 | 4.21% |
| BUY | retest1 | 2023-07-03 00:00:00 | 199.10 | 2023-07-13 00:00:00 | 199.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-04 00:00:00 | 209.05 | 2023-12-13 00:00:00 | 218.25 | PARTIAL | 0.50 | 4.40% |
| BUY | retest1 | 2023-12-04 00:00:00 | 209.05 | 2024-01-08 00:00:00 | 223.45 | TARGET_HIT | 0.50 | 6.89% |
| BUY | retest1 | 2024-01-31 00:00:00 | 247.60 | 2024-02-09 00:00:00 | 263.65 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2024-01-31 00:00:00 | 247.60 | 2024-02-28 00:00:00 | 261.75 | TARGET_HIT | 0.50 | 5.71% |
| BUY | retest1 | 2024-04-25 00:00:00 | 268.65 | 2024-04-30 00:00:00 | 282.67 | PARTIAL | 0.50 | 5.22% |
| BUY | retest1 | 2024-04-25 00:00:00 | 268.65 | 2024-05-06 00:00:00 | 268.65 | STOP_HIT | 0.50 | 0.00% |

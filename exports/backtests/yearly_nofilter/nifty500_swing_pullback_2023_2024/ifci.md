# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 63.09
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 8.97% / 7.34%
- **Sum % (uncompounded):** 62.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 1 | 3 | 3 | 8.97% | 62.8% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 1 | 3 | 3 | 8.97% | 62.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 1 | 3 | 3 | 8.97% | 62.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 00:00:00 | 13.65 | 11.87 | 12.60 | Stage2 pullback-breakout RSI=69 vol=2.9x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 00:00:00 | 14.65 | 11.99 | 13.17 | T1 booked 50% @ 14.65 |
| Stop hit — per-position SL triggered | 2023-08-16 00:00:00 | 14.05 | 12.13 | 13.73 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 15.25 | 12.36 | 14.20 | Stage2 pullback-breakout RSI=66 vol=2.5x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 16.65 | 12.45 | 14.65 | T1 booked 50% @ 16.65 |
| Target hit | 2023-10-23 00:00:00 | 21.30 | 15.11 | 23.02 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 25.00 | 16.19 | 23.10 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-11-29 00:00:00 | 25.25 | 16.99 | 24.08 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 45.80 | 31.16 | 41.98 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 42.18 | 32.14 | 43.73 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 46.10 | 32.66 | 43.41 | Stage2 pullback-breakout RSI=57 vol=2.3x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 00:00:00 | 50.97 | 33.67 | 45.69 | T1 booked 50% @ 50.97 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-27 00:00:00 | 13.65 | 2023-08-07 00:00:00 | 14.65 | PARTIAL | 0.50 | 7.34% |
| BUY | retest1 | 2023-07-27 00:00:00 | 13.65 | 2023-08-16 00:00:00 | 14.05 | STOP_HIT | 0.50 | 2.93% |
| BUY | retest1 | 2023-08-31 00:00:00 | 15.25 | 2023-09-04 00:00:00 | 16.65 | PARTIAL | 0.50 | 9.20% |
| BUY | retest1 | 2023-08-31 00:00:00 | 15.25 | 2023-10-23 00:00:00 | 21.30 | TARGET_HIT | 0.50 | 39.67% |
| BUY | retest1 | 2023-11-13 00:00:00 | 25.00 | 2023-11-29 00:00:00 | 25.25 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest1 | 2024-04-03 00:00:00 | 45.80 | 2024-04-15 00:00:00 | 42.18 | STOP_HIT | 1.00 | -7.90% |
| BUY | retest1 | 2024-04-23 00:00:00 | 46.10 | 2024-05-03 00:00:00 | 50.97 | PARTIAL | 0.50 | 10.55% |

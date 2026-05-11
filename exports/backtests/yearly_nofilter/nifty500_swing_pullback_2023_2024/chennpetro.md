# Chennai Petroleum Corporation Ltd. (CHENNPETRO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1052.80
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 5.60% / 7.08%
- **Sum % (uncompounded):** 22.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 5.60% | 22.4% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 5.60% | 22.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 5.60% | 22.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 00:00:00 | 422.20 | 324.96 | 385.05 | Stage2 pullback-breakout RSI=64 vol=4.5x ATR=14.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 452.11 | 333.19 | 413.06 | T1 booked 50% @ 452.11 |
| Target hit | 2023-10-06 00:00:00 | 482.90 | 362.48 | 492.06 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 665.45 | 430.13 | 606.07 | Stage2 pullback-breakout RSI=69 vol=1.8x ATR=28.05 |
| Stop hit — per-position SL triggered | 2023-12-14 00:00:00 | 684.70 | 453.60 | 651.41 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 913.80 | 654.98 | 879.54 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=48.80 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 895.80 | 680.63 | 906.02 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-29 00:00:00 | 422.20 | 2023-09-08 00:00:00 | 452.11 | PARTIAL | 0.50 | 7.08% |
| BUY | retest1 | 2023-08-29 00:00:00 | 422.20 | 2023-10-06 00:00:00 | 482.90 | TARGET_HIT | 0.50 | 14.38% |
| BUY | retest1 | 2023-11-30 00:00:00 | 665.45 | 2023-12-14 00:00:00 | 684.70 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest1 | 2024-03-27 00:00:00 | 913.80 | 2024-04-12 00:00:00 | 895.80 | STOP_HIT | 1.00 | -1.97% |

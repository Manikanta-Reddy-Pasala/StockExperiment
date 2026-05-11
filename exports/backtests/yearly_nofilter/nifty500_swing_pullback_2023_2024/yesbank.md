# Yes Bank Ltd. (YESBANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 22.94
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
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 1
- **Target hits / Stop hits / Partials:** 4 / 1 / 5
- **Avg / median % per leg:** 5.91% / 5.52%
- **Sum % (uncompounded):** 59.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 9 | 90.0% | 4 | 1 | 5 | 5.91% | 59.1% |
| BUY @ 2nd Alert (retest1) | 10 | 9 | 90.0% | 4 | 1 | 5 | 5.91% | 59.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 9 | 90.0% | 4 | 1 | 5 | 5.91% | 59.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 17.35 | 16.79 | 16.96 | Stage2 pullback-breakout RSI=63 vol=4.0x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 18.09 | 16.80 | 17.11 | T1 booked 50% @ 18.09 |
| Target hit | 2023-09-12 00:00:00 | 17.55 | 16.88 | 17.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 17.95 | 16.92 | 16.71 | Stage2 pullback-breakout RSI=69 vol=3.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 00:00:00 | 18.94 | 16.98 | 17.35 | T1 booked 50% @ 18.94 |
| Target hit | 2023-12-20 00:00:00 | 20.50 | 17.74 | 20.53 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 00:00:00 | 22.65 | 17.98 | 20.96 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 00:00:00 | 24.45 | 18.25 | 21.99 | T1 booked 50% @ 24.45 |
| Target hit | 2024-01-30 00:00:00 | 23.90 | 19.17 | 24.11 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 00:00:00 | 25.40 | 19.40 | 24.07 | Stage2 pullback-breakout RSI=60 vol=2.5x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 00:00:00 | 27.49 | 19.50 | 24.62 | T1 booked 50% @ 27.49 |
| Target hit | 2024-02-20 00:00:00 | 26.60 | 20.28 | 26.81 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 00:00:00 | 25.20 | 21.58 | 24.29 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 00:00:00 | 27.09 | 21.80 | 25.02 | T1 booked 50% @ 27.09 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 25.20 | 21.91 | 25.14 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-01 00:00:00 | 17.35 | 2023-09-04 00:00:00 | 18.09 | PARTIAL | 0.50 | 4.29% |
| BUY | retest1 | 2023-09-01 00:00:00 | 17.35 | 2023-09-12 00:00:00 | 17.55 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2023-11-08 00:00:00 | 17.95 | 2023-11-13 00:00:00 | 18.94 | PARTIAL | 0.50 | 5.51% |
| BUY | retest1 | 2023-11-08 00:00:00 | 17.95 | 2023-12-20 00:00:00 | 20.50 | TARGET_HIT | 0.50 | 14.21% |
| BUY | retest1 | 2024-01-01 00:00:00 | 22.65 | 2024-01-08 00:00:00 | 24.45 | PARTIAL | 0.50 | 7.93% |
| BUY | retest1 | 2024-01-01 00:00:00 | 22.65 | 2024-01-30 00:00:00 | 23.90 | TARGET_HIT | 0.50 | 5.52% |
| BUY | retest1 | 2024-02-06 00:00:00 | 25.40 | 2024-02-07 00:00:00 | 27.49 | PARTIAL | 0.50 | 8.23% |
| BUY | retest1 | 2024-02-06 00:00:00 | 25.40 | 2024-02-20 00:00:00 | 26.60 | TARGET_HIT | 0.50 | 4.72% |
| BUY | retest1 | 2024-04-22 00:00:00 | 25.20 | 2024-04-29 00:00:00 | 27.09 | PARTIAL | 0.50 | 7.50% |
| BUY | retest1 | 2024-04-22 00:00:00 | 25.20 | 2024-05-03 00:00:00 | 25.20 | STOP_HIT | 0.50 | 0.00% |

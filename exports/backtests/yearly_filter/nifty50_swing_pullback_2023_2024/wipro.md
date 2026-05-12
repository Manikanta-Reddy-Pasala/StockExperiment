# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 197.91
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 2.95% / 3.48%
- **Sum % (uncompounded):** 14.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 2.95% | 14.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 2.95% | 14.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 2.95% | 14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 210.58 | 201.25 | 207.35 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=3.73 |
| Stop hit — per-position SL triggered | 2023-10-13 00:00:00 | 204.98 | 201.36 | 207.28 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 00:00:00 | 209.38 | 200.06 | 200.55 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 00:00:00 | 216.66 | 200.72 | 205.63 | T1 booked 50% @ 216.66 |
| Target hit | 2024-01-11 00:00:00 | 224.10 | 205.42 | 224.67 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 232.73 | 205.69 | 225.44 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 244.97 | 206.10 | 227.52 | T1 booked 50% @ 244.97 |
| Stop hit — per-position SL triggered | 2024-01-29 00:00:00 | 236.50 | 208.94 | 234.05 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-11 00:00:00 | 210.58 | 2023-10-13 00:00:00 | 204.98 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest1 | 2023-12-06 00:00:00 | 209.38 | 2023-12-14 00:00:00 | 216.66 | PARTIAL | 0.50 | 3.48% |
| BUY | retest1 | 2023-12-06 00:00:00 | 209.38 | 2024-01-11 00:00:00 | 224.10 | TARGET_HIT | 0.50 | 7.03% |
| BUY | retest1 | 2024-01-12 00:00:00 | 232.73 | 2024-01-15 00:00:00 | 244.97 | PARTIAL | 0.50 | 5.26% |
| BUY | retest1 | 2024-01-12 00:00:00 | 232.73 | 2024-01-29 00:00:00 | 236.50 | STOP_HIT | 0.50 | 1.62% |

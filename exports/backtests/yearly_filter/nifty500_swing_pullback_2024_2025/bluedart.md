# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 5303.50
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.02% | 0.1% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.02% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.02% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 00:00:00 | 8138.15 | 6760.24 | 7701.97 | Stage2 pullback-breakout RSI=69 vol=1.6x ATR=239.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 00:00:00 | 8617.72 | 6880.93 | 8062.24 | T1 booked 50% @ 8617.72 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 8138.15 | 6960.55 | 8235.02 | SL hit (bars_held=13) |

### Cycle 2 — BUY (started 2024-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 00:00:00 | 8175.55 | 7030.52 | 8041.00 | Stage2 pullback-breakout RSI=54 vol=1.8x ATR=299.32 |
| Stop hit — per-position SL triggered | 2024-08-14 00:00:00 | 7839.65 | 7127.20 | 8033.28 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 00:00:00 | 8640.90 | 7434.41 | 8267.98 | Stage2 pullback-breakout RSI=59 vol=3.7x ATR=285.17 |
| Stop hit — per-position SL triggered | 2024-10-16 00:00:00 | 8492.90 | 7538.51 | 8430.99 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-01 00:00:00 | 8138.15 | 2024-07-11 00:00:00 | 8617.72 | PARTIAL | 0.50 | 5.89% |
| BUY | retest1 | 2024-07-01 00:00:00 | 8138.15 | 2024-07-19 00:00:00 | 8138.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 00:00:00 | 8175.55 | 2024-08-14 00:00:00 | 7839.65 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2024-10-01 00:00:00 | 8640.90 | 2024-10-16 00:00:00 | 8492.90 | STOP_HIT | 1.00 | -1.71% |

# MARUTI (MARUTI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 13521.00
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
- **Avg / median % per leg:** 5.62% / 3.27%
- **Sum % (uncompounded):** 28.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 5.62% | 28.1% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 5.62% | 28.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 5.62% | 28.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 9789.05 | 8935.58 | 9488.40 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=149.82 |
| Stop hit — per-position SL triggered | 2023-07-14 00:00:00 | 9603.65 | 9015.35 | 9663.07 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 10003.80 | 9178.88 | 9605.22 | Stage2 pullback-breakout RSI=69 vol=3.3x ATR=163.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 10330.82 | 9190.34 | 9674.36 | T1 booked 50% @ 10330.82 |
| Stop hit — per-position SL triggered | 2023-09-21 00:00:00 | 10284.30 | 9337.02 | 10206.10 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 10186.90 | 9906.61 | 10047.62 | Stage2 pullback-breakout RSI=54 vol=1.6x ATR=177.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 00:00:00 | 10542.37 | 9913.89 | 10103.84 | T1 booked 50% @ 10542.37 |
| Target hit | 2024-04-12 00:00:00 | 12266.55 | 10605.61 | 12270.86 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 9789.05 | 2023-07-14 00:00:00 | 9603.65 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2023-08-31 00:00:00 | 10003.80 | 2023-09-01 00:00:00 | 10330.82 | PARTIAL | 0.50 | 3.27% |
| BUY | retest1 | 2023-08-31 00:00:00 | 10003.80 | 2023-09-21 00:00:00 | 10284.30 | STOP_HIT | 0.50 | 2.80% |
| BUY | retest1 | 2024-01-31 00:00:00 | 10186.90 | 2024-02-01 00:00:00 | 10542.37 | PARTIAL | 0.50 | 3.49% |
| BUY | retest1 | 2024-01-31 00:00:00 | 10186.90 | 2024-04-12 00:00:00 | 12266.55 | TARGET_HIT | 0.50 | 20.41% |

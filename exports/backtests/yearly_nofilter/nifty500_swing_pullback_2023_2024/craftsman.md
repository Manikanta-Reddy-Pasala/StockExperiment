# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 9200.50
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 5.04% / 1.87%
- **Sum % (uncompounded):** 25.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 3 | 1 | 5.04% | 25.2% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 3 | 1 | 5.04% | 25.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 3 | 1 | 5.04% | 25.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 4097.65 | 3348.47 | 3967.68 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=117.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 00:00:00 | 4332.50 | 3368.05 | 4036.41 | T1 booked 50% @ 4332.50 |
| Target hit | 2023-09-08 00:00:00 | 4817.75 | 3818.78 | 4824.31 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 4791.55 | 4046.12 | 4591.80 | Stage2 pullback-breakout RSI=60 vol=5.1x ATR=175.35 |
| Stop hit — per-position SL triggered | 2023-11-13 00:00:00 | 4798.05 | 4131.93 | 4779.44 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 5107.00 | 4148.27 | 4812.91 | Stage2 pullback-breakout RSI=67 vol=3.2x ATR=160.05 |
| Stop hit — per-position SL triggered | 2023-12-01 00:00:00 | 5100.75 | 4239.82 | 4990.84 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 5307.20 | 4317.76 | 5085.98 | Stage2 pullback-breakout RSI=63 vol=1.5x ATR=159.15 |
| Stop hit — per-position SL triggered | 2023-12-29 00:00:00 | 5406.25 | 4414.68 | 5249.19 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 4097.65 | 2023-07-14 00:00:00 | 4332.50 | PARTIAL | 0.50 | 5.73% |
| BUY | retest1 | 2023-07-12 00:00:00 | 4097.65 | 2023-09-08 00:00:00 | 4817.75 | TARGET_HIT | 0.50 | 17.57% |
| BUY | retest1 | 2023-10-30 00:00:00 | 4791.55 | 2023-11-13 00:00:00 | 4798.05 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2023-11-16 00:00:00 | 5107.00 | 2023-12-01 00:00:00 | 5100.75 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-12-14 00:00:00 | 5307.20 | 2023-12-29 00:00:00 | 5406.25 | STOP_HIT | 1.00 | 1.87% |

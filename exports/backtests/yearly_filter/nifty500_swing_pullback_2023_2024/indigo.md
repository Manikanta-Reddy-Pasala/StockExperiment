# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 4342.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 0.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.07% | 0.3% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.07% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.07% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 2535.50 | 2293.87 | 2430.19 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=55.67 |
| Stop hit — per-position SL triggered | 2023-10-09 00:00:00 | 2451.99 | 2295.66 | 2434.31 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 2559.30 | 2331.60 | 2484.21 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=63.76 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 2631.25 | 2353.93 | 2538.71 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 3066.60 | 2577.68 | 2973.86 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=76.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 3219.81 | 2589.00 | 3005.85 | T1 booked 50% @ 3219.81 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 3066.60 | 2609.92 | 3044.08 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-03-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 00:00:00 | 3218.55 | 2707.45 | 3128.74 | Stage2 pullback-breakout RSI=60 vol=8.6x ATR=89.35 |
| Stop hit — per-position SL triggered | 2024-03-14 00:00:00 | 3084.53 | 2721.33 | 3140.80 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-06 00:00:00 | 2535.50 | 2023-10-09 00:00:00 | 2451.99 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2023-11-06 00:00:00 | 2559.30 | 2023-11-20 00:00:00 | 2631.25 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest1 | 2024-02-01 00:00:00 | 3066.60 | 2024-02-05 00:00:00 | 3219.81 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-01 00:00:00 | 3066.60 | 2024-02-09 00:00:00 | 3066.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-11 00:00:00 | 3218.55 | 2024-03-14 00:00:00 | 3084.53 | STOP_HIT | 1.00 | -4.16% |

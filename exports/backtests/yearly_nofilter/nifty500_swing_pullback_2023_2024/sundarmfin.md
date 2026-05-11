# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 4746.40
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
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 3
- **Avg / median % per leg:** 5.63% / 6.61%
- **Sum % (uncompounded):** 39.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 5.63% | 39.4% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 5.63% | 39.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 2 | 3 | 5.63% | 39.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 00:00:00 | 2659.20 | 2455.03 | 2605.60 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=47.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 00:00:00 | 2753.82 | 2458.03 | 2620.00 | T1 booked 50% @ 2753.82 |
| Target hit | 2023-11-10 00:00:00 | 3183.80 | 2686.69 | 3204.35 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 3373.75 | 2746.10 | 3208.63 | Stage2 pullback-breakout RSI=63 vol=5.7x ATR=108.69 |
| Stop hit — per-position SL triggered | 2023-12-11 00:00:00 | 3210.72 | 2793.81 | 3291.68 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-12-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 00:00:00 | 3496.25 | 2806.41 | 3317.35 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=117.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 00:00:00 | 3731.47 | 2832.15 | 3412.41 | T1 booked 50% @ 3731.47 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 3496.25 | 2847.12 | 3443.83 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-02-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 00:00:00 | 3698.05 | 3046.42 | 3589.56 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=122.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 00:00:00 | 3942.37 | 3116.68 | 3724.06 | T1 booked 50% @ 3942.37 |
| Target hit | 2024-03-11 00:00:00 | 3978.85 | 3261.41 | 4052.50 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-11 00:00:00 | 2659.20 | 2023-09-12 00:00:00 | 2753.82 | PARTIAL | 0.50 | 3.56% |
| BUY | retest1 | 2023-09-11 00:00:00 | 2659.20 | 2023-11-10 00:00:00 | 3183.80 | TARGET_HIT | 0.50 | 19.73% |
| BUY | retest1 | 2023-11-29 00:00:00 | 3373.75 | 2023-12-11 00:00:00 | 3210.72 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest1 | 2023-12-13 00:00:00 | 3496.25 | 2023-12-18 00:00:00 | 3731.47 | PARTIAL | 0.50 | 6.73% |
| BUY | retest1 | 2023-12-13 00:00:00 | 3496.25 | 2023-12-20 00:00:00 | 3496.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-05 00:00:00 | 3698.05 | 2024-02-19 00:00:00 | 3942.37 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2024-02-05 00:00:00 | 3698.05 | 2024-03-11 00:00:00 | 3978.85 | TARGET_HIT | 0.50 | 7.59% |

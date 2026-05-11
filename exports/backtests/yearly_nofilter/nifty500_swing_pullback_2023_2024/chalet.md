# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 770.55
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
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 0
- **Target hits / Stop hits / Partials:** 3 / 1 / 4
- **Avg / median % per leg:** 12.15% / 8.60%
- **Sum % (uncompounded):** 97.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 8 | 100.0% | 3 | 1 | 4 | 12.15% | 97.2% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 3 | 1 | 4 | 12.15% | 97.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 3 | 1 | 4 | 12.15% | 97.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 427.15 | 376.50 | 421.61 | Stage2 pullback-breakout RSI=55 vol=3.2x ATR=12.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 00:00:00 | 451.83 | 380.21 | 430.39 | T1 booked 50% @ 451.83 |
| Target hit | 2023-09-21 00:00:00 | 540.40 | 432.50 | 542.88 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 566.90 | 440.71 | 548.84 | Stage2 pullback-breakout RSI=62 vol=5.1x ATR=18.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 00:00:00 | 603.15 | 445.22 | 560.65 | T1 booked 50% @ 603.15 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 581.30 | 458.14 | 580.58 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 00:00:00 | 614.60 | 495.69 | 587.95 | Stage2 pullback-breakout RSI=63 vol=7.2x ATR=18.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 00:00:00 | 651.46 | 498.60 | 597.85 | T1 booked 50% @ 651.46 |
| Target hit | 2024-02-29 00:00:00 | 810.50 | 607.23 | 818.29 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 00:00:00 | 794.40 | 627.47 | 771.71 | Stage2 pullback-breakout RSI=55 vol=3.5x ATR=37.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 00:00:00 | 870.00 | 633.67 | 789.07 | T1 booked 50% @ 870.00 |
| Target hit | 2024-05-06 00:00:00 | 862.70 | 686.09 | 865.08 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 427.15 | 2023-07-07 00:00:00 | 451.83 | PARTIAL | 0.50 | 5.78% |
| BUY | retest1 | 2023-06-28 00:00:00 | 427.15 | 2023-09-21 00:00:00 | 540.40 | TARGET_HIT | 0.50 | 26.51% |
| BUY | retest1 | 2023-10-03 00:00:00 | 566.90 | 2023-10-06 00:00:00 | 603.15 | PARTIAL | 0.50 | 6.39% |
| BUY | retest1 | 2023-10-03 00:00:00 | 566.90 | 2023-10-19 00:00:00 | 581.30 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2023-12-12 00:00:00 | 614.60 | 2023-12-14 00:00:00 | 651.46 | PARTIAL | 0.50 | 6.00% |
| BUY | retest1 | 2023-12-12 00:00:00 | 614.60 | 2024-02-29 00:00:00 | 810.50 | TARGET_HIT | 0.50 | 31.87% |
| BUY | retest1 | 2024-03-20 00:00:00 | 794.40 | 2024-03-26 00:00:00 | 870.00 | PARTIAL | 0.50 | 9.52% |
| BUY | retest1 | 2024-03-20 00:00:00 | 794.40 | 2024-05-06 00:00:00 | 862.70 | TARGET_HIT | 0.50 | 8.60% |

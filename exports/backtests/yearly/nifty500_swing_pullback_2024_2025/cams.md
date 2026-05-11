# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 836.30
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.63% / 0.00%
- **Sum % (uncompounded):** 11.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 1.63% | 11.4% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 1.63% | 11.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 1.63% | 11.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 05:30:00 | 909.38 | 664.67 | 841.26 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=44.40 |
| Stop hit — per-position SL triggered | 2024-09-02 05:30:00 | 883.62 | 684.96 | 863.20 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 05:30:00 | 904.06 | 703.10 | 871.95 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=29.52 |
| Stop hit — per-position SL triggered | 2024-09-30 05:30:00 | 881.82 | 721.50 | 886.59 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 05:30:00 | 920.12 | 735.55 | 887.10 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=34.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 05:30:00 | 988.13 | 742.02 | 904.91 | T1 booked 50% @ 988.13 |
| Stop hit — per-position SL triggered | 2024-10-22 05:30:00 | 920.12 | 747.25 | 908.01 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-11-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 05:30:00 | 908.53 | 759.20 | 895.70 | Stage2 pullback-breakout RSI=53 vol=2.7x ATR=37.62 |
| Stop hit — per-position SL triggered | 2024-11-19 05:30:00 | 907.62 | 774.64 | 910.42 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-11-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 05:30:00 | 970.01 | 782.00 | 917.73 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=35.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 05:30:00 | 1041.14 | 795.63 | 963.83 | T1 booked 50% @ 1041.14 |
| Target hit | 2024-12-20 05:30:00 | 990.20 | 820.26 | 1007.22 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-19 05:30:00 | 909.38 | 2024-09-02 05:30:00 | 883.62 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest1 | 2024-09-16 05:30:00 | 904.06 | 2024-09-30 05:30:00 | 881.82 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest1 | 2024-10-14 05:30:00 | 920.12 | 2024-10-17 05:30:00 | 988.13 | PARTIAL | 0.50 | 7.39% |
| BUY | retest1 | 2024-10-14 05:30:00 | 920.12 | 2024-10-22 05:30:00 | 920.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-04 05:30:00 | 908.53 | 2024-11-19 05:30:00 | 907.62 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest1 | 2024-11-27 05:30:00 | 970.01 | 2024-12-05 05:30:00 | 1041.14 | PARTIAL | 0.50 | 7.33% |
| BUY | retest1 | 2024-11-27 05:30:00 | 970.01 | 2024-12-20 05:30:00 | 990.20 | TARGET_HIT | 0.50 | 2.08% |

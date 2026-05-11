# Kfin Technologies Ltd. (KFINTECH)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 903.25
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 3
- **Avg / median % per leg:** 7.08% / 7.93%
- **Sum % (uncompounded):** 42.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 1 | 2 | 3 | 7.08% | 42.5% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 1 | 2 | 3 | 7.08% | 42.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 1 | 2 | 3 | 7.08% | 42.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 00:00:00 | 734.30 | 613.29 | 708.15 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=25.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 00:00:00 | 785.80 | 622.98 | 732.72 | T1 booked 50% @ 785.80 |
| Stop hit — per-position SL triggered | 2024-07-16 00:00:00 | 753.10 | 628.15 | 739.96 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 798.60 | 637.96 | 750.49 | Stage2 pullback-breakout RSI=65 vol=3.4x ATR=31.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 00:00:00 | 861.93 | 642.13 | 768.48 | T1 booked 50% @ 861.93 |
| Target hit | 2024-09-06 00:00:00 | 989.60 | 722.13 | 994.74 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-10-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 00:00:00 | 1137.35 | 772.00 | 1042.76 | Stage2 pullback-breakout RSI=64 vol=7.5x ATR=53.29 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 1057.41 | 777.48 | 1043.59 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-11-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 00:00:00 | 1066.70 | 844.52 | 1013.59 | Stage2 pullback-breakout RSI=59 vol=2.0x ATR=43.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 00:00:00 | 1152.92 | 850.18 | 1035.01 | T1 booked 50% @ 1152.92 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-01 00:00:00 | 734.30 | 2024-07-10 00:00:00 | 785.80 | PARTIAL | 0.50 | 7.01% |
| BUY | retest1 | 2024-07-01 00:00:00 | 734.30 | 2024-07-16 00:00:00 | 753.10 | STOP_HIT | 0.50 | 2.56% |
| BUY | retest1 | 2024-07-29 00:00:00 | 798.60 | 2024-07-31 00:00:00 | 861.93 | PARTIAL | 0.50 | 7.93% |
| BUY | retest1 | 2024-07-29 00:00:00 | 798.60 | 2024-09-06 00:00:00 | 989.60 | TARGET_HIT | 0.50 | 23.92% |
| BUY | retest1 | 2024-10-01 00:00:00 | 1137.35 | 2024-10-04 00:00:00 | 1057.41 | STOP_HIT | 1.00 | -7.03% |
| BUY | retest1 | 2024-11-22 00:00:00 | 1066.70 | 2024-11-26 00:00:00 | 1152.92 | PARTIAL | 0.50 | 8.08% |

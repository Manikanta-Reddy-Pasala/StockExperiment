# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 810.85
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 3.08% / 7.72%
- **Sum % (uncompounded):** 18.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.08% | 18.5% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.08% | 18.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.08% | 18.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 00:00:00 | 688.05 | 376.62 | 642.92 | Stage2 pullback-breakout RSI=67 vol=2.3x ATR=30.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 00:00:00 | 748.10 | 387.31 | 668.39 | T1 booked 50% @ 748.10 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 688.05 | 440.20 | 760.99 | SL hit (bars_held=17) |

### Cycle 2 — BUY (started 2023-10-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 00:00:00 | 822.60 | 508.74 | 775.57 | Stage2 pullback-breakout RSI=64 vol=5.3x ATR=33.41 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 772.49 | 517.42 | 782.34 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 1063.45 | 702.42 | 1020.38 | Stage2 pullback-breakout RSI=62 vol=3.2x ATR=41.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 00:00:00 | 1145.56 | 728.48 | 1057.41 | T1 booked 50% @ 1145.56 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 1063.45 | 735.62 | 1063.15 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 978.55 | 825.45 | 933.49 | Stage2 pullback-breakout RSI=59 vol=3.0x ATR=39.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 1057.83 | 837.60 | 971.88 | T1 booked 50% @ 1057.83 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-21 00:00:00 | 688.05 | 2023-08-24 00:00:00 | 748.10 | PARTIAL | 0.50 | 8.73% |
| BUY | retest1 | 2023-08-21 00:00:00 | 688.05 | 2023-09-13 00:00:00 | 688.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 00:00:00 | 822.60 | 2023-10-23 00:00:00 | 772.49 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest1 | 2024-01-11 00:00:00 | 1063.45 | 2024-01-20 00:00:00 | 1145.56 | PARTIAL | 0.50 | 7.72% |
| BUY | retest1 | 2024-01-11 00:00:00 | 1063.45 | 2024-01-24 00:00:00 | 1063.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-16 00:00:00 | 978.55 | 2024-04-26 00:00:00 | 1057.83 | PARTIAL | 0.50 | 8.10% |

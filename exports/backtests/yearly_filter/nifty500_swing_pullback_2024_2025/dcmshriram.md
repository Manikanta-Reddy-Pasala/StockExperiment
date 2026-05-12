# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1217.70
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.36% / 5.88%
- **Sum % (uncompounded):** 1.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.36% | 1.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.36% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.36% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 1052.10 | 982.34 | 1012.51 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=30.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 00:00:00 | 1114.01 | 986.82 | 1033.58 | T1 booked 50% @ 1114.01 |
| Target hit | 2024-09-06 00:00:00 | 1123.05 | 1016.13 | 1125.49 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 1121.75 | 1026.29 | 1051.00 | Stage2 pullback-breakout RSI=63 vol=4.8x ATR=41.35 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 1059.73 | 1028.91 | 1064.17 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-12-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 00:00:00 | 1151.95 | 1065.90 | 1106.66 | Stage2 pullback-breakout RSI=58 vol=2.7x ATR=43.60 |
| Stop hit — per-position SL triggered | 2025-01-06 00:00:00 | 1086.55 | 1067.63 | 1107.11 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-29 00:00:00 | 1052.10 | 2024-08-06 00:00:00 | 1114.01 | PARTIAL | 0.50 | 5.88% |
| BUY | retest1 | 2024-07-29 00:00:00 | 1052.10 | 2024-09-06 00:00:00 | 1123.05 | TARGET_HIT | 0.50 | 6.74% |
| BUY | retest1 | 2024-10-16 00:00:00 | 1121.75 | 2024-10-22 00:00:00 | 1059.73 | STOP_HIT | 1.00 | -5.53% |
| BUY | retest1 | 2024-12-31 00:00:00 | 1151.95 | 2025-01-06 00:00:00 | 1086.55 | STOP_HIT | 1.00 | -5.68% |

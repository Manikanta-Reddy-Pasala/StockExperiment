# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 729.15
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 1.62% / -0.84%
- **Sum % (uncompounded):** 14.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 1.62% | 14.6% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 1.62% | 14.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 3 | 33.3% | 1 | 6 | 2 | 1.62% | 14.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 05:30:00 | 1090.65 | 857.49 | 1067.97 | Stage2 pullback-breakout RSI=57 vol=2.3x ATR=32.77 |
| Stop hit — per-position SL triggered | 2023-07-21 05:30:00 | 1041.49 | 865.16 | 1062.17 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-08-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 05:30:00 | 1146.25 | 886.00 | 1079.56 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=41.85 |
| Stop hit — per-position SL triggered | 2023-08-21 05:30:00 | 1131.40 | 910.18 | 1117.83 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-08-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 05:30:00 | 1177.35 | 915.03 | 1124.77 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=37.82 |
| Stop hit — per-position SL triggered | 2023-09-06 05:30:00 | 1167.45 | 939.04 | 1152.66 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-09-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 05:30:00 | 1142.40 | 965.46 | 1127.65 | Stage2 pullback-breakout RSI=53 vol=2.0x ATR=37.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 05:30:00 | 1217.94 | 980.14 | 1145.66 | T1 booked 50% @ 1217.94 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 1142.40 | 997.07 | 1174.18 | SL hit (bars_held=16) |

### Cycle 5 — BUY (started 2023-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 05:30:00 | 1217.55 | 1005.03 | 1170.24 | Stage2 pullback-breakout RSI=59 vol=3.9x ATR=47.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 05:30:00 | 1312.28 | 1019.62 | 1211.28 | T1 booked 50% @ 1312.28 |
| Target hit | 2023-12-12 05:30:00 | 1432.65 | 1110.78 | 1443.05 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2023-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 05:30:00 | 1518.60 | 1122.05 | 1456.17 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=53.24 |
| Stop hit — per-position SL triggered | 2023-12-21 05:30:00 | 1438.75 | 1136.66 | 1469.02 | SL hit (bars_held=4) |

### Cycle 7 — BUY (started 2024-02-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 05:30:00 | 1674.95 | 1238.93 | 1531.86 | Stage2 pullback-breakout RSI=68 vol=1.5x ATR=62.22 |
| Stop hit — per-position SL triggered | 2024-02-13 05:30:00 | 1581.62 | 1260.33 | 1588.22 | SL hit (bars_held=5) |

### Cycle 8 — BUY (started 2024-04-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 05:30:00 | 1508.50 | 1346.88 | 1436.59 | Stage2 pullback-breakout RSI=60 vol=4.8x ATR=52.19 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-17 05:30:00 | 1090.65 | 2023-07-21 05:30:00 | 1041.49 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest1 | 2023-08-04 05:30:00 | 1146.25 | 2023-08-21 05:30:00 | 1131.40 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2023-08-23 05:30:00 | 1177.35 | 2023-09-06 05:30:00 | 1167.45 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest1 | 2023-09-28 05:30:00 | 1142.40 | 2023-10-11 05:30:00 | 1217.94 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2023-09-28 05:30:00 | 1142.40 | 2023-10-23 05:30:00 | 1142.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-31 05:30:00 | 1217.55 | 2023-11-08 05:30:00 | 1312.28 | PARTIAL | 0.50 | 7.78% |
| BUY | retest1 | 2023-10-31 05:30:00 | 1217.55 | 2023-12-12 05:30:00 | 1432.65 | TARGET_HIT | 0.50 | 17.67% |
| BUY | retest1 | 2023-12-15 05:30:00 | 1518.60 | 2023-12-21 05:30:00 | 1438.75 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest1 | 2024-02-06 05:30:00 | 1674.95 | 2024-02-13 05:30:00 | 1581.62 | STOP_HIT | 1.00 | -5.57% |

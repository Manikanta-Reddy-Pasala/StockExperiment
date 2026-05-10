# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1248.40
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 1 / 6 / 1
- **Avg / median % per leg:** -0.68% / -1.08%
- **Sum % (uncompounded):** -5.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 6 | 1 | -0.68% | -5.4% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 6 | 1 | -0.68% | -5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 6 | 1 | -0.68% | -5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 954.90 | 919.57 | 927.69 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=24.22 |
| Stop hit — per-position SL triggered | 2025-07-10 05:30:00 | 944.55 | 922.51 | 942.00 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 1028.35 | 940.06 | 984.40 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=25.46 |
| Stop hit — per-position SL triggered | 2025-09-17 05:30:00 | 1033.65 | 949.50 | 1019.55 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 1076.50 | 960.20 | 1042.83 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=23.62 |
| Stop hit — per-position SL triggered | 2025-10-07 05:30:00 | 1041.07 | 961.87 | 1043.09 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 1073.50 | 969.22 | 1026.58 | Stage2 pullback-breakout RSI=63 vol=2.6x ATR=24.80 |
| Stop hit — per-position SL triggered | 2025-11-07 05:30:00 | 1036.31 | 975.94 | 1047.50 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 1053.80 | 992.15 | 1015.08 | Stage2 pullback-breakout RSI=60 vol=3.5x ATR=23.74 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 1018.18 | 996.46 | 1037.05 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 1119.40 | 1002.66 | 1052.97 | Stage2 pullback-breakout RSI=67 vol=1.5x ATR=29.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 05:30:00 | 1178.56 | 1010.63 | 1092.48 | T1 booked 50% @ 1178.56 |
| Target hit | 2026-03-04 05:30:00 | 1168.40 | 1045.63 | 1200.38 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2026-03-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 05:30:00 | 1225.00 | 1053.74 | 1194.35 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=35.99 |
| Stop hit — per-position SL triggered | 2026-03-13 05:30:00 | 1171.02 | 1054.63 | 1189.49 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 954.90 | 2025-07-10 05:30:00 | 944.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest1 | 2025-09-03 05:30:00 | 1028.35 | 2025-09-17 05:30:00 | 1033.65 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest1 | 2025-10-03 05:30:00 | 1076.50 | 2025-10-07 05:30:00 | 1041.07 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2025-10-28 05:30:00 | 1073.50 | 2025-11-07 05:30:00 | 1036.31 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest1 | 2025-12-31 05:30:00 | 1053.80 | 2026-01-08 05:30:00 | 1018.18 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest1 | 2026-01-28 05:30:00 | 1119.40 | 2026-02-04 05:30:00 | 1178.56 | PARTIAL | 0.50 | 5.29% |
| BUY | retest1 | 2026-01-28 05:30:00 | 1119.40 | 2026-03-04 05:30:00 | 1168.40 | TARGET_HIT | 0.50 | 4.38% |
| BUY | retest1 | 2026-03-12 05:30:00 | 1225.00 | 2026-03-13 05:30:00 | 1171.02 | STOP_HIT | 1.00 | -4.41% |

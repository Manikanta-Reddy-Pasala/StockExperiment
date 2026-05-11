# Axis Bank Ltd. (AXISBANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1268.30
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 1.46% / 2.66%
- **Sum % (uncompounded):** 7.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.46% | 7.3% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.46% | 7.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.46% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 1060.15 | 951.71 | 1011.90 | Stage2 pullback-breakout RSI=67 vol=1.5x ATR=17.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 00:00:00 | 1096.03 | 954.44 | 1026.11 | T1 booked 50% @ 1096.03 |
| Target hit | 2023-12-22 00:00:00 | 1088.30 | 977.04 | 1093.57 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-03-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 00:00:00 | 1125.00 | 1022.01 | 1089.59 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=24.11 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 1088.84 | 1024.45 | 1093.34 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 1127.00 | 1032.61 | 1064.37 | Stage2 pullback-breakout RSI=66 vol=3.5x ATR=25.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 00:00:00 | 1177.47 | 1036.13 | 1087.35 | T1 booked 50% @ 1177.47 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 1127.00 | 1040.23 | 1104.68 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-29 00:00:00 | 1060.15 | 2023-12-01 00:00:00 | 1096.03 | PARTIAL | 0.50 | 3.38% |
| BUY | retest1 | 2023-11-29 00:00:00 | 1060.15 | 2023-12-22 00:00:00 | 1088.30 | TARGET_HIT | 0.50 | 2.66% |
| BUY | retest1 | 2024-03-06 00:00:00 | 1125.00 | 2024-03-12 00:00:00 | 1088.84 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest1 | 2024-04-25 00:00:00 | 1127.00 | 2024-04-30 00:00:00 | 1177.47 | PARTIAL | 0.50 | 4.48% |
| BUY | retest1 | 2024-04-25 00:00:00 | 1127.00 | 2024-05-07 00:00:00 | 1127.00 | STOP_HIT | 0.50 | 0.00% |

# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1728.90
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
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 3
- **Avg / median % per leg:** 7.27% / 8.66%
- **Sum % (uncompounded):** 43.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 1 | 3 | 7.27% | 43.6% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 1 | 3 | 7.27% | 43.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 2 | 1 | 3 | 7.27% | 43.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 05:30:00 | 1058.65 | 924.24 | 964.51 | Stage2 pullback-breakout RSI=68 vol=11.6x ATR=39.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 05:30:00 | 1138.11 | 931.42 | 1012.34 | T1 booked 50% @ 1138.11 |
| Target hit | 2025-12-03 05:30:00 | 1085.20 | 953.06 | 1092.18 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-12-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 05:30:00 | 1227.10 | 969.12 | 1122.88 | Stage2 pullback-breakout RSI=67 vol=6.9x ATR=51.59 |
| Stop hit — per-position SL triggered | 2025-12-31 05:30:00 | 1218.80 | 996.94 | 1206.71 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 1181.00 | 1030.04 | 1151.70 | Stage2 pullback-breakout RSI=55 vol=2.4x ATR=51.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 05:30:00 | 1283.28 | 1041.97 | 1191.87 | T1 booked 50% @ 1283.28 |
| Target hit | 2026-03-20 05:30:00 | 1376.70 | 1129.61 | 1413.92 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 1465.40 | 1152.74 | 1394.59 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=66.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 05:30:00 | 1597.92 | 1166.51 | 1432.25 | T1 booked 50% @ 1597.92 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-12 05:30:00 | 1058.65 | 2025-11-18 05:30:00 | 1138.11 | PARTIAL | 0.50 | 7.51% |
| BUY | retest1 | 2025-11-12 05:30:00 | 1058.65 | 2025-12-03 05:30:00 | 1085.20 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2025-12-16 05:30:00 | 1227.10 | 2025-12-31 05:30:00 | 1218.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2026-02-03 05:30:00 | 1181.00 | 2026-02-11 05:30:00 | 1283.28 | PARTIAL | 0.50 | 8.66% |
| BUY | retest1 | 2026-02-03 05:30:00 | 1181.00 | 2026-03-20 05:30:00 | 1376.70 | TARGET_HIT | 0.50 | 16.57% |
| BUY | retest1 | 2026-04-08 05:30:00 | 1465.40 | 2026-04-15 05:30:00 | 1597.92 | PARTIAL | 0.50 | 9.04% |

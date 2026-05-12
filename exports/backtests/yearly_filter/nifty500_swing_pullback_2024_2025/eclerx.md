# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1630.00
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 4.69% / 7.45%
- **Sum % (uncompounded):** 37.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 4.69% | 37.5% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 4.69% | 37.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 4.69% | 37.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 00:00:00 | 1259.25 | 1145.48 | 1194.19 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=33.28 |
| Stop hit — per-position SL triggered | 2024-07-18 00:00:00 | 1238.10 | 1156.08 | 1233.50 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 00:00:00 | 1242.93 | 1162.46 | 1217.02 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=35.35 |
| Stop hit — per-position SL triggered | 2024-08-14 00:00:00 | 1189.91 | 1167.24 | 1228.59 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-08-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 00:00:00 | 1349.35 | 1169.81 | 1241.39 | Stage2 pullback-breakout RSI=66 vol=4.5x ATR=50.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 00:00:00 | 1449.92 | 1181.34 | 1299.61 | T1 booked 50% @ 1449.92 |
| Stop hit — per-position SL triggered | 2024-09-09 00:00:00 | 1349.35 | 1202.72 | 1375.65 | SL hit (bars_held=15) |

### Cycle 4 — BUY (started 2024-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 00:00:00 | 1439.80 | 1222.55 | 1377.34 | Stage2 pullback-breakout RSI=63 vol=12.1x ATR=58.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 00:00:00 | 1556.93 | 1242.04 | 1445.83 | T1 booked 50% @ 1556.93 |
| Target hit | 2024-10-21 00:00:00 | 1464.55 | 1268.80 | 1490.10 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-11-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 00:00:00 | 1519.95 | 1286.94 | 1466.29 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=56.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 00:00:00 | 1633.36 | 1290.48 | 1483.07 | T1 booked 50% @ 1633.36 |
| Target hit | 2024-12-20 00:00:00 | 1804.48 | 1413.79 | 1809.82 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 00:00:00 | 1259.25 | 2024-07-18 00:00:00 | 1238.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2024-08-06 00:00:00 | 1242.93 | 2024-08-14 00:00:00 | 1189.91 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2024-08-19 00:00:00 | 1349.35 | 2024-08-27 00:00:00 | 1449.92 | PARTIAL | 0.50 | 7.45% |
| BUY | retest1 | 2024-08-19 00:00:00 | 1349.35 | 2024-09-09 00:00:00 | 1349.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 00:00:00 | 1439.80 | 2024-10-07 00:00:00 | 1556.93 | PARTIAL | 0.50 | 8.14% |
| BUY | retest1 | 2024-09-25 00:00:00 | 1439.80 | 2024-10-21 00:00:00 | 1464.55 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2024-11-05 00:00:00 | 1519.95 | 2024-11-06 00:00:00 | 1633.36 | PARTIAL | 0.50 | 7.46% |
| BUY | retest1 | 2024-11-05 00:00:00 | 1519.95 | 2024-12-20 00:00:00 | 1804.48 | TARGET_HIT | 0.50 | 18.72% |

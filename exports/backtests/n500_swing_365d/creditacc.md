# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1497.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 1.15% / 2.98%
- **Sum % (uncompounded):** 9.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 5 | 2 | 1.15% | 9.2% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 5 | 2 | 1.15% | 9.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 5 | 2 | 1.15% | 9.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 05:30:00 | 1202.90 | 1075.98 | 1161.17 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=44.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 05:30:00 | 1291.57 | 1087.03 | 1203.22 | T1 booked 50% @ 1291.57 |
| Target hit | 2025-07-28 05:30:00 | 1273.70 | 1118.62 | 1283.38 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 05:30:00 | 1360.20 | 1178.14 | 1332.01 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=40.68 |
| Stop hit — per-position SL triggered | 2025-10-01 05:30:00 | 1400.80 | 1196.83 | 1358.92 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 05:30:00 | 1422.50 | 1220.62 | 1359.92 | Stage2 pullback-breakout RSI=59 vol=7.3x ATR=47.58 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 1351.13 | 1236.01 | 1388.78 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 1331.30 | 1261.59 | 1290.30 | Stage2 pullback-breakout RSI=56 vol=2.7x ATR=43.60 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 1265.90 | 1265.34 | 1292.84 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2026-01-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-21 05:30:00 | 1355.20 | 1266.24 | 1298.78 | Stage2 pullback-breakout RSI=58 vol=15.2x ATR=53.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 05:30:00 | 1462.24 | 1268.06 | 1313.17 | T1 booked 50% @ 1462.24 |
| Stop hit — per-position SL triggered | 2026-01-27 05:30:00 | 1355.20 | 1269.81 | 1320.73 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2026-02-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 05:30:00 | 1338.80 | 1273.28 | 1292.24 | Stage2 pullback-breakout RSI=58 vol=1.5x ATR=44.76 |
| Stop hit — per-position SL triggered | 2026-02-27 05:30:00 | 1271.66 | 1273.20 | 1289.66 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-25 05:30:00 | 1202.90 | 2025-07-04 05:30:00 | 1291.57 | PARTIAL | 0.50 | 7.37% |
| BUY | retest1 | 2025-06-25 05:30:00 | 1202.90 | 2025-07-28 05:30:00 | 1273.70 | TARGET_HIT | 0.50 | 5.89% |
| BUY | retest1 | 2025-09-17 05:30:00 | 1360.20 | 2025-10-01 05:30:00 | 1400.80 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest1 | 2025-10-24 05:30:00 | 1422.50 | 2025-11-06 05:30:00 | 1351.13 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest1 | 2026-01-05 05:30:00 | 1331.30 | 2026-01-20 05:30:00 | 1265.90 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest1 | 2026-01-21 05:30:00 | 1355.20 | 2026-01-22 05:30:00 | 1462.24 | PARTIAL | 0.50 | 7.90% |
| BUY | retest1 | 2026-01-21 05:30:00 | 1355.20 | 2026-01-27 05:30:00 | 1355.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 05:30:00 | 1338.80 | 2026-02-27 05:30:00 | 1271.66 | STOP_HIT | 1.00 | -5.02% |

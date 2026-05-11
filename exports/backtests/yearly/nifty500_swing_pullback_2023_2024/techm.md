# Tech Mahindra Ltd. (TECHM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1463.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 2.26% / 2.63%
- **Sum % (uncompounded):** 11.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.26% | 11.3% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.26% | 11.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.26% | 11.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 1256.75 | 1113.01 | 1205.13 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=25.79 |
| Stop hit — per-position SL triggered | 2023-09-18 00:00:00 | 1289.80 | 1127.99 | 1248.50 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 1264.70 | 1158.88 | 1214.96 | Stage2 pullback-breakout RSI=69 vol=3.0x ATR=24.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 00:00:00 | 1314.34 | 1160.35 | 1223.64 | T1 booked 50% @ 1314.34 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 1264.70 | 1163.70 | 1236.31 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 1308.05 | 1178.49 | 1256.39 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=30.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 1369.87 | 1180.07 | 1264.13 | T1 booked 50% @ 1369.87 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 1308.05 | 1182.91 | 1274.94 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 1278.75 | 1228.59 | 1231.01 | Stage2 pullback-breakout RSI=58 vol=7.1x ATR=35.87 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-04 00:00:00 | 1256.75 | 2023-09-18 00:00:00 | 1289.80 | STOP_HIT | 1.00 | 2.63% |
| BUY | retest1 | 2023-12-14 00:00:00 | 1264.70 | 2023-12-15 00:00:00 | 1314.34 | PARTIAL | 0.50 | 3.93% |
| BUY | retest1 | 2023-12-14 00:00:00 | 1264.70 | 2023-12-20 00:00:00 | 1264.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-12 00:00:00 | 1308.05 | 2024-01-15 00:00:00 | 1369.87 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2024-01-12 00:00:00 | 1308.05 | 2024-01-17 00:00:00 | 1308.05 | STOP_HIT | 0.50 | 0.00% |

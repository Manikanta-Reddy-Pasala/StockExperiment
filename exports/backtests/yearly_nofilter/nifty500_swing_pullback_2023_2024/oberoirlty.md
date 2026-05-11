# Oberoi Realty Ltd. (OBEROIRLTY)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1670.20
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 4.35% / 4.96%
- **Sum % (uncompounded):** 34.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 1 | 4 | 3 | 4.35% | 34.8% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 4 | 3 | 4.35% | 34.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 4 | 3 | 4.35% | 34.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 1040.05 | 932.64 | 991.86 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=26.34 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 1056.25 | 944.70 | 1032.58 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 1127.30 | 980.03 | 1089.80 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=27.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 1183.20 | 991.07 | 1118.42 | T1 booked 50% @ 1183.20 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 1127.30 | 995.97 | 1127.88 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2023-11-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 00:00:00 | 1164.70 | 1031.95 | 1121.04 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=32.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 00:00:00 | 1228.95 | 1035.70 | 1139.27 | T1 booked 50% @ 1228.95 |
| Target hit | 2023-12-20 00:00:00 | 1381.15 | 1130.39 | 1410.44 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-22 00:00:00 | 1378.15 | 1224.71 | 1342.67 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=44.87 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 1354.20 | 1238.15 | 1357.60 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 1424.75 | 1249.12 | 1358.16 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=42.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 00:00:00 | 1508.86 | 1256.12 | 1391.81 | T1 booked 50% @ 1508.86 |
| Stop hit — per-position SL triggered | 2024-03-28 00:00:00 | 1424.75 | 1258.30 | 1399.81 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 1040.05 | 2023-07-21 00:00:00 | 1056.25 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest1 | 2023-08-30 00:00:00 | 1127.30 | 2023-09-08 00:00:00 | 1183.20 | PARTIAL | 0.50 | 4.96% |
| BUY | retest1 | 2023-08-30 00:00:00 | 1127.30 | 2023-09-13 00:00:00 | 1127.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-01 00:00:00 | 1164.70 | 2023-11-03 00:00:00 | 1228.95 | PARTIAL | 0.50 | 5.52% |
| BUY | retest1 | 2023-11-01 00:00:00 | 1164.70 | 2023-12-20 00:00:00 | 1381.15 | TARGET_HIT | 0.50 | 18.58% |
| BUY | retest1 | 2024-02-22 00:00:00 | 1378.15 | 2024-03-06 00:00:00 | 1354.20 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest1 | 2024-03-21 00:00:00 | 1424.75 | 2024-03-27 00:00:00 | 1508.86 | PARTIAL | 0.50 | 5.90% |
| BUY | retest1 | 2024-03-21 00:00:00 | 1424.75 | 2024-03-28 00:00:00 | 1424.75 | STOP_HIT | 0.50 | 0.00% |

# Nestle India Ltd. (NESTLEIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1482.40
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
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 0 / 7 / 1
- **Avg / median % per leg:** -1.17% / -2.17%
- **Sum % (uncompounded):** -9.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.17% | -9.3% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.17% | -9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.17% | -9.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 1154.97 | 1024.55 | 1130.30 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=16.71 |
| Stop hit — per-position SL triggered | 2023-07-10 00:00:00 | 1129.91 | 1026.78 | 1131.47 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 1131.12 | 1060.74 | 1108.64 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=16.32 |
| Stop hit — per-position SL triggered | 2023-10-03 00:00:00 | 1115.76 | 1067.53 | 1122.78 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 1206.62 | 1078.00 | 1151.84 | Stage2 pullback-breakout RSI=69 vol=4.1x ATR=20.83 |
| Stop hit — per-position SL triggered | 2023-10-26 00:00:00 | 1175.37 | 1082.92 | 1168.45 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-12-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 00:00:00 | 1274.48 | 1125.82 | 1231.63 | Stage2 pullback-breakout RSI=66 vol=4.0x ATR=22.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 00:00:00 | 1319.00 | 1136.51 | 1260.35 | T1 booked 50% @ 1319.00 |
| Stop hit — per-position SL triggered | 2024-01-08 00:00:00 | 1309.65 | 1148.54 | 1296.49 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 00:00:00 | 1305.75 | 1189.38 | 1277.74 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=23.33 |
| Stop hit — per-position SL triggered | 2024-03-19 00:00:00 | 1270.76 | 1195.16 | 1281.76 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 1311.18 | 1200.56 | 1285.22 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=27.43 |
| Stop hit — per-position SL triggered | 2024-04-04 00:00:00 | 1270.04 | 1204.03 | 1286.11 | SL hit (bars_held=4) |

### Cycle 7 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 1281.33 | 1210.20 | 1262.68 | Stage2 pullback-breakout RSI=54 vol=2.5x ATR=31.08 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 1234.71 | 1211.99 | 1256.21 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 1154.97 | 2023-07-10 00:00:00 | 1129.91 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest1 | 2023-09-15 00:00:00 | 1131.12 | 2023-10-03 00:00:00 | 1115.76 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2023-10-19 00:00:00 | 1206.62 | 2023-10-26 00:00:00 | 1175.37 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest1 | 2023-12-19 00:00:00 | 1274.48 | 2023-12-29 00:00:00 | 1319.00 | PARTIAL | 0.50 | 3.49% |
| BUY | retest1 | 2023-12-19 00:00:00 | 1274.48 | 2024-01-08 00:00:00 | 1309.65 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2024-03-11 00:00:00 | 1305.75 | 2024-03-19 00:00:00 | 1270.76 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest1 | 2024-03-28 00:00:00 | 1311.18 | 2024-04-04 00:00:00 | 1270.04 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest1 | 2024-04-25 00:00:00 | 1281.33 | 2024-05-03 00:00:00 | 1234.71 | STOP_HIT | 1.00 | -3.64% |

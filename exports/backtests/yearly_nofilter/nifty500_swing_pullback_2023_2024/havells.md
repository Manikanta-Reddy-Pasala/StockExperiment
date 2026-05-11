# Havells India Ltd. (HAVELLS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1241.60
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
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 3
- **Avg / median % per leg:** 1.18% / 2.41%
- **Sum % (uncompounded):** 9.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 0 | 5 | 3 | 1.18% | 9.4% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 0 | 5 | 3 | 1.18% | 9.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 0 | 5 | 3 | 1.18% | 9.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 00:00:00 | 1334.40 | 1265.98 | 1300.83 | Stage2 pullback-breakout RSI=59 vol=4.2x ATR=31.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 00:00:00 | 1397.61 | 1267.76 | 1310.77 | T1 booked 50% @ 1397.61 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 1334.40 | 1268.11 | 1310.06 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 1339.80 | 1275.70 | 1305.67 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=26.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 00:00:00 | 1392.30 | 1279.43 | 1322.54 | T1 booked 50% @ 1392.30 |
| Stop hit — per-position SL triggered | 2023-09-06 00:00:00 | 1353.00 | 1282.68 | 1335.37 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 00:00:00 | 1372.40 | 1306.32 | 1325.39 | Stage2 pullback-breakout RSI=69 vol=1.5x ATR=23.75 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 1336.78 | 1306.61 | 1326.37 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2024-01-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 00:00:00 | 1397.70 | 1311.72 | 1351.80 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=27.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 00:00:00 | 1451.73 | 1316.69 | 1372.66 | T1 booked 50% @ 1451.73 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 1431.35 | 1321.27 | 1392.69 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 1544.15 | 1368.89 | 1493.22 | Stage2 pullback-breakout RSI=61 vol=3.1x ATR=40.05 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1484.07 | 1380.82 | 1508.25 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-18 00:00:00 | 1334.40 | 2023-07-20 00:00:00 | 1397.61 | PARTIAL | 0.50 | 4.74% |
| BUY | retest1 | 2023-07-18 00:00:00 | 1334.40 | 2023-07-21 00:00:00 | 1334.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-23 00:00:00 | 1339.80 | 2023-08-31 00:00:00 | 1392.30 | PARTIAL | 0.50 | 3.92% |
| BUY | retest1 | 2023-08-23 00:00:00 | 1339.80 | 2023-09-06 00:00:00 | 1353.00 | STOP_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2023-12-19 00:00:00 | 1372.40 | 2023-12-20 00:00:00 | 1336.78 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest1 | 2024-01-03 00:00:00 | 1397.70 | 2024-01-11 00:00:00 | 1451.73 | PARTIAL | 0.50 | 3.87% |
| BUY | retest1 | 2024-01-03 00:00:00 | 1397.70 | 2024-01-17 00:00:00 | 1431.35 | STOP_HIT | 0.50 | 2.41% |
| BUY | retest1 | 2024-04-02 00:00:00 | 1544.15 | 2024-04-15 00:00:00 | 1484.07 | STOP_HIT | 1.00 | -3.89% |

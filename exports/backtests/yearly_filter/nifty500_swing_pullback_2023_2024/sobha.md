# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1385.70
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
- **Avg / median % per leg:** 13.12% / 11.95%
- **Sum % (uncompounded):** 104.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 13.12% | 105.0% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 13.12% | 105.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 13.12% | 105.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 686.83 | 571.89 | 641.30 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=28.00 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 644.83 | 573.68 | 644.99 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 00:00:00 | 742.51 | 578.83 | 665.42 | Stage2 pullback-breakout RSI=69 vol=3.5x ATR=32.81 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 693.29 | 592.40 | 706.41 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-11-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 00:00:00 | 807.14 | 607.53 | 728.28 | Stage2 pullback-breakout RSI=70 vol=4.9x ATR=33.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 873.31 | 618.21 | 768.07 | T1 booked 50% @ 873.31 |
| Target hit | 2024-02-12 00:00:00 | 1312.59 | 859.74 | 1339.87 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 00:00:00 | 1440.54 | 893.96 | 1355.65 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=86.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 00:00:00 | 1612.69 | 917.60 | 1404.25 | T1 booked 50% @ 1612.69 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 1440.54 | 959.57 | 1471.24 | SL hit (bars_held=11) |

### Cycle 5 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 1364.66 | 991.88 | 1355.26 | Stage2 pullback-breakout RSI=50 vol=3.0x ATR=94.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 1554.51 | 1023.29 | 1410.55 | T1 booked 50% @ 1554.51 |
| Target hit | 2024-05-10 00:00:00 | 1651.70 | 1158.62 | 1677.79 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-29 00:00:00 | 686.83 | 2023-10-04 00:00:00 | 644.83 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest1 | 2023-10-10 00:00:00 | 742.51 | 2023-10-23 00:00:00 | 693.29 | STOP_HIT | 1.00 | -6.63% |
| BUY | retest1 | 2023-11-09 00:00:00 | 807.14 | 2023-11-16 00:00:00 | 873.31 | PARTIAL | 0.50 | 8.20% |
| BUY | retest1 | 2023-11-09 00:00:00 | 807.14 | 2024-02-12 00:00:00 | 1312.59 | TARGET_HIT | 0.50 | 62.62% |
| BUY | retest1 | 2024-02-21 00:00:00 | 1440.54 | 2024-02-27 00:00:00 | 1612.69 | PARTIAL | 0.50 | 11.95% |
| BUY | retest1 | 2024-02-21 00:00:00 | 1440.54 | 2024-03-06 00:00:00 | 1440.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 00:00:00 | 1364.66 | 2024-04-03 00:00:00 | 1554.51 | PARTIAL | 0.50 | 13.91% |
| BUY | retest1 | 2024-03-21 00:00:00 | 1364.66 | 2024-05-10 00:00:00 | 1651.70 | TARGET_HIT | 0.50 | 21.03% |

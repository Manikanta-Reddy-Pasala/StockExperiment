# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1304.50
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 2.02% / 3.52%
- **Sum % (uncompounded):** 18.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 4 | 3 | 2.02% | 18.2% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 4 | 3 | 2.02% | 18.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 2 | 4 | 3 | 2.02% | 18.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 00:00:00 | 1011.25 | 795.12 | 962.21 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=34.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 00:00:00 | 1079.76 | 802.81 | 986.57 | T1 booked 50% @ 1079.76 |
| Target hit | 2023-08-08 00:00:00 | 1057.85 | 851.49 | 1068.19 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 1168.55 | 870.57 | 1076.04 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=50.97 |
| Stop hit — per-position SL triggered | 2023-08-24 00:00:00 | 1092.09 | 875.13 | 1080.27 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 00:00:00 | 1111.35 | 931.51 | 1061.63 | Stage2 pullback-breakout RSI=63 vol=2.8x ATR=33.58 |
| Stop hit — per-position SL triggered | 2023-10-31 00:00:00 | 1060.98 | 940.63 | 1083.07 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 1113.40 | 953.55 | 1054.65 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=41.29 |
| Stop hit — per-position SL triggered | 2023-12-05 00:00:00 | 1122.45 | 967.67 | 1084.56 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 1193.35 | 990.37 | 1120.11 | Stage2 pullback-breakout RSI=66 vol=3.1x ATR=42.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 00:00:00 | 1278.95 | 1006.41 | 1176.76 | T1 booked 50% @ 1278.95 |
| Target hit | 2024-01-23 00:00:00 | 1235.40 | 1034.37 | 1238.57 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-05-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 00:00:00 | 1389.80 | 1150.13 | 1330.23 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=44.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 00:00:00 | 1478.29 | 1153.25 | 1342.97 | T1 booked 50% @ 1478.29 |
| Stop hit — per-position SL triggered | 2024-05-08 00:00:00 | 1389.80 | 1162.25 | 1371.04 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-07 00:00:00 | 1011.25 | 2023-07-12 00:00:00 | 1079.76 | PARTIAL | 0.50 | 6.78% |
| BUY | retest1 | 2023-07-07 00:00:00 | 1011.25 | 2023-08-08 00:00:00 | 1057.85 | TARGET_HIT | 0.50 | 4.61% |
| BUY | retest1 | 2023-08-22 00:00:00 | 1168.55 | 2023-08-24 00:00:00 | 1092.09 | STOP_HIT | 1.00 | -6.54% |
| BUY | retest1 | 2023-10-23 00:00:00 | 1111.35 | 2023-10-31 00:00:00 | 1060.98 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest1 | 2023-11-20 00:00:00 | 1113.40 | 2023-12-05 00:00:00 | 1122.45 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest1 | 2023-12-28 00:00:00 | 1193.35 | 2024-01-08 00:00:00 | 1278.95 | PARTIAL | 0.50 | 7.17% |
| BUY | retest1 | 2023-12-28 00:00:00 | 1193.35 | 2024-01-23 00:00:00 | 1235.40 | TARGET_HIT | 0.50 | 3.52% |
| BUY | retest1 | 2024-05-02 00:00:00 | 1389.80 | 2024-05-03 00:00:00 | 1478.29 | PARTIAL | 0.50 | 6.37% |
| BUY | retest1 | 2024-05-02 00:00:00 | 1389.80 | 2024-05-08 00:00:00 | 1389.80 | STOP_HIT | 0.50 | 0.00% |

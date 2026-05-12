# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1631.80
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
- **Avg / median % per leg:** 4.00% / 5.07%
- **Sum % (uncompounded):** 23.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 1 | 3 | 4.00% | 24.0% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 1 | 3 | 4.00% | 24.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 2 | 1 | 3 | 4.00% | 24.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 1121.85 | 951.36 | 1085.04 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=26.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 1175.11 | 961.39 | 1103.62 | T1 booked 50% @ 1175.11 |
| Target hit | 2023-10-18 00:00:00 | 1218.45 | 1019.57 | 1218.93 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 00:00:00 | 1166.65 | 1055.81 | 1139.50 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=29.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 00:00:00 | 1225.75 | 1061.10 | 1157.01 | T1 booked 50% @ 1225.75 |
| Target hit | 2024-01-10 00:00:00 | 1216.25 | 1091.04 | 1227.54 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 1299.45 | 1097.83 | 1240.51 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=33.00 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 1249.94 | 1101.64 | 1249.48 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 1193.30 | 1118.91 | 1157.05 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=30.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 00:00:00 | 1254.50 | 1120.73 | 1170.81 | T1 booked 50% @ 1254.50 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 1121.85 | 2023-09-08 00:00:00 | 1175.11 | PARTIAL | 0.50 | 4.75% |
| BUY | retest1 | 2023-08-31 00:00:00 | 1121.85 | 2023-10-18 00:00:00 | 1218.45 | TARGET_HIT | 0.50 | 8.61% |
| BUY | retest1 | 2023-12-08 00:00:00 | 1166.65 | 2023-12-14 00:00:00 | 1225.75 | PARTIAL | 0.50 | 5.07% |
| BUY | retest1 | 2023-12-08 00:00:00 | 1166.65 | 2024-01-10 00:00:00 | 1216.25 | TARGET_HIT | 0.50 | 4.25% |
| BUY | retest1 | 2024-01-16 00:00:00 | 1299.45 | 2024-01-18 00:00:00 | 1249.94 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest1 | 2024-04-30 00:00:00 | 1193.30 | 2024-05-02 00:00:00 | 1254.50 | PARTIAL | 0.50 | 5.13% |

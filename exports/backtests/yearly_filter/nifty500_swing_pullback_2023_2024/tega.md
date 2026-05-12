# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1639.90
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
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 0 / 7 / 4
- **Avg / median % per leg:** 1.84% / 2.23%
- **Sum % (uncompounded):** 20.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 0 | 7 | 4 | 1.84% | 20.3% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 0 | 7 | 4 | 1.84% | 20.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 0 | 7 | 4 | 1.84% | 20.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 995.40 | 797.19 | 968.65 | Stage2 pullback-breakout RSI=54 vol=2.1x ATR=44.15 |
| Stop hit — per-position SL triggered | 2023-09-14 00:00:00 | 947.45 | 813.50 | 967.01 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 00:00:00 | 933.70 | 832.22 | 902.63 | Stage2 pullback-breakout RSI=55 vol=4.2x ATR=36.17 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 879.45 | 834.88 | 907.20 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 974.15 | 839.95 | 911.63 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=38.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 00:00:00 | 1050.49 | 853.65 | 972.50 | T1 booked 50% @ 1050.49 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 1001.70 | 864.30 | 1000.31 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 1038.50 | 884.91 | 1004.63 | Stage2 pullback-breakout RSI=60 vol=4.5x ATR=34.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 00:00:00 | 1107.05 | 889.79 | 1016.13 | T1 booked 50% @ 1107.05 |
| Stop hit — per-position SL triggered | 2023-12-21 00:00:00 | 1038.50 | 891.62 | 1021.66 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 1186.00 | 956.39 | 1137.04 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=44.73 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 1118.91 | 966.06 | 1142.91 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2024-02-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 00:00:00 | 1239.80 | 973.17 | 1159.68 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=55.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 00:00:00 | 1351.33 | 976.55 | 1174.30 | T1 booked 50% @ 1351.33 |
| Stop hit — per-position SL triggered | 2024-02-26 00:00:00 | 1239.80 | 996.92 | 1224.38 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 1421.15 | 1066.99 | 1303.25 | Stage2 pullback-breakout RSI=70 vol=2.3x ATR=57.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 00:00:00 | 1535.62 | 1086.47 | 1368.88 | T1 booked 50% @ 1535.62 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 1452.85 | 1119.27 | 1447.39 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 995.40 | 2023-09-14 00:00:00 | 947.45 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest1 | 2023-10-18 00:00:00 | 933.70 | 2023-10-23 00:00:00 | 879.45 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest1 | 2023-11-02 00:00:00 | 974.15 | 2023-11-13 00:00:00 | 1050.49 | PARTIAL | 0.50 | 7.84% |
| BUY | retest1 | 2023-11-02 00:00:00 | 974.15 | 2023-11-22 00:00:00 | 1001.70 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2023-12-15 00:00:00 | 1038.50 | 2023-12-20 00:00:00 | 1107.05 | PARTIAL | 0.50 | 6.60% |
| BUY | retest1 | 2023-12-15 00:00:00 | 1038.50 | 2023-12-21 00:00:00 | 1038.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 00:00:00 | 1186.00 | 2024-02-09 00:00:00 | 1118.91 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest1 | 2024-02-14 00:00:00 | 1239.80 | 2024-02-15 00:00:00 | 1351.33 | PARTIAL | 0.50 | 9.00% |
| BUY | retest1 | 2024-02-14 00:00:00 | 1239.80 | 2024-02-26 00:00:00 | 1239.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-16 00:00:00 | 1421.15 | 2024-04-24 00:00:00 | 1535.62 | PARTIAL | 0.50 | 8.05% |
| BUY | retest1 | 2024-04-16 00:00:00 | 1421.15 | 2024-05-07 00:00:00 | 1452.85 | STOP_HIT | 0.50 | 2.23% |

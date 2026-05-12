# Tata Consumer Products Ltd. (TATACONSUM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1176.20
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -0.99% / -2.58%
- **Sum % (uncompounded):** -6.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.99% | -6.9% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.99% | -6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.99% | -6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 00:00:00 | 868.11 | 795.42 | 837.11 | Stage2 pullback-breakout RSI=65 vol=3.4x ATR=14.90 |
| Stop hit — per-position SL triggered | 2023-09-07 00:00:00 | 845.76 | 795.95 | 838.16 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 00:00:00 | 878.38 | 798.34 | 844.76 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=16.43 |
| Stop hit — per-position SL triggered | 2023-09-21 00:00:00 | 853.74 | 801.62 | 852.73 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 00:00:00 | 889.54 | 803.14 | 857.47 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=17.30 |
| Stop hit — per-position SL triggered | 2023-09-28 00:00:00 | 863.59 | 805.29 | 862.10 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2023-10-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 00:00:00 | 902.14 | 811.51 | 869.58 | Stage2 pullback-breakout RSI=67 vol=3.5x ATR=16.67 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 877.14 | 814.62 | 876.38 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 929.20 | 836.54 | 911.18 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=14.60 |
| Stop hit — per-position SL triggered | 2023-12-14 00:00:00 | 938.78 | 846.25 | 928.81 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2023-12-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 00:00:00 | 954.29 | 849.17 | 933.13 | Stage2 pullback-breakout RSI=67 vol=3.3x ATR=14.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 00:00:00 | 984.03 | 850.32 | 936.10 | T1 booked 50% @ 984.03 |
| Stop hit — per-position SL triggered | 2023-12-21 00:00:00 | 954.29 | 851.48 | 939.07 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-06 00:00:00 | 868.11 | 2023-09-07 00:00:00 | 845.76 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest1 | 2023-09-13 00:00:00 | 878.38 | 2023-09-21 00:00:00 | 853.74 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest1 | 2023-09-25 00:00:00 | 889.54 | 2023-09-28 00:00:00 | 863.59 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2023-10-13 00:00:00 | 902.14 | 2023-10-19 00:00:00 | 877.14 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2023-11-30 00:00:00 | 929.20 | 2023-12-14 00:00:00 | 938.78 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest1 | 2023-12-19 00:00:00 | 954.29 | 2023-12-20 00:00:00 | 984.03 | PARTIAL | 0.50 | 3.12% |
| BUY | retest1 | 2023-12-19 00:00:00 | 954.29 | 2023-12-21 00:00:00 | 954.29 | STOP_HIT | 0.50 | 0.00% |

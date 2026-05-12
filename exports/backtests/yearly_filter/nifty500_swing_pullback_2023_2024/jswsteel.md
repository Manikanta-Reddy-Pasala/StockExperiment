# JSW Steel Ltd. (JSWSTEEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1268.60
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
- **Avg / median % per leg:** -1.37% / -3.06%
- **Sum % (uncompounded):** -10.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.37% | -10.9% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.37% | -10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.37% | -10.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 783.40 | 711.49 | 750.19 | Stage2 pullback-breakout RSI=69 vol=2.6x ATR=16.69 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 802.15 | 719.61 | 780.51 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 00:00:00 | 831.50 | 734.93 | 804.86 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=17.80 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 804.81 | 736.41 | 805.70 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 806.40 | 742.75 | 792.86 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=16.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 00:00:00 | 838.49 | 747.75 | 805.06 | T1 booked 50% @ 838.49 |
| Stop hit — per-position SL triggered | 2023-09-15 00:00:00 | 806.40 | 749.63 | 806.70 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 836.60 | 784.58 | 820.22 | Stage2 pullback-breakout RSI=56 vol=2.0x ATR=20.14 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 806.39 | 785.22 | 819.58 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2024-02-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 00:00:00 | 828.25 | 787.71 | 819.05 | Stage2 pullback-breakout RSI=55 vol=3.0x ATR=19.39 |
| Stop hit — per-position SL triggered | 2024-02-28 00:00:00 | 799.16 | 789.04 | 817.04 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 871.95 | 794.33 | 819.54 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=23.21 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 845.25 | 801.28 | 848.93 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 905.80 | 805.15 | 858.63 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=23.67 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 870.29 | 809.08 | 869.13 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 783.40 | 2023-07-13 00:00:00 | 802.15 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest1 | 2023-08-10 00:00:00 | 831.50 | 2023-08-14 00:00:00 | 804.81 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest1 | 2023-09-01 00:00:00 | 806.40 | 2023-09-12 00:00:00 | 838.49 | PARTIAL | 0.50 | 3.98% |
| BUY | retest1 | 2023-09-01 00:00:00 | 806.40 | 2023-09-15 00:00:00 | 806.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-07 00:00:00 | 836.60 | 2024-02-09 00:00:00 | 806.39 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest1 | 2024-02-21 00:00:00 | 828.25 | 2024-02-28 00:00:00 | 799.16 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-04-01 00:00:00 | 871.95 | 2024-04-16 00:00:00 | 845.25 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest1 | 2024-04-25 00:00:00 | 905.80 | 2024-05-03 00:00:00 | 870.29 | STOP_HIT | 1.00 | -3.92% |

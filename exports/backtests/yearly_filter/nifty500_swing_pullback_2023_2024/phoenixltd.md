# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1803.20
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
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 3
- **Avg / median % per leg:** 1.72% / 0.00%
- **Sum % (uncompounded):** 13.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 1.72% | 13.8% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 1.72% | 13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 5 | 3 | 1.72% | 13.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 00:00:00 | 807.28 | 714.62 | 778.61 | Stage2 pullback-breakout RSI=65 vol=3.3x ATR=21.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 00:00:00 | 851.25 | 717.99 | 786.46 | T1 booked 50% @ 851.25 |
| Stop hit — per-position SL triggered | 2023-07-20 00:00:00 | 807.28 | 723.47 | 803.14 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-07-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 00:00:00 | 849.58 | 728.57 | 813.39 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=25.50 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 831.03 | 740.12 | 835.22 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 962.15 | 795.39 | 913.09 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=30.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 00:00:00 | 1022.28 | 799.22 | 926.95 | T1 booked 50% @ 1022.28 |
| Stop hit — per-position SL triggered | 2023-10-16 00:00:00 | 962.15 | 800.84 | 930.33 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-01-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 00:00:00 | 1244.43 | 925.18 | 1143.12 | Stage2 pullback-breakout RSI=69 vol=3.5x ATR=47.40 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 1173.33 | 950.29 | 1199.95 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 1239.45 | 977.09 | 1202.62 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=61.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 00:00:00 | 1363.00 | 984.02 | 1225.15 | T1 booked 50% @ 1363.00 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 1239.45 | 1002.33 | 1276.02 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-07 00:00:00 | 807.28 | 2023-07-13 00:00:00 | 851.25 | PARTIAL | 0.50 | 5.45% |
| BUY | retest1 | 2023-07-07 00:00:00 | 807.28 | 2023-07-20 00:00:00 | 807.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-27 00:00:00 | 849.58 | 2023-08-10 00:00:00 | 831.03 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest1 | 2023-10-11 00:00:00 | 962.15 | 2023-10-13 00:00:00 | 1022.28 | PARTIAL | 0.50 | 6.25% |
| BUY | retest1 | 2023-10-11 00:00:00 | 962.15 | 2023-10-16 00:00:00 | 962.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-05 00:00:00 | 1244.43 | 2024-01-17 00:00:00 | 1173.33 | STOP_HIT | 1.00 | -5.71% |
| BUY | retest1 | 2024-02-02 00:00:00 | 1239.45 | 2024-02-06 00:00:00 | 1363.00 | PARTIAL | 0.50 | 9.97% |
| BUY | retest1 | 2024-02-02 00:00:00 | 1239.45 | 2024-02-13 00:00:00 | 1239.45 | STOP_HIT | 0.50 | 0.00% |

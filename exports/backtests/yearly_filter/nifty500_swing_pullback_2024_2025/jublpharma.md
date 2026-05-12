# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1014.40
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 10.69% / 8.01%
- **Sum % (uncompounded):** 42.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 10.69% | 42.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 10.69% | 42.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 10.69% | 42.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 00:00:00 | 749.55 | 613.67 | 733.08 | Stage2 pullback-breakout RSI=57 vol=10.1x ATR=30.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 00:00:00 | 809.56 | 625.23 | 746.68 | T1 booked 50% @ 809.56 |
| Target hit | 2024-10-03 00:00:00 | 1118.50 | 761.96 | 1120.46 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 00:00:00 | 1222.00 | 791.79 | 1140.19 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=54.44 |
| Stop hit — per-position SL triggered | 2024-10-18 00:00:00 | 1140.34 | 803.66 | 1153.68 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 1211.75 | 831.70 | 1140.29 | Stage2 pullback-breakout RSI=60 vol=2.2x ATR=62.95 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1117.32 | 865.77 | 1187.79 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-19 00:00:00 | 749.55 | 2024-08-01 00:00:00 | 809.56 | PARTIAL | 0.50 | 8.01% |
| BUY | retest1 | 2024-07-19 00:00:00 | 749.55 | 2024-10-03 00:00:00 | 1118.50 | TARGET_HIT | 0.50 | 49.22% |
| BUY | retest1 | 2024-10-15 00:00:00 | 1222.00 | 2024-10-18 00:00:00 | 1140.34 | STOP_HIT | 1.00 | -6.68% |
| BUY | retest1 | 2024-10-31 00:00:00 | 1211.75 | 2024-11-13 00:00:00 | 1117.32 | STOP_HIT | 1.00 | -7.79% |

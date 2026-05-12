# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1649.80
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 3.58% / 5.67%
- **Sum % (uncompounded):** 17.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 3.58% | 17.9% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 3.58% | 17.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 3.58% | 17.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 00:00:00 | 1155.65 | 1093.71 | 1119.49 | Stage2 pullback-breakout RSI=59 vol=4.5x ATR=26.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 00:00:00 | 1209.47 | 1096.42 | 1136.95 | T1 booked 50% @ 1209.47 |
| Target hit | 2023-10-23 00:00:00 | 1221.20 | 1124.34 | 1234.30 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 1231.58 | 1129.08 | 1215.46 | Stage2 pullback-breakout RSI=54 vol=2.8x ATR=35.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 00:00:00 | 1302.97 | 1130.75 | 1223.26 | T1 booked 50% @ 1302.97 |
| Target hit | 2023-11-23 00:00:00 | 1304.90 | 1157.70 | 1308.87 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 1258.70 | 1199.51 | 1238.71 | Stage2 pullback-breakout RSI=53 vol=1.8x ATR=35.26 |
| Stop hit — per-position SL triggered | 2024-02-05 00:00:00 | 1205.81 | 1200.38 | 1235.86 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-14 00:00:00 | 1155.65 | 2023-09-20 00:00:00 | 1209.47 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2023-09-14 00:00:00 | 1155.65 | 2023-10-23 00:00:00 | 1221.20 | TARGET_HIT | 0.50 | 5.67% |
| BUY | retest1 | 2023-11-02 00:00:00 | 1231.58 | 2023-11-03 00:00:00 | 1302.97 | PARTIAL | 0.50 | 5.80% |
| BUY | retest1 | 2023-11-02 00:00:00 | 1231.58 | 2023-11-23 00:00:00 | 1304.90 | TARGET_HIT | 0.50 | 5.95% |
| BUY | retest1 | 2024-01-31 00:00:00 | 1258.70 | 2024-02-05 00:00:00 | 1205.81 | STOP_HIT | 1.00 | -4.20% |

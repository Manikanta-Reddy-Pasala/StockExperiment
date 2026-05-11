# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1674.10
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 1.92% / 5.02%
- **Sum % (uncompounded):** 7.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 1.92% | 7.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 1.92% | 7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 1.92% | 7.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 05:30:00 | 1449.85 | 1234.92 | 1407.95 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=39.53 |
| Stop hit — per-position SL triggered | 2024-07-22 05:30:00 | 1390.56 | 1238.49 | 1409.12 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-08-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 05:30:00 | 1443.80 | 1269.07 | 1379.66 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=45.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 05:30:00 | 1535.28 | 1283.46 | 1433.44 | T1 booked 50% @ 1535.28 |
| Target hit | 2024-10-03 05:30:00 | 1516.25 | 1334.63 | 1563.63 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-02-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 05:30:00 | 1438.65 | 1316.39 | 1358.07 | Stage2 pullback-breakout RSI=64 vol=2.5x ATR=49.62 |
| Stop hit — per-position SL triggered | 2025-03-13 05:30:00 | 1444.70 | 1328.16 | 1411.17 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-18 05:30:00 | 1449.85 | 2024-07-22 05:30:00 | 1390.56 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest1 | 2024-08-27 05:30:00 | 1443.80 | 2024-09-05 05:30:00 | 1535.28 | PARTIAL | 0.50 | 6.34% |
| BUY | retest1 | 2024-08-27 05:30:00 | 1443.80 | 2024-10-03 05:30:00 | 1516.25 | TARGET_HIT | 0.50 | 5.02% |
| BUY | retest1 | 2025-02-27 05:30:00 | 1438.65 | 2025-03-13 05:30:00 | 1444.70 | STOP_HIT | 1.00 | 0.42% |

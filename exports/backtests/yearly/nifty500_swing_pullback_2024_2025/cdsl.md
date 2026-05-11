# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1259.30
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 3
- **Avg / median % per leg:** 4.93% / 7.07%
- **Sum % (uncompounded):** 29.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 2 | 3 | 4.93% | 29.6% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 2 | 3 | 4.93% | 29.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 2 | 3 | 4.93% | 29.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 05:30:00 | 1282.63 | 977.61 | 1194.45 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=49.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 05:30:00 | 1382.58 | 990.52 | 1232.81 | T1 booked 50% @ 1382.58 |
| Target hit | 2024-09-09 05:30:00 | 1372.60 | 1058.40 | 1395.66 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 05:30:00 | 1444.70 | 1074.28 | 1392.56 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=51.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 05:30:00 | 1546.87 | 1086.59 | 1418.43 | T1 booked 50% @ 1546.87 |
| Stop hit — per-position SL triggered | 2024-09-30 05:30:00 | 1444.70 | 1113.75 | 1450.76 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2024-10-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 05:30:00 | 1471.85 | 1130.69 | 1430.18 | Stage2 pullback-breakout RSI=57 vol=2.6x ATR=56.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 05:30:00 | 1585.28 | 1142.02 | 1452.68 | T1 booked 50% @ 1585.28 |
| Stop hit — per-position SL triggered | 2024-10-22 05:30:00 | 1471.85 | 1165.68 | 1493.58 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-09 05:30:00 | 1282.63 | 2024-08-16 05:30:00 | 1382.58 | PARTIAL | 0.50 | 7.79% |
| BUY | retest1 | 2024-08-09 05:30:00 | 1282.63 | 2024-09-09 05:30:00 | 1372.60 | TARGET_HIT | 0.50 | 7.01% |
| BUY | retest1 | 2024-09-16 05:30:00 | 1444.70 | 2024-09-19 05:30:00 | 1546.87 | PARTIAL | 0.50 | 7.07% |
| BUY | retest1 | 2024-09-16 05:30:00 | 1444.70 | 2024-09-30 05:30:00 | 1444.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 05:30:00 | 1471.85 | 2024-10-14 05:30:00 | 1585.28 | PARTIAL | 0.50 | 7.71% |
| BUY | retest1 | 2024-10-09 05:30:00 | 1471.85 | 2024-10-22 05:30:00 | 1471.85 | STOP_HIT | 0.50 | 0.00% |

# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2023-09-26 00:00:00 → 2026-05-11 00:00:00 (649 bars)
- **Last close:** 1613.70
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
- **Avg / median % per leg:** 5.41% / 8.25%
- **Sum % (uncompounded):** 21.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 5.41% | 21.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 5.41% | 21.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 5.41% | 21.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 00:00:00 | 1052.29 | 844.09 | 983.23 | Stage2 pullback-breakout RSI=62 vol=3.7x ATR=43.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 00:00:00 | 1139.08 | 849.66 | 1009.12 | T1 booked 50% @ 1139.08 |
| Target hit | 2024-09-19 00:00:00 | 1314.43 | 989.32 | 1316.87 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 1467.34 | 1044.54 | 1327.20 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=59.11 |
| Stop hit — per-position SL triggered | 2024-10-18 00:00:00 | 1378.68 | 1052.20 | 1346.05 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-12-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 00:00:00 | 1472.25 | 1142.41 | 1361.78 | Stage2 pullback-breakout RSI=66 vol=3.5x ATR=53.88 |
| Stop hit — per-position SL triggered | 2024-12-12 00:00:00 | 1391.44 | 1147.99 | 1372.80 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-24 00:00:00 | 1052.29 | 2024-07-26 00:00:00 | 1139.08 | PARTIAL | 0.50 | 8.25% |
| BUY | retest1 | 2024-07-24 00:00:00 | 1052.29 | 2024-09-19 00:00:00 | 1314.43 | TARGET_HIT | 0.50 | 24.91% |
| BUY | retest1 | 2024-10-16 00:00:00 | 1467.34 | 2024-10-18 00:00:00 | 1378.68 | STOP_HIT | 1.00 | -6.04% |
| BUY | retest1 | 2024-12-10 00:00:00 | 1472.25 | 2024-12-12 00:00:00 | 1391.44 | STOP_HIT | 1.00 | -5.49% |

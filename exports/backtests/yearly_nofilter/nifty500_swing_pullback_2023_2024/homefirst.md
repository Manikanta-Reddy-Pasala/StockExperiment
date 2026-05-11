# Home First Finance Company India Ltd. (HOMEFIRST)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1177.40
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.07% / -3.52%
- **Sum % (uncompounded):** 0.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.07% | 0.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.07% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.07% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 00:00:00 | 867.55 | 781.57 | 838.71 | Stage2 pullback-breakout RSI=62 vol=4.3x ATR=28.35 |
| Stop hit — per-position SL triggered | 2023-08-29 00:00:00 | 825.02 | 787.01 | 844.53 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-10-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 00:00:00 | 846.15 | 798.85 | 835.74 | Stage2 pullback-breakout RSI=53 vol=2.2x ATR=25.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 00:00:00 | 896.63 | 803.13 | 850.79 | T1 booked 50% @ 896.63 |
| Target hit | 2023-11-22 00:00:00 | 914.50 | 832.28 | 926.80 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 1025.55 | 840.64 | 930.11 | Stage2 pullback-breakout RSI=69 vol=3.3x ATR=37.80 |
| Stop hit — per-position SL triggered | 2023-12-21 00:00:00 | 989.40 | 856.83 | 979.46 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 1009.15 | 875.29 | 966.31 | Stage2 pullback-breakout RSI=62 vol=5.2x ATR=35.34 |
| Stop hit — per-position SL triggered | 2024-01-25 00:00:00 | 956.14 | 879.56 | 971.99 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-17 00:00:00 | 867.55 | 2023-08-29 00:00:00 | 825.02 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest1 | 2023-10-05 00:00:00 | 846.15 | 2023-10-16 00:00:00 | 896.63 | PARTIAL | 0.50 | 5.97% |
| BUY | retest1 | 2023-10-05 00:00:00 | 846.15 | 2023-11-22 00:00:00 | 914.50 | TARGET_HIT | 0.50 | 8.08% |
| BUY | retest1 | 2023-12-07 00:00:00 | 1025.55 | 2023-12-21 00:00:00 | 989.40 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest1 | 2024-01-19 00:00:00 | 1009.15 | 2024-01-25 00:00:00 | 956.14 | STOP_HIT | 1.00 | -5.25% |

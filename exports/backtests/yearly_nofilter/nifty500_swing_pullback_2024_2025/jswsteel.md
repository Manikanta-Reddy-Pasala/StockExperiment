# JSW Steel Ltd. (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.52% / 3.62%
- **Sum % (uncompounded):** 9.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.52% | 9.1% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.52% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.52% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 00:00:00 | 928.25 | 862.09 | 911.29 | Stage2 pullback-breakout RSI=55 vol=1.6x ATR=21.48 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 896.03 | 863.18 | 912.17 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-08-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 00:00:00 | 917.35 | 864.99 | 905.14 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=25.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 00:00:00 | 968.29 | 870.01 | 918.52 | T1 booked 50% @ 968.29 |
| Stop hit — per-position SL triggered | 2024-08-27 00:00:00 | 944.00 | 870.75 | 920.95 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 00:00:00 | 954.80 | 878.27 | 932.37 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=21.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 00:00:00 | 997.71 | 885.23 | 953.54 | T1 booked 50% @ 997.71 |
| Target hit | 2024-10-16 00:00:00 | 989.35 | 902.62 | 996.12 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-11-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 00:00:00 | 999.50 | 911.07 | 974.45 | Stage2 pullback-breakout RSI=57 vol=3.4x ATR=26.48 |
| Stop hit — per-position SL triggered | 2024-11-12 00:00:00 | 959.78 | 914.73 | 978.18 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-31 00:00:00 | 928.25 | 2024-08-02 00:00:00 | 896.03 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest1 | 2024-08-12 00:00:00 | 917.35 | 2024-08-26 00:00:00 | 968.29 | PARTIAL | 0.50 | 5.55% |
| BUY | retest1 | 2024-08-12 00:00:00 | 917.35 | 2024-08-27 00:00:00 | 944.00 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest1 | 2024-09-12 00:00:00 | 954.80 | 2024-09-24 00:00:00 | 997.71 | PARTIAL | 0.50 | 4.49% |
| BUY | retest1 | 2024-09-12 00:00:00 | 954.80 | 2024-10-16 00:00:00 | 989.35 | TARGET_HIT | 0.50 | 3.62% |
| BUY | retest1 | 2024-11-05 00:00:00 | 999.50 | 2024-11-12 00:00:00 | 959.78 | STOP_HIT | 1.00 | -3.97% |

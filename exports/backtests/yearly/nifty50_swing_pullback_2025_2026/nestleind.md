# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1482.40
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 1.82% / 2.98%
- **Sum % (uncompounded):** 9.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 1.82% | 9.1% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 1.82% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 1.82% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 05:30:00 | 1221.40 | 1176.66 | 1184.84 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=22.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 05:30:00 | 1265.70 | 1177.65 | 1193.58 | T1 booked 50% @ 1265.70 |
| Target hit | 2025-11-25 05:30:00 | 1263.20 | 1199.60 | 1266.29 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-12-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 05:30:00 | 1272.60 | 1208.56 | 1248.72 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=18.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 05:30:00 | 1310.51 | 1213.23 | 1267.47 | T1 booked 50% @ 1310.51 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 1299.10 | 1217.03 | 1281.27 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 1332.40 | 1227.73 | 1297.93 | Stage2 pullback-breakout RSI=62 vol=4.2x ATR=26.89 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 1292.06 | 1228.27 | 1296.46 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-15 05:30:00 | 1221.40 | 2025-10-16 05:30:00 | 1265.70 | PARTIAL | 0.50 | 3.63% |
| BUY | retest1 | 2025-10-15 05:30:00 | 1221.40 | 2025-11-25 05:30:00 | 1263.20 | TARGET_HIT | 0.50 | 3.42% |
| BUY | retest1 | 2025-12-26 05:30:00 | 1272.60 | 2026-01-05 05:30:00 | 1310.51 | PARTIAL | 0.50 | 2.98% |
| BUY | retest1 | 2025-12-26 05:30:00 | 1272.60 | 2026-01-09 05:30:00 | 1299.10 | STOP_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2026-01-30 05:30:00 | 1332.40 | 2026-02-01 05:30:00 | 1292.06 | STOP_HIT | 1.00 | -3.03% |

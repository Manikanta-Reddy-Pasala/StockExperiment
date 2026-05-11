# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1104.10
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
- **Winners / losers:** 5 / 0
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.20% / 2.70%
- **Sum % (uncompounded):** 15.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 1 | 2 | 2 | 3.20% | 16.0% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 1 | 2 | 2 | 3.20% | 16.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 5 | 100.0% | 1 | 2 | 2 | 3.20% | 16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 05:30:00 | 1321.95 | 1157.21 | 1264.47 | Stage2 pullback-breakout RSI=67 vol=3.3x ATR=34.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 05:30:00 | 1391.28 | 1166.34 | 1298.00 | T1 booked 50% @ 1391.28 |
| Stop hit — per-position SL triggered | 2023-07-27 05:30:00 | 1355.65 | 1181.13 | 1350.97 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 05:30:00 | 1313.10 | 1269.35 | 1280.33 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=32.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 05:30:00 | 1377.56 | 1272.87 | 1305.83 | T1 booked 50% @ 1377.56 |
| Target hit | 2023-12-20 05:30:00 | 1348.60 | 1284.65 | 1354.81 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 05:30:00 | 1371.95 | 1290.23 | 1338.42 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=35.73 |
| Stop hit — per-position SL triggered | 2024-01-20 05:30:00 | 1379.90 | 1299.61 | 1370.76 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 05:30:00 | 1321.95 | 2023-07-19 05:30:00 | 1391.28 | PARTIAL | 0.50 | 5.24% |
| BUY | retest1 | 2023-07-12 05:30:00 | 1321.95 | 2023-07-27 05:30:00 | 1355.65 | STOP_HIT | 0.50 | 2.55% |
| BUY | retest1 | 2023-11-24 05:30:00 | 1313.10 | 2023-12-04 05:30:00 | 1377.56 | PARTIAL | 0.50 | 4.91% |
| BUY | retest1 | 2023-11-24 05:30:00 | 1313.10 | 2023-12-20 05:30:00 | 1348.60 | TARGET_HIT | 0.50 | 2.70% |
| BUY | retest1 | 2024-01-08 05:30:00 | 1371.95 | 2024-01-20 05:30:00 | 1379.90 | STOP_HIT | 1.00 | 0.58% |

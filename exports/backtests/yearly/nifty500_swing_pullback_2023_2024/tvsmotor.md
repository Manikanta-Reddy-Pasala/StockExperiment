# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 3695.20
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 1.70% / 2.74%
- **Sum % (uncompounded):** 8.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 1.70% | 8.5% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 1.70% | 8.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 1.70% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 05:30:00 | 1384.00 | 1173.05 | 1333.94 | Stage2 pullback-breakout RSI=62 vol=4.3x ATR=33.74 |
| Stop hit — per-position SL triggered | 2023-08-07 05:30:00 | 1333.39 | 1189.46 | 1350.87 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-08-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 05:30:00 | 1415.55 | 1213.65 | 1357.48 | Stage2 pullback-breakout RSI=68 vol=2.2x ATR=29.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 05:30:00 | 1474.85 | 1225.32 | 1395.85 | T1 booked 50% @ 1474.85 |
| Stop hit — per-position SL triggered | 2023-09-13 05:30:00 | 1454.40 | 1237.49 | 1426.55 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-02-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 05:30:00 | 2138.75 | 1649.26 | 2042.98 | Stage2 pullback-breakout RSI=68 vol=2.7x ATR=55.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 05:30:00 | 2250.34 | 1695.53 | 2105.55 | T1 booked 50% @ 2250.34 |
| Stop hit — per-position SL triggered | 2024-03-13 05:30:00 | 2138.75 | 1737.77 | 2179.66 | SL hit (bars_held=18) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-25 05:30:00 | 1384.00 | 2023-08-07 05:30:00 | 1333.39 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest1 | 2023-08-30 05:30:00 | 1415.55 | 2023-09-06 05:30:00 | 1474.85 | PARTIAL | 0.50 | 4.19% |
| BUY | retest1 | 2023-08-30 05:30:00 | 1415.55 | 2023-09-13 05:30:00 | 1454.40 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest1 | 2024-02-16 05:30:00 | 2138.75 | 2024-03-01 05:30:00 | 2250.34 | PARTIAL | 0.50 | 5.22% |
| BUY | retest1 | 2024-02-16 05:30:00 | 2138.75 | 2024-03-13 05:30:00 | 2138.75 | STOP_HIT | 0.50 | 0.00% |

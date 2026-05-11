# Voltas Ltd. (VOLTAS)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1324.80
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
- **Avg / median % per leg:** 3.58% / 6.61%
- **Sum % (uncompounded):** 14.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.58% | 14.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.58% | 14.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.58% | 14.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 00:00:00 | 1582.55 | 1240.73 | 1488.64 | Stage2 pullback-breakout RSI=63 vol=8.2x ATR=52.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 00:00:00 | 1687.13 | 1265.92 | 1556.01 | T1 booked 50% @ 1687.13 |
| Target hit | 2024-09-30 00:00:00 | 1845.10 | 1401.02 | 1845.80 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 1881.30 | 1443.60 | 1822.60 | Stage2 pullback-breakout RSI=60 vol=3.1x ATR=49.52 |
| Stop hit — per-position SL triggered | 2024-10-21 00:00:00 | 1807.02 | 1455.25 | 1826.39 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 1766.95 | 1488.97 | 1751.69 | Stage2 pullback-breakout RSI=51 vol=2.0x ATR=58.31 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1679.49 | 1498.26 | 1742.08 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-12 00:00:00 | 1582.55 | 2024-08-22 00:00:00 | 1687.13 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2024-08-12 00:00:00 | 1582.55 | 2024-09-30 00:00:00 | 1845.10 | TARGET_HIT | 0.50 | 16.59% |
| BUY | retest1 | 2024-10-16 00:00:00 | 1881.30 | 2024-10-21 00:00:00 | 1807.02 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest1 | 2024-11-07 00:00:00 | 1766.95 | 2024-11-13 00:00:00 | 1679.49 | STOP_HIT | 1.00 | -4.95% |

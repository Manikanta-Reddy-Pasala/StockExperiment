# Eternal Ltd. (ETERNAL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 248.65
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
- **Avg / median % per leg:** 3.51% / 7.72%
- **Sum % (uncompounded):** 21.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.51% | 21.1% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.51% | 21.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.51% | 21.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 99.80 | 70.18 | 91.58 | Stage2 pullback-breakout RSI=70 vol=2.2x ATR=3.74 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 97.90 | 72.89 | 96.17 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 102.95 | 73.45 | 97.10 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=3.42 |
| Stop hit — per-position SL triggered | 2023-09-25 00:00:00 | 97.81 | 74.74 | 98.16 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 116.50 | 82.44 | 107.73 | Stage2 pullback-breakout RSI=66 vol=3.2x ATR=4.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 00:00:00 | 125.65 | 83.23 | 110.36 | T1 booked 50% @ 125.65 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 118.15 | 86.54 | 116.96 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 139.55 | 102.33 | 133.23 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 00:00:00 | 150.32 | 105.06 | 138.18 | T1 booked 50% @ 150.32 |
| Target hit | 2024-03-11 00:00:00 | 154.85 | 115.61 | 158.91 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-30 00:00:00 | 99.80 | 2023-09-13 00:00:00 | 97.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest1 | 2023-09-15 00:00:00 | 102.95 | 2023-09-25 00:00:00 | 97.81 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest1 | 2023-11-03 00:00:00 | 116.50 | 2023-11-07 00:00:00 | 125.65 | PARTIAL | 0.50 | 7.85% |
| BUY | retest1 | 2023-11-03 00:00:00 | 116.50 | 2023-11-20 00:00:00 | 118.15 | STOP_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2024-01-31 00:00:00 | 139.55 | 2024-02-09 00:00:00 | 150.32 | PARTIAL | 0.50 | 7.72% |
| BUY | retest1 | 2024-01-31 00:00:00 | 139.55 | 2024-03-11 00:00:00 | 154.85 | TARGET_HIT | 0.50 | 10.96% |

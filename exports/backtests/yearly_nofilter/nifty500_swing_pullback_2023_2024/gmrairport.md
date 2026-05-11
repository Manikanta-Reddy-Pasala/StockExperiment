# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 98.16
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 4
- **Avg / median % per leg:** 10.52% / 5.91%
- **Sum % (uncompounded):** 84.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 2 | 2 | 4 | 10.52% | 84.2% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 2 | 2 | 4 | 10.52% | 84.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 2 | 4 | 10.52% | 84.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 00:00:00 | 46.50 | 41.72 | 44.39 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 00:00:00 | 48.59 | 41.78 | 44.75 | T1 booked 50% @ 48.59 |
| Target hit | 2023-09-12 00:00:00 | 59.50 | 46.22 | 60.11 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 00:00:00 | 57.65 | 49.96 | 56.48 | Stage2 pullback-breakout RSI=55 vol=3.0x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 00:00:00 | 60.87 | 50.73 | 57.49 | T1 booked 50% @ 60.87 |
| Target hit | 2024-01-23 00:00:00 | 75.50 | 59.20 | 80.66 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 85.20 | 60.54 | 80.02 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 00:00:00 | 93.09 | 63.81 | 86.79 | T1 booked 50% @ 93.09 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 85.20 | 64.03 | 86.74 | SL hit (bars_held=13) |

### Cycle 4 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 85.15 | 70.32 | 82.41 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 90.18 | 70.53 | 83.20 | T1 booked 50% @ 90.18 |
| Stop hit — per-position SL triggered | 2024-04-30 00:00:00 | 85.15 | 70.84 | 83.75 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-24 00:00:00 | 46.50 | 2023-07-25 00:00:00 | 48.59 | PARTIAL | 0.50 | 4.50% |
| BUY | retest1 | 2023-07-24 00:00:00 | 46.50 | 2023-09-12 00:00:00 | 59.50 | TARGET_HIT | 0.50 | 27.96% |
| BUY | retest1 | 2023-11-10 00:00:00 | 57.65 | 2023-11-24 00:00:00 | 60.87 | PARTIAL | 0.50 | 5.58% |
| BUY | retest1 | 2023-11-10 00:00:00 | 57.65 | 2024-01-23 00:00:00 | 75.50 | TARGET_HIT | 0.50 | 30.96% |
| BUY | retest1 | 2024-02-02 00:00:00 | 85.20 | 2024-02-20 00:00:00 | 93.09 | PARTIAL | 0.50 | 9.27% |
| BUY | retest1 | 2024-02-02 00:00:00 | 85.20 | 2024-02-21 00:00:00 | 85.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-25 00:00:00 | 85.15 | 2024-04-26 00:00:00 | 90.18 | PARTIAL | 0.50 | 5.91% |
| BUY | retest1 | 2024-04-25 00:00:00 | 85.15 | 2024-04-30 00:00:00 | 85.15 | STOP_HIT | 0.50 | 0.00% |

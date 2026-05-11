# Trident Ltd. (TRIDENT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 25.97
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
- **Avg / median % per leg:** 2.91% / 5.39%
- **Sum % (uncompounded):** 17.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.91% | 17.4% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.91% | 17.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.91% | 17.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 00:00:00 | 36.80 | 34.98 | 35.57 | Stage2 pullback-breakout RSI=57 vol=3.0x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 39.12 | 35.09 | 36.12 | T1 booked 50% @ 39.12 |
| Stop hit — per-position SL triggered | 2023-11-21 00:00:00 | 37.30 | 35.18 | 36.54 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 00:00:00 | 37.85 | 35.36 | 36.64 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=0.95 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 36.43 | 35.46 | 36.72 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 37.40 | 35.53 | 36.54 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 00:00:00 | 39.42 | 35.63 | 37.21 | T1 booked 50% @ 39.42 |
| Target hit | 2024-01-23 00:00:00 | 42.95 | 36.81 | 43.12 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 41.15 | 38.62 | 39.07 | Stage2 pullback-breakout RSI=56 vol=2.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 38.39 | 38.73 | 39.61 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-07 00:00:00 | 36.80 | 2023-11-15 00:00:00 | 39.12 | PARTIAL | 0.50 | 6.31% |
| BUY | retest1 | 2023-11-07 00:00:00 | 36.80 | 2023-11-21 00:00:00 | 37.30 | STOP_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2023-12-11 00:00:00 | 37.85 | 2023-12-20 00:00:00 | 36.43 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2024-01-02 00:00:00 | 37.40 | 2024-01-05 00:00:00 | 39.42 | PARTIAL | 0.50 | 5.39% |
| BUY | retest1 | 2024-01-02 00:00:00 | 37.40 | 2024-01-23 00:00:00 | 42.95 | TARGET_HIT | 0.50 | 14.84% |
| BUY | retest1 | 2024-04-03 00:00:00 | 41.15 | 2024-04-15 00:00:00 | 38.39 | STOP_HIT | 1.00 | -6.71% |

# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 63.09
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -1.84% / 0.00%
- **Sum % (uncompounded):** -11.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.84% | -11.0% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.84% | -11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.84% | -11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 00:00:00 | 64.58 | 43.63 | 61.16 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 00:00:00 | 70.66 | 44.43 | 62.21 | T1 booked 50% @ 70.66 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 64.58 | 46.02 | 66.42 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 00:00:00 | 76.75 | 52.69 | 73.28 | Stage2 pullback-breakout RSI=57 vol=3.4x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-09-06 00:00:00 | 70.34 | 54.11 | 73.35 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-09-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 00:00:00 | 74.37 | 55.06 | 72.14 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-09-19 00:00:00 | 69.21 | 55.53 | 71.83 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 65.22 | 56.92 | 58.85 | Stage2 pullback-breakout RSI=61 vol=3.0x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 59.99 | 57.13 | 60.15 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 64.55 | 57.38 | 60.35 | Stage2 pullback-breakout RSI=57 vol=3.1x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-12-09 00:00:00 | 66.41 | 58.11 | 63.46 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 00:00:00 | 64.58 | 2024-07-12 00:00:00 | 70.66 | PARTIAL | 0.50 | 9.41% |
| BUY | retest1 | 2024-07-08 00:00:00 | 64.58 | 2024-07-23 00:00:00 | 64.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 00:00:00 | 76.75 | 2024-09-06 00:00:00 | 70.34 | STOP_HIT | 1.00 | -8.35% |
| BUY | retest1 | 2024-09-16 00:00:00 | 74.37 | 2024-09-19 00:00:00 | 69.21 | STOP_HIT | 1.00 | -6.94% |
| BUY | retest1 | 2024-11-06 00:00:00 | 65.22 | 2024-11-11 00:00:00 | 59.99 | STOP_HIT | 1.00 | -8.03% |
| BUY | retest1 | 2024-11-25 00:00:00 | 64.55 | 2024-12-09 00:00:00 | 66.41 | STOP_HIT | 1.00 | 2.88% |

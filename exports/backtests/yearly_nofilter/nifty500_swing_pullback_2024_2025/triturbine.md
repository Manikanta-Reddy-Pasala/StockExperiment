# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 585.00
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
- **Avg / median % per leg:** 4.09% / 1.47%
- **Sum % (uncompounded):** 20.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 4.09% | 20.5% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 4.09% | 20.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 4.09% | 20.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 00:00:00 | 617.70 | 486.84 | 582.44 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=26.10 |
| Stop hit — per-position SL triggered | 2024-07-10 00:00:00 | 626.75 | 500.07 | 610.96 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-10-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 00:00:00 | 731.20 | 593.51 | 713.86 | Stage2 pullback-breakout RSI=53 vol=4.4x ATR=32.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 00:00:00 | 797.16 | 598.56 | 727.32 | T1 booked 50% @ 797.16 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 731.20 | 610.87 | 752.15 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 764.15 | 624.04 | 687.57 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=38.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 00:00:00 | 840.31 | 626.03 | 700.56 | T1 booked 50% @ 840.31 |
| Stop hit — per-position SL triggered | 2024-11-29 00:00:00 | 764.15 | 631.16 | 725.77 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-26 00:00:00 | 617.70 | 2024-07-10 00:00:00 | 626.75 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest1 | 2024-10-08 00:00:00 | 731.20 | 2024-10-11 00:00:00 | 797.16 | PARTIAL | 0.50 | 9.02% |
| BUY | retest1 | 2024-10-08 00:00:00 | 731.20 | 2024-10-22 00:00:00 | 731.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 764.15 | 2024-11-26 00:00:00 | 840.31 | PARTIAL | 0.50 | 9.97% |
| BUY | retest1 | 2024-11-25 00:00:00 | 764.15 | 2024-11-29 00:00:00 | 764.15 | STOP_HIT | 0.50 | 0.00% |

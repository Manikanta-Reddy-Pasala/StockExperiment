# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 26610.00
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
- **Avg / median % per leg:** -0.52% / 0.00%
- **Sum % (uncompounded):** -3.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.52% | -3.1% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.52% | -3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.52% | -3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 00:00:00 | 28596.80 | 25913.96 | 27661.45 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=577.04 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 27731.24 | 25983.27 | 27807.00 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 00:00:00 | 29156.60 | 26383.85 | 27950.47 | Stage2 pullback-breakout RSI=63 vol=3.1x ATR=789.01 |
| Stop hit — per-position SL triggered | 2024-09-05 00:00:00 | 29664.10 | 26700.11 | 29084.71 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 00:00:00 | 29184.70 | 27014.82 | 28854.19 | Stage2 pullback-breakout RSI=55 vol=1.6x ATR=599.17 |
| Stop hit — per-position SL triggered | 2024-10-03 00:00:00 | 28285.94 | 27089.77 | 28865.94 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-12-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 00:00:00 | 29053.85 | 27641.64 | 28449.39 | Stage2 pullback-breakout RSI=59 vol=2.6x ATR=584.44 |
| Stop hit — per-position SL triggered | 2024-12-24 00:00:00 | 28177.19 | 27669.24 | 28480.99 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-12-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 00:00:00 | 29263.50 | 27694.24 | 28565.51 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=624.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 00:00:00 | 30511.61 | 27839.75 | 29213.26 | T1 booked 50% @ 30511.61 |
| Stop hit — per-position SL triggered | 2025-01-10 00:00:00 | 29263.50 | 27885.14 | 29248.68 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 00:00:00 | 28596.80 | 2024-07-19 00:00:00 | 27731.24 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2024-08-22 00:00:00 | 29156.60 | 2024-09-05 00:00:00 | 29664.10 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest1 | 2024-09-26 00:00:00 | 29184.70 | 2024-10-03 00:00:00 | 28285.94 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2024-12-19 00:00:00 | 29053.85 | 2024-12-24 00:00:00 | 28177.19 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest1 | 2024-12-27 00:00:00 | 29263.50 | 2025-01-07 00:00:00 | 30511.61 | PARTIAL | 0.50 | 4.27% |
| BUY | retest1 | 2024-12-27 00:00:00 | 29263.50 | 2025-01-10 00:00:00 | 29263.50 | STOP_HIT | 0.50 | 0.00% |

# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 4535.40
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 1.98% / 4.20%
- **Sum % (uncompounded):** 11.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.98% | 11.9% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.98% | 11.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.98% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 4812.70 | 4540.33 | 4711.02 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=100.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 5014.63 | 4552.60 | 4777.97 | T1 booked 50% @ 5014.63 |
| Target hit | 2023-10-13 00:00:00 | 5260.30 | 4720.93 | 5263.57 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 5192.80 | 4761.33 | 5067.84 | Stage2 pullback-breakout RSI=57 vol=10.9x ATR=124.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 00:00:00 | 5440.88 | 4778.30 | 5138.58 | T1 booked 50% @ 5440.88 |
| Target hit | 2023-11-21 00:00:00 | 5277.65 | 4833.62 | 5300.07 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 5476.30 | 4885.82 | 5335.16 | Stage2 pullback-breakout RSI=62 vol=1.5x ATR=129.24 |
| Stop hit — per-position SL triggered | 2023-12-11 00:00:00 | 5282.44 | 4896.98 | 5356.07 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 6002.05 | 5130.27 | 5791.97 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=179.67 |
| Stop hit — per-position SL triggered | 2024-02-02 00:00:00 | 5732.55 | 5146.05 | 5815.39 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 4812.70 | 2023-09-05 00:00:00 | 5014.63 | PARTIAL | 0.50 | 4.20% |
| BUY | retest1 | 2023-08-31 00:00:00 | 4812.70 | 2023-10-13 00:00:00 | 5260.30 | TARGET_HIT | 0.50 | 9.30% |
| BUY | retest1 | 2023-11-03 00:00:00 | 5192.80 | 2023-11-08 00:00:00 | 5440.88 | PARTIAL | 0.50 | 4.78% |
| BUY | retest1 | 2023-11-03 00:00:00 | 5192.80 | 2023-11-21 00:00:00 | 5277.65 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2023-12-07 00:00:00 | 5476.30 | 2023-12-11 00:00:00 | 5282.44 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest1 | 2024-01-31 00:00:00 | 6002.05 | 2024-02-02 00:00:00 | 5732.55 | STOP_HIT | 1.00 | -4.49% |

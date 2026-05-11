# India Cements Ltd. (INDIACEM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 408.80
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
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.37% / 5.31%
- **Sum % (uncompounded):** 8.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.37% | 8.2% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.37% | 8.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.37% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 229.15 | 211.44 | 216.15 | Stage2 pullback-breakout RSI=63 vol=3.9x ATR=7.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 00:00:00 | 244.21 | 211.74 | 218.50 | T1 booked 50% @ 244.21 |
| Stop hit — per-position SL triggered | 2023-08-25 00:00:00 | 229.15 | 214.60 | 232.29 | SL hit (bars_held=11) |

### Cycle 2 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 260.95 | 216.02 | 237.10 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=10.35 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 245.42 | 218.33 | 243.10 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 00:00:00 | 229.70 | 219.61 | 217.85 | Stage2 pullback-breakout RSI=65 vol=4.3x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 00:00:00 | 241.89 | 220.06 | 222.35 | T1 booked 50% @ 241.89 |
| Target hit | 2023-12-20 00:00:00 | 248.15 | 225.59 | 252.84 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 273.10 | 228.85 | 258.10 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=10.48 |
| Stop hit — per-position SL triggered | 2024-01-08 00:00:00 | 257.38 | 229.49 | 258.68 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 229.15 | 2023-08-10 00:00:00 | 244.21 | PARTIAL | 0.50 | 6.57% |
| BUY | retest1 | 2023-08-09 00:00:00 | 229.15 | 2023-08-25 00:00:00 | 229.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-01 00:00:00 | 260.95 | 2023-09-12 00:00:00 | 245.42 | STOP_HIT | 1.00 | -5.95% |
| BUY | retest1 | 2023-11-28 00:00:00 | 229.70 | 2023-11-30 00:00:00 | 241.89 | PARTIAL | 0.50 | 5.31% |
| BUY | retest1 | 2023-11-28 00:00:00 | 229.70 | 2023-12-20 00:00:00 | 248.15 | TARGET_HIT | 0.50 | 8.03% |
| BUY | retest1 | 2024-01-04 00:00:00 | 273.10 | 2024-01-08 00:00:00 | 257.38 | STOP_HIT | 1.00 | -5.76% |

# Polycab India Ltd. (POLYCAB)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 9083.00
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
- **Avg / median % per leg:** 2.30% / 4.19%
- **Sum % (uncompounded):** 13.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.30% | 13.8% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.30% | 13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.30% | 13.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 05:30:00 | 6786.00 | 5896.49 | 6601.59 | Stage2 pullback-breakout RSI=56 vol=2.4x ATR=201.47 |
| Stop hit — per-position SL triggered | 2024-09-03 05:30:00 | 6767.15 | 5979.97 | 6708.40 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 05:30:00 | 6811.00 | 6021.43 | 6703.41 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=148.10 |
| Stop hit — per-position SL triggered | 2024-09-19 05:30:00 | 6588.85 | 6060.46 | 6690.68 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-09-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 05:30:00 | 6747.40 | 6089.35 | 6678.97 | Stage2 pullback-breakout RSI=54 vol=1.6x ATR=141.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 05:30:00 | 7030.45 | 6098.95 | 6714.76 | T1 booked 50% @ 7030.45 |
| Target hit | 2024-10-17 05:30:00 | 7120.55 | 6243.83 | 7140.65 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-11-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 05:30:00 | 7044.00 | 6330.71 | 6673.20 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=201.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 05:30:00 | 7447.34 | 6378.14 | 6923.97 | T1 booked 50% @ 7447.34 |
| Target hit | 2024-12-20 05:30:00 | 7178.25 | 6496.15 | 7269.85 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-20 05:30:00 | 6786.00 | 2024-09-03 05:30:00 | 6767.15 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-11 05:30:00 | 6811.00 | 2024-09-19 05:30:00 | 6588.85 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest1 | 2024-09-26 05:30:00 | 6747.40 | 2024-09-27 05:30:00 | 7030.45 | PARTIAL | 0.50 | 4.19% |
| BUY | retest1 | 2024-09-26 05:30:00 | 6747.40 | 2024-10-17 05:30:00 | 7120.55 | TARGET_HIT | 0.50 | 5.53% |
| BUY | retest1 | 2024-11-27 05:30:00 | 7044.00 | 2024-12-04 05:30:00 | 7447.34 | PARTIAL | 0.50 | 5.73% |
| BUY | retest1 | 2024-11-27 05:30:00 | 7044.00 | 2024-12-20 05:30:00 | 7178.25 | TARGET_HIT | 0.50 | 1.91% |

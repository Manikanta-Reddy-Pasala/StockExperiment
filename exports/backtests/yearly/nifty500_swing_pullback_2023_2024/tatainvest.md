# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (910 bars)
- **Last close:** 717.60
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -5.16% / -4.33%
- **Sum % (uncompounded):** -15.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.16% | -15.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.16% | -15.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.16% | -15.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 05:30:00 | 251.20 | 224.79 | 246.09 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=4.89 |
| Stop hit — per-position SL triggered | 2023-09-13 05:30:00 | 243.87 | 225.21 | 246.08 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-01-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 05:30:00 | 448.04 | 308.83 | 424.57 | Stage2 pullback-breakout RSI=70 vol=3.7x ATR=12.94 |
| Stop hit — per-position SL triggered | 2024-01-17 05:30:00 | 428.63 | 313.80 | 428.07 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-04-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 05:30:00 | 747.80 | 453.11 | 695.49 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=41.08 |
| Stop hit — per-position SL triggered | 2024-04-09 05:30:00 | 686.18 | 460.63 | 698.27 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-05-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 05:30:00 | 687.41 | 496.87 | 668.87 | Stage2 pullback-breakout RSI=54 vol=2.4x ATR=26.84 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-11 05:30:00 | 251.20 | 2023-09-13 05:30:00 | 243.87 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2024-01-11 05:30:00 | 448.04 | 2024-01-17 05:30:00 | 428.63 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest1 | 2024-04-04 05:30:00 | 747.80 | 2024-04-09 05:30:00 | 686.18 | STOP_HIT | 1.00 | -8.24% |

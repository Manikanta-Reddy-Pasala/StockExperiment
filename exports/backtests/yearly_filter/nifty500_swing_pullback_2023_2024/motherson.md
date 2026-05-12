# Samvardhana Motherson International Ltd. (MOTHERSON)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 130.21
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 4
- **Avg / median % per leg:** 3.16% / 4.96%
- **Sum % (uncompounded):** 25.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 0 | 4 | 4 | 3.16% | 25.3% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 0 | 4 | 4 | 3.16% | 25.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 0 | 4 | 4 | 3.16% | 25.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 61.70 | 58.77 | 60.29 | Stage2 pullback-breakout RSI=55 vol=2.7x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 00:00:00 | 64.76 | 59.06 | 61.64 | T1 booked 50% @ 64.76 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 61.70 | 59.41 | 62.88 | SL hit (bars_held=15) |

### Cycle 2 — BUY (started 2023-12-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 00:00:00 | 67.97 | 59.71 | 63.76 | Stage2 pullback-breakout RSI=68 vol=4.2x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 00:00:00 | 71.55 | 59.82 | 64.40 | T1 booked 50% @ 71.55 |
| Stop hit — per-position SL triggered | 2024-01-03 00:00:00 | 67.97 | 60.00 | 65.22 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-02-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 00:00:00 | 79.03 | 64.96 | 76.45 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 00:00:00 | 84.00 | 65.72 | 78.11 | T1 booked 50% @ 84.00 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 79.03 | 65.87 | 78.34 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 82.83 | 68.60 | 79.25 | Stage2 pullback-breakout RSI=65 vol=2.1x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 87.69 | 69.71 | 82.26 | T1 booked 50% @ 87.69 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 85.23 | 70.55 | 84.03 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-29 00:00:00 | 61.70 | 2023-12-11 00:00:00 | 64.76 | PARTIAL | 0.50 | 4.96% |
| BUY | retest1 | 2023-11-29 00:00:00 | 61.70 | 2023-12-20 00:00:00 | 61.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-29 00:00:00 | 67.97 | 2024-01-01 00:00:00 | 71.55 | PARTIAL | 0.50 | 5.27% |
| BUY | retest1 | 2023-12-29 00:00:00 | 67.97 | 2024-01-03 00:00:00 | 67.97 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-28 00:00:00 | 79.03 | 2024-03-05 00:00:00 | 84.00 | PARTIAL | 0.50 | 6.28% |
| BUY | retest1 | 2024-02-28 00:00:00 | 79.03 | 2024-03-06 00:00:00 | 79.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-16 00:00:00 | 82.83 | 2024-04-26 00:00:00 | 87.69 | PARTIAL | 0.50 | 5.86% |
| BUY | retest1 | 2024-04-16 00:00:00 | 82.83 | 2024-05-06 00:00:00 | 85.23 | STOP_HIT | 0.50 | 2.90% |

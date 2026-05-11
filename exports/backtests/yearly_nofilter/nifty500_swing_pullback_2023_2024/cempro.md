# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 922.85
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 6.34% / 7.71%
- **Sum % (uncompounded):** 50.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.34% | 50.7% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.34% | 50.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.34% | 50.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 00:00:00 | 190.30 | 139.17 | 178.65 | Stage2 pullback-breakout RSI=69 vol=6.4x ATR=7.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 00:00:00 | 204.97 | 140.39 | 182.71 | T1 booked 50% @ 204.97 |
| Target hit | 2023-09-18 00:00:00 | 222.45 | 158.83 | 223.49 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 225.60 | 169.04 | 218.95 | Stage2 pullback-breakout RSI=58 vol=2.6x ATR=7.26 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 214.71 | 170.84 | 217.35 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 224.80 | 174.46 | 211.13 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 00:00:00 | 241.96 | 176.81 | 219.16 | T1 booked 50% @ 241.96 |
| Target hit | 2023-12-20 00:00:00 | 268.20 | 199.11 | 276.95 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 00:00:00 | 311.95 | 210.84 | 287.67 | Stage2 pullback-breakout RSI=70 vol=4.5x ATR=12.00 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 293.94 | 214.40 | 292.21 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 357.15 | 266.55 | 331.41 | Stage2 pullback-breakout RSI=63 vol=3.2x ATR=17.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 00:00:00 | 391.97 | 274.41 | 352.93 | T1 booked 50% @ 391.97 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 357.15 | 281.31 | 364.57 | SL hit (bars_held=15) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-08 00:00:00 | 190.30 | 2023-08-10 00:00:00 | 204.97 | PARTIAL | 0.50 | 7.71% |
| BUY | retest1 | 2023-08-08 00:00:00 | 190.30 | 2023-09-18 00:00:00 | 222.45 | TARGET_HIT | 0.50 | 16.89% |
| BUY | retest1 | 2023-10-17 00:00:00 | 225.60 | 2023-10-23 00:00:00 | 214.71 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest1 | 2023-11-08 00:00:00 | 224.80 | 2023-11-13 00:00:00 | 241.96 | PARTIAL | 0.50 | 7.64% |
| BUY | retest1 | 2023-11-08 00:00:00 | 224.80 | 2023-12-20 00:00:00 | 268.20 | TARGET_HIT | 0.50 | 19.31% |
| BUY | retest1 | 2024-01-10 00:00:00 | 311.95 | 2024-01-16 00:00:00 | 293.94 | STOP_HIT | 1.00 | -5.77% |
| BUY | retest1 | 2024-04-16 00:00:00 | 357.15 | 2024-04-29 00:00:00 | 391.97 | PARTIAL | 0.50 | 9.75% |
| BUY | retest1 | 2024-04-16 00:00:00 | 357.15 | 2024-05-09 00:00:00 | 357.15 | STOP_HIT | 0.50 | 0.00% |

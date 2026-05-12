# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 342.85
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 8.60% / -3.81%
- **Sum % (uncompounded):** 43.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 8.60% | 43.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 8.60% | 43.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 8.60% | 43.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 156.70 | 120.02 | 145.43 | Stage2 pullback-breakout RSI=64 vol=3.2x ATR=7.28 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 145.78 | 121.48 | 146.89 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 138.55 | 123.60 | 137.64 | Stage2 pullback-breakout RSI=50 vol=1.9x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 149.14 | 125.32 | 139.78 | T1 booked 50% @ 149.14 |
| Target hit | 2024-02-12 00:00:00 | 212.35 | 149.32 | 218.56 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 225.00 | 170.06 | 211.89 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=10.67 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 209.00 | 173.54 | 216.36 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 220.30 | 175.47 | 215.15 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=9.97 |
| Stop hit — per-position SL triggered | 2024-05-08 00:00:00 | 211.90 | 180.00 | 219.06 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-16 00:00:00 | 156.70 | 2023-10-23 00:00:00 | 145.78 | STOP_HIT | 1.00 | -6.97% |
| BUY | retest1 | 2023-11-15 00:00:00 | 138.55 | 2023-12-04 00:00:00 | 149.14 | PARTIAL | 0.50 | 7.64% |
| BUY | retest1 | 2023-11-15 00:00:00 | 138.55 | 2024-02-12 00:00:00 | 212.35 | TARGET_HIT | 0.50 | 53.27% |
| BUY | retest1 | 2024-04-03 00:00:00 | 225.00 | 2024-04-15 00:00:00 | 209.00 | STOP_HIT | 1.00 | -7.11% |
| BUY | retest1 | 2024-04-23 00:00:00 | 220.30 | 2024-05-08 00:00:00 | 211.90 | STOP_HIT | 1.00 | -3.81% |

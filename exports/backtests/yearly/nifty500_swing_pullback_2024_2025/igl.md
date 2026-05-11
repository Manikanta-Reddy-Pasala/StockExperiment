# Indraprastha Gas Ltd. (IGL)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 165.99
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.25% / 4.86%
- **Sum % (uncompounded):** 9.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.25% | 9.0% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.25% | 9.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.25% | 9.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 05:30:00 | 251.85 | 222.85 | 236.76 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=7.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 05:30:00 | 266.41 | 223.61 | 241.17 | T1 booked 50% @ 266.41 |
| Target hit | 2024-08-05 05:30:00 | 264.10 | 232.57 | 266.13 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-08-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 05:30:00 | 276.40 | 238.81 | 269.67 | Stage2 pullback-breakout RSI=59 vol=2.5x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-09-09 05:30:00 | 266.66 | 240.83 | 271.19 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 05:30:00 | 274.05 | 242.28 | 268.78 | Stage2 pullback-breakout RSI=55 vol=6.4x ATR=7.71 |
| Stop hit — per-position SL triggered | 2024-10-01 05:30:00 | 279.20 | 245.30 | 272.63 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-28 05:30:00 | 251.85 | 2024-07-02 05:30:00 | 266.41 | PARTIAL | 0.50 | 5.78% |
| BUY | retest1 | 2024-06-28 05:30:00 | 251.85 | 2024-08-05 05:30:00 | 264.10 | TARGET_HIT | 0.50 | 4.86% |
| BUY | retest1 | 2024-08-30 05:30:00 | 276.40 | 2024-09-09 05:30:00 | 266.66 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest1 | 2024-09-17 05:30:00 | 274.05 | 2024-10-01 05:30:00 | 279.20 | STOP_HIT | 1.00 | 1.88% |

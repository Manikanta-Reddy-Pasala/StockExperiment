# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 583.60
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
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 15.53% / 7.84%
- **Sum % (uncompounded):** 77.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 15.53% | 77.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 15.53% | 77.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 15.53% | 77.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 00:00:00 | 245.99 | 234.34 | 235.07 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=9.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 00:00:00 | 265.28 | 234.80 | 239.26 | T1 booked 50% @ 265.28 |
| Target hit | 2024-08-02 00:00:00 | 259.40 | 240.90 | 266.48 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-08-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 00:00:00 | 299.75 | 243.10 | 271.37 | Stage2 pullback-breakout RSI=66 vol=5.2x ATR=16.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 00:00:00 | 333.22 | 243.93 | 276.62 | T1 booked 50% @ 333.22 |
| Target hit | 2024-10-22 00:00:00 | 463.05 | 310.67 | 472.43 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 00:00:00 | 480.20 | 344.75 | 446.57 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=22.27 |
| Stop hit — per-position SL triggered | 2024-12-18 00:00:00 | 474.05 | 357.65 | 467.69 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-05 00:00:00 | 245.99 | 2024-07-09 00:00:00 | 265.28 | PARTIAL | 0.50 | 7.84% |
| BUY | retest1 | 2024-07-05 00:00:00 | 245.99 | 2024-08-02 00:00:00 | 259.40 | TARGET_HIT | 0.50 | 5.45% |
| BUY | retest1 | 2024-08-13 00:00:00 | 299.75 | 2024-08-14 00:00:00 | 333.22 | PARTIAL | 0.50 | 11.17% |
| BUY | retest1 | 2024-08-13 00:00:00 | 299.75 | 2024-10-22 00:00:00 | 463.05 | TARGET_HIT | 0.50 | 54.48% |
| BUY | retest1 | 2024-12-04 00:00:00 | 480.20 | 2024-12-18 00:00:00 | 474.05 | STOP_HIT | 1.00 | -1.28% |

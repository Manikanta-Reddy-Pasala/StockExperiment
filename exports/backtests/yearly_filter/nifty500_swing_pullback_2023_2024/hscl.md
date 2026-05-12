# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (909 bars)
- **Last close:** 614.60
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.73% / 7.84%
- **Sum % (uncompounded):** 10.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.73% | 10.9% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.73% | 10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.73% | 10.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 00:00:00 | 265.35 | 177.53 | 252.27 | Stage2 pullback-breakout RSI=64 vol=3.4x ATR=9.19 |
| Stop hit — per-position SL triggered | 2023-11-30 00:00:00 | 251.56 | 182.62 | 258.19 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 00:00:00 | 307.35 | 199.46 | 284.88 | Stage2 pullback-breakout RSI=66 vol=2.4x ATR=12.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 00:00:00 | 331.45 | 203.98 | 295.01 | T1 booked 50% @ 331.45 |
| Target hit | 2024-02-12 00:00:00 | 352.85 | 246.00 | 365.86 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 367.70 | 278.12 | 329.71 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=15.98 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 343.73 | 286.47 | 351.67 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-21 00:00:00 | 265.35 | 2023-11-30 00:00:00 | 251.56 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest1 | 2023-12-26 00:00:00 | 307.35 | 2024-01-01 00:00:00 | 331.45 | PARTIAL | 0.50 | 7.84% |
| BUY | retest1 | 2023-12-26 00:00:00 | 307.35 | 2024-02-12 00:00:00 | 352.85 | TARGET_HIT | 0.50 | 14.80% |
| BUY | retest1 | 2024-04-23 00:00:00 | 367.70 | 2024-05-09 00:00:00 | 343.73 | STOP_HIT | 1.00 | -6.52% |

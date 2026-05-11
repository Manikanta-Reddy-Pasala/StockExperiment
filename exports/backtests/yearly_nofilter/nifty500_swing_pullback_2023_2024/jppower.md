# Jaiprakash Power Ventures Ltd. (JPPOWER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 18.57
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
- **Avg / median % per leg:** 14.73% / 11.40%
- **Sum % (uncompounded):** 73.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 14.73% | 73.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 14.73% | 73.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 14.73% | 73.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 10.40 | 7.78 | 9.58 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 00:00:00 | 11.59 | 7.87 | 10.01 | T1 booked 50% @ 11.59 |
| Target hit | 2023-12-01 00:00:00 | 12.85 | 8.85 | 13.05 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 00:00:00 | 14.75 | 9.76 | 13.77 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 00:00:00 | 16.25 | 9.88 | 14.17 | T1 booked 50% @ 16.25 |
| Target hit | 2024-02-21 00:00:00 | 19.80 | 12.41 | 19.85 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 19.95 | 14.20 | 17.89 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 18.81 | 14.40 | 18.35 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-02 00:00:00 | 10.40 | 2023-11-06 00:00:00 | 11.59 | PARTIAL | 0.50 | 11.40% |
| BUY | retest1 | 2023-11-02 00:00:00 | 10.40 | 2023-12-01 00:00:00 | 12.85 | TARGET_HIT | 0.50 | 23.56% |
| BUY | retest1 | 2024-01-01 00:00:00 | 14.75 | 2024-01-03 00:00:00 | 16.25 | PARTIAL | 0.50 | 10.18% |
| BUY | retest1 | 2024-01-01 00:00:00 | 14.75 | 2024-02-21 00:00:00 | 19.80 | TARGET_HIT | 0.50 | 34.24% |
| BUY | retest1 | 2024-04-30 00:00:00 | 19.95 | 2024-05-07 00:00:00 | 18.81 | STOP_HIT | 1.00 | -5.72% |

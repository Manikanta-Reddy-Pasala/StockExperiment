# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1243.00
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
- **Avg / median % per leg:** 0.43% / 0.61%
- **Sum % (uncompounded):** 1.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 0.43% | 1.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 0.43% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 0.43% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 00:00:00 | 437.75 | 415.99 | 431.53 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=11.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 00:00:00 | 460.31 | 416.58 | 434.18 | T1 booked 50% @ 460.31 |
| Target hit | 2024-07-19 00:00:00 | 440.40 | 421.33 | 452.99 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 00:00:00 | 474.15 | 438.35 | 463.57 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=14.59 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 452.27 | 440.54 | 466.31 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-10-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 00:00:00 | 492.20 | 441.46 | 466.06 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=18.73 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 494.95 | 446.34 | 483.44 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-02 00:00:00 | 437.75 | 2024-07-04 00:00:00 | 460.31 | PARTIAL | 0.50 | 5.15% |
| BUY | retest1 | 2024-07-02 00:00:00 | 437.75 | 2024-07-19 00:00:00 | 440.40 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-11 00:00:00 | 474.15 | 2024-10-22 00:00:00 | 452.27 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest1 | 2024-10-28 00:00:00 | 492.20 | 2024-11-11 00:00:00 | 494.95 | STOP_HIT | 1.00 | 0.56% |

# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 218.59
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 9.19% / 7.67%
- **Sum % (uncompounded):** 55.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 9.19% | 55.1% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 9.19% | 55.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 9.19% | 55.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 00:00:00 | 70.83 | 59.60 | 68.61 | Stage2 pullback-breakout RSI=54 vol=2.3x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 00:00:00 | 78.06 | 60.13 | 70.08 | T1 booked 50% @ 78.06 |
| Target hit | 2024-01-17 00:00:00 | 104.29 | 75.37 | 105.71 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 114.77 | 85.65 | 111.62 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=3.81 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 109.06 | 86.73 | 112.13 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 117.69 | 89.01 | 109.00 | Stage2 pullback-breakout RSI=64 vol=3.0x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 00:00:00 | 126.71 | 89.74 | 112.10 | T1 booked 50% @ 126.71 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 117.69 | 91.61 | 116.37 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 122.49 | 94.28 | 118.49 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 116.33 | 95.04 | 118.88 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-27 00:00:00 | 70.83 | 2023-11-02 00:00:00 | 78.06 | PARTIAL | 0.50 | 10.21% |
| BUY | retest1 | 2023-10-27 00:00:00 | 70.83 | 2024-01-17 00:00:00 | 104.29 | TARGET_HIT | 0.50 | 47.24% |
| BUY | retest1 | 2024-03-05 00:00:00 | 114.77 | 2024-03-12 00:00:00 | 109.06 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest1 | 2024-04-02 00:00:00 | 117.69 | 2024-04-04 00:00:00 | 126.71 | PARTIAL | 0.50 | 7.67% |
| BUY | retest1 | 2024-04-02 00:00:00 | 117.69 | 2024-04-15 00:00:00 | 117.69 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-30 00:00:00 | 122.49 | 2024-05-06 00:00:00 | 116.33 | STOP_HIT | 1.00 | -5.03% |

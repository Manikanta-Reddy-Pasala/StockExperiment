# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1408.90
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
- **Avg / median % per leg:** 2.64% / 6.14%
- **Sum % (uncompounded):** 15.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.64% | 15.9% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.64% | 15.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.64% | 15.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 00:00:00 | 2306.30 | 1807.22 | 2203.90 | Stage2 pullback-breakout RSI=59 vol=9.6x ATR=95.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 00:00:00 | 2498.03 | 1830.23 | 2268.77 | T1 booked 50% @ 2498.03 |
| Stop hit — per-position SL triggered | 2024-09-02 00:00:00 | 2306.30 | 1852.78 | 2311.81 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-09-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 00:00:00 | 2484.95 | 1883.11 | 2343.40 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=103.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 00:00:00 | 2692.07 | 1899.26 | 2408.53 | T1 booked 50% @ 2692.07 |
| Target hit | 2024-10-07 00:00:00 | 2637.65 | 2030.53 | 2704.05 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 2924.95 | 2073.38 | 2738.25 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=120.37 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 2744.40 | 2106.53 | 2796.19 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-11-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 00:00:00 | 2796.20 | 2232.37 | 2678.24 | Stage2 pullback-breakout RSI=56 vol=3.3x ATR=125.64 |
| Stop hit — per-position SL triggered | 2024-12-11 00:00:00 | 2775.10 | 2288.09 | 2767.26 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-21 00:00:00 | 2306.30 | 2024-08-27 00:00:00 | 2498.03 | PARTIAL | 0.50 | 8.31% |
| BUY | retest1 | 2024-08-21 00:00:00 | 2306.30 | 2024-09-02 00:00:00 | 2306.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 00:00:00 | 2484.95 | 2024-09-12 00:00:00 | 2692.07 | PARTIAL | 0.50 | 8.33% |
| BUY | retest1 | 2024-09-10 00:00:00 | 2484.95 | 2024-10-07 00:00:00 | 2637.65 | TARGET_HIT | 0.50 | 6.14% |
| BUY | retest1 | 2024-10-16 00:00:00 | 2924.95 | 2024-10-22 00:00:00 | 2744.40 | STOP_HIT | 1.00 | -6.17% |
| BUY | retest1 | 2024-11-27 00:00:00 | 2796.20 | 2024-12-11 00:00:00 | 2775.10 | STOP_HIT | 1.00 | -0.75% |

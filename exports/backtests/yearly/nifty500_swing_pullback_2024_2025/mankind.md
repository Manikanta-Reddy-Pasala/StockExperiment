# Mankind Pharma Ltd. (MANKIND)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 2430.80
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 4.15% / 6.51%
- **Sum % (uncompounded):** 16.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 2 | 2 | 4.15% | 16.6% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 2 | 2 | 4.15% | 16.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 2 | 2 | 4.15% | 16.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 05:30:00 | 2627.80 | 2268.51 | 2578.28 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=95.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 05:30:00 | 2819.77 | 2289.99 | 2631.07 | T1 booked 50% @ 2819.77 |
| Stop hit — per-position SL triggered | 2024-11-12 05:30:00 | 2627.80 | 2304.63 | 2640.54 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-12-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 05:30:00 | 2805.95 | 2370.41 | 2642.23 | Stage2 pullback-breakout RSI=66 vol=5.4x ATR=91.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 05:30:00 | 2988.49 | 2386.02 | 2709.59 | T1 booked 50% @ 2988.49 |
| Stop hit — per-position SL triggered | 2025-01-06 05:30:00 | 2884.55 | 2429.71 | 2819.27 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-30 05:30:00 | 2627.80 | 2024-11-06 05:30:00 | 2819.77 | PARTIAL | 0.50 | 7.31% |
| BUY | retest1 | 2024-10-30 05:30:00 | 2627.80 | 2024-11-12 05:30:00 | 2627.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-18 05:30:00 | 2805.95 | 2024-12-23 05:30:00 | 2988.49 | PARTIAL | 0.50 | 6.51% |
| BUY | retest1 | 2024-12-18 05:30:00 | 2805.95 | 2025-01-06 05:30:00 | 2884.55 | STOP_HIT | 0.50 | 2.80% |

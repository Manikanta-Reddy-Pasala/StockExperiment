# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 903.50
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
- **Avg / median % per leg:** 0.41% / 4.76%
- **Sum % (uncompounded):** 1.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.41% | 1.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.41% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.41% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 05:30:00 | 878.95 | 834.09 | 857.13 | Stage2 pullback-breakout RSI=58 vol=2.8x ATR=23.34 |
| Stop hit — per-position SL triggered | 2025-07-14 05:30:00 | 843.94 | 836.48 | 860.49 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2025-07-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 05:30:00 | 902.40 | 837.48 | 865.47 | Stage2 pullback-breakout RSI=65 vol=3.1x ATR=21.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 05:30:00 | 945.37 | 841.63 | 888.19 | T1 booked 50% @ 945.37 |
| Target hit | 2025-08-19 05:30:00 | 946.05 | 862.09 | 950.36 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 970.80 | 878.38 | 923.90 | Stage2 pullback-breakout RSI=66 vol=5.1x ATR=25.85 |
| Stop hit — per-position SL triggered | 2025-10-07 05:30:00 | 932.03 | 879.66 | 927.28 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-02 05:30:00 | 878.95 | 2025-07-14 05:30:00 | 843.94 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest1 | 2025-07-16 05:30:00 | 902.40 | 2025-07-23 05:30:00 | 945.37 | PARTIAL | 0.50 | 4.76% |
| BUY | retest1 | 2025-07-16 05:30:00 | 902.40 | 2025-08-19 05:30:00 | 946.05 | TARGET_HIT | 0.50 | 4.84% |
| BUY | retest1 | 2025-10-03 05:30:00 | 970.80 | 2025-10-07 05:30:00 | 932.03 | STOP_HIT | 1.00 | -3.99% |

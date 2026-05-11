# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (836 bars)
- **Last close:** 20743.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 2 / 0 / 2
- **Avg / median % per leg:** 23.52% / 32.49%
- **Sum % (uncompounded):** 94.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 23.52% | 94.1% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 23.52% | 94.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 23.52% | 94.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 00:00:00 | 2675.00 | 1759.91 | 2554.27 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=113.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 00:00:00 | 2902.85 | 1801.52 | 2645.96 | T1 booked 50% @ 2902.85 |
| Target hit | 2023-10-16 00:00:00 | 3830.05 | 2473.07 | 3842.81 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-02-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 00:00:00 | 4422.65 | 2560.88 | 3813.20 | Stage2 pullback-breakout RSI=67 vol=2.7x ATR=218.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 00:00:00 | 4860.06 | 2584.84 | 3923.26 | T1 booked 50% @ 4860.06 |
| Target hit | 2024-03-13 00:00:00 | 5859.65 | 3211.79 | 5939.96 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-04 00:00:00 | 2675.00 | 2023-08-10 00:00:00 | 2902.85 | PARTIAL | 0.50 | 8.52% |
| BUY | retest1 | 2023-08-04 00:00:00 | 2675.00 | 2023-10-16 00:00:00 | 3830.05 | TARGET_HIT | 0.50 | 43.18% |
| BUY | retest1 | 2024-02-14 00:00:00 | 4422.65 | 2024-02-15 00:00:00 | 4860.06 | PARTIAL | 0.50 | 9.89% |
| BUY | retest1 | 2024-02-14 00:00:00 | 4422.65 | 2024-03-13 00:00:00 | 5859.65 | TARGET_HIT | 0.50 | 32.49% |

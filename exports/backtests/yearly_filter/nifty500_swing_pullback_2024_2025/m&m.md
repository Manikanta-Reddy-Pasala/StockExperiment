# Mahindra & Mahindra Ltd. (M&M)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 3261.90
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
- **Avg / median % per leg:** -0.68% / 0.47%
- **Sum % (uncompounded):** -2.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.68% | -2.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.68% | -2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.68% | -2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 00:00:00 | 2840.45 | 2250.19 | 2768.82 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=77.60 |
| Stop hit — per-position SL triggered | 2024-08-22 00:00:00 | 2724.05 | 2270.17 | 2765.39 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-09-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 00:00:00 | 2950.85 | 2363.97 | 2775.20 | Stage2 pullback-breakout RSI=68 vol=4.5x ATR=68.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 00:00:00 | 3087.07 | 2377.80 | 2827.35 | T1 booked 50% @ 3087.07 |
| Target hit | 2024-10-17 00:00:00 | 2964.60 | 2487.37 | 3053.07 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 00:00:00 | 3211.10 | 2680.12 | 3027.31 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=79.68 |
| Stop hit — per-position SL triggered | 2025-01-06 00:00:00 | 3091.58 | 2689.39 | 3048.84 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-16 00:00:00 | 2840.45 | 2024-08-22 00:00:00 | 2724.05 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest1 | 2024-09-20 00:00:00 | 2950.85 | 2024-09-24 00:00:00 | 3087.07 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2024-09-20 00:00:00 | 2950.85 | 2024-10-17 00:00:00 | 2964.60 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-01-02 00:00:00 | 3211.10 | 2025-01-06 00:00:00 | 3091.58 | STOP_HIT | 1.00 | -3.72% |

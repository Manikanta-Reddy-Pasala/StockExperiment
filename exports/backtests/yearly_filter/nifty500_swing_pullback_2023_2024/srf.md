# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2787.90
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.22% / 2.71%
- **Sum % (uncompounded):** 0.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.22% | 0.9% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.22% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.22% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-03-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 00:00:00 | 2450.05 | 2353.70 | 2381.28 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=46.81 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 2379.84 | 2355.70 | 2391.12 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 2538.35 | 2361.80 | 2424.57 | Stage2 pullback-breakout RSI=67 vol=3.4x ATR=58.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 2654.78 | 2374.26 | 2494.62 | T1 booked 50% @ 2654.78 |
| Stop hit — per-position SL triggered | 2024-04-08 00:00:00 | 2607.20 | 2383.20 | 2529.90 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 2645.55 | 2387.87 | 2546.34 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=62.44 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 2551.89 | 2391.97 | 2554.80 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-07 00:00:00 | 2450.05 | 2024-03-13 00:00:00 | 2379.84 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest1 | 2024-03-21 00:00:00 | 2538.35 | 2024-04-02 00:00:00 | 2654.78 | PARTIAL | 0.50 | 4.59% |
| BUY | retest1 | 2024-03-21 00:00:00 | 2538.35 | 2024-04-08 00:00:00 | 2607.20 | STOP_HIT | 0.50 | 2.71% |
| BUY | retest1 | 2024-04-10 00:00:00 | 2645.55 | 2024-04-15 00:00:00 | 2551.89 | STOP_HIT | 1.00 | -3.54% |

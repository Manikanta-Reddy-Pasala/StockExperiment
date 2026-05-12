# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 560.85
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 2.48% / 8.80%
- **Sum % (uncompounded):** 9.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 2.48% | 9.9% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 2.48% | 9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 2.48% | 9.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 00:00:00 | 509.95 | 349.75 | 478.91 | Stage2 pullback-breakout RSI=64 vol=3.8x ATR=22.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 00:00:00 | 554.80 | 358.48 | 499.34 | T1 booked 50% @ 554.80 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 509.95 | 367.02 | 511.73 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 740.80 | 503.61 | 709.89 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=38.08 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 683.69 | 523.54 | 722.46 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-12-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 00:00:00 | 710.50 | 538.99 | 690.42 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=31.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 00:00:00 | 773.14 | 558.29 | 725.74 | T1 booked 50% @ 773.14 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-22 00:00:00 | 509.95 | 2024-07-29 00:00:00 | 554.80 | PARTIAL | 0.50 | 8.80% |
| BUY | retest1 | 2024-07-22 00:00:00 | 509.95 | 2024-08-05 00:00:00 | 509.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 00:00:00 | 740.80 | 2024-11-13 00:00:00 | 683.69 | STOP_HIT | 1.00 | -7.71% |
| BUY | retest1 | 2024-12-02 00:00:00 | 710.50 | 2024-12-16 00:00:00 | 773.14 | PARTIAL | 0.50 | 8.82% |

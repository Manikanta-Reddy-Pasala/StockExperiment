# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 2528.90
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
- **Avg / median % per leg:** 1.63% / 6.69%
- **Sum % (uncompounded):** 6.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.63% | 6.5% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.63% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.63% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 05:30:00 | 2638.80 | 2319.89 | 2604.60 | Stage2 pullback-breakout RSI=53 vol=5.7x ATR=91.55 |
| Stop hit — per-position SL triggered | 2025-08-07 05:30:00 | 2501.48 | 2326.68 | 2589.78 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-08-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 05:30:00 | 2602.20 | 2336.97 | 2570.57 | Stage2 pullback-breakout RSI=53 vol=4.1x ATR=87.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 05:30:00 | 2776.36 | 2341.46 | 2591.31 | T1 booked 50% @ 2776.36 |
| Target hit | 2025-09-12 05:30:00 | 2867.00 | 2434.59 | 2869.43 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 05:30:00 | 2951.00 | 2494.16 | 2843.43 | Stage2 pullback-breakout RSI=61 vol=1.9x ATR=101.00 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 2799.50 | 2512.72 | 2853.00 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-04 05:30:00 | 2638.80 | 2025-08-07 05:30:00 | 2501.48 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest1 | 2025-08-14 05:30:00 | 2602.20 | 2025-08-18 05:30:00 | 2776.36 | PARTIAL | 0.50 | 6.69% |
| BUY | retest1 | 2025-08-14 05:30:00 | 2602.20 | 2025-09-12 05:30:00 | 2867.00 | TARGET_HIT | 0.50 | 10.18% |
| BUY | retest1 | 2025-10-07 05:30:00 | 2951.00 | 2025-10-14 05:30:00 | 2799.50 | STOP_HIT | 1.00 | -5.13% |

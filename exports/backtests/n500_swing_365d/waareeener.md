# Waaree Energies Ltd. (WAAREEENER)

## Backtest Summary

- **Window:** 2024-10-28 05:30:00 → 2026-05-08 05:30:00 (378 bars)
- **Last close:** 3230.10
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
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.37% / 0.00%
- **Sum % (uncompounded):** 1.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.37% | 1.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.37% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.37% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 05:30:00 | 3417.30 | 2760.73 | 3144.93 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=134.16 |
| Stop hit — per-position SL triggered | 2025-09-01 05:30:00 | 3216.06 | 2771.43 | 3172.37 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 3477.80 | 2803.95 | 3218.87 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=131.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 05:30:00 | 3740.81 | 2813.26 | 3268.48 | T1 booked 50% @ 3740.81 |
| Stop hit — per-position SL triggered | 2025-09-19 05:30:00 | 3477.80 | 2857.34 | 3402.14 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-10-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 05:30:00 | 3484.50 | 2935.21 | 3388.72 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=120.78 |
| Stop hit — per-position SL triggered | 2025-10-29 05:30:00 | 3477.60 | 2992.23 | 3477.84 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-28 05:30:00 | 3417.30 | 2025-09-01 05:30:00 | 3216.06 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest1 | 2025-09-10 05:30:00 | 3477.80 | 2025-09-11 05:30:00 | 3740.81 | PARTIAL | 0.50 | 7.56% |
| BUY | retest1 | 2025-09-10 05:30:00 | 3477.80 | 2025-09-19 05:30:00 | 3477.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-14 05:30:00 | 3484.50 | 2025-10-29 05:30:00 | 3477.60 | STOP_HIT | 1.00 | -0.20% |

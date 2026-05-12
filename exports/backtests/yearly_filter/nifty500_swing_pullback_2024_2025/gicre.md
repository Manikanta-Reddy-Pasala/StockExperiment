# General Insurance Corporation of India (GICRE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 391.20
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -2.13% / -2.76%
- **Sum % (uncompounded):** -8.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -2.13% | -8.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -2.13% | -8.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -2.13% | -8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 00:00:00 | 402.60 | 328.17 | 382.89 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=16.41 |
| Stop hit — per-position SL triggered | 2024-07-22 00:00:00 | 391.50 | 335.65 | 396.63 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 410.50 | 337.55 | 392.83 | Stage2 pullback-breakout RSI=58 vol=7.2x ATR=22.24 |
| Stop hit — per-position SL triggered | 2024-08-06 00:00:00 | 377.14 | 342.05 | 397.45 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-10-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 00:00:00 | 399.70 | 362.08 | 390.55 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=13.06 |
| Stop hit — per-position SL triggered | 2024-10-21 00:00:00 | 380.11 | 363.48 | 390.34 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-11-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 00:00:00 | 378.85 | 363.88 | 368.89 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=13.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 00:00:00 | 406.43 | 365.50 | 380.10 | T1 booked 50% @ 406.43 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-05 00:00:00 | 402.60 | 2024-07-22 00:00:00 | 391.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest1 | 2024-07-26 00:00:00 | 410.50 | 2024-08-06 00:00:00 | 377.14 | STOP_HIT | 1.00 | -8.13% |
| BUY | retest1 | 2024-10-14 00:00:00 | 399.70 | 2024-10-21 00:00:00 | 380.11 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest1 | 2024-11-22 00:00:00 | 378.85 | 2024-11-29 00:00:00 | 406.43 | PARTIAL | 0.50 | 7.28% |

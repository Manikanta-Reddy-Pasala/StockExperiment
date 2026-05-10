# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 141.76
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 1.70% / 2.62%
- **Sum % (uncompounded):** 10.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.70% | 10.2% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.70% | 10.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.70% | 10.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 106.58 | 100.85 | 104.16 | Stage2 pullback-breakout RSI=56 vol=3.1x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 05:30:00 | 114.24 | 101.26 | 106.64 | T1 booked 50% @ 114.24 |
| Target hit | 2025-07-11 05:30:00 | 108.75 | 102.24 | 109.92 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 05:30:00 | 104.05 | 102.98 | 102.13 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 100.80 | 103.03 | 102.80 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-10-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 05:30:00 | 107.51 | 103.11 | 103.61 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 103.74 | 103.15 | 103.83 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2026-02-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 05:30:00 | 106.54 | 103.47 | 103.94 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 05:30:00 | 111.86 | 103.54 | 104.52 | T1 booked 50% @ 111.86 |
| Target hit | 2026-03-23 05:30:00 | 109.33 | 106.18 | 116.93 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-05-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 05:30:00 | 139.63 | 111.03 | 129.67 | Stage2 pullback-breakout RSI=66 vol=2.2x ATR=5.72 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 106.58 | 2025-06-30 05:30:00 | 114.24 | PARTIAL | 0.50 | 7.18% |
| BUY | retest1 | 2025-06-24 05:30:00 | 106.58 | 2025-07-11 05:30:00 | 108.75 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2025-09-17 05:30:00 | 104.05 | 2025-09-26 05:30:00 | 100.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2025-10-10 05:30:00 | 107.51 | 2025-10-14 05:30:00 | 103.74 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2026-02-23 05:30:00 | 106.54 | 2026-02-24 05:30:00 | 111.86 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-23 05:30:00 | 106.54 | 2026-03-23 05:30:00 | 109.33 | TARGET_HIT | 0.50 | 2.62% |

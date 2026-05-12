# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2024-09-04 00:00:00 → 2026-05-11 00:00:00 (416 bars)
- **Last close:** 136.31
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -0.84% / -3.12%
- **Sum % (uncompounded):** -4.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.84% | -4.2% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.84% | -4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.84% | -4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 00:00:00 | 115.85 | 101.15 | 106.64 | Stage2 pullback-breakout RSI=68 vol=3.3x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-07-11 00:00:00 | 109.85 | 102.13 | 109.92 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2025-09-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 00:00:00 | 104.05 | 102.91 | 102.13 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-09-26 00:00:00 | 100.80 | 102.97 | 102.80 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 00:00:00 | 107.51 | 103.05 | 103.61 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-10-14 00:00:00 | 103.74 | 103.09 | 103.83 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2026-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 00:00:00 | 106.54 | 103.45 | 103.94 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 00:00:00 | 111.86 | 103.51 | 104.52 | T1 booked 50% @ 111.86 |
| Target hit | 2026-03-23 00:00:00 | 109.33 | 106.16 | 116.93 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-05-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 00:00:00 | 139.63 | 111.02 | 129.67 | Stage2 pullback-breakout RSI=66 vol=2.2x ATR=5.72 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 00:00:00 | 115.85 | 2025-07-11 00:00:00 | 109.85 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest1 | 2025-09-17 00:00:00 | 104.05 | 2025-09-26 00:00:00 | 100.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2025-10-10 00:00:00 | 107.51 | 2025-10-14 00:00:00 | 103.74 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2026-02-23 00:00:00 | 106.54 | 2026-02-24 00:00:00 | 111.86 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-23 00:00:00 | 106.54 | 2026-03-23 00:00:00 | 109.33 | TARGET_HIT | 0.50 | 2.62% |

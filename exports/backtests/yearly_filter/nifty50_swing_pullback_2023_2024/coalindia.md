# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 455.85
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 8.94% / 3.09%
- **Sum % (uncompounded):** 62.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 8.94% | 62.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 8.94% | 62.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 8.94% | 62.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 00:00:00 | 240.35 | 228.86 | 230.93 | Stage2 pullback-breakout RSI=68 vol=2.5x ATR=3.90 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 234.49 | 228.92 | 231.28 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 236.90 | 229.40 | 231.17 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 244.23 | 229.58 | 232.76 | T1 booked 50% @ 244.23 |
| Target hit | 2024-01-18 00:00:00 | 375.65 | 295.35 | 375.97 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 00:00:00 | 398.80 | 297.24 | 378.77 | Stage2 pullback-breakout RSI=68 vol=2.0x ATR=10.15 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 383.58 | 298.02 | 378.44 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 412.80 | 300.89 | 382.98 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=12.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 437.94 | 306.47 | 396.34 | T1 booked 50% @ 437.94 |
| Target hit | 2024-02-21 00:00:00 | 432.95 | 323.32 | 437.54 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-05-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 00:00:00 | 474.60 | 368.52 | 449.82 | Stage2 pullback-breakout RSI=67 vol=4.8x ATR=11.98 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 456.62 | 369.44 | 450.85 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-01 00:00:00 | 240.35 | 2023-08-02 00:00:00 | 234.49 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest1 | 2023-09-01 00:00:00 | 236.90 | 2023-09-04 00:00:00 | 244.23 | PARTIAL | 0.50 | 3.09% |
| BUY | retest1 | 2023-09-01 00:00:00 | 236.90 | 2024-01-18 00:00:00 | 375.65 | TARGET_HIT | 0.50 | 58.57% |
| BUY | retest1 | 2024-01-20 00:00:00 | 398.80 | 2024-01-23 00:00:00 | 383.58 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest1 | 2024-01-29 00:00:00 | 412.80 | 2024-02-05 00:00:00 | 437.94 | PARTIAL | 0.50 | 6.09% |
| BUY | retest1 | 2024-01-29 00:00:00 | 412.80 | 2024-02-21 00:00:00 | 432.95 | TARGET_HIT | 0.50 | 4.88% |
| BUY | retest1 | 2024-05-03 00:00:00 | 474.60 | 2024-05-06 00:00:00 | 456.62 | STOP_HIT | 1.00 | -3.79% |

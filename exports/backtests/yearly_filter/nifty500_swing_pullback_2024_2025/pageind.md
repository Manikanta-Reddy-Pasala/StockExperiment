# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 36845.00
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 3
- **Avg / median % per leg:** 1.23% / 2.19%
- **Sum % (uncompounded):** 9.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 0 | 5 | 3 | 1.23% | 9.9% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 0 | 5 | 3 | 1.23% | 9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 5 | 3 | 1.23% | 9.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 00:00:00 | 40541.55 | 37408.72 | 39224.14 | Stage2 pullback-breakout RSI=66 vol=2.1x ATR=926.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 00:00:00 | 42393.59 | 37752.93 | 40450.38 | T1 booked 50% @ 42393.59 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 41429.50 | 37925.65 | 41005.48 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 42520.55 | 38481.74 | 41415.66 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=956.53 |
| Stop hit — per-position SL triggered | 2024-09-04 00:00:00 | 41085.76 | 38576.53 | 41481.93 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 00:00:00 | 43260.70 | 38729.99 | 41416.13 | Stage2 pullback-breakout RSI=64 vol=3.2x ATR=1011.80 |
| Stop hit — per-position SL triggered | 2024-09-25 00:00:00 | 41742.99 | 39085.25 | 42211.11 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-10-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 00:00:00 | 43935.40 | 39359.08 | 42274.78 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=1107.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 00:00:00 | 46150.50 | 39594.72 | 43319.47 | T1 booked 50% @ 46150.50 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 43935.40 | 39869.50 | 44028.85 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 45064.10 | 40271.43 | 43702.32 | Stage2 pullback-breakout RSI=61 vol=2.5x ATR=1111.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 00:00:00 | 47287.92 | 40348.36 | 44111.90 | T1 booked 50% @ 47287.92 |
| Stop hit — per-position SL triggered | 2024-11-18 00:00:00 | 45064.10 | 40621.09 | 44773.27 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 00:00:00 | 40541.55 | 2024-07-30 00:00:00 | 42393.59 | PARTIAL | 0.50 | 4.57% |
| BUY | retest1 | 2024-07-15 00:00:00 | 40541.55 | 2024-08-05 00:00:00 | 41429.50 | STOP_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2024-08-30 00:00:00 | 42520.55 | 2024-09-04 00:00:00 | 41085.76 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest1 | 2024-09-12 00:00:00 | 43260.70 | 2024-09-25 00:00:00 | 41742.99 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-10-09 00:00:00 | 43935.40 | 2024-10-15 00:00:00 | 46150.50 | PARTIAL | 0.50 | 5.04% |
| BUY | retest1 | 2024-10-09 00:00:00 | 43935.40 | 2024-10-22 00:00:00 | 43935.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-07 00:00:00 | 45064.10 | 2024-11-08 00:00:00 | 47287.92 | PARTIAL | 0.50 | 4.93% |
| BUY | retest1 | 2024-11-07 00:00:00 | 45064.10 | 2024-11-18 00:00:00 | 45064.10 | STOP_HIT | 0.50 | 0.00% |

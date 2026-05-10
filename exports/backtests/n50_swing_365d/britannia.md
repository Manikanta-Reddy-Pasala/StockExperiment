# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 5520.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.89% / 2.31%
- **Sum % (uncompounded):** 4.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.89% | 4.5% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.89% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.89% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 05:30:00 | 5765.50 | 5464.06 | 5572.26 | Stage2 pullback-breakout RSI=59 vol=3.0x ATR=142.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 05:30:00 | 6050.02 | 5488.33 | 5718.72 | T1 booked 50% @ 6050.02 |
| Target hit | 2025-09-23 05:30:00 | 5934.00 | 5566.93 | 6006.55 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 05:30:00 | 6013.50 | 5659.66 | 5930.61 | Stage2 pullback-breakout RSI=56 vol=3.6x ATR=124.36 |
| Stop hit — per-position SL triggered | 2025-11-11 05:30:00 | 5826.95 | 5672.08 | 5967.67 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2026-01-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 05:30:00 | 6185.00 | 5763.83 | 6027.24 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=107.06 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 6024.41 | 5766.51 | 6027.83 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 6019.00 | 5789.18 | 5893.64 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=136.50 |
| Stop hit — per-position SL triggered | 2026-02-25 05:30:00 | 6158.00 | 5820.36 | 6039.27 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-26 05:30:00 | 5765.50 | 2025-09-04 05:30:00 | 6050.02 | PARTIAL | 0.50 | 4.93% |
| BUY | retest1 | 2025-08-26 05:30:00 | 5765.50 | 2025-09-23 05:30:00 | 5934.00 | TARGET_HIT | 0.50 | 2.92% |
| BUY | retest1 | 2025-11-06 05:30:00 | 6013.50 | 2025-11-11 05:30:00 | 5826.95 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest1 | 2026-01-07 05:30:00 | 6185.00 | 2026-01-08 05:30:00 | 6024.41 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest1 | 2026-02-11 05:30:00 | 6019.00 | 2026-02-25 05:30:00 | 6158.00 | STOP_HIT | 1.00 | 2.31% |

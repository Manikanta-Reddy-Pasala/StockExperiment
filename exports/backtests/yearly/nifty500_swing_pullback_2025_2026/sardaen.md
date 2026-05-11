# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 589.15
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -3.20% / -3.68%
- **Sum % (uncompounded):** -16.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.20% | -16.0% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.20% | -16.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.20% | -16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 05:30:00 | 452.70 | 445.27 | 439.29 | Stage2 pullback-breakout RSI=58 vol=5.5x ATR=14.51 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 436.05 | 445.23 | 442.94 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 05:30:00 | 626.15 | 478.57 | 578.00 | Stage2 pullback-breakout RSI=69 vol=4.4x ATR=24.74 |
| Stop hit — per-position SL triggered | 2025-09-18 05:30:00 | 589.04 | 482.04 | 582.29 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 05:30:00 | 520.50 | 501.00 | 498.12 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=17.92 |
| Stop hit — per-position SL triggered | 2025-12-30 05:30:00 | 516.70 | 502.52 | 510.76 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2026-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 05:30:00 | 529.45 | 500.17 | 494.44 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=18.82 |
| Stop hit — per-position SL triggered | 2026-02-09 05:30:00 | 501.22 | 500.60 | 499.34 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 532.20 | 501.77 | 508.94 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=16.89 |
| Stop hit — per-position SL triggered | 2026-03-13 05:30:00 | 530.35 | 505.78 | 530.27 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-11 05:30:00 | 452.70 | 2025-07-25 05:30:00 | 436.05 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest1 | 2025-09-15 05:30:00 | 626.15 | 2025-09-18 05:30:00 | 589.04 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest1 | 2025-12-15 05:30:00 | 520.50 | 2025-12-30 05:30:00 | 516.70 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2026-02-05 05:30:00 | 529.45 | 2026-02-09 05:30:00 | 501.22 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest1 | 2026-02-25 05:30:00 | 532.20 | 2026-03-13 05:30:00 | 530.35 | STOP_HIT | 1.00 | -0.35% |

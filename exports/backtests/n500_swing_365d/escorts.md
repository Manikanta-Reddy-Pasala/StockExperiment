# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 3146.60
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
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -1.24% / -3.49%
- **Sum % (uncompounded):** -6.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.24% | -6.2% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.24% | -6.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.24% | -6.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 05:30:00 | 3644.00 | 3410.59 | 3504.79 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=100.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 05:30:00 | 3845.64 | 3417.78 | 3543.26 | T1 booked 50% @ 3845.64 |
| Stop hit — per-position SL triggered | 2025-09-05 05:30:00 | 3644.00 | 3420.49 | 3557.28 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-09-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 05:30:00 | 3794.70 | 3445.24 | 3645.76 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=111.67 |
| Stop hit — per-position SL triggered | 2025-09-23 05:30:00 | 3627.20 | 3452.61 | 3658.90 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-11-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 05:30:00 | 3847.80 | 3521.83 | 3659.58 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=98.31 |
| Stop hit — per-position SL triggered | 2025-12-05 05:30:00 | 3700.33 | 3536.67 | 3709.59 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2026-01-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 05:30:00 | 3820.60 | 3561.98 | 3705.83 | Stage2 pullback-breakout RSI=62 vol=3.2x ATR=88.84 |
| Stop hit — per-position SL triggered | 2026-01-14 05:30:00 | 3687.35 | 3585.28 | 3769.99 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-01 05:30:00 | 3644.00 | 2025-09-04 05:30:00 | 3845.64 | PARTIAL | 0.50 | 5.53% |
| BUY | retest1 | 2025-09-01 05:30:00 | 3644.00 | 2025-09-05 05:30:00 | 3644.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 05:30:00 | 3794.70 | 2025-09-23 05:30:00 | 3627.20 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest1 | 2025-11-27 05:30:00 | 3847.80 | 2025-12-05 05:30:00 | 3700.33 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest1 | 2026-01-01 05:30:00 | 3820.60 | 2026-01-14 05:30:00 | 3687.35 | STOP_HIT | 1.00 | -3.49% |

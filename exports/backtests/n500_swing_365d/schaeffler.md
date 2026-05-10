# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 4247.90
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.06% / 2.80%
- **Sum % (uncompounded):** 0.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.06% | 0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.06% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.06% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 4047.30 | 3651.87 | 3992.11 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=115.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 05:30:00 | 4277.70 | 3679.10 | 4031.27 | T1 booked 50% @ 4277.70 |
| Stop hit — per-position SL triggered | 2025-07-24 05:30:00 | 4160.50 | 3734.65 | 4152.98 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 4286.20 | 3838.76 | 4087.26 | Stage2 pullback-breakout RSI=67 vol=2.3x ATR=116.15 |
| Stop hit — per-position SL triggered | 2025-10-10 05:30:00 | 4111.98 | 3856.52 | 4130.56 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 4278.70 | 3870.71 | 4010.85 | Stage2 pullback-breakout RSI=66 vol=8.9x ATR=119.15 |
| Stop hit — per-position SL triggered | 2025-11-07 05:30:00 | 4099.97 | 3877.86 | 4036.76 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 4047.30 | 2025-07-09 05:30:00 | 4277.70 | PARTIAL | 0.50 | 5.69% |
| BUY | retest1 | 2025-06-30 05:30:00 | 4047.30 | 2025-07-24 05:30:00 | 4160.50 | STOP_HIT | 0.50 | 2.80% |
| BUY | retest1 | 2025-10-03 05:30:00 | 4286.20 | 2025-10-10 05:30:00 | 4111.98 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest1 | 2025-11-03 05:30:00 | 4278.70 | 2025-11-07 05:30:00 | 4099.97 | STOP_HIT | 1.00 | -4.18% |

# Larsen & Toubro Ltd. (LT)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 3974.50
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
- **Avg / median % per leg:** 0.71% / 0.78%
- **Sum % (uncompounded):** 2.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.71% | 2.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.71% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.71% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 05:30:00 | 3665.10 | 3523.62 | 3526.57 | Stage2 pullback-breakout RSI=63 vol=4.9x ATR=62.25 |
| Stop hit — per-position SL triggered | 2025-08-13 05:30:00 | 3693.70 | 3535.07 | 3604.60 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 3667.80 | 3545.28 | 3584.57 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=53.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 05:30:00 | 3775.58 | 3555.22 | 3634.78 | T1 booked 50% @ 3775.58 |
| Stop hit — per-position SL triggered | 2025-09-30 05:30:00 | 3667.80 | 3557.57 | 3641.70 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 4074.10 | 3710.65 | 4008.99 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=63.22 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 4038.70 | 3743.81 | 4039.13 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-30 05:30:00 | 3665.10 | 2025-08-13 05:30:00 | 3693.70 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest1 | 2025-09-16 05:30:00 | 3667.80 | 2025-09-26 05:30:00 | 3775.58 | PARTIAL | 0.50 | 2.94% |
| BUY | retest1 | 2025-09-16 05:30:00 | 3667.80 | 2025-09-30 05:30:00 | 3667.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 05:30:00 | 4074.10 | 2025-12-29 05:30:00 | 4038.70 | STOP_HIT | 1.00 | -0.87% |

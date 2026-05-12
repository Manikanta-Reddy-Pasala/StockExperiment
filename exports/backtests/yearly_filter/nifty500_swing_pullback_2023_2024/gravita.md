# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1763.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 4.00% / 2.47%
- **Sum % (uncompounded):** 19.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 0 | 3 | 2 | 4.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 3 | 2 | 4.00% | 20.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 4.00% | 20.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 00:00:00 | 627.90 | 493.21 | 609.92 | Stage2 pullback-breakout RSI=63 vol=3.0x ATR=17.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 663.61 | 496.11 | 615.55 | T1 booked 50% @ 663.61 |
| Stop hit — per-position SL triggered | 2023-07-24 00:00:00 | 627.90 | 503.77 | 630.04 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 00:00:00 | 679.10 | 507.15 | 638.12 | Stage2 pullback-breakout RSI=69 vol=1.6x ATR=24.38 |
| Stop hit — per-position SL triggered | 2023-08-09 00:00:00 | 695.85 | 523.72 | 667.28 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 992.55 | 860.26 | 876.49 | Stage2 pullback-breakout RSI=65 vol=6.7x ATR=52.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 00:00:00 | 1098.39 | 869.36 | 945.47 | T1 booked 50% @ 1098.39 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 1004.05 | 878.10 | 984.54 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-13 00:00:00 | 627.90 | 2023-07-17 00:00:00 | 663.61 | PARTIAL | 0.50 | 5.69% |
| BUY | retest1 | 2023-07-13 00:00:00 | 627.90 | 2023-07-24 00:00:00 | 627.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-26 00:00:00 | 679.10 | 2023-08-09 00:00:00 | 695.85 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest1 | 2024-03-27 00:00:00 | 992.55 | 2024-04-04 00:00:00 | 1098.39 | PARTIAL | 0.50 | 10.66% |
| BUY | retest1 | 2024-03-27 00:00:00 | 992.55 | 2024-04-12 00:00:00 | 1004.05 | STOP_HIT | 0.50 | 1.16% |

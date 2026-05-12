# TCS (TCS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 2394.40
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.76% / -2.11%
- **Sum % (uncompounded):** -7.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.76% | -7.0% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.76% | -7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.76% | -7.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 00:00:00 | 3638.35 | 3353.52 | 3553.51 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=58.54 |
| Stop hit — per-position SL triggered | 2023-10-12 00:00:00 | 3550.55 | 3360.61 | 3563.20 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 3497.85 | 3369.50 | 3403.76 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=52.96 |
| Stop hit — per-position SL triggered | 2023-12-01 00:00:00 | 3511.65 | 3382.00 | 3464.36 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 3882.80 | 3469.01 | 3732.96 | Stage2 pullback-breakout RSI=65 vol=2.9x ATR=74.40 |
| Stop hit — per-position SL triggered | 2024-01-29 00:00:00 | 3801.00 | 3506.77 | 3811.84 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 00:00:00 | 4192.25 | 3655.38 | 4077.36 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=81.01 |
| Stop hit — per-position SL triggered | 2024-03-19 00:00:00 | 4070.74 | 3679.03 | 4099.12 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-09 00:00:00 | 3638.35 | 2023-10-12 00:00:00 | 3550.55 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest1 | 2023-11-16 00:00:00 | 3497.85 | 2023-12-01 00:00:00 | 3511.65 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest1 | 2024-01-12 00:00:00 | 3882.80 | 2024-01-29 00:00:00 | 3801.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2024-03-12 00:00:00 | 4192.25 | 2024-03-19 00:00:00 | 4070.74 | STOP_HIT | 1.00 | -2.90% |

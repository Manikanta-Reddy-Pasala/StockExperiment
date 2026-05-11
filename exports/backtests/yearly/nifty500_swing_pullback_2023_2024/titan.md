# Titan Company Ltd. (TITAN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 4509.00
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -0.26% / 1.04%
- **Sum % (uncompounded):** -1.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.26% | -1.0% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.26% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.26% | -1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 00:00:00 | 3017.25 | 2743.42 | 2977.50 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=54.39 |
| Stop hit — per-position SL triggered | 2023-08-28 00:00:00 | 3048.55 | 2773.05 | 3028.11 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 3274.50 | 2937.17 | 3212.71 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=60.51 |
| Stop hit — per-position SL triggered | 2023-11-17 00:00:00 | 3338.85 | 2970.84 | 3265.31 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-03-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 00:00:00 | 3765.90 | 3314.89 | 3654.56 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=82.85 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 3641.63 | 3348.01 | 3701.48 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 3706.70 | 3366.62 | 3663.71 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=74.09 |
| Stop hit — per-position SL triggered | 2024-04-09 00:00:00 | 3679.35 | 3402.43 | 3712.31 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-11 00:00:00 | 3017.25 | 2023-08-28 00:00:00 | 3048.55 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest1 | 2023-11-03 00:00:00 | 3274.50 | 2023-11-17 00:00:00 | 3338.85 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest1 | 2024-03-01 00:00:00 | 3765.90 | 2024-03-13 00:00:00 | 3641.63 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest1 | 2024-03-22 00:00:00 | 3706.70 | 2024-04-09 00:00:00 | 3679.35 | STOP_HIT | 1.00 | -0.74% |

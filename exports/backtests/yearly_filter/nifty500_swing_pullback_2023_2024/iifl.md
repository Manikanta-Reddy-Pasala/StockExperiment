# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 453.40
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
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.43% / 0.00%
- **Sum % (uncompounded):** -2.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.43% | -2.2% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.43% | -2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.43% | -2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 590.88 | 465.04 | 558.47 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=20.80 |
| Stop hit — per-position SL triggered | 2023-08-30 00:00:00 | 559.68 | 471.07 | 563.02 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 589.32 | 472.25 | 565.52 | Stage2 pullback-breakout RSI=61 vol=1.9x ATR=20.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 629.46 | 479.83 | 575.64 | T1 booked 50% @ 629.46 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 589.32 | 480.65 | 574.32 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 614.34 | 529.20 | 594.23 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=20.41 |
| Stop hit — per-position SL triggered | 2023-12-18 00:00:00 | 624.68 | 538.33 | 615.11 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 00:00:00 | 604.48 | 560.37 | 584.32 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=21.62 |
| Stop hit — per-position SL triggered | 2024-03-05 00:00:00 | 572.05 | 560.04 | 574.09 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-22 00:00:00 | 590.88 | 2023-08-30 00:00:00 | 559.68 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest1 | 2023-08-31 00:00:00 | 589.32 | 2023-09-11 00:00:00 | 629.46 | PARTIAL | 0.50 | 6.81% |
| BUY | retest1 | 2023-08-31 00:00:00 | 589.32 | 2023-09-12 00:00:00 | 589.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-04 00:00:00 | 614.34 | 2023-12-18 00:00:00 | 624.68 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest1 | 2024-03-01 00:00:00 | 604.48 | 2024-03-05 00:00:00 | 572.05 | STOP_HIT | 1.00 | -5.37% |

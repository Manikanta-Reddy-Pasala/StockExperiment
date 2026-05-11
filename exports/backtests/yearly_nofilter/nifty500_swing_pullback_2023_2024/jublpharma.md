# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1014.40
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 2.73% / 6.39%
- **Sum % (uncompounded):** 19.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 2.73% | 19.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 2.73% | 19.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 3 | 3 | 2.73% | 19.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 00:00:00 | 392.20 | 350.67 | 379.71 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=12.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 00:00:00 | 417.25 | 352.66 | 390.05 | T1 booked 50% @ 417.25 |
| Target hit | 2023-09-12 00:00:00 | 432.80 | 372.74 | 447.97 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 453.05 | 389.70 | 420.52 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=17.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 00:00:00 | 487.46 | 393.43 | 438.62 | T1 booked 50% @ 487.46 |
| Stop hit — per-position SL triggered | 2023-12-11 00:00:00 | 453.05 | 394.06 | 440.31 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 592.25 | 442.72 | 563.45 | Stage2 pullback-breakout RSI=62 vol=6.8x ATR=24.75 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 555.12 | 449.11 | 567.06 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-02-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 00:00:00 | 610.25 | 454.65 | 575.05 | Stage2 pullback-breakout RSI=64 vol=2.6x ATR=25.73 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 571.65 | 460.44 | 583.67 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 636.90 | 489.27 | 579.69 | Stage2 pullback-breakout RSI=69 vol=8.0x ATR=23.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 00:00:00 | 683.73 | 502.21 | 624.21 | T1 booked 50% @ 683.73 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-07 00:00:00 | 392.20 | 2023-08-10 00:00:00 | 417.25 | PARTIAL | 0.50 | 6.39% |
| BUY | retest1 | 2023-08-07 00:00:00 | 392.20 | 2023-09-12 00:00:00 | 432.80 | TARGET_HIT | 0.50 | 10.35% |
| BUY | retest1 | 2023-12-01 00:00:00 | 453.05 | 2023-12-08 00:00:00 | 487.46 | PARTIAL | 0.50 | 7.59% |
| BUY | retest1 | 2023-12-01 00:00:00 | 453.05 | 2023-12-11 00:00:00 | 453.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 00:00:00 | 592.25 | 2024-02-09 00:00:00 | 555.12 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest1 | 2024-02-15 00:00:00 | 610.25 | 2024-02-21 00:00:00 | 571.65 | STOP_HIT | 1.00 | -6.33% |
| BUY | retest1 | 2024-04-04 00:00:00 | 636.90 | 2024-04-18 00:00:00 | 683.73 | PARTIAL | 0.50 | 7.35% |

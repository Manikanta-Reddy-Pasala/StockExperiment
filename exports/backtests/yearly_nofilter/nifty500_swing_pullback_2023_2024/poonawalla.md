# Poonawalla Fincorp Ltd. (POONAWALLA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 462.30
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** 2.24% / 0.85%
- **Sum % (uncompounded):** 15.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 5 | 1 | 2.24% | 15.7% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 5 | 1 | 2.24% | 15.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 5 | 1 | 2.24% | 15.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 00:00:00 | 358.75 | 314.75 | 344.02 | Stage2 pullback-breakout RSI=63 vol=3.4x ATR=9.47 |
| Stop hit — per-position SL triggered | 2023-07-19 00:00:00 | 368.95 | 320.17 | 359.55 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 388.95 | 352.86 | 384.54 | Stage2 pullback-breakout RSI=51 vol=1.6x ATR=11.09 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 372.32 | 354.89 | 379.76 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 388.95 | 355.26 | 364.63 | Stage2 pullback-breakout RSI=63 vol=3.6x ATR=13.36 |
| Stop hit — per-position SL triggered | 2023-11-17 00:00:00 | 368.91 | 357.42 | 375.62 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 409.30 | 358.47 | 374.58 | Stage2 pullback-breakout RSI=67 vol=3.0x ATR=13.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 00:00:00 | 435.77 | 364.24 | 403.61 | T1 booked 50% @ 435.77 |
| Target hit | 2024-02-02 00:00:00 | 467.70 | 393.52 | 473.88 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 00:00:00 | 471.95 | 411.29 | 462.64 | Stage2 pullback-breakout RSI=53 vol=3.5x ATR=17.71 |
| Stop hit — per-position SL triggered | 2024-03-27 00:00:00 | 475.25 | 416.65 | 466.42 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 494.60 | 419.03 | 470.41 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=17.57 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 498.80 | 426.06 | 484.80 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 00:00:00 | 358.75 | 2023-07-19 00:00:00 | 368.95 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest1 | 2023-10-06 00:00:00 | 388.95 | 2023-10-19 00:00:00 | 372.32 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2023-11-08 00:00:00 | 388.95 | 2023-11-17 00:00:00 | 368.91 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2023-11-30 00:00:00 | 409.30 | 2023-12-14 00:00:00 | 435.77 | PARTIAL | 0.50 | 6.47% |
| BUY | retest1 | 2023-11-30 00:00:00 | 409.30 | 2024-02-02 00:00:00 | 467.70 | TARGET_HIT | 0.50 | 14.27% |
| BUY | retest1 | 2024-03-12 00:00:00 | 471.95 | 2024-03-27 00:00:00 | 475.25 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest1 | 2024-04-03 00:00:00 | 494.60 | 2024-04-19 00:00:00 | 498.80 | STOP_HIT | 1.00 | 0.85% |

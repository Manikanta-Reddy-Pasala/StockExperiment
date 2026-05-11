# Sun TV Network Ltd. (SUNTV)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 573.55
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.35% / -0.25%
- **Sum % (uncompounded):** 2.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.35% | 2.5% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.35% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.35% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 568.25 | 479.92 | 539.17 | Stage2 pullback-breakout RSI=68 vol=3.6x ATR=16.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 00:00:00 | 600.56 | 482.17 | 549.10 | T1 booked 50% @ 600.56 |
| Target hit | 2023-09-18 00:00:00 | 593.60 | 500.94 | 595.37 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 622.15 | 509.19 | 597.24 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=17.13 |
| Stop hit — per-position SL triggered | 2023-10-16 00:00:00 | 596.46 | 519.35 | 616.05 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-10-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 00:00:00 | 661.80 | 521.98 | 622.68 | Stage2 pullback-breakout RSI=68 vol=2.8x ATR=19.57 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 632.44 | 525.69 | 628.85 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 672.25 | 542.41 | 645.07 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=16.11 |
| Stop hit — per-position SL triggered | 2023-11-29 00:00:00 | 670.60 | 554.50 | 660.53 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 630.70 | 605.77 | 609.17 | Stage2 pullback-breakout RSI=57 vol=4.7x ATR=18.36 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 603.15 | 605.92 | 609.87 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 634.95 | 606.43 | 612.51 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=17.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 00:00:00 | 669.86 | 610.48 | 636.33 | T1 booked 50% @ 669.86 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-23 00:00:00 | 568.25 | 2023-08-25 00:00:00 | 600.56 | PARTIAL | 0.50 | 5.69% |
| BUY | retest1 | 2023-08-23 00:00:00 | 568.25 | 2023-09-18 00:00:00 | 593.60 | TARGET_HIT | 0.50 | 4.46% |
| BUY | retest1 | 2023-10-03 00:00:00 | 622.15 | 2023-10-16 00:00:00 | 596.46 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest1 | 2023-10-18 00:00:00 | 661.80 | 2023-10-23 00:00:00 | 632.44 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest1 | 2023-11-13 00:00:00 | 672.25 | 2023-11-29 00:00:00 | 670.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-04-10 00:00:00 | 630.70 | 2024-04-15 00:00:00 | 603.15 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest1 | 2024-04-25 00:00:00 | 634.95 | 2024-05-09 00:00:00 | 669.86 | PARTIAL | 0.50 | 5.50% |

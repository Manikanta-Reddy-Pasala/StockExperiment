# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1507.30
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 7.82% / 0.00%
- **Sum % (uncompounded):** 54.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 7.82% | 54.7% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 7.82% | 54.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 7.82% | 54.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 590.30 | 492.17 | 562.35 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=20.29 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 559.87 | 499.73 | 572.48 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 610.50 | 507.81 | 570.90 | Stage2 pullback-breakout RSI=68 vol=3.7x ATR=17.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 00:00:00 | 645.58 | 509.16 | 577.87 | T1 booked 50% @ 645.58 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 610.50 | 519.35 | 611.55 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-10-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 00:00:00 | 671.30 | 532.73 | 616.76 | Stage2 pullback-breakout RSI=69 vol=6.5x ATR=23.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 00:00:00 | 718.47 | 537.34 | 635.91 | T1 booked 50% @ 718.47 |
| Target hit | 2023-12-21 00:00:00 | 1064.15 | 691.83 | 1067.42 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 1230.40 | 882.38 | 1197.35 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=67.29 |
| Stop hit — per-position SL triggered | 2024-03-07 00:00:00 | 1181.15 | 912.20 | 1194.92 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 1313.65 | 941.90 | 1158.21 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=64.95 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 1216.23 | 963.84 | 1209.59 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-28 00:00:00 | 590.30 | 2023-08-10 00:00:00 | 559.87 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2023-08-30 00:00:00 | 610.50 | 2023-08-31 00:00:00 | 645.58 | PARTIAL | 0.50 | 5.75% |
| BUY | retest1 | 2023-08-30 00:00:00 | 610.50 | 2023-09-12 00:00:00 | 610.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-05 00:00:00 | 671.30 | 2023-10-10 00:00:00 | 718.47 | PARTIAL | 0.50 | 7.03% |
| BUY | retest1 | 2023-10-05 00:00:00 | 671.30 | 2023-12-21 00:00:00 | 1064.15 | TARGET_HIT | 0.50 | 58.52% |
| BUY | retest1 | 2024-02-23 00:00:00 | 1230.40 | 2024-03-07 00:00:00 | 1181.15 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest1 | 2024-04-02 00:00:00 | 1313.65 | 2024-04-12 00:00:00 | 1216.23 | STOP_HIT | 1.00 | -7.42% |

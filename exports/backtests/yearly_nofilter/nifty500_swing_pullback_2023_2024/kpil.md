# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1275.70
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
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.97% / 1.51%
- **Sum % (uncompounded):** 13.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 4 | 2 | 1.97% | 13.8% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 4 | 2 | 1.97% | 13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 4 | 2 | 1.97% | 13.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 555.60 | 507.64 | 533.48 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=14.83 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 562.60 | 512.19 | 548.14 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 660.45 | 542.69 | 632.22 | Stage2 pullback-breakout RSI=68 vol=1.6x ATR=20.07 |
| Stop hit — per-position SL triggered | 2023-09-14 00:00:00 | 656.30 | 554.17 | 650.54 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 649.90 | 562.89 | 641.29 | Stage2 pullback-breakout RSI=55 vol=7.3x ATR=21.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 00:00:00 | 692.68 | 570.67 | 650.30 | T1 booked 50% @ 692.68 |
| Target hit | 2023-10-25 00:00:00 | 666.30 | 577.81 | 669.32 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 696.15 | 593.80 | 652.52 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=22.65 |
| Stop hit — per-position SL triggered | 2023-12-18 00:00:00 | 668.40 | 602.55 | 671.73 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-12-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 00:00:00 | 709.20 | 606.50 | 663.46 | Stage2 pullback-breakout RSI=65 vol=11.7x ATR=23.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 00:00:00 | 755.48 | 612.24 | 688.19 | T1 booked 50% @ 755.48 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 719.90 | 622.90 | 716.20 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 555.60 | 2023-07-17 00:00:00 | 562.60 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest1 | 2023-08-31 00:00:00 | 660.45 | 2023-09-14 00:00:00 | 656.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2023-10-03 00:00:00 | 649.90 | 2023-10-16 00:00:00 | 692.68 | PARTIAL | 0.50 | 6.58% |
| BUY | retest1 | 2023-10-03 00:00:00 | 649.90 | 2023-10-25 00:00:00 | 666.30 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2023-12-04 00:00:00 | 696.15 | 2023-12-18 00:00:00 | 668.40 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest1 | 2023-12-29 00:00:00 | 709.20 | 2024-01-05 00:00:00 | 755.48 | PARTIAL | 0.50 | 6.53% |
| BUY | retest1 | 2023-12-29 00:00:00 | 709.20 | 2024-01-18 00:00:00 | 719.90 | STOP_HIT | 0.50 | 1.51% |

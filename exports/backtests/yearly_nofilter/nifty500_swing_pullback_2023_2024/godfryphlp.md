# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 2424.80
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 2
- **Avg / median % per leg:** 2.61% / -1.49%
- **Sum % (uncompounded):** 28.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 2 | 7 | 2 | 2.61% | 28.7% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 2 | 7 | 2 | 2.61% | 28.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 4 | 36.4% | 2 | 7 | 2 | 2.61% | 28.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 00:00:00 | 574.52 | 551.79 | 557.25 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=12.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 00:00:00 | 599.12 | 552.63 | 564.76 | T1 booked 50% @ 599.12 |
| Target hit | 2023-09-05 00:00:00 | 699.58 | 589.17 | 701.81 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 00:00:00 | 710.83 | 601.89 | 695.02 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=20.97 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 679.37 | 608.09 | 701.18 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 00:00:00 | 754.57 | 612.45 | 707.61 | Stage2 pullback-breakout RSI=66 vol=9.1x ATR=27.90 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 712.72 | 623.40 | 726.28 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-11-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 00:00:00 | 786.92 | 629.86 | 732.17 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=34.04 |
| Stop hit — per-position SL triggered | 2023-11-03 00:00:00 | 735.86 | 632.23 | 735.03 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 700.60 | 642.86 | 694.60 | Stage2 pullback-breakout RSI=51 vol=1.9x ATR=18.98 |
| Stop hit — per-position SL triggered | 2023-12-18 00:00:00 | 690.13 | 648.09 | 696.06 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 717.75 | 653.15 | 699.99 | Stage2 pullback-breakout RSI=59 vol=3.9x ATR=17.50 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 710.48 | 660.52 | 718.58 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 753.72 | 665.24 | 725.59 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=21.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 00:00:00 | 796.68 | 668.60 | 739.92 | T1 booked 50% @ 796.68 |
| Target hit | 2024-03-13 00:00:00 | 977.52 | 736.32 | 983.62 | Trail-exit close<EMA20 |

### Cycle 8 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 1117.37 | 769.40 | 1020.00 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=56.07 |
| Stop hit — per-position SL triggered | 2024-04-10 00:00:00 | 1033.26 | 786.68 | 1039.46 | SL hit (bars_held=6) |

### Cycle 9 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 1149.90 | 807.90 | 1040.74 | Stage2 pullback-breakout RSI=67 vol=6.0x ATR=47.94 |
| Stop hit — per-position SL triggered | 2024-05-02 00:00:00 | 1077.99 | 819.55 | 1061.90 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-26 00:00:00 | 574.52 | 2023-07-27 00:00:00 | 599.12 | PARTIAL | 0.50 | 4.28% |
| BUY | retest1 | 2023-07-26 00:00:00 | 574.52 | 2023-09-05 00:00:00 | 699.58 | TARGET_HIT | 0.50 | 21.77% |
| BUY | retest1 | 2023-09-25 00:00:00 | 710.83 | 2023-10-04 00:00:00 | 679.37 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2023-10-10 00:00:00 | 754.57 | 2023-10-23 00:00:00 | 712.72 | STOP_HIT | 1.00 | -5.55% |
| BUY | retest1 | 2023-11-01 00:00:00 | 786.92 | 2023-11-03 00:00:00 | 735.86 | STOP_HIT | 1.00 | -6.49% |
| BUY | retest1 | 2023-12-04 00:00:00 | 700.60 | 2023-12-18 00:00:00 | 690.13 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2024-01-02 00:00:00 | 717.75 | 2024-01-16 00:00:00 | 710.48 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest1 | 2024-01-25 00:00:00 | 753.72 | 2024-01-31 00:00:00 | 796.68 | PARTIAL | 0.50 | 5.70% |
| BUY | retest1 | 2024-01-25 00:00:00 | 753.72 | 2024-03-13 00:00:00 | 977.52 | TARGET_HIT | 0.50 | 29.69% |
| BUY | retest1 | 2024-04-02 00:00:00 | 1117.37 | 2024-04-10 00:00:00 | 1033.26 | STOP_HIT | 1.00 | -7.53% |
| BUY | retest1 | 2024-04-25 00:00:00 | 1149.90 | 2024-05-02 00:00:00 | 1077.99 | STOP_HIT | 1.00 | -6.25% |

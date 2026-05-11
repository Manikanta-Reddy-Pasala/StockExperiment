# Lodha Developers Ltd. (LODHA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 936.35
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
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 1.63% / 2.21%
- **Sum % (uncompounded):** 13.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 1 | 5 | 2 | 1.63% | 13.0% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 5 | 2 | 1.63% | 13.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 5 | 2 | 1.63% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 00:00:00 | 721.40 | 546.14 | 684.27 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=29.32 |
| Stop hit — per-position SL triggered | 2023-08-07 00:00:00 | 726.75 | 563.82 | 713.32 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 701.75 | 585.45 | 688.24 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=27.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 756.65 | 591.61 | 706.59 | T1 booked 50% @ 756.65 |
| Target hit | 2023-09-22 00:00:00 | 740.60 | 607.86 | 748.24 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 813.95 | 627.68 | 773.67 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=34.99 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 761.46 | 642.87 | 789.91 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 850.25 | 650.76 | 787.57 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=37.17 |
| Stop hit — per-position SL triggered | 2023-11-16 00:00:00 | 869.00 | 670.01 | 829.69 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-02-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 00:00:00 | 1117.55 | 842.07 | 1085.31 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=53.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 00:00:00 | 1223.65 | 880.80 | 1145.25 | T1 booked 50% @ 1223.65 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 1117.55 | 886.15 | 1146.01 | SL hit (bars_held=15) |

### Cycle 6 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 1235.15 | 946.06 | 1172.66 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=52.12 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 1156.97 | 966.98 | 1196.22 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-24 00:00:00 | 721.40 | 2023-08-07 00:00:00 | 726.75 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest1 | 2023-09-04 00:00:00 | 701.75 | 2023-09-08 00:00:00 | 756.65 | PARTIAL | 0.50 | 7.82% |
| BUY | retest1 | 2023-09-04 00:00:00 | 701.75 | 2023-09-22 00:00:00 | 740.60 | TARGET_HIT | 0.50 | 5.54% |
| BUY | retest1 | 2023-10-11 00:00:00 | 813.95 | 2023-10-25 00:00:00 | 761.46 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest1 | 2023-11-02 00:00:00 | 850.25 | 2023-11-16 00:00:00 | 869.00 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest1 | 2024-02-20 00:00:00 | 1117.55 | 2024-03-07 00:00:00 | 1223.65 | PARTIAL | 0.50 | 9.49% |
| BUY | retest1 | 2024-02-20 00:00:00 | 1117.55 | 2024-03-12 00:00:00 | 1117.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 00:00:00 | 1235.15 | 2024-05-06 00:00:00 | 1156.97 | STOP_HIT | 1.00 | -6.33% |
